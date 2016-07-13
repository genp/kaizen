import numpy as np
import skimage.feature
import skimage.color
import skimage.transform
import os

#set logging level to 2 to suppress caffe output
os.environ['GLOG_minloglevel'] = '2' 

import caffe
import tempfile
import re

class BaseFeature:
    def extract_many(self, img):
        codes = np.array([self.extract(i) for i in img])
        return codes

    def reduce(self, codes, type = "subsample", output_dim = 1024):
        '''
        "codes" should be an array of codes for either a single or multiple images in the form of:
        (N, c) where "N" is the number of images and "c" is the array of codes.  Can also take in input of the form:
        (c,) where "c" is the array of codes

        "type" indicates the process to perform on the given feature.
        Currently supported operations: subsample, normalization

        "output_dim" is the number of dimensions requested for output of a dimensionality reduction operation.
        Not needed for non dimensionality reduction operations (ie "normalization")
        '''

        if type == "subsample":
            if output_dim < codes.shape[1]:
                return codes[0:output_dim]
            else:
                raise ValueError('output_dim is larger than the codes! ')

        elif type == "normalization":

        #subsample
        #normalization
        #power norm  ?

class ColorHist(BaseFeature):
    def set_params(self, bins=4):
        self.bins = bins

    def extract(self, img):
        pixels = np.reshape(img, (img.shape[0]*img.shape[1],-1))
        hist,e = np.histogramdd(pixels, bins=self.bins, range=3*[[0,255]], normed=True)
        hist = np.reshape(hist, (-1)) # Make it 1-D
        return hist


class HoGDalal(BaseFeature):
    def set_params(self, ori=9, px_per_cell=(8,8), cells_per_block=(2,2), window_size=40):
        self.ori = ori
        self.px_per_cell = px_per_cell
        self.cells_per_block = cells_per_block
        self.window_size = window_size

    def extract(self, img):
        flat_img = flatten(img)
        flat_img = skimage.transform.resize(img[:,:,1], (self.window_size, self.window_size))
        hog_feat = skimage.feature.hog(flat_img, orientations=self.ori, pixels_per_cell=self.px_per_cell,
                                       cells_per_block=self.cells_per_block)
        hog_feat = np.reshape(hog_feat, (-1))
        return hog_feat

class TinyImage(BaseFeature):
    def set_params(self, flatten=False):
        self.flatten = flatten

    def extract(self, img):
        if self.flatten:
            img = flatten(img)

        tiny = skimage.transform.resize(img, (32,32))
        tiny = np.reshape(tiny, (-1))
        return tiny


#Darius
#CNN code may not work on Multi-GPU machines.
#
class CNN:

    max_batch_size = 500
    net = {}

    def initialize_cnn(self, batch_size, network):
        temp = tempfile.NamedTemporaryFile()

        def_path = "caffemodels/" + self.model +"/train.prototxt"
        weight_path = "caffemodels/" + self.model + "/weights.caffemodel"
        #go through and edit batch size
        arch = open(def_path,'r').readlines()
        for i in range(len(arch)):
            if "batch_size" in arch[i]:
                arch[i] = re.sub('\d+',str(batch_size),arch[i])
            if "height" in arch[i]:
                self.h = re.findall('\d+',arch[i])[0]
            if "width" in arch[i]:
                self.w = re.findall('\d+',arch[i])[0]
        temp.writelines(arch)
        temp.seek(0)
        # self.net[network] = caffe.Net(str(temp.name),str(weight_path),caffe.TEST)
        # self.transformer = caffe.io.Transformer({'data': self.net[network].blobs['data'].data.shape})
        # self.transformer.set_transpose('data', self.transpose)
        # self.transformer.set_channel_swap('data',self.channel_swap)

        temp.close()

    def set_params(self, model = "caffenet", layer_name = "fc7", transpose = (2,0,1), channel_swap = (2,1,0), initialize = True):
        '''
        Parameters
        ------------
        "model" is the folder name where the model specs and weights live. 
        ie model = "VGG", "GoogleNet", "BVLC_Reference_Caffenet"
        
        "layer_name" is the layer name used for extraction 
        ie layer_name = "fc7" (for VGG)
        
        see below for better idea of what "transpose" and "channel_swap" are used for
        http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

        set "initialize" to False when using extract_many.  Initialize makes single-patch feature extraction
        significantly faster
        '''
        self.model = model
        self.layer_name = layer_name
        self.transpose = transpose
        self.channel_swap = channel_swap

        if initialize:
            self.initialize_cnn(1,"one")
            self.initialize_cnn(self.max_batch_size,"many")
    #assume that we're getting a single image
    #Img comes in format (x,y,c)
    def extract(self, img):
        
        img = img[0:int(self.w),0:int(self.h)]
        img = self.transformer.preprocess('data',img)
        if len(img.shape) == 3:
            img = np.expand_dims(img,axis=0)
        self.net["one"].set_input_arrays(img, np.array([1],dtype=np.float32))
        p = self.net["one"].forward()
        feat = self.net["one"].blobs[self.layer_name].data[...].reshape(-1)
        feat = np.reshape(feat, (-1))
        return feat
    
    #expecting an array of images
    def extract_many(self, img):

        codes = np.array([])
        if img.shape[0] > self.max_batch_size:
            print 'exceeded max batch size. splitting into minibatches'
            self.initialize_cnn(self.max_batch_size,"many")
            for i in range(int(np.round(img.shape[0]/self.max_batch_size))):
                print 'minibatch: ' + str(i)
                tim = img[i*500:(i+1)*500,0:int(self.w),0:int(self.h)]

                #Lots of repeated code
                tim = np.array([self.transformer.preprocess('data',i) for i in tim])
                self.net["many"].set_input_arrays(tim, np.ones(self.max_batch_size,dtype=np.float32))
                p = self.net["many"].forward()
                codes = np.append(codes,self.net["many"].blobs[self.layer_name].data[...])
            if np.round(img.shape[0]/self.max_batch_size) * self.max_batch_size < img.shape[0]:
                mult = np.round(img.shape[0]/self.max_batch_size) * self.max_batch_size
                print 'final minibatch'
                self.initialize_cnn(img.shape[0]-mult,"many")
                tim = img[mult:img.shape[0],0:int(self.w),0:int(self.h)]
                #Lots of repeated code
                tim = np.array([self.transformer.preprocess('data',i) for i in tim])
                self.net["many"].set_input_arrays(tim, np.ones(img.shape[0]-mult,dtype=np.float32))
                p = self.net["many"].forward()
                codes = np.append(codes,self.net["many"].blobs[self.layer_name].data[...])
            codes = codes.reshape(np.append(-1,self.net["many"].blobs[self.layer_name].data.shape[1:]))
        else:
            self.initialize_cnn(img.shape[0],"many")
            img = img[:,0:int(self.w),0:int(self.h)]
            img = np.array([self.transformer.preprocess('data',i) for i in img])
            self.net["many"].set_input_arrays(img, np.ones(img.shape[0],dtype=np.float32))
            p = self.net["many"].forward()
            codes = self.net["many"].blobs[self.layer_name].data[...]
        return codes

    def extract_many_pad(self, img):
        
        codes = np.array([])
        if img.shape[0] > self.max_batch_size:
            print 'exceeded max batch size. splitting into minibatches'
            self.initialize_cnn(self.max_batch_size,"many")
            mult = np.round(img.shape[0]/self.max_batch_size) * self.max_batch_size

            for i in range(int(np.round(img.shape[0]/self.max_batch_size))):
                print 'minibatch: ' + str(i)
                tim = img[i*self.max_batch_size:(i+1)*self.max_batch_size,0:int(self.w),0:int(self.h)]
                #Lots of repeated code
                tim = np.array([self.transformer.preprocess('data',i) for i in tim])
                self.net["many"].set_input_arrays(tim, np.ones(self.max_batch_size,dtype=np.float32))
                p = self.net["many"].forward()
                codes = np.append(codes,self.net["many"].blobs[self.layer_name].data[...])
            if np.round(img.shape[0]/self.max_batch_size) * self.max_batch_size < img.shape[0]:
                print 'final minibatch'
                tim = img[mult:img.shape[0],0:int(self.w),0:int(self.h)]
                tim = np.array([self.transformer.preprocess('data',i) for i in tim])
                tim = np.vstack((tim, np.zeros(np.append(self.max_batch_size-(img.shape[0]-mult),self.net["many"].blobs['data'].data.shape[1:]))))

                #Lots of repeated code
                self.net["many"].set_input_arrays(tim.astype(np.float32), np.ones(self.max_batch_size,dtype=np.float32))
                p = self.net["many"].forward()
                codes = np.append(codes,self.net["many"].blobs[self.layer_name].data[...][0:img.shape[0]-mult])
            codes = codes.reshape(np.append(-1,self.net["many"].blobs[self.layer_name].data.shape[1:]))
        else:
            self.initialize_cnn(img.shape[0],"many")
            img = img[:,0:int(self.w),0:int(self.h)]
            img = np.array([self.transformer.preprocess('data',i) for i in img])
            self.net["many"].set_input_arrays(img, np.ones(img.shape[0],dtype=np.float32))
            p = self.net["many"].forward()
            codes = self.net["many"].blobs[self.layer_name].data[...]
        return codes


def flatten(img):
    if img.shape[2] > 1:
        Y = 0.2125*img[:,:,0] + 0.7154*img[:,:,1] + 0.0721*img[:,:,2]
    else:
        Y = img
    return Y

kinds = [ColorHist, HoGDalal, TinyImage, CNN]
