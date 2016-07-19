import re
import tempfile

import caffe
import numpy as np
import skimage.feature
import skimage.color
import skimage.transform


class BaseFeature:

    @classmethod
    def _set_params(self, set_params_func):
        def set_params_internal(self, **kwargs):
            # set up reduce params
            self.use_reduce = kwargs.get('use_reduce', False)
            for key in ('ops', 'output_dim', 'alpha'): 
                setattr(self, key, kwargs.get(key))            
            set_params_func(self, **kwargs)
        return set_params_internal

    # Decorator for applying dimensionality reduction and normalization to output of feature extract functions
    @classmethod
    def _reduce(self, extract_func):
        def process(self, *args): 
            '''
            "codes" should be a numpy array of codes for either a single or multiple images of shape:
            (N, c) where "N" is the number of images and "c" is the length of codes.  

            "ops" indicates the processes to perform on the given feature.
            Currently supported operations: subsample, normalization (normalize), power normalization (power_norm)

            "output_dim" is the number of dimensions requested for output of a dimensionality reduction operation.
            Not needed for non dimensionality reduction operations (ie "normalization")
            
            "alpha" is the power for the power normalization operation
            '''
            codes = extract_func(self, *args)            
            if not self.use_reduce:
                return codes
            output_codes = codes if len(codes.shape) > 1 else codes.reshape(1,len(codes))
            for op in self.ops:
                if op == "subsample":
                        if self.output_dim <= output_codes.shape[1]:
                            output_codes = output_codes[:,0:self.output_dim]
                        else:
                            raise ValueError('output_dim is larger than the codes! ')
                elif op == "normalize":
                    mean = np.mean(output_codes, 1)
                    std = np.std(output_codes, 1)
                    norm = np.divide((output_codes - mean[:, np.newaxis]),std[:, np.newaxis])
                    output_codes = norm

                elif op == "power_norm":
                    pownorm = lambda x: np.power(np.abs(x), self.alpha)
                    pw = pownorm(output_codes)
                    norm = np.linalg.norm(pw, axis=1)
                    output_codes = np.divide(pw,norm[:, np.newaxis])
            if output_codes.shape[0] == 1:
                output_codes = np.reshape(output_codes, -1)
            return output_codes
        return process

    def extract_many(self, img):
        ex_func = self._reduce(self.extract)
        codes = np.array([ex_func(i) for i in img])
        return codes


class ColorHist(BaseFeature):
    @BaseFeature._set_params
    def set_params(self, **kwargs):
        self.bins = kwargs.get('bins', 4)
    
    @BaseFeature._reduce
    def extract(self, img):
        pixels = np.reshape(img, (img.shape[0]*img.shape[1],-1))
        hist,e = np.histogramdd(pixels, bins=self.bins, range=3*[[0,255]], normed=True)
        hist = np.reshape(hist, (-1)) # Make it 1-D
        return hist


class HoGDalal(BaseFeature):
    @BaseFeature._set_params
    def set_params(self, **kwargs): 
        self.ori = kwargs.get('ori', 9)
        self.px_per_cell = kwargs.get('px_per_cell', (8,8))
        self.cells_per_block = kwargs.get('cells_per_block', (2,2))
        self.window_size = kwargs.get('window_size',40)

    @BaseFeature._reduce
    def extract(self, img):
        flat_img = flatten(img)
        flat_img = skimage.transform.resize(img[:,:,1], (self.window_size, self.window_size))
        hog_feat = skimage.feature.hog(flat_img, orientations=self.ori, pixels_per_cell=self.px_per_cell,
                                       cells_per_block=self.cells_per_block)
        hog_feat = np.reshape(hog_feat, (-1))
        return hog_feat

class TinyImage(BaseFeature):
    @BaseFeature._set_params
    def set_params(self, **kwargs):
        self.flatten = kwargs.get('flatten', False)

    @BaseFeature._reduce
    def extract(self, img):
        if self.flatten:
            img = flatten(img)

        tiny = skimage.transform.resize(img, (32,32))
        tiny = np.reshape(tiny, (-1))
        return tiny


# Darius - CNN code may not work on Multi-GPU machines.
class CNN(BaseFeature):

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

        self.net[network] = caffe.Net(str(temp.name),str(weight_path),caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': self.net[network].blobs['data'].data.shape})
        self.transformer.set_transpose('data', self.transpose)
        self.transformer.set_channel_swap('data',self.channel_swap)

        temp.close()

    @BaseFeature._set_params
    def set_params(self, **kwargs):
        
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
        self.model = kwargs.get('model', "caffenet")
        self.layer_name = kwargs.get('layer_name', "fc7")
        self.transpose = kwargs.get('transpose', (2,0,1))
        self.channel_swap = kwargs.get('channel_swap', (2,1,0))
        self.initialize = kwargs.get('initialize', False)

        if self.initialize:
            self.initialize_cnn(1,"one")
            self.initialize_cnn(self.max_batch_size,"many")

    #assume that we're getting a single image
    #Img comes in format (x,y,c)
    @BaseFeature._reduce
    def extract(self, img):
        
        img = self.transformer.preprocess('data',img)
        if len(img.shape) == 3:
            img = np.expand_dims(img,axis=0)
        self.net["one"].set_input_arrays(img, np.array([1],dtype=np.float32))
        p = self.net["one"].forward()
        feat = self.net["one"].blobs[self.layer_name].data[...].reshape(-1)
        feat = np.reshape(feat, (-1))
        return feat
    
    #expecting an array of images
    @BaseFeature._reduce
    def extract_many(self, img):

        codes = np.array([])
        if img.shape[0] > self.max_batch_size:
            print 'exceeded max batch size. splitting into minibatches'
            self.initialize_cnn(self.max_batch_size,"many")
            for i in range(int(np.round(img.shape[0]/self.max_batch_size))):
                print 'minibatch: ' + str(i)
                tim = img[i*500:(i+1)*500,:,:]

                #Lots of repeated code
                tim = np.array([self.transformer.preprocess('data',i) for i in tim])
                self.net["many"].set_input_arrays(tim, np.ones(self.max_batch_size,dtype=np.float32))
                p = self.net["many"].forward()
                codes = np.append(codes,self.net["many"].blobs[self.layer_name].data[...])
            if np.round(img.shape[0]/self.max_batch_size) * self.max_batch_size < img.shape[0]:
                mult = np.round(img.shape[0]/self.max_batch_size) * self.max_batch_size
                print 'final minibatch'
                self.initialize_cnn(img.shape[0]-mult,"many")
                tim = img[mult:img.shape[0],:,:]
                #Lots of repeated code
                tim = np.array([self.transformer.preprocess('data',i) for i in tim])
                self.net["many"].set_input_arrays(tim, np.ones(img.shape[0]-mult,dtype=np.float32))
                p = self.net["many"].forward()
                codes = np.append(codes,self.net["many"].blobs[self.layer_name].data[...])
            codes = codes.reshape(np.append(-1,self.net["many"].blobs[self.layer_name].data.shape[1:]))
        else:
            self.initialize_cnn(img.shape[0],"many")
            img = img[:,:,:]
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
                tim = img[i*self.max_batch_size:(i+1)*self.max_batch_size,:,:]
                #Lots of repeated code
                tim = np.array([self.transformer.preprocess('data',i) for i in tim])
                self.net["many"].set_input_arrays(tim, np.ones(self.max_batch_size,dtype=np.float32))
                p = self.net["many"].forward()
                codes = np.append(codes,self.net["many"].blobs[self.layer_name].data[...])
            if np.round(img.shape[0]/self.max_batch_size) * self.max_batch_size < img.shape[0]:
                print 'final minibatch'
                tim = img[mult:img.shape[0],:,:]
                tim = np.array([self.transformer.preprocess('data',i) for i in tim])
                tim = np.vstack((tim, np.zeros(np.append(self.max_batch_size-(img.shape[0]-mult),self.net["many"].blobs['data'].data.shape[1:]))))

                #Lots of repeated code
                self.net["many"].set_input_arrays(tim.astype(np.float32), np.ones(self.max_batch_size,dtype=np.float32))
                p = self.net["many"].forward()
                codes = np.append(codes,self.net["many"].blobs[self.layer_name].data[...][0:img.shape[0]-mult])
            codes = codes.reshape(np.append(-1,self.net["many"].blobs[self.layer_name].data.shape[1:]))
        else:
            self.initialize_cnn(img.shape[0],"many")
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
