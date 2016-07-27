import re
import tempfile
import warnings

import numpy as np
import skimage.feature
import skimage.color
import skimage.transform

def reduce(reducible_feature, codes):
    '''
    "codes" should be a numpy array of codes for either a single or multiple images of shape:
    (N, c) where "N" is the number of images and "c" is the length of codes.  

    reducible_feature should be a class in the extract module with these member parameters: 

    "ops" indicates the processes to perform on the given feature.
    Currently supported operations: subsample, normalization (normalize), power normalization (power_norm)

    "output_dim" is the number of dimensions requested for output of a dimensionality reduction operation.
    Not needed for non dimensionality reduction operations (ie "normalization")
    
    "alpha" is the power for the power normalization operation
    '''
    output_codes = codes if len(codes.shape) > 1 else codes.reshape(1,len(codes))

    for op in reducible_feature.ops:

        if op == "subsample":
            odim = reducible_feature.output_dim
            if odim <= output_codes.shape[1]:
                output_codes = output_codes[:,0:odim]
            else:
                raise ValueError('output_dim is larger than the codes! ')
        elif op == "normalize":
            mean = np.mean(output_codes, 1)
            std = np.std(output_codes, 1)
            norm = np.divide((output_codes - mean[:, np.newaxis]),std[:, np.newaxis])
            output_codes = norm

        elif op == "power_norm":
            alpha = reducible_feature.alpha
            pownorm = lambda x: np.power(np.abs(x), alpha)
            pw = pownorm(output_codes)
            norm = np.linalg.norm(pw, axis=1)
            if not np.any(norm):
                warnings.warn("Power norm not evaluated due to 0 value norm")
                continue
            output_codes = np.divide(pw,norm[:, np.newaxis])
            output_codes = np.nan_to_num(output_codes)

    if output_codes.shape[0] == 1:
        output_codes = np.reshape(output_codes, -1)
    return output_codes

def maybe_reduce(f):
    def maybe_reducing_f(self, *args):
        if self.use_reduce:
            return reduce(self, f(self, *args))
        return f(self, *args)
    return maybe_reducing_f


class ReducibleFeature:

    def set_params(self, **kwargs):
        self.use_reduce = kwargs.get('use_reduce', False)
        for key in ('ops', 'output_dim', 'alpha'): 
            setattr(self, key, kwargs.get(key))
        self.params = kwargs    
    
    def extract_many(self, img):
        codes = np.array([self.extract(i) for i in img])
        return codes


class ColorHist(ReducibleFeature):
    def set_params(self, **kwargs):
        ReducibleFeature.set_params(self, **kwargs)
        self.bins = kwargs.get('bins', 4)
    
    @maybe_reduce
    def extract(self, img):
        pixels = np.reshape(img, (img.shape[0]*img.shape[1],-1))
        hist,e = np.histogramdd(pixels, bins=self.bins, range=3*[[0,255]], normed=True)
        hist = np.reshape(hist, (-1)) # Make it 1-D
        return hist


class HoGDalal(ReducibleFeature):
    def set_params(self, **kwargs):
        ReducibleFeature.set_params(self, **kwargs)
        self.ori = kwargs.get('ori', 9)
        self.px_per_cell = kwargs.get('px_per_cell', (8,8))
        self.cells_per_block = kwargs.get('cells_per_block', (2,2))
        self.window_size = kwargs.get('window_size',40)

    @maybe_reduce
    def extract(self, img):
        flat_img = flatten(img)
        flat_img = skimage.transform.resize(img[:,:,1], (self.window_size, self.window_size))
        hog_feat = skimage.feature.hog(flat_img, orientations=self.ori, pixels_per_cell=self.px_per_cell,
                                       cells_per_block=self.cells_per_block)
        hog_feat = np.reshape(hog_feat, (-1))
        return hog_feat

class TinyImage(ReducibleFeature):
    def set_params(self, **kwargs):
        ReducibleFeature.set_params(self, **kwargs)
        self.flatten = kwargs.get('flatten', False)

    @maybe_reduce
    def extract(self, img):
        if self.flatten:
            img = flatten(img)

        tiny = skimage.transform.resize(img, (32,32))
        tiny = np.reshape(tiny, (-1))
        return tiny

class MultiNet:
   def __init__(self, single, many):
       self.single = single
       self.many = many


class CNN_Model:
   def __init__(self, net, xform):
       self.net = net
       self.xform = xform

# Darius - CNN code may not work on Multi-GPU machines.
class CNN(ReducibleFeature):

    MANY_BATCH_SIZE = 500
    CACHE = {}

    def get_networks(self):
        key = self.cache_key()
        if not key in CNN.CACHE.keys():
            self.populate_cache(key)
        self.single = CNN.CACHE[key].single
        self.many = CNN.CACHE[key].many

    def cache_key(self):
        key = str(self.params)
        
    def populate_cache(self, key):
        single = self.create_model(1)
        many = self.create_model(CNN.MANY_BATCH_SIZE)
        CNN.CACHE[key] = MultiNet(single, many)

    def create_model(self, batch_size):
        print 'creating model'
        import caffe
        import config
        if config.USE_GPU:                
                caffe.set_device(config.GPU_DEVICE_ID)
                caffe.set_mode_gpu()
        temp = tempfile.NamedTemporaryFile(delete = False)

        def_path = "caffemodels/" + self.model +"/train.prototxt"
        weight_path = "caffemodels/" + self.model + "/weights.caffemodel"
        #go through and edit batch size
        arch = open(def_path,'r').readlines()
        for i in range(len(arch)):
            if "batch_size" in arch[i]:
                arch[i] = re.sub('\d+',str(batch_size),arch[i])
        temp.writelines(arch)
        temp.close()

        net = caffe.Net(str(temp.name), str(weight_path), caffe.TEST)
        xform = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        xform.set_transpose('data', self.transpose)
        xform.set_channel_swap('data',self.channel_swap)
        
        # TODO delete temp file

        return CNN_Model(net, xform)


    def set_params(self, **kwargs):

        '''
        Parameters
        ------------
        "model" is the folder name where the model specs and weights live. 
        ie model = "VGG", "GoogleNet", "BVLC_Reference_Caffenet"
        
        "layer_name" is the layer name used for extraction 
        ie layer_name = "fc7" (for VGG)
        
        see below for better idea of what "transpose" and
        "channel_swap" are used for
        http://nbviewer.jupyter.org/github/BVLC/caffe/blob/master/examples/00-classification.ipynb

        set "initialize" to False when using extract_many.  Initialize
        makes single-patch feature extraction significantly faster

        '''

        ReducibleFeature.set_params(self, **kwargs)        
        self.model = kwargs.get('model', "caffenet")
        self.layer_name = kwargs.get('layer_name', "fc7")
        self.transpose = kwargs.get('transpose', (2,0,1))
        self.channel_swap = kwargs.get('channel_swap', (2,1,0))


    #assume that we're getting a single image
    #Img comes in format (x,y,c)
    @maybe_reduce
    def extract(self, img):
        # check that network is initialized
        self.get_networks()

        img = self.single.xform.preprocess('data',img)
        if len(img.shape) == 3:
            img = np.expand_dims(img,axis=0)
        self.single.net.set_input_arrays(img, np.array([1], dtype=np.float32))
        p = self.single.net.forward()
        feat = self.single.net.blobs[self.layer_name].data[...].reshape(-1)
        feat = np.reshape(feat, (-1))
        return feat
    
    @maybe_reduce
    def extract_many(self, imgs):
        '''
        imgs is a list of app.models.Patch.image, which are ndarrays of shape (x,y,3)
        '''
        self.get_networks()

        if len(imgs) > CNN.MANY_BATCH_SIZE:
            print 'exceeded max batch size. splitting into {} minibatches'.format(int(len(imgs)/CNN.MANY_BATCH_SIZE)+1)
            codes = np.asarray([])
            for i in range(int(len(imgs)/CNN.MANY_BATCH_SIZE)+1):
                tim = imgs[i*CNN.MANY_BATCH_SIZE:min(len(imgs),(i+1)*CNN.MANY_BATCH_SIZE)]
                tim = np.array([self.many.xform.preprocess('data',i) for i in tim])
                num_imgs = len(tim)
                if num_imgs < CNN.MANY_BATCH_SIZE:
                    tim = np.vstack((tim, np.zeros(np.append(CNN.MANY_BATCH_SIZE-num_imgs,self.many.net.blobs['data'].data.shape[1:]),dtype=np.float32)))                 
                self.many.net.set_input_arrays(tim, np.ones(CNN.MANY_BATCH_SIZE,dtype=np.float32))
                p = self.many.net.forward()
                codes = np.append(codes,self.many.net.blobs[self.layer_name].data[...])
            codes = codes.reshape(np.append(-1,self.many.net.blobs[self.layer_name].data.shape[1:]))
            codes = codes[:len(imgs), :]
        else:
            tim = np.array([self.many.xform.preprocess('data',i) for i in imgs])
            num_imgs = len(tim)
            if num_imgs < CNN.MANY_BATCH_SIZE:
                tim = np.vstack((tim, np.zeros(np.append(CNN.MANY_BATCH_SIZE-num_imgs,self.many.net.blobs['data'].data.shape[1:]),dtype=np.float32)))
            self.many.net.set_input_arrays(tim, np.ones(tim.shape[0],dtype=np.float32))
            p = self.many.net.forward()
            codes = self.many.net.blobs[self.layer_name].data[...]
            if num_imgs < CNN.MANY_BATCH_SIZE:
                codes = codes[:num_imgs,:]
        return codes


def flatten(img):
    if img.shape[2] > 1:
        Y = 0.2125*img[:,:,0] + 0.7154*img[:,:,1] + 0.0721*img[:,:,2]
    else:
        Y = img
    return Y

kinds = [ColorHist, HoGDalal, TinyImage, CNN]
