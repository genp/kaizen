import numpy as np
import skimage.feature
import skimage.color
import skimage.transform

class ColorHist:
    def set_params(self, bins=4):
        pass
  
    def extract(self, img):
        pixels = np.reshape(img, (img.shape[0]*img.shape[1],-1))
        hist,e = np.histogramdd(pixels, bins=self.bins, range=3*[[0,255]], normed=True)
        hist = np.reshape(hist, (-1)) # Make it 1-D
        return hist

class HoGDalal:
    def set_params(self, ori=9, px_per_cell=(8,8), cells_per_block=(2,2), window_size=160):
        self.ori = ori
        self.px_per_cell = px_per_cell
        self.cells_per_block = cells_per_block
        self.window_size = window_size

    def extract(self, img):
        flat_img = skimage.color.rgb2gray(img)
        flat_img = skimage.transform.resize(flat_img, (self.window_size, self.window_size))
        hog_feat = skimage.feature.hog(flat_img, orientations=self.ori, pixels_per_cell=self.px_per_cell,
                                       cells_per_block=self.cells_per_block)
        return hog_feat

class TinyImage:
    def set_params(self, color_space='lab'):
        self.color_space = color_space

    def extract(self, img):
        if self.color_space=='lab':
            img = skimage.color.rgb2lab(img)
        elif self.color_space=='gray' or self.color_space=='grey' :
            img = skimage.color.rgb2gray(img)

        tiny = skimage.transform.resize(img, (32,32))
        tiny = np.reshape(tiny, (-1))
        return tiny

