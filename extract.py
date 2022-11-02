import math
import re
import tempfile
import warnings

import numpy as np
import skimage.feature
import skimage.color
import skimage.transform
from functools import reduce
import config
import random
import os


def reduce(reducible_feature, codes):
    """
    "codes" should be a numpy array of codes for either a single or multiple images of shape:
    (N, c) where "N" is the number of images and "c" is the length of codes.

    reducible_feature should be a class in the extract module with these member parameters:

    "ops" indicates the processes to perform on the given feature.
    Currently supported operations: subsample, normalization (normalize), power normalization (power_norm)

    "output_dim" is the number of dimensions requested for output of a dimensionality reduction operation.
    Not needed for non dimensionality reduction operations (ie "normalization")

    "alpha" is the power for the power normalization operation
    """
    output_codes = codes if len(codes.shape) > 1 else codes.reshape(1, len(codes))
    output_codes = (
        output_codes
        if len(codes.shape) == 2
        else codes.reshape(output_codes.shape[0], -1)
    )

    for op in reducible_feature.ops:

        if op == "subsample":
            odim = reducible_feature.output_dim
            if odim <= output_codes.shape[1]:
                output_codes = output_codes[:, 0:odim]
            else:
                raise ValueError("output_dim is larger than the codes! ")
        elif op == "normalize":
            mean = np.mean(output_codes, 1)
            std = np.std(output_codes, 1)
            norm = np.divide((output_codes - mean[:, np.newaxis]), std[:, np.newaxis])
            output_codes = norm

        elif op == "power_norm":
            alpha = reducible_feature.alpha
            pownorm = lambda x: np.power(np.abs(x), alpha)
            pw = pownorm(output_codes)
            norm = np.linalg.norm(pw, axis=1)
            if not np.any(norm):
                warnings.warn("Power norm not evaluated due to 0 value norm")
                continue
            output_codes = np.divide(pw, norm[:, np.newaxis])
            output_codes = np.nan_to_num(output_codes)

    # if output_codes.shape[0] == 1:
    #     output_codes = np.reshape(output_codes, -1)
    return output_codes


def maybe_reduce(f):
    def maybe_reducing_f(self, *args):
        if self.use_reduce:
            return reduce(self, f(self, *args))
        return f(self, *args)

    return maybe_reducing_f


class ReducibleFeature:
    def set_params(self, **kwargs):
        self.use_reduce = kwargs.get("use_reduce", False)
        for key in ("ops", "output_dim", "alpha"):
            setattr(self, key, kwargs.get(key))
        self.params = kwargs

    def extract_many(self, img):
        codes = np.array([self.extract(i) for i in img])
        return codes


class ColorHist(ReducibleFeature):
    def set_params(self, **kwargs):
        ReducibleFeature.set_params(self, **kwargs)
        self.bins = kwargs.get("bins", 4)

    @maybe_reduce
    def extract(self, img):
        pixels = np.reshape(img, (img.shape[0] * img.shape[1], -1))
        hist, e = np.histogramdd(
            pixels, bins=self.bins, range=3 * [[0, 255]], normed=True
        )
        hist = np.reshape(hist, (-1))  # Make it 1-D
        return hist


class TinyImage(ReducibleFeature):
    def set_params(self, **kwargs):
        ReducibleFeature.set_params(self, **kwargs)
        self.flatten = kwargs.get("flatten", False)

    @maybe_reduce
    def extract(self, img):
        if self.flatten:
            img = flatten(img)

        tiny = skimage.transform.resize(img, (32, 32))
        tiny = np.reshape(tiny, (-1))
        return tiny


class MultiNet:
    def __init__(self, single, many):
        self.single = single
        self.many = many


class Network_Model:
    def __init__(self, net, xform):
        self.net = net
        self.xform = xform


# Neural Network feature extractors from the timm library of NN models
import timm
import torch
import numpy # need to import as numpy for type checking
from PIL import Image

class TimmModel(ReducibleFeature):
    def __init__(self, **kwargs):
        self.set_params(**kwargs)

    def create_model():
        pass

    def set_params(self, **kwargs):

        """
        Parameters
        ------------
        "model" is the model name from the timm library.
        use
        > model_names = timm.list_models(pretrained=True)
        > pprint(model_names)
        to list options
        """

        ReducibleFeature.set_params(self, **kwargs)
        self.model_name = kwargs.get(
            "model", "vit_base_patch16_224"
        )  # Default model ViT base
        self.model = timm.create_model(
            self.model_name, pretrained=True, num_classes=0
        )  # intialized pretrained model without classification output
        # TODO: load local weights instead of pretrained
        # forward evaluation only
        self.model.eval()

        # model image pre-processing transform
        self.config = timm.data.resolve_data_config({}, model=self.model_name)
        self.transform = timm.data.transforms_factory.create_transform(**self.config)

        self.many_batch_size = kwargs.get("batch_size", 500)

        # if CUDA available, use first gpu
        # Not implemented: multi gpu support
        if torch.cuda.is_available():
            self.GPU_DEVICE_ID = torch.device("cuda:0")
            print(f"Using GPU {self.GPU_DEVICE_ID}")

    @maybe_reduce
    def extract(self, img):
        """
        Input is a single image
        <img> is a PIL image
        <output_features> is a 1xF numpy array
        """

        return self.extract_many([img])

    @maybe_reduce
    def extract_many(self, imgs):
        """
        Input is an array of images
        <imgs> is a list of PIL images
        <output_features> is a numpy array, NxF
        """

        # pre-process input images
        tensors = []
        for img in imgs:
            # catch when a numpy array is passed instead of a PIL image
            # print(f'img type in extract_many {type(img)}')
            if type(img) == numpy.ndarray:
                img = Image.fromarray(img)

            tensor = self.transform(img).unsqueeze(0)
            tensors.append(tensor)
        tensors = torch.vstack(tensors)

        # extract features from just before the classification head
        features = self.model.forward_features(tensors)
        # the func below isn't actually doing classification because the last
        # linear layer was not loaded with the model
        pooled_features = self.model.forward_head(
            features
        )
        output_features = pooled_features.detach().cpu().numpy()

        # print(output_features.shape)
        # output_features = output_features.squeeze()
        # print(output_features.shape)

        return output_features


def flatten(img):
    if img.shape[2] > 1:
        Y = 0.2125 * img[:, :, 0] + 0.7154 * img[:, :, 1] + 0.0721 * img[:, :, 2]
    else:
        Y = img
    return Y


kinds = [ColorHist, TinyImage, TimmModel]
