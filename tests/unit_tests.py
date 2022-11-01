#!/usr/bin/env python
import numpy as np
import os

import app

from extract import TimmModel
from apptimer import AppTimer
import unittest
from app import models


class TestExtractTimmModel(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestExtractTimmModel, self).__init__(*args, **kwargs)
        print("Initializing unit test for TimmModel feature extractor...")

        self.c = TimmModel()
        # print(vars(self.c))

        p = models.Patch.query.all()[0]
        self.img = p.image
        # print(f'self.img type in TestExtractTimmModel {type(self.img)}')
        self.repeat_img_n = 30
        self.many = np.repeat(np.expand_dims(self.img, axis=0), self.repeat_img_n, axis=0)

    def test_output_extract(self):
        print('test_output_extract...')
        feat = self.c.extract(self.img)
        print(f'extracted feature shape = {feat.shape}')
        # Single image feauutres are expected to be shape 1xN
        self.assertEqual(feat.shape[0], 1)
        self.assertGreater(feat.shape[1], 1)
        self.assertEqual(len(feat.shape), 2)

    def test_output_extract_reduce(self):
        self.c.use_reduce = True

        self.c.ops = ['normalize']
        self.c.output_dim = 100 # expect the feature size to be longer than 100 dimensions

        print('test_output_extract_reduce...')
        feat = self.c.extract(self.img)
        print(f'extracted feature shape = {feat.shape}')
        # normalized feature should sum to 0
        self.assertAlmostEqual(np.sum(feat), 0, places=3)


        self.c.ops = ['subsample', 'power_norm']
        self.c.output_dim = 100 # expect the feature size to be longer than 100 dimensions
        self.c.alpha = 2.5 # power normalize exponent = 2.5

        print('test_output_extract_reduce...')
        feat = self.c.extract(self.img)
        print(f'extracted feature shape = {feat.shape}')
        # Single image feauutres are expected to be shape 1xN
        self.assertEqual(feat.shape[0], 1)
        self.assertGreater(feat.shape[1], 1)
        self.assertEqual(len(feat.shape), 2)


    def test_output_extract_many(self):
        print('test_output_extract_many...')
        feats = self.c.extract_many(self.many)
        print(f'extract many feature shape = {feats.shape}')
        self.assertEqual(feats.shape[0], self.repeat_img_n)




# def time_extract_tests(model_class=TimmModel):
#     a = AppTimer()

#     img = np.random.rand(257, 257, 3)
#     img_many = np.expand_dims(img, axis=0)
#     img_many = np.repeat(img_many, 300, axis=0)
#     img_manymore = np.repeat(img_many, 2, axis=0)

#     c = model_class()

#     log = open("time_log_extract.txt", "w+")

#     a.start()
#     c.set_params(initialize=True)
#     print("Test #1: Set Params", file=log)
#     a.stop(log)

#     a.start()
#     out = c.extract(img)
#     print("Test #2: Single image extraction using extract()", file=log)
#     a.stop(log)

#     a.start()
#     for i in img_many:
#         out = c.extract(i)
#         print("Test #3: Multiple image extraction using extract()", file=log)
#         a.stop(log)

#     a.start()
#     out = c.extract_many(img_many)
#     print("Test #4: Multiple image extraction using extract_many()", file=log)
#     a.stop(log)

#     a.start()
#     out = c.extract_many_pad(img_many)
#     print("Test #5: Multiple image extraction using extract_many_pad()", file=log)
#     a.stop(log)

#     a.start()
#     for i in img_manymore:
#         out = c.extract(i)
#         print(
#             "Test #6: Multiple image extraction using extract() batch_size > max batch size",
#             file=log,
#         )
#         a.stop(log)

#     a.start()
#     out = c.extract_many(img_manymore)
#     print(
#         "Test #7: Multiple image extraction using extract_many() batch_size > max batch size",
#         file=log,
#     )
#     a.stop(log)

#     a.start()
#     out = c.extract_many_pad(img_manymore)
#     print(
#         "Test #8: Multiple image extraction using extract_many_pad() batch_size > max batch size ",
#         file=log,
#     )
#     a.stop(log)

#     log.close()


# def reduce_tests(model_class=TimmModel):
#     a = AppTimer()

#     img = 255 * np.random.rand(457, 457, 3)
#     img_many = np.expand_dims(img, axis=0)
#     img_many = np.repeat(img_many, 300, axis=0)
#     img_manymore = np.repeat(img_many, 2, axis=0)

#     print(np.min(img))
#     print(np.max(img))

#     c = model_class()

#     log = open("reduce_log_extract.txt", "w+")

#     a.start()
#     c.set_params(
#         initialize=True,
#         use_reduce=True,
#         ops=["subsample", "power_norm"],
#         output_dim=200,
#         alpha=2.5,
#     )  # model= 'VGG'
#     print("Test #1: Set Params", file=log)
#     a.stop(log)

#     a.start()
#     out = c.extract(img)
#     print(np.min(out))
#     print(np.max(out))
#     print(
#         "Min and max values of extract feature output: ({}, {})".format(
#             np.min(out), np.max(out)
#         )
#     )
#     print("Test #2: Single image extraction using extract()", file=log)
#     a.stop(log)

#     a.start()

#     out2 = c.extract_many(img_many)
#     print(
#         "Min and max values of extract many feature output: ({}, {})".format(
#             np.min(out2), np.max(out2)
#         )
#     )
#     print("Test #3: Multiple image extraction using extract_many()", file=log)
#     a.stop(log)

#     print(
#         "Check that output of extract and extract many is the same: {}".format(
#             np.allclose(out, out2[0])
#         )
#     )

#     img = np.zeros((257, 257, 3))
#     img_many = np.expand_dims(img, axis=0)

#     a.start()
#     out = c.extract(img)
#     print(np.min(out))
#     print(np.max(out))
#     print(
#         "Min and max values of extract feature output on blank image: ({}, {})".format(
#             np.min(out), np.max(out)
#         )
#     )
#     print("Test #4: Single image extraction using extract()", file=log)
#     a.stop(log)

#     a.start()

#     out2 = c.extract_many(img_many)
#     print(
#         "Min and max values of extract many feature output on blank image: ({}, {})".format(
#             np.min(out2), np.max(out2)
#         )
#     )
#     print("Test #5: Multiple image extraction using extract_many()", file=log)
#     a.stop(log)

#     log.close()

#     print(
#         "Check that output of extract and extract many is the same: {}".format(
#             np.allclose(out, out2[0])
#         )
#     )

# # TODO update to test a generic featureSpec
# def extract_tests():
#     # assumes one dataset and a blob with a least 10 patches loaded into database
#     d = models.Dataset.query.get(1)
#     print(d)
#     blob = models.Blob.query.get(1)
#     print(blob)
#     imgs = [p.image for p in blob.patches[:10]]
#     fs = models.FeatureSpec.query.filter(
#         models.FeatureSpec.name == "CNN_CaffeNet_redux"
#     ).first()
#     print(fs)
#     feat = fs.instance.extract_many(imgs)
#     pfeat = fs.instance.extract(imgs[0])
#     print(feat.shape)
#     print(pfeat.shape)
#     print(max(feat[0]))
#     print(min(feat[0]))
#     print(max(pfeat))
#     print(min(pfeat))
#     print(
#         "extract_many and extract return same result: {}".format(
#             np.allclose(pfeat, feat[0], atol=1e-6)
#         )
#     )

# # TODO update to test a generic featureSpec
# def extract_multi_batch_tests():
#     # assumes one dataset and a blob with a least 10 patches loaded into database
#     d = models.Dataset.query.get(1)
#     print(d)
#     blob = models.Blob.query.get(1)
#     print(blob)
#     imgs = [p.image for p in blob.patches]
#     fs = models.FeatureSpec.query.filter(
#         models.FeatureSpec.name == "CNN_CaffeNet_redux"
#     ).first()
#     print(fs)
#     feat = fs.instance.extract_many(imgs)

#     big_imgs = []
#     for i in range(5):
#         big_imgs += imgs
#     big_feat = fs.instance.extract_many(big_imgs)

#     print(np.min(big_feat))
#     print(np.min(feat))
#     print(np.max(big_feat))
#     print(np.max(feat))
#     print(
#         "extract_many one and multi batch return same result: {}".format(
#             np.allclose(big_feat[0 : len(feat)], feat, atol=1e-6)
#         )
#     )


# def analyze_blob_test(ds_id, blob_id):
#     ds = models.Dataset.query.get(ds_id)
#     blob = models.Blob.query.get(blob_id)
#     ds.create_blob_features(blob)


# def analyze_patch_test(ds_id, patch_id):
#     ds = models.Dataset.query.get(ds_id)
#     patch = models.Patch.query.get(patch_id)
#     ds.featurespecs[0].analyze_patch(patch)


# def add_examples_test(k_id):
#     # expects to read definition file
#     import tasks

#     k = models.Keyword.query.get(k_id)
#     tasks.add_examples(k)


# def video_features_test():
#     cwd = os.path.dirname(os.path.abspath(__file__))

#     ps = models.PatchSpec.query.filter(models.PatchSpec.name == "Default").first()
#     fs = models.FeatureSpec.query.filter(models.FeatureSpec.name == "TinyImage").first()

#     blob = models.Blob(os.path.join(cwd, "test_video.mp4"))
#     patches = ps.create_blob_patches(blob)
#     patch_list = [p for p in patches]
#     assert len(patch_list) == 1, len(patch_list)
#     patch = patch_list[0]
#     assert isinstance(patch.data, np.ndarray), type(patch.data)
#     assert patch.data.shape == (88, 1080, 1080, 3), patch.data.shape

#     feature_data = fs.instance.extract_many(patch.data)
#     assert isinstance(feature_data, np.ndarray), type(feature_data)
#     assert feature_data.shape == (88, 32 * 32 * 3), feature_data.shape

#     feature = app.models.Feature(patch=patch, spec=fs, data=feature_data)

#     blob2 = models.Blob(os.path.join(cwd, "test_video.mp4"))
#     patches2 = ps.create_blob_patches(blob2)
#     patch_list2 = [p for p in patches2]
#     assert len(patch_list2) == 1, len(patch_list2)
#     patch2 = patch_list2[0]
#     assert isinstance(patch2.data, np.ndarray), type(patch2.data)
#     assert patch2.data.shape == (77, 980, 980, 3), patch2.data.shape

#     feature_data2 = fs.instance.extract_many(patch2.data)
#     assert isinstance(feature_data2, np.ndarray)
#     assert feature_data2.shape == (77, 32 * 32 * 3), feature_data2.shape

#     feature2 = app.models.Feature(patch=patch2, spec=fs, data=feature_data2)

#     squared_distance = feature.squared_distance(feature2)

#     assert squared_distance < 0.0001


# if __name__ == "__main__":
#     time_tests()
