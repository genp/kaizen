#!/usr/bin/env python
import numpy as np
import os

import config
import caffe
from extract import CNN
from apptimer import AppTimer
import unittest
from app import db, models

class TestExtractCNN(unittest.TestCase):

	def __init__(self, *args, **kwargs):
		super(TestExtractCNN, self).__init__(*args, **kwargs)
		print 'initializing unit test'
		self.c = CNN()
		self.c.set_params(initialize = True)
		p = models.Patch.query.all()[0]
		self.img = p.image
		self.many = np.repeat(np.expand_dims(self.img, axis=0),300,axis=0)

	def test_output_extract(self):		
		self.assertEqual(self.c.extract(self.img).shape,(4096,))

	def test_output_extract_many(self):		
		self.assertEqual(self.c.extract_many(self.many).shape,(300,4096))
		self.assertEqual(self.c.extract_many(np.repeat(self.many,2,axis=0)).shape,(600,4096))

	def test_output_extract_many_pad(self):		
		self.assertEqual(self.c.extract_many_pad(self.many).shape,(300,4096))
		self.assertEqual(self.c.extract_many(np.repeat(self.many,2,axis=0)).shape,(600,4096))


def time_tests():
	a = AppTimer()

	img = np.random.rand(257,257,3)
	img_many = np.expand_dims(img, axis=0)
	img_many = np.repeat(img_many,300,axis=0)
	img_manymore = np.repeat(img_many,2,axis=0)

	c = CNN()

	log=open('time_log_extract.txt', 'w+')

	a.start()
	c.set_params(initialize = True)        
	print >> log, "Test #1: Set Params"
	a.stop(log)


	a.start()
	out = c.extract(img)
	print >> log, "Test #2: Single image extraction using extract()"
	a.stop(log)

	a.start()
	for i in img_many:
		out = c.extract(i)
	print >> log, "Test #3: Multiple image extraction using extract()"
	a.stop(log)

	a.start()
	out = c.extract_many(img_many)
	print >> log, "Test #4: Multiple image extraction using extract_many()"
	a.stop(log)

	a.start()
	out = c.extract_many_pad(img_many)
	print >> log, "Test #5: Multiple image extraction using extract_many_pad()"
	a.stop(log)

	a.start()
	for i in img_manymore:
		out = c.extract(i)
	print >> log, "Test #6: Multiple image extraction using extract() batch_size > max batch size"
	a.stop(log)


	a.start()
	out = c.extract_many(img_manymore)
	print >> log, "Test #7: Multiple image extraction using extract_many() batch_size > max batch size"
	a.stop(log)


	a.start()
	out = c.extract_many_pad(img_manymore)
	print >> log, "Test #8: Multiple image extraction using extract_many_pad() batch_size > max batch size "
	a.stop(log)

	log.close()

def reduce_tests():
	a = AppTimer()

<<<<<<< HEAD
	img = np.round(255 * np.random.rand(256,256,3))

    #img = 255*np.ones((256,256,3))
    #img = np.zeros((257,257,3))
	#img = np.random.rand(157,157,3)

=======
	img = 255 * np.random.rand(457,457,3)
>>>>>>> 98a93ca2277fd87dd8d94482749f362890d6e32e
	img_many = np.expand_dims(img, axis=0)
	img_many = np.repeat(img_many,300,axis=0)
	img_manymore = np.repeat(img_many,2,axis=0)

	print np.min(img)
	print np.max(img)


	c = CNN()
        
	log=open('reduce_log_extract.txt', 'w+')

	a.start()
	c.set_params(initialize = True, use_reduce = True, ops = ["subsample", "power_norm"], output_dim = 200, alpha = 2.5) # model= 'VGG'
	print >> log, "Test #1: Set Params"
	a.stop(log)


	a.start()
	out = c.extract(img)
	print np.min(out)
	print np.max(out)
	print "Min and max values of extract feature output: ({}, {})".format(np.min(out),  np.max(out))
	print >> log, "Test #2: Single image extraction using extract()"
	a.stop(log)


	a.start()

	out2 = c.extract_many(img_many)
	print "Min and max values of extract many feature output: ({}, {})".format(np.min(out2),  np.max(out2))
	print >> log, "Test #3: Multiple image extraction using extract_many()"
	a.stop(log)

        print "Check that output of extract and extract many is the same: {}".format(np.allclose(out, out2[0]))


        img = np.zeros((257,257,3))
	img_many = np.expand_dims(img, axis=0)

	a.start()
	out = c.extract(img)
	print np.min(out)
	print np.max(out)
	print "Min and max values of extract feature output on blank image: ({}, {})".format(np.min(out),  np.max(out))
	print >> log, "Test #4: Single image extraction using extract()"
	a.stop(log)


	a.start()

	out2 = c.extract_many(img_many)
	print "Min and max values of extract many feature output on blank image: ({}, {})".format(np.min(out2),  np.max(out2))
	print >> log, "Test #5: Multiple image extraction using extract_many()"
	a.stop(log)

	log.close()

        print "Check that output of extract and extract many is the same: {}".format(np.allclose(out, out2[0]))

def extract_tests():
    # assumes one dataset and a blob with a least 10 patches loaded into database
    d = app.models.Dataset.query.get(1)
    blob = app.models.Blob.query.get(1)
    imgs = [p.image for p in blob.patches[:10]]
    fs = app.models.FeatureSpec.query.get(6)
    feat = fs.instance.extract_many(imgs)
    pfeat = fs.analyze_patch(blob.patches[0])
    
    print 'extract_many and feature_spec.analyze_patch return same result: {}'.format(np.allclose(pfeat.vector, feat[0], atol=1e-6))
        


if __name__ == '__main__':
    #unittest.main()
    time_tests()
    reduce_tests()
