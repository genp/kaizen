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

	img = np.random.rand(257,257,3)
        # img = 255*np.ones((257,257,3))
        # img = np.zeros((257,257,3))
	img_many = np.expand_dims(img, axis=0)
	img_many = np.repeat(img_many,300,axis=0)
	img_manymore = np.repeat(img_many,2,axis=0)

	c = CNN()
        
	log=open('reduce_log_extract.txt', 'w+')

	a.start()
	c.set_params(initialize = True, use_reduce = True, ops = ["subsample", "power_norm"], output_dim = 200, alpha = 2.5)        
	print >> log, "Test #1: Set Params"
	a.stop(log)

	a.start()        
	out1 = c.extract(img)        
	print >> log, "Test #2: Single image extraction using extract()"
	a.stop(log)


	a.start()
	out2 = c.extract_many(img_many)
	print >> log, "Test #3: Multiple image extraction using extract_many()"
	a.stop(log)

	log.close()

        print all(out1 == out2[0])
        print out1.shape
        print out1
        print sum(out1-out2[0])

if __name__ == '__main__':
    #unittest.main()
    #time_tests()
    reduce_tests()
