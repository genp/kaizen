import nrrd
import numpy as np
import matplotlib.pyplot as plt
import caffe
import os
import lmdb
import sys
import boto3
import scipy.misc
import csv

urls = np.array([])
seg_path = "/path/to/segmentations"

def upload_to_s3():
	'''
	Goes through a directory of NRRD files (containing 512x512 CT scan images in NDARRAY format) and
	clips (hounsfield width = 100 level = 50) and uploads the CT scans to s3 
	'''
	s3 = boto3.resource('s3')
	for root, dirs, files in os.walk(seg_path):
		for fname in files:
			if ".nrrd" in fname and "label" not in fname:
				print fname

				data = nrrd.read(os.path.join(root, fname.replace('\\','')))[0]
				w= 100
				l = 50
				h = np.clip(data, l - (w/2), l + (w/2))
				for ind in range(data.shape[2]):
					print ind
					img = np.tile(h[:,:,ind].reshape((1,512,512)),(3,1,1)).astype(np.uint8)
					scipy.misc.imsave('temp.jpg', img)
					urls = np.append(urls, "https://protect3dsegjpeg.s3.amazonaws.com/" + fname.replace(".nrrd","")+"-"+ str(ind) + ".jpg")
					s3.meta.client.upload_file("temp.jpg", "protect3dsegjpeg",fname.replace(".nrrd","")+"-"+ str(ind) + ".jpg")


	os.remove("temp.jpg")


def bound(img):
    label = np.where(img != 0)
    return np.min(label[0]), np.max(label[0]), np.min(label[1]), np.max(label[1])

def overlap_squares(patch_a, patch_b, overlap):
    '''
    checks for overlap of bboxes specified as tuples (x, y, size)
    check if squares overlap by IoU >= overlap

    modified to return a float value for intersection over union as opposed to a boolean value
    '''    
    # intersection
    y_in = np.intersect1d(range(patch_a[1], patch_a[1]+patch_a[2]), range(patch_b[1], patch_b[1]+patch_b[2]))
    x_in = np.intersect1d(range(patch_a[0], patch_a[0]+patch_a[2]), range(patch_b[0], patch_b[0]+patch_b[2]))
    intersection = float(len(y_in))*float(len(x_in))

    # union
    union = float(patch_a[2])**2+float(patch_b[2])**2 - intersection

    # print intersection/union
    return intersection/union


def write_keyword_file(keyword, rng = (0,10000), iou = (.5,.6)):
	'''
	writes label data to a file with the name "_KEYWORD.csv"

	format: image_url, x, y, h, w, label (t/f ; 0/1)
	'''
	f = open("_" + keyword + ".csv", 'w+')
	writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	i = 0
	for root, dirs, files in os.walk(seg_path):
		for fname in files:
			if ".nrrd" in fname and "label" in fname:
				print fname

				labels = nrrd.read(os.path.join(root, fname.replace('\\','')))[0]
				indices = np.where(np.max(np.max(labels,axis=0),axis=0) == 1)
				for ind in indices[0]:
					print ind
					corners = bound(labels[:,:,ind])
					url = urls[i+ind]
					side = max(corners[1] - corners[0], corners[3] - corners[2])
					pad = 10
					if side > rng[0] and side <= rng[1]:
						for x in range((corners[0]+side+pad) - (corners[0]-pad)):
							for y in range((corners[2] + side + pad) - (corners[2] - pad) ):
								intersect = overlap_squares((corners[0]-pad,corners[2]-pad,side), (x+corners[0]-pad,y+corners[2]-pad,side), iou)
								if intersect <= iou[1] and intersect > iou[0]:
									writer.writerow([str(url), str(corners[0]-pad+x), str(y+corners[2]-pad), str(side + pad),str(side + pad), str(1)])
						writer.writerow([str(url), str(corners[0]-pad), str(corners[2]-pad), str(side + pad),str(side + pad), str(1)])
				i = i + labels.shape[2]
			
	f.close()


#upload_to_s3()

for root, dirs, files in os.walk(seg_path):
	for fname in files:
		if ".nrrd" in fname and "label" not in fname:
			print fname
			data = nrrd.read(os.path.join(root, fname.replace('\\','')))[0]
			for ind in range(data.shape[2]):
				print ind
				urls = np.append(urls, "https://protect3dsegjpeg.s3.amazonaws.com/" + fname.replace(".nrrd","")+"-"+ str(ind) + ".jpg")

print "done with urls; proceeding to write keyword file."

i = (.91,.92)
write_keyword_file("hemorrhage_0-20",rng = (0,20), iou = i)
write_keyword_file("hemorrhage_20-50",rng = (20,50), iou = i)
write_keyword_file("hemorrhage_50-100",rng = (50,100), iou = i)
write_keyword_file("hemorrhage_100-1000",rng = (100,1000), iou = i)


f = open("urls.csv", 'w+')
writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
for u in urls:
	writer.writerow([u])
f.close()


