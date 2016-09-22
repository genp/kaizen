import openslide as o
import numpy as np
import os
import boto3
import csv
import gc
import psutil
import multiprocessing as mp
import resource

urls = np.array([])
seg_path = "/xvdf/CAMELYON16/TrainingData/Train_01/"
label_path = "/xvdf/CAMELYON16/TrainingData/Ground_Truth/Mask/"
level = 2
subdiv = 3

def upload_to_s3():
	'''
	Goes through a directory of TIF files (containing whole-slide images) and
	uses the 3rd level to upload to s3
	'''
	
	urls = np.array([])
	s3 = boto3.resource('s3')
	for root, dirs, files in os.walk(seg_path):
		for fname in files:
			if ".tif" in fname and "label" not in fname:
				print fname		
				data = o.OpenSlide(os.path.join(root, fname.replace('\\','')))
				print data.level_dimensions[level][0]

				print "proceeding to subpatch"
				w, l = data.level_dimensions[level]
				wn = w/subdiv
				ln = l/subdiv
				i = 0
				for x in range(subdiv):
					for y in range(subdiv):
						patch = data.read_region((x*wn,y*ln),level,(wn,ln))
						patch.save("temp.jpg")
						savename = fname.upper().replace(".TIF","") + "-" + str(i) + ".jpg"
						s3.meta.client.upload_file("temp.jpg", "camelyonjpeg-seg" + str(level),savename)
						print "uploaded: " + savename
						i = i+1

				'''
				if data.level_dimensions[level][0] > 65500 or data.level_dimensions[level][1] > 65500:
					print "exceeded dimensions, proceeding to subpatch"
					w, l = data.level_dimensions[level]
					wn = w/subdiv
					ln = l/subdiv
					i = 0
					for x in range(subdiv):
						for y in range(subdiv):
							patch = data.read_region((x*wn,y*ln),level,(wn,ln))
							patch.save("temp.jpg")
							savename = fname.upper().replace(".TIF","") + "-" + str(i) + ".jpg"
							s3.meta.client.upload_file("temp.jpg", "camelyonjpeg" + str(level),savename)
							print "uploaded: " + savename
							i = i+1
				else:
					img = data.read_region((0,0),level,data.level_dimensions[level])
					img.save("temp.jpg")
					urls = np.append(urls, "https://camelyonjpeg.s3.amazonaws.com/" + fname.replace(".tif",".jpg"))
					s3.meta.client.upload_file("temp.jpg", "camelyonjpeg" + str(level),fname.upper().replace(".TIF",".jpg"))
					print "uploaded: "+  fname
					data = []
					img = []
					gc.collect()
				'''

	os.remove("temp.jpg")


def bound(img):
	label = np.array(np.where(img != 0))

	if label.size != 0:
		return np.min(label[0]), np.max(label[0]), np.min(label[1]), np.max(label[1])
	else:
		return 0,0,0,0

def compute_label(labels,writer,threshold,url,loc):
	print loc
	annot = np.array(labels.read_region((loc[0],loc[1]),level,(loc[2],loc[3])))[:,:,:3]
	print annot.shape

	print "mem 2"
	print psutil.virtual_memory()
	verts = bound(annot)
	x1, x2, y1, y2 = verts
	patch_size = [128,256]

	for p in patch_size:
		for x in range(int(np.ceil((x2-x1)/float(p)))):
			for y in range(int(np.ceil((y2-y1)/float(p)))):
				print p, x, y
				patch = annot[x1+x*p:x1+(x+1)*p, y1+y*p:y1+(y+1)*p]
				overlap = np.divide(np.count_nonzero(patch == 255),p*p)
				if overlap >= threshold:
					writer.writerow([str(url), str(x1+x*p),str(y1+y*p), str(p),str(p), str(1)])

def write_keyword_file(keyword, threshold = .8):
	'''
	writes label data to a file with the name "_KEYWORD.csv"

	format: image_url, y,x, h, w, label (t/f ; 0/1)
	'''
	f = open("_" + keyword + ".csv", 'w+')
	writer = csv.writer(f, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
	for root, dirs, files in os.walk(label_path):
		for fname in files:
				print fname

				i = 0
				labels = o.OpenSlide(os.path.join(root, fname.replace('\\','')))

				w,l = labels.level_dimensions[level]
				wn = w/subdiv
				ln = l/subdiv

				print "proceeding to subpatch"
				for x in range(subdiv):
					for y in range(subdiv):
						url = "https://camelyonjpeg-seg" +str(level) + ".s3.amazonaws.com/" + fname.replace("_Mask","").upper().replace(".TIF","") + "-" + str(i) + ".jpg"
						print "mem 1"
						print psutil.virtual_memory()
						proc = mp.Process(target = compute_label(labels, writer,threshold,url,(x*wn,y*ln,wn,ln)))
						proc.start()
						proc.join()
						print "mem 3"
						print psutil.virtual_memory()
						print i
						i = i+1

				'''
				if w > 65500 or l > 65500:
					print "exceeded 65500 pixels, proceeding to subpatch"
					for x in range(subdiv):
						for y in range(subdiv):
							url = "https://camelyonjpeg" +str(level) + ".s3.amazonaws.com/" + fname.replace("_Mask","").upper().replace(".TIF","") + "-" + str(i) + ".jpg"
							print "mem 1"
							print psutil.virtual_memory()
							proc = mp.Process(target = compute_label(labels, writer,threshold,url,(x*wn,y*ln,wn,ln)))
							proc.start()
							proc.join()
							print "mem 3"
							print psutil.virtual_memory()
							print i
							i = i+1
				else:
					url = "https://camelyonjpeg" +str(level) + ".s3.amazonaws.com/" + fname.replace("_Mask","").upper().replace(".TIF",".jpg")
					print "mem 1"
					print psutil.virtual_memory()
					proc = mp.Process(target = compute_label(labels, writer,threshold,url,(0,0,labels.level_dimensions[level][0],labels.level_dimensions[level][1])))
					proc.start()
					proc.join()
					print "mem 3"
					print psutil.virtual_memory()
				'''
		
	f.close()


upload_to_s3()
#write_keyword_file("level_1")
