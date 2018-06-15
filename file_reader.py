import numpy as np
import scipy
from scipy.ndimage import imread
from scipy.misc import imresize
import matplotlib.pyplot as plt
import cPickle as pickle
#import cv2
import cPickle as pickle
from utils import *
import os
import sys


dirname = 'BenchmarkIMAGES/'
crop_size = (1024,1024)
mode = 'RGB'
num_images = 300 

save_dir=''
save_name = 'MIT300_pickle'
default_size = (1024, 1024)
N_splits = num_images
	

def collect_images(dirname,num_images,crop_size=None, mode='RGB'):
	imglist = []
	if crop_size is not None:
		for i in xrange(num_images):
			fname = dirname + 'i'+str((i+1))+'.jpg'
			img = imresize(imread(fname, mode=mode), crop_size)
			imglist.append(img)
	
		#we turn imglist into a flat array and return it
		imglist = np.array(imglist)
	if crop_size is None:
		for i in xrange(num_images):
			fname = dirname + 'i' + (i+1) +'jpg'
			img = imread(fname, mode=mode)
			imglist.append(img)
	
	imglist = np.array(imglist)
	return imglist


def collect_files_and_images(rootdir, crop_size = default_size, mode='RGB', save = True, save_dir = None):
	filelist = []
	print "IN FUCNTION"
	for subdir, dirs, files in os.walk(rootdir):
		# this will save them all in one enormous file
		print subdir
		print dirs
		for file in files:
			print file
			fname = os.fsdecode(file)
			if filename.endswith(".jpg"):
				if crop_size is not None:
					img = imresize(imread(filename, mode=mode), crop_size)
				if crop_size is None:
					img = imread(filename, mode=mode)
				filelist.append(img)

	filelist = np.array(filelist)
	if save and save_dir is not None:
		save_images(filelist, save_dir, save_name = "_data")
	return filelist

def print_dirs_files(rootdir):
	for subdir, dirs, files in os.walk(rootdir):
		print subdir
		print "  "
		print dirs
		print "  "
		print files
		print "  "

def combine_file_in_directory(crop_size = default_size,verbose=False, mode='RGB', save=True, save_dir='./', make_dir_name = None):
	rootdir ='./'
	for subdir, dirs, files in os.walk('./'):
		for file in files:
			filename = os.path.basename(file)
			if '.py' not in filename and '.pyc' not in filename:
				if verbose:
					print "combined: " + str(rootdir+'/'+filename)
				arr = load(rootdir + '/' + filename)
				img_type = filename.split('_')[-1]
				if img_type == 'images':
					img_arr.append(arr)
					if verbose:
						print "added to images"
				if img_type == 'output':
					out_arr.append(arr)
					if verbose:
						print "added to outputs"
		img_arr = np.concatenate(img_arr)
		out_arr = np.concatenate(out_arr)

		#img_arr = np.array(img_arr)
		#out_arr = np.array(out_arr)
		#print img_arr.shape	
		#print out_arr.shape
		if verbose:
			print "images shape" + str(img_arr.shape)
			print "outputs shape" + str(out_arr.shape)
		if save:
			if make_dir_name == '':
				save_array(img_arr, rootdir + 'images_'+ add_name)
				save_array(out_arr, rootdir + 'ouputs_'+ add_name)
			
			if make_dir_name !='':
				save_array(img_arr, rootdir + make_dir_name + '/' + 'images_' + add_name)
				save_array(out_arr, rootdir + make_dir_name + '/' + 'outputs_' + add_name)
		return img_arr, out_arr



def save_images_per_directory(rootdir, crop_size = default_size, mode='RGB', save=True, save_dir='./', make_dir_name = None): 
	if make_dir_name is not None:
		assert type(make_dir_name) == str, 'make directory name must be a string'
		save_dir = save_dir +  make_dir_name + '/'
		if not os.path.exists(make_dir_name):
			try:
				os.makedirs(make_dir_name)
			except OSError as e:
				if e.errno!= errno.EEXIST:
					print "error found: " + str(e)
					raise
				else:
					print "directory probably already exists despite check"
					raise

	print os.walk(rootdir)
	if not save:
		total_list = []
	for subdir, dirs, files in os.walk(rootdir):
		filelist = []
		
		for file in files:
			filename = os.path.basename(file)
			if file.endswith(".jpg") or file.endswith(".jpeg"):
				if crop_size is not None:
					#print "IN IMAGE LOOP"
					#print subdir
					print filename
					img = imresize(imread(subdir + '/' + filename, mode=mode), crop_size)
				if crop_size is None:
					img = imread(filename, mode=mode)
				filelist.append(img)

		splits = subdir.split("/")
		#get the last split
		name = splits[-1]
		if name=="":
			name = splits[0]
		name = "_images"
		if len(dirs) ==0:
			name = splits[-2]
			name = name + "_output"
			#name = "_output"
		filelist = np.array(filelist)
		if save:
			print name
			#print filelist.shape
			save_array(filelist,save_dir + name)
			print "SAVED: " + name
			#print save_dir+name
		if not save:
			total_list.append(filelist)
			print "PROCESSED: " + name
	if not save:
		total_list = np.array(total_list)
		return total_list

def combine_arrays_into_one(rootdir, save=True, add_name="combined", make_dir_name='', verbose=True):
	if make_dir_name != '':
		if not os.path.exists(rootdir + make_dir_name):
			try:
				os.makedirs(rootdir + make_dir_name)
			except OSError as e:
				if e.errno!= errno.EEXIST:
					print "error found: " + str(e)
					raise
				else:
					print "directory probably already exists despite check"
					raise
		
	for subdir, dirs, files in os.walk(rootdir):
		img_arr = []
		out_arr = []
		for file in files:
			filename = os.path.basename(file)
			if '.' not in filename:
				if verbose:
					print "combined: " + str(rootdir+'/'+filename)
				arr = load(rootdir + '/' + filename)
				img_type = filename.split('_')[-1]
				if img_type == 'images':
					img_arr.append(arr)
					if verbose:
						print "added to images"
				if img_type == 'output':
					out_arr.append(arr)
					if verbose:
						print "added to outputs"
		img_arr = np.concatenate(img_arr)
		out_arr = np.concatenate(out_arr)

		#img_arr = np.array(img_arr)
		#out_arr = np.array(out_arr)
		#print img_arr.shape	
		#print out_arr.shape
		if verbose:
			print "images shape" + str(img_arr.shape)
			print "outputs shape" + str(out_arr.shape)
		if save:
			if make_dir_name == '':
				save_array(img_arr, rootdir + 'images_'+ add_name)
				save_array(out_arr, rootdir + 'ouputs_'+ add_name)
			
			if make_dir_name !='':
				save_array(img_arr, rootdir + make_dir_name + '/' + 'images_' + add_name)
				save_array(out_arr, rootdir + make_dir_name + '/' + 'outputs_' + add_name)
		return img_arr, out_arr


def read_image(num, dirnme = dirname, mode='RGB'):
	fname = dirname + 'i' + str(num) +'.jpg'
	return imread(fname, mode=mode)
	
def save_images(imgarray, save_dir, save_name, N_splits = None):
	if N_splits is None:
		fname = save_dir+save_name
		save(imgarray, fname)
	if N_splits is not None:
		N = len(imgarray)/N_splits
		for i in xrange(N_splits):
			arr = imgarray[(N*i):(N*(i+1)), :,:,:]
			fname = save_dir + save_name +'_'+i
			save(arr, fname)

def split_img_on_colour(img, mode='RGB'):
	
	if mode== 'RGB':
		red = img[0,:,:]
		blue = img[1,:,:]
		green = img[2,:,:]
		return [red, blue, green]


def main():
	#get the commandline arguments
	#put in defaults
	dirname = 'BenchmarkIMAGES/'
	save_name ='Data/'
	#default crop size
	crop_size = (100,100)
	mode = 'RGB'
	if len(sys.argv) >=2:
		dirname = sys.argv[1]
	if len(sys.argv)>=3:
		save_name = sys.argv[2]
	if len(sys.argv)>=4:
		val = int(sys.argv[3])
		crop_size = (val, val)
	if len(sys.argv)>=5:
		mode = sys.argv[5]

	save_images_per_directory(dirname, crop_size = crop_size, mode=mode, save=True, save_dir=save_name, make_dir_name = None)
	#combine saved iamges
	combine_arrays_into_one('', add_name='combined', make_dir_name='', verbose=True)
	#combine_file_in_directory()


if __name__ == '__main__':
	print "File Reader Activated"
	main()
	
	
