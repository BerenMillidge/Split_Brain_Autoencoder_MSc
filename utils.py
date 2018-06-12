#from __future__ import division
import numpy as np
import scipy
import matplotlib.pyplot as plt
import cPickle as pickle
from skimage import exposure
import sys
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, TerminateOnNaN, ReduceLROnPlateau
import os

def get_run_num():
	if len(sys.argv)>1:
		return sys.argv[1]


#pickle loading and saving functionality

def save(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load(fname):
	return pickle.load(open(fname, 'rb'))

def save_array(obj, fname):
	pickle.dump(obj, open(fname, 'wb'))

def load_array(fname):
	return pickle.load(open(fname, 'rb'))


def serialize_class_object(f):
	try:
		return dict((k,v) for k,v in f.__dict__.iteritems() if not callable(v) and not k.startswith('__'))
	except Exception as err:
		print "Exception in Serialization: " + str(err)
		return {"Error" : err}



def build_callbacks(save_path, min_delta = 1e-4, patience = 10, histogram_freq=0):
	
	checkpointer = ModelCheckpoint(filepath=os.path.join(save_path, "_weights"), monitor="val_loss",save_best_only=True, save_weights_only=True)
	
	early_stopper = EarlyStopping(monitor='val_loss', min_delta=min_delta, patience=patience*2)

	epoch_logger = CSVLogger(os.path.join(save_path, "epoch_logs.csv"))
	
	#batch_logger= BatchLossCSVLogger(os.path.join(save_path, "batch_logs.csv"))
	
	tensorboard = TensorBoard(log_dir=(os.path.join(save_path, '_tensorboard_logs')), histogram_freq=histogram_freq, write_grads=(histogram_freq>0))

	terminator = TerminateOnNaN()
	
	reduceLR = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience= patience, verbose=1, mode='auto', min_lr = 1e-8)

	return [checkpointer, early_stopper, epoch_logger, tensorboard, terminator, reduceLR]


def calculate_variance_of_errmap(errmaps):
	shape = errmaps.shape
	assert len(shape)==3, 'error map must be two dimensinoal (+ number of error maps)'
	variance_map = np.zeros((shape[1], shape[2]))
	for i in xrange(shape[1]):
		for j in xrange(shape[2]):
			var_list = errmaps[:,i,j]
			var_list = np.array(var_list)
			var = np.var(var_list)
			variance_map[i][j] = var
	return variance_map

def apply_var_map_to_sal_map(salmap, varmap):
	shape = salmap.shape
	assert len(shape)==2, 'salience map must be two dimensional'
	assert shape == varmap.shape,'maps must both be the same shape'
	normed_salmap = np.zeros(shape)
	for i in xrange(shape[0]):
		for j in xrange(shape[1]):
			normed_salmap[i][j] = float(salmap[i][j])/float(varmap[i][j])
	return normed_salmap

def norm_error_map(errmap):
	shape = errmap.shape
	assert len(shape) == 2 or len(shape)==3, 'must be 2d with a possible number of 2d errmaps'
	if len(shape) == 2:
		total = np.sum(errmap)
		print total
		normed_map = np.zeros(shape)
		for i in xrange(shape[0]):
			for j in xrange(shape[1]):
				print errmap[i][j]
				normed_map[i][j] = float(errmap[i][j])/float(total)
		return normed_map
	if len(shape) ==3:
		normed_maps = []
		for i in xrange(len(errmap)):
			emap = errmap[i]
			total = np.sum(emap)
			normed_map = np.zeros((shape[1], shape[2]))
			for j in xrange(shape[1]):
				for k in xrange(shape[2]):
					normed_map[j][k] = float(emap[j][k])/float(total)
			normed_maps.append(normed_map)
		normed_maps = np.array(normed_maps)
		return normed_maps

def show_colour_splits(img, show_original = True):
	if show_original:
		print "ORIGINAL:"
		plt.imshow(img)
		plt.show()
	print "RED:"
	plt.imshow(img[:,:,0])
	plt.show()
	print "GREEN:"
	plt.imshow(img[:,:,1])
	plt.show()
	print "BLUE:"
	plt.imshow(img[:,:,2])
	plt.show()


def index_distance(indices1, indices2):
	assert len(indices1) == len(indices2),'indices must have same dimension'
	total = 0
	for i in xrange(len(indices1)):
		total += (indices1[i] - indices2[i]) **2
	return np.sqrt(total)

def max_index_in_array(arr):
	maxval = 0
	shape = arr.shape
	indices = [0,0]

	for i in xrange(shape[0]):
		for j in xrange(shape[1]):
			if arr[i][j]>maxval:
				maxval = arr[i][j]
				indices=[i,j]
	
	return indices


def first_fixations_max(err_map, N, gauss_smooth=True, sigma=2):
	fix_map = np.zeros(err_map.shape)
	for i in xrange(N):
		j,k = max_index_in_array(err_map)
		fix_map[j][k] = 255
		err_map[j][k] = 0
	if gauss_smooth:
		fix_map = gaussian_filter(fix_map, sigma)
	return fix_map
		

def compare_two_images(img1, img2, title1 = "", title2 = "", reshape=False):


		if reshape:
			assert img1.shape == img2.shape, 'images are not of same shape'
			shape=img1.shape
			img1 = np.reshape(img1, (shape[0], shape[1]))
			img2 = np.reshape(img2, (shape[0], shape[1]))		

		plt.subplot(121)
		plt.imshow(img1, cmap='gray')
		plt.title(title1)
		plt.xticks([])
		plt.yticks([])

		#plot filtered image
		plt.subplot(122)
		plt.imshow(img2, cmap='gray')
		plt.title(title2)
		plt.xticks([])
		plt.yticks([])

		plt.show()


def compare_images(imgs, titles = None, break_num = 10, reshape=True):
	N = len(imgs)
	imgs = list(imgs)
	if titles is not None: 
		assert N == len(titles), 'images and titles must be of same length'
	if reshape:
		shape = imgs[0].shape
		for i in xrange(N):
			print imgs[i].shape
			imgs[i] = np.reshape(imgs[i],(shape[0], shape[1]))

	if N<break_num:
		for i in xrange(N):
			plt.subplot(1,N, i+1)
			plt.imshow(imgs[i], cmap='gray')
			plt.title(titles[i])
			plt.xticks([])
			plt.yticks([])
	if N>break_num:
		div = N/break_num
		for n in xrange(div):
			for i in xrange(N):
				plt.subplot(1,N,i+1)
				plt.imshow(imgs[(div*n)+i], cmap='gray')
				plt.title(titles[(div*n)+i])
				plt.xticks([])
				plt.yticks([])
				plt.show()

	plt.show()
	


def load_and_show_colour_split_images(fname):
	img = plt.imread(fname)
	red = img[:,:,0]
	green = img[:,:,1]
	blue=img[:,:,2]
	
	"""
	#colour img
	plt.subplot(111)
	plt.imshow(img)
	plt.title('Original Colour Image')
	
	#red
	plt.subplot(212)
	plt.imshow(red)
	plt.title('Red Channel')
	
	#green
	plt.subplot(221)
	plt.imshow(green)
	plt.title('Green Channel')
	
	#blue
	plt.subplot(222)
	plt.imshow(blue)
	plt.title('Blue Channel')
	
	#show
	plt.show()
	"""
	fig = plt.figure()

	#originalcolour
	ax1 = fig.add_subplot(221)
	plt.imshow(img)
	plt.title('Original Colour Image')
	plt.xticks([])
	plt.yticks([])

	#red
	ax2 = fig.add_subplot(222)
	plt.imshow(red)
	plt.title('Red Channel')
	plt.xticks([])
	plt.yticks([])

	#green
	ax3 = fig.add_subplot(223)
	plt.imshow(green)
	plt.title('Green Channel')
	plt.xticks([])
	plt.yticks([])

	##blue
	ax4 = fig.add_subplot(224)
	plt.imshow(blue)
	plt.title('Blue Channel')
	plt.xticks([])
	plt.yticks([])

	plt.tight_layout()
	plt.show(fig)

def show_two_images(img1, img2, title1, title2):
	fig = plt.figure()

	ax1 = fig.add_subplot(221)
	plt.imshow(img1)
	plt.title(title1)
	plt.xticks([])
	plt.yticks([])

	ax2=fig.add_subplot(222)
	plt.imshow(img2)
	plt.title(title2)
	plt.xticks([])
	plt.yticks([])

	plt.tight_layout()
	plt.show(fig)
			

def mean_map(err_map1, err_map2):
	shape = err_map1.shape
	assert shape == err_map2.shape, 'Error maps are not compatible'
	avg_map = np.zeros(shape)
	for i in xrange(shape[0]):
		for j in xrange(shape[1]):
			avg_map[i][j] = (err_map1[i][j] + err_map2[i][j])/2.

	return avg_map


def get_mean_maps(err_maps):
	shape = err_maps[0].shape
	assert len(shape) ==2, 'error maps must be two dimensional!'
	N = len(err_maps)
	for z in xrange(N):
		assert shape==err_maps[z].shape, 'Error map ' + str(z+1) + ' is not compatible'

	avg_map = np.zeros(shape)
	for i in xrange(shape[0]):
		for j in xrange(shape[1]):
			for k in xrange(N):
				err_map = err_maps[k]
				avg_map[i][j] += err_map[i][j]
	# now we divide
	for n in xrange(shape[0]):
		for m in xrange(shape[1]):
			avg_map[n][m] = avg_map[n][m]/N
	return avg_map

def mean_maps(all_err_maps):
	assert len(all_err_maps)>0 and type(all_err_maps) == tuple, "all error maps must be a tuple of error maps containing at least one!"
	num_maps = len(all_err_maps)
	shape = all_err_maps[0].shape
	N = shape[0]
	avg_maps = []
	for i in xrange(num_maps):
		assert all_err_maps[i].shape == shape, "error maps must all be same shape to be averaged"
	
	for j in xrange(N):
		err_maps = []
		for k in xrange(num_maps):
			errmaps = all_err_maps[k]
			if len(errmaps) ==4:
				errmaps = np.reshape(errmaps, (N, shape[1], shape[2]))
			err_maps.append(errmaps[j])
		avg_maps.append(get_mean_maps(err_maps))
	avg_maps = np.array(avg_maps)
	return avg_maps	

def reshape_into_image(img):
	shape = img.shape
	print shape
	print len(img)
	if len(shape) ==4:
		return np.reshape(img, (len(img), shape[1], shape[2]))
	if len(shape)==3:
		if shape[2] > 3:
			return img
		if shape[2] >1 and shape[2] <=3:
			return img
		if shape[2] ==1:
			return np.reshape(img, (shape[0], shape[1]))
	#return np.reshape(img, (len(img), shape, shape))
	return img

def process_img_array(imgarray, f):
	imglist = []
	for i in xrange(len(imgarray)):
		newimg = f(imgarray[i])
		imglist.append(newimg)
	imglist = np.array(imglist)
	return imglist


def split_image_dataset_into_halves(imgs):
	shape = imgs.shape
	if len(shape) == 4:
		width = shape[2]
		half1 = imgs[:,:,0:width/2,:]
		half2 = imgs[:,:,width/2:width,:]
		return half1, half2

	if len(shape)==3:
		width = shape[2]
		half1 = imgs[:,:,0:width/2]
		half2 = imgs[:,:,width/2:width]
		return half1, half2


def split_dataset_by_colour(data, reshape=True):
	red = data[:,:,:,0]
	blue = data[:,:,:,1]
	green = data[:,:,:,2]
	if reshape:
		red = np.reshape(red, (red.shape[0], red.shape[1], red.shape[2], 1))
		blue = np.reshape(blue, (blue.shape[0], blue.shape[1], blue.shape[2], 1))
		green = np.reshape(green, (green.shape[0], green.shape[1], green.shape[2],1))
	print red.shape
	return [red, blue, green]

def split_img_by_colour(img, reshape = False):
	red = img[:,:,0]
	blue = img[:,:,1]
	green = img[:,:,2]
	if reshape:
		red = np.reshape(red, (red.shape[0], red.shape[1], red.shape[2], 1))
		blue = np.reshape(blue, (blue.shape[0], blue.shape[1], blue.shape[2], 1))
		green = np.reshape(green, (green.shape[0], green.shape[1], green.shape[2],1))
	print red.shape
	return [red, blue, green]

def split_into_test_train(data, frac_train = 0.9, frac_test = 0.1):
	assert frac_train + frac_test == 1, 'fractions must add up to one'
	length = len(data)
	train = data[0:int(frac_train*length)]
	test = data[int(frac_train*length): length]
	return train, test

def get_error(err_map, sal_map, accuracy = True, verbose = True):
	salience_error_map = np.abs(err_map - sal_map)
	error = np.sum(salience_error_map)
	print "ERROR: " + str(error)
	if accuracy: 
		shape = salience_error_map.shape
		assert len(shape) == 2, ' salience error map should be two dimensional'
		dim = shape[0] * shape[1]
		acc = float(error) / float(dim)
		print "Accuracy (normalised error): " + str(accuracy)
		return error, accuracy
	
	return error


def get_errors(mean_maps, sal_maps, total_error = True, verbose = False, save=True, save_name = '_errors', err_list = True, avg_error = True):
	errmaps = np.abs(mean_maps - sal_maps)
	N = len(errmaps)
	ret = []
	if total_error:
		total = float(np.sum(errmaps))/float(N)
		ret.append(total_error)

	if avg_error:
		if total_error:
			avg = total/float(N)
		if not total_error:
			avg= float(np.sum(errmaps))/float(N)
		ret.append(avg)

	if err_list:
		errslist = []
		for i in xrange(N):
			err = float(np.sum(errmaps[i]))
			errslist.append(err)

		errslist = np.array(errslist)
		ret.append(errslist)

	if save:
		save_array(ret, save_name)

	return ret



def process_hyperparams_error(results_file, saliences_file, params, param_name, err_list=False, save=True, save_name=None, info=True):

	res_dict = {}
	if err_list:
		list_dict = {}

	res = load_array(results_file)
	N = len(res)
	assert N == len(params) and type(res) == list, "Your number of paramaters and results do not match"
	assert type(param_name) == str and len(param_name)>=1

	shape = res[0][3].shape
	sals = load_array(saliences_file)
	sal_shape = sals.shape
	print shape
	print sal_shape
	assert shape==sal_shape, "results and salience shapes must be the same. You probably have the wrong salience file"
	for i in xrange(N):
		assert res[i][3].shape == shape, "all results should be of the same shape!"
		m_maps = res[i][3]
		ret = get_errors(m_maps, sals, save=False, err_list=err_list)
		avg_error = ret[1]
		key_string = param_name + " " + str(params[i])
		res_dict[avg_error] = key_string
		if err_list:
			errs = ret[2]
			list_dict[key_string] = errs

	
	if save:
		if save_name is None:
			save_name= param_name + "_error_dict"
		if err_list:
			save_array((res_dict, list_dict), save_name)
		if not err_list:
			save_array(res_dict, save_name)

	if info:
		print "Hyperparamter results: \n"
		print_res_dict(res_dict)
		mu, std = get_average_loss_std(res_dict)
		print "Mean loss: " + str(mu) + "\n"
		print "Standard Deviation of loss: " + str(std) + "\n"
		name, loss = get_min_loss(res_dict)
		print "The minimum loss is: " + name + ": " + str(loss)
		print " "
	
	if err_list:
		return (res_dict, list_dict)
	return res_dict

def get_average_loss_std(res_dict):
	ks = res_dict.keys()
	mu = np.mean(ks)
	sigma = np.std(ks)
	return (mu, sigma)

def print_res_dict(res_dict):
	for k,v in res_dict.iteritems():
		print str(k) + " : " + str(v) + "\n"


def get_min_loss(res_dict, name=True):
	min_loss = min(res_dict.keys())
	if name:
		return (res_dict[min_loss], min_loss)
	return min_loss


def gaussian_filter(img, sigma = 4):
	img = reshape_into_image(img)
	return scipy.ndimage.filters.gaussian_filter(img, sigma)


def compare_saliences(smaps1, smaps2, maps=True, verbose = True, save=False,save_name=None, show=False, N = 10, start =0):
	shape = smaps1.shape
	if verbose:
		print "Shape smap1: " + str(shape)
		print "Shape smap2: " + str(smaps2.shape)
	assert shape== smaps2.shape, 'saliency maps must be same shape'

	errslist = []
	if maps:
		maplist = []
	for i in xrange(len(smaps1)):
		errmap = np.absolute(smaps1[i] -smaps2[i])
		err = np.sum(errmap) / (shape[1] * shape[2])
		if maps:
			maplist.append(errmap)
		errslist.append(err)
	if maps:
		mapslist = np.array(mapslist)
		if save and save_name is not None:
			save(mapslist, save_name + '_maps')
			save(errslist, save_name+'_errors')
		return errslist, mapslist

	if save and save_name is not None:
		save(mapslist, save_name + '_maps')
		save(errslist, save_name+'_errors')

	if show:
		for i in xrange(N):
			compare_two_images(smaps1[start+i], smaps2[start+i],'predicted','actual')
	
	return errslist
	


def get_amplitude_spectrum(img, mult = 255, img_type = 'uint8', show = False, type_convert=True):
	# first we get the fft of the image
	img_amp = np.fft.fft2(img)
	#then we turn it to the amplitude spectrum
	img_amp = np.fft.fftshift(np.abs(img_amp))
	#we ten take logarithms
	img_amp = np.log(img_amp + 1e-8)
	#we resscale to -1:+1 for displays
	img_amp = (((img_amp - np.min(img_amp))*2)/np.ptp(img_amp)) -1
	#we then multiply it out and cast it to type displayable in matplotlib
	if type_convert:
		img_amp = (img_amp * mult).astype(img_type)

	else:
		img_amp = img_amp * mult

	#we then show if we want to
	if show:
		plt.imshow(img_amp)
		plt.show()

	#and then return
	return img_amp

def get_fft(img):
	return np.fft.fft2(img)

def get_magnitude_spectrum(img, show=False, type_convert=True, img_type='uint8'):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	#print magnitude_spectrum
	if type_convert:
		magnitude_spectrum = magnitude_spectrum.astype(img_type)
	
	if show:
		#we plot the original image
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Original Image')
		plt.xticks([])
		plt.yticks([])
	
	#transformed image
		plt.subplot(131)
		plt.imshow(magnitude_spectrum, cmap='gray')
		plt.title('Magnitude Spectrum')
		plt.xticks([])
		plt.yticks([])
		plt.show()

	#we then return the magnitude spectrum
	return magnitude_spectrum

def get_fft_shift(img):
	f = np.fft.fft2(img)
	return np.fft.fftshift(f)
		


def high_pass_filter(img, filter_width = 10, show = False):

	fshift = get_fft_shift(img)

	rows, cols = img.shape
	crow, ccol = rows/2, cols/2
	fshift[crow-filter_width: crow+filter_width, ccol-filter_width: ccol+filter_width] = 0
	#we start to transform it back
	f_ishift = np.fft.ifftshift(fshift)
	img_back = np.fft.ifft2(f_ishift)
	img_back = np.abs(img_back)

	if show:
		#get original image
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Input Image')
		plt.xticks([])
		plt.yticks([])

		#plot filtered image
		plt.subplot(122)
		plt.imshow(img_back, cmap='gray')
		plt.title('Image after HPF')
		plt.xticks([])
		plt.yticks([])
		
		plt.show()

	return img_back


def butter2d_lp(shape, f, n, pxd=1):
    """Designs an n-th order lowpass 2D Butterworth filter with cutoff
   frequency f. pxd defines the number of pixels per unit of frequency (e.g.,
   degrees of visual angle)."""
    pxd = float(pxd)
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)  * cols / pxd
    y = np.linspace(-0.5, 0.5, rows)  * rows / pxd
    radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
    filt = 1 / (1.0 + (radius / f)**(2*n))
    return filt
 
def butter2d_bp(shape, cutin, cutoff, n, pxd=1):
    """Designs an n-th order bandpass 2D Butterworth filter with cutin and
   cutoff frequencies. pxd defines the number of pixels per unit of frequency
   (e.g., degrees of visual angle)."""
    return butter2d_lp(shape,cutoff,n,pxd) - butter2d_lp(shape,cutin,n,pxd)
 
def butter2d_hp(shape, f, n, pxd=1):
    """Designs an n-th order highpass 2D Butterworth filter with cutin
   frequency f. pxd defines the number of pixels per unit of frequency (e.g.,
   degrees of visual angle)."""
    return 1. - butter2d_lp(shape, f, n, pxd)
 
def ideal2d_lp(shape, f, pxd=1):
    """Designs an ideal filter with cutoff frequency f. pxd defines the number
   of pixels per unit of frequency (e.g., degrees of visual angle)."""
    pxd = float(pxd)
    rows, cols = shape
    x = np.linspace(-0.5, 0.5, cols)  * cols / pxd
    y = np.linspace(-0.5, 0.5, rows)  * rows / pxd
    radius = np.sqrt((x**2)[np.newaxis] + (y**2)[:, np.newaxis])
    filt = np.ones(shape)
    filt[radius>f] = 0
    return filt
 
def ideal2d_bp(shape, cutin, cutoff, pxd=1):
    """Designs an ideal filter with cutin and cutoff frequencies. pxd defines
   the number of pixels per unit of frequency (e.g., degrees of visual
   angle)."""
    return ideal2d_lp(shape,cutoff,pxd) - ideal2d_lp(shape,cutin,pxd)
 
def ideal2d_hp(shape, f, n, pxd=1):
    """Designs an ideal filter with cutin frequency f. pxd defines the number
   of pixels per unit of frequency (e.g., degrees of visual angle)."""
    return 1. - ideal2d_lp(shape, f, n, pxd)
 
def bandpass(data, highpass, lowpass, n, pxd, eq='histogram'):
    """Designs then applies a 2D bandpass filter to the data array. If n is
   None, and ideal filter (with perfectly sharp transitions) is used
   instead."""
    fft = np.fft.fftshift(np.fft.fft2(data))
    if n:
        H = butter2d_bp(data.shape, highpass, lowpass, n, pxd)
    else:
        H = ideal2d_bp(data.shape, highpass, lowpass, pxd)
    fft_new = fft * H
    new_image = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_new)))    
    if eq == 'histogram':
        new_image = exposure.equalize_hist(new_image)
    return new_image
	
def log_transformed_fft(img, show= False):
	fft = np.fft.fftshift(np.fft.fft2(img))
	log = np.log(np.abs(fft))
	if show:
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Input Image')
		plt.xticks([])
		plt.yticks([])

		#plot filtered image
		plt.subplot(122)
		plt.imshow(new_img, cmap='gray')
		plt.title('Image after Lowpass Filter')
		plt.xticks([])
		plt.yticks([])
		
		plt.show()
	return log

def lowpass_filter(img, show = False):
	fft = np.fft.fftshift(np.fft.fft2(img))
	filt = butter2d_lp(img.shape, 0.2, 2, pxd = 43)
	fft_new = fft * filt
	new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_new)))
	new_img = exposure.equalize_hist(new_img)
	if show:
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Input Image')
		plt.xticks([])
		plt.yticks([])

		plt.subplot(122)
		plt.imshow(new_img, cmap='gray')
		plt.title('Image after Low pass filter')
		plt.xticks([])
		plt.yticks([])
		
		plt.show()
	return new_img

def highpass_filter(img, show = False):
	fft = np.fft.fftshift(np.fft.fft2(img))
	filt = butter2d_hp(img.shape, 0.2,2, pxd=43)
	fft_new = fft * filt
	new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_new)))
	new_img = exposure.equalize_hist(new_img)
	if show:
		
		#get original image
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Input Image')
		plt.xticks([])
		plt.yticks([])

		#plot filtered image
		plt.subplot(122)
		plt.imshow(new_img, cmap='gray')
		plt.title('Image after Highpass filter')
		plt.xticks([])
		plt.yticks([])
		
		plt.show()
	return new_img

def bandpass_filter(img, show = False):
	fft = np.fft.fftshift(np.fft.fft2(img))
	filt = butter2d_bp(img.shape, 1.50001, 1.50002,2, pxd=43)
	fft_new = fft * filt
	new_img = np.abs(np.fft.ifft2(np.fft.ifftshift(fft_new)))
	new_img = exposure.equalize_hist(new_img)
	if show:
		plt.subplot(121)
		plt.imshow(img, cmap='gray')
		plt.title('Input Image')
		plt.xticks([])
		plt.yticks([])

		#plot filtered image
		plt.subplot(122)
		plt.imshow(new_img, cmap='gray')
		plt.title('Image after Bandpass Filter')
		plt.xticks([])
		plt.yticks([])
		
		plt.show()
	return new_img

	
def filter_dataset(dataset, f, N = -1):
	if N == -1:
		N = len(dataset)

	fftlist = []
	for i in xrange(N):
		fftlist.append(f(dataset[i]))
	fftlist = np.array(fftlist)
	return fftlist


def split_dataset_spatial_frequency(dataset, save=True, save_name=None):
	lp = filter_dataset(dataset, lowpass_filter)
	hp = filter_dataset(dataset, highpass_filter)
	bp = filter_dataset(dataset, bandpass_filter)
	res = (lp, hp, bp)
	if save:
		if save_name is None:
			save_name = 'spatial_frequency_split_lp_hp_bp'
		save_array(res, save_name)
	return res
