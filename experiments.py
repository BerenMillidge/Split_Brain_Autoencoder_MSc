from keras.datasets import cifar10, mnist
from matplotlib import pyplot as plt
from scipy.misc import toimage
import numpy as np
from keras.datasets import cifar10
from keras.layers import *
from keras.models import Model
from keras.constraints import maxnorm
from keras import optimizers
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
from keras.callbacks import TensorBoard
from file_reader import *
from utils import *
from autoencoder import *
import scipy
import sys

#seed = 8
#np.random.seed(seed)

def normalise(data):
	return data.astype('float32')/255.0


def load_colour_split_cifar(test_up_to = None):

	(xtrain, ytrain), (xtest, ytest) = cifar10.load_data()

	xtrain = normalise(xtrain)
	xtest = normalise(xtest)

	redtrain, greentrain, bluetrain = split_dataset_by_colour(xtrain)
	redtest, greentest, bluetest = split_dataset_by_colour(xtest)


	redtrain = np.reshape(redtrain, (len(redtrain), 32,32,1))
	greentrain = np.reshape(greentrain, (len(greentrain), 32,32,1))
	bluetrain = np.reshape(bluetrain, (len(bluetrain), 32,32,1))
	redtest = np.reshape(redtest, (len(redtest), 32,32,1))
	greentest = np.reshape(greentest, (len(greentest), 32,32,1))
	bluetest = np.reshape(bluetest, (len(bluetest), 32,32,1))

	if test_up_to is not None:
		redtrain = redtrain[0:test_up_to,:,:,:]
		greentrain = greentrain[0:test_up_to,:,:,:]
		bluetrain = bluetrain[0:test_up_to,:,:,:]
		redtest = redtest[test_up_to:test_up_to*2,:,:,:]
		bluetest = bluetest[test_up_to:test_up_to*2,:,:,:]
		greentest = greentest[test_up_to:test_up_to*2,:,:,:]

	return redtrain, greentrain, bluetrain, redtest, greentest, bluetest


def load_half_split_cifar(col = 1, test_up_to = None):
	(xtrain, ytrain), (xtest,ytest) = cifar10.load_data()
	xtrain = normalise(xtrain)
	xtest = normalise(xtest)
	
	redtrain, greentrain, bluetrain = split_dataset_by_colour(xtrain)
	redtest, greentest, bluetest = split_dataset_by_colour(xtest)

	redtrain = np.reshape(redtrain, (len(redtrain), 32,32,1))
	greentrain = np.reshape(greentrain, (len(greentrain), 32,32,1))
	bluetrain = np.reshape(bluetrain, (len(bluetrain), 32,32,1))
	redtest = np.reshape(redtest, (len(redtest), 32,32,1))
	greentest = np.reshape(greentest, (len(greentest), 32,32,1))
	bluetest = np.reshape(bluetest, (len(bluetest), 32,32,1))

	if test_up_to is not None:
		redtrain = redtrain[0:test_up_to,:,:,:]
		greentrain = greentrain[0:test_up_to,:,:,:]
		bluetrain = bluetrain[0:test_up_to,:,:,:]
		redtest = redtest[test_up_to:test_up_to*2,:,:,:]
		bluetest = bluetest[test_up_to:test_up_to*2,:,:,:]
		greentest = greentest[test_up_to:test_up_to*2,:,:,:]

	half1train, half2train = split_image_dataset_into_halves(redtrain)
	half1test, half2test = split_image_dataset_into_halves(redtest)
	
	return half1train, half2train, half1test, half2test

def load_spatial_frequency_split_cifar(test_up_to=None):
	(xtrain, ytrain), (xtest,ytest) = cifar10.load_data()
	xtrain = normalise(xtrain)
	xtest = normalise(xtest)
	
	redtrain, greentrain, bluetrain = split_dataset_by_colour(xtrain)
	redtest, greentest, bluetest = split_dataset_by_colour(xtest)

	redtrain = np.reshape(redtrain, (len(redtrain), 32,32,1))
	greentrain = np.reshape(greentrain, (len(greentrain), 32,32,1))
	bluetrain = np.reshape(bluetrain, (len(bluetrain), 32,32,1))
	redtest = np.reshape(redtest, (len(redtest), 32,32,1))
	greentest = np.reshape(greentest, (len(greentest), 32,32,1))
	bluetest = np.reshape(bluetest, (len(bluetest), 32,32,1))

	if test_up_to is not None:
		redtrain = redtrain[0:test_up_to,:,:,:]
		greentrain = greentrain[0:test_up_to,:,:,:]
		bluetrain = bluetrain[0:test_up_to,:,:,:]
		redtest = redtest[test_up_to:test_up_to*2,:,:,:]
		bluetest = bluetest[test_up_to:test_up_to*2,:,:,:]
		greentest = greentest[test_up_to:test_up_to*2,:,:,:]

	lptrain = filter_dataset(redtrain, lowpass_filter)
	lptest = filter_dataset(redtest, lowpass_filter)
	hptrain = filter_dataset(redtrain, highpass_filter)
	hptest = filter_dataset(redtest, highpass_filter)

	#bptrain = filter_dataset(redtrain, bandpass_filter)
	#bptest = filter_dataset(redtest, bandpass_filter)

	return lptrain, lptest, hptrain, hptest
		

def run_colour_experiments(epochs = 1, save=True, test_up_to=None):

	redtrain, greentrain, bluetrain, redtest, greentest, bluetest = load_colour_split_cifar(test_up_to=test_up_to)
	#compare images here
	#for i in xrange(10):
	#	compare_two_images(redtrain[i], greentrain[i], reshape=True)

	a1 = Hemisphere(redtrain, greentrain, redtest, greentest,verbose=True)
	a2 = Hemisphere(greentrain, redtrain, greentest, redtest)

	
	a1.train(epochs=epochs, get_weights=True)
	a2.train(epochs=epochs)

	a1.plot_results()
	a2.plot_results()

	errmap1 = a1.get_error_maps()
	errmap2 = a2.get_error_maps()

	a1.plot_error_maps(errmap1)
	a2.plot_error_maps(errmap2)

	errmaps = [errmap1, errmap2]

	if save:
		save(errmaps, 'colour_red_green_errormaps')

	return errmaps
	

def run_half_split_experiments(epochs = 1, save=True,test_up_to=None, history=True):
	
	half1train, half2train, half1test, half2test = load_half_split_cifar(test_up_to=test_up_to)

	a1 = Hemisphere(half1train, half2train, half1test, half2test)
	a2 = Hemisphere(half2train, half1train, half2test, half1test)

	his1 =a1.train(epochs=epochs)
	his2 = a2.train(epochs=epochs)
	
	a1.plot_results()
	a2.plot_results()

	errmap1 = a1.get_error_maps()
	errmap2 = a2.get_error_maps()

	a1.plot_error_maps(errmap1)
	a2.plot_error_maps(errmap2)

	errmaps = [errmap1, errmap2]

	if save:
		save(errmaps, 'colour_red_green_errormaps_split_half')

	if history:
		his1 = serialize_class_object(his1)
		his2 = serialize_class_object(his2)
		return (errmaps, his1, his2)

	return errmaps

def run_spatial_frequency_split_experiments(epochs=1, save=True, test_up_to=None):
	
	lptrain, lptest, hptrain, hptest = load_spatial_frequency_split_cifar(test_up_to=test_up_to)

	a1 = Hemisphere(lptrain, hptrain, lptest, hptest)
	a2 = Hemisphere(hptrain, lptrain, hptest, lptest)

	a1.train(epochs=10)
	a2.train(epochs=10)

	a1.plot_results()
	a2.plot_results()

	errmap1 = a1.get_error_maps()
	errmap2 = a2.get_error_maps()

	a1.plot_error_maps(errmap1)
	a2.plot_error_maps(errmap2)

	errmaps = [errmap1, errmap2]

	if save:
		save(errmaps, 'colour_red_green_errormaps')

	return errmaps


def run_benchmark_image_set_experiments(epochs=100, save=True, test_up_to=None):
	imgs = load('BenchmarkDATA/BenchmarkIMAGES_images')
	imgs= normalise(imgs)
	print imgs.shape
	red, green,blue = split_dataset_by_colour(imgs)
	print red.shape
	redtrain, redtest = split_into_test_train(red)
	print redtrain.shape
	greentrain, greentest = split_into_test_train(green)
	print redtrain.shape
	print redtest.shape

	a1 = Hemisphere(redtrain, redtrain, redtest, redtest)
	print "hemisphere initialised"
	
	a2 = Hemisphere(greentrain, greentrain, greentest, greentest)
	print "second hemisphere initialised"
	

	a1.train(epochs=epochs)
	print "a1 trained"
	
	a2.train(epochs=epochs)
	print "a2 trained"

	a1.plot_results()
	a2.plot_results()

	preds1, errmap1 = a1.get_error_maps(return_preds = True)
	preds2, errmap2 = a2.get_error_maps(return_preds=True)

	print errmap1[0]
	
	a1.plot_error_maps(errmap1, predictions=preds1)
	a2.plot_error_maps(errmap2,predictions=preds2)
	
	mean_maps = mean_map(errmap1, errmap2)
	a1.plot_error_maps(mean_maps)

	if save:
		save_array(mean_maps, 'benchmark_red_green_error_maps')
	return mean_maps


def run_spatial_frequency_split_experiments_images_from_file(fname, epochs=100, save=True, test_up_to=None, preview=False, verbose=False, param_name=None, param=None, save_name=None, test_all=False):
	imgs = load_array(fname)
	lp,hp,bp = imgs
	print lp.shape
	lp = normalise(lp)
	hp = normalise(hp)
	bp = normalise(bp)
	lptrain, lptest=  split_into_test_train(lp)
	hptrain, hptest = split_into_test_train(hp)

	if preview:
		for i in xrange(10):
			compare_two_images(lptrain[i], hptrain[i], reshape=True)

	if param_name is None or param is None:
		a1 = Hemisphere(lptrain, hptrain, lptest, hptest)
	if param_name is not None and param is not None:
		a1 = Hemisphere(lptrain, hptrain, lptest, hptest, param_name=param)
	if verbose:
		print "hemisphere initialised"
	if param_name is None or param is None:
		a2 = Hemisphere(lptrain, hptrain, lptest, hptest)
	if param_name is not None and param is not None:
		a2 = Hemisphere(lptrain, hptrain, lptest, hptest, param_name=param)
	if verbose:
		print "second hemisphere initialised"

	if test_all:
		a1=Hemisphere(lp, hp, lp, hp)
		a2=Hemisphere(hp,lp,hp,lp)
	

	a1.train(epochs=epochs)
	if verbose:
		print "a1 trained"
	
	a2.train(epochs=epochs)
	if verbose:
		print "a2 trained"

	a1.plot_results()
	a2.plot_results()

	preds1, errmap1 = a1.get_error_maps(return_preds = True)
	preds2, errmap2 = a2.get_error_maps(return_preds=True)

	if save:
		if save_name is None:
			save_array((redtest, preds1, errmap1),fname+'spfreq_imgs_preds_errmaps')
		if save_name is not None:
			save_array((redtest, preds1, errmap1), save_name + 'spfreq_imgs_preds_errmaps')

	if verbose:
		print errmap1[0]
	
	a1.plot_error_maps(errmap1, predictions=preds1)
	a2.plot_error_maps(errmap2,predictions=preds2)
	
	mean_maps = mean_map(errmap1, errmap2)
	a1.plot_error_maps(mean_maps)

	if save:
		if save_name is None:
			save_array(mean_maps, 'benchmark_spfreq_red_green_error_maps')
		if save_name is not None:
			save_array(mean_maps, save_name + '_mean_maps')
	return mean_maps


def run_colour_split_experiments_images_from_file_with_all_hyperparams(fname,epochs=100, save=False, test_up_to=None, verbose = False,lrate=0.001,decay=1e-6, momentum=0.9, nesterov=True, shuffle=True, loss='binary_crossentropy',activation='relu', dropout=0.3, padding='same', save_name = None, batch_size=25, optimizer=None):
	imgs = load(fname)
	imgs= normalise(imgs)
	red, green,blue = split_dataset_by_colour(imgs)
	redtrain, redtest = split_into_test_train(red)
	greentrain, greentest = split_into_test_train(green)

	if optimizer is None:
		optimizer = optimizers.SGD(lr =lrate, decay=decay, momentum = momentum, nesterov = nesterov)

	a1=Hemisphere(redtrain, greentrain, redtest, greentest, batch_size=batch_size, dropout=dropout,activation=activation, padding=padding, optimizer =optimizer, epochs=epochs, loss=loss)
	a2=Hemisphere(greentrain, redtrain, greentest, redtest, batch_size=batch_size, dropout=dropout,activation=activation, padding=padding, optimizer =optimizer, epochs=epochs, loss=loss)
	

	a1.train(epochs=epochs)
	if verbose:
		print "a1 trained"
	
	a2.train(epochs=epochs)
	if verbose:
		print "a2 trained"
	preds1, errmap1 = a1.get_error_maps(return_preds = True)
	preds2, errmap2 = a2.get_error_maps(return_preds=True)

	if save:
		if save_name is None:
			save_array((redtest, preds1, errmap1),fname+'_imgs_preds_errmaps')
		if save_name is not None:
			save_array((redtest, preds1, errmap1), save_name + '_imgs_preds_errmaps')

	if verbose:
		print errmap1[0]
	
	mean_maps = mean_map(errmap1, errmap2)

	if save:
		if save_name is None:
			save_array(mean_maps, 'benchmark_red_green_error_maps')
		if save_name is not None:
			save_array(mean_maps, save_name + '_mean_maps')
	return redtest, preds1, errmap1, mean_maps



def run_colour_split_experiments_images_from_file(fname,epochs=100, save=True, test_up_to=None, preview = False, verbose = False, param_name= None, param = None, save_name = None, test_all=False):
	imgs = load(fname)
	imgs= normalise(imgs)
	red, green,blue = split_dataset_by_colour(imgs)
	redtrain, redtest = split_into_test_train(red)
	greentrain, greentest = split_into_test_train(green)

	if preview:
		for i in xrange(10):
			compare_two_images(redtrain[i], greentrain[i], reshape=True)

	if param_name is None or param is None:
		a1 = Hemisphere(redtrain, greentrain, redtest, greentest)
	if param_name is not None and param is not None:
		a1 = Hemisphere(redtrain, greentrain, redtest, greentest, param_name=param)
	if verbose:
		print "hemisphere initialised"
	if param_name is None or param is None:
		a2 = Hemisphere(greentrain, redtrain, greentest, redtest)
	if param_name is not None and param is not None:
		a2 = Hemisphere(greentrain, redtrain, greentest, redtest, param_name=param)
	if verbose:
		print "second hemisphere initialised"

	if test_all:
		a1=Hemisphere(red, green, red, green)
		a2=Hemisphere(green,red,green,red)
	

	a1.train(epochs=epochs)
	if verbose:
		print "a1 trained"
	
	a2.train(epochs=epochs)
	if verbose:
		print "a2 trained"

	a1.plot_results()
	a2.plot_results()

	preds1, errmap1 = a1.get_error_maps(return_preds = True)
	preds2, errmap2 = a2.get_error_maps(return_preds=True)

	if save:
		if save_name is None:
			save_array((redtest, preds1, errmap1),fname+'_imgs_preds_errmaps')
		if save_name is not None:
			save_array((redtest, preds1, errmap1), save_name + '_imgs_preds_errmaps')

	if verbose:
		print errmap1[0]
	
	a1.plot_error_maps(errmap1, predictions=preds1)
	a2.plot_error_maps(errmap2,predictions=preds2)
	
	mean_maps = mean_map(errmap1, errmap2)
	a1.plot_error_maps(mean_maps)

	if save:
		if save_name is None:
			save_array(mean_maps, 'benchmark_red_green_error_maps')
		if save_name is not None:
			save_array(mean_maps, save_name + '_mean_maps')
	return mean_maps



def run_all_colour_split_experiments_images_from_file(fname,epochs=100, save=True, test_up_to=None, preview = False, verbose = False, param_name= None, param = None, save_name = None, test_all=False):
	imgs = load(fname)
	imgs= normalise(imgs)
	red, green,blue = split_dataset_by_colour(imgs)
	redtrain, redtest = split_into_test_train(red)
	greentrain, greentest = split_into_test_train(green)
	bluetrain, bluetest = split_into_test_train(blue)

	a1 = Hemisphere(redtrain, greentrain, redtest, greentest)
	a2 = Hemisphere(greentrain, redtrain, greentest, redtest)
	a3 = Hemisphere(redtrain, bluetrain, redtest, bluetest)
	a4 = Hemisphere(bluetrain, redtrain, bluetest, redtest)
	a5 = Hemisphere(greentrain, bluetrain, greentest, bluetest)
	a6 = Hemisphere(bluetrain, greentrain, bluetest, greentest)
	
	a1.train(epochs=epochs)
	if verbose:
		print "a1 trained"
	
	a2.train(epochs=epochs)
	if verbose:
		print "a2 trained"

	a3.train(epochs=epochs)
	if verbose:
		print "a3 trained"

	a4.train(epochs=epochs)
	if verbose:
		print "a4 trained"
	
	a5.train(epochs=epochs)
	if verbose:
		print "a25trained"

	a6.train(epochs=epochs)
	if verbose:
		print "a6 trained"

	a1.plot_results()
	a2.plot_results()
	a3.plot_results()
	a4.plot_results()
	a5.plot_results()
	a6.plot_results()


	preds1, errmap1 = a1.get_error_maps(return_preds = True)
	preds2, errmap2 = a2.get_error_maps(return_preds=True)
	preds3, errmap3 = a3.get_error_maps(return_preds = True)
	preds4, errmap4 = a4.get_error_maps(return_preds=True)
	preds5, errmap5 = a5.get_error_maps(return_preds = True)
	preds6, errmap6 = a6.get_error_maps(return_preds=True)

	if save:
		if save_name is None:
			save_array((redtest, preds1, errmap1),fname+'_imgs_preds_errmaps')
		if save_name is not None:
			save_array((redtest, preds1, errmap1), save_name + '_imgs_preds_errmaps')

	#if verbose:
	#	print errmap1[0]
	
	#a1.plot_error_maps(errmap1, predictions=preds1)
	#a2.plot_error_maps(errmap2,predictions=preds2)
	
	err_maps=(errmap1, errmap2, errmap3, errmap4, errmap5, errmap6)
	mean_map = mean_maps(err_maps)
	a1.plot_error_maps(mean_map)

	if save:
		if save_name is None:
			save_array(mean_map, 'benchmark_red_green_error_maps_all')
		if save_name is not None:
			save_array(mean_map, save_name + '_mean_maps')
	return mean_map


def compare_error_map_to_salience_map(err_fname, sal_fname, start = 100, gauss=False):
	tup = load(err_fname)
	errmap = tup[2]
	print "ERRMAP:"
	print errmap.shape
	salmap = load(sal_fname)
	N = int(len(salmap)/10)
	salmap=salmap[1710:1900,:,:,0]
	shape = salmap.shape
	salmap = np.reshape(salmap, (shape[0], shape[1], shape[2])) 
	print "SALMAP:"
	print salmap.shape
	preds = tup[1]
	test = tup[0]
	preds = np.reshape(preds, (shape[0], shape[1], shape[2]))
	test = np.reshape(test,(shape[0], shape[1], shape[2]))
	
	for i in xrange(50):
		if not gauss:
			imgs = (test[start + i], preds[start + i], errmap[start + i], salmap[start + i])
			titles=('test image', 'prediction', 'error map', 'target salience map')
			compare_images(imgs, titles)
		if gauss:
			sigma=2
			errm = gaussian_filter(errmap[start+i],sigma)
			imgs = (test[start + i], preds[start + i],errm, salmap[start + i])
			titles=('test image', 'prediction', 'error map', 'target salience map')
			compare_images(imgs, titles)
	#compare_saliences(errmap, salmap)

def compare_mean_map_to_salience_map(mmap_fname, sal_fname, start = 100, gauss=False, N = 50):
	mmap = load(mmap_fname)
	print "MEAN MAP:"
	print mmap.shape
	salmap = load(sal_fname)
	salmap=salmap[1710:1900,:,:,0]
	shape = salmap.shape
	salmap = np.reshape(salmap, (shape[0], shape[1], shape[2])) 
	print "SALMAP:"
	print salmap.shape
	for i in xrange(N):
		if not gauss:
			compare_two_images(mmap[start + i], salmap[start+i], 'mean error map', 'target salience map')
		if gauss:
			sigma = 2
			errm = gaussian_filter(mmap[start+i])
			compare_two_images(errm, salmap[start+i], 'mean error map', 'target salience map')
	



	
def hyperparam_grid_search(param_name, param_list, input_fname, save_base, epochs=100, error=True, error_list = False, sal_map_fname = 'testsaliences_combined', fn=run_colour_split_experiments_images_from_file):
	N = len(param_list)
	for i in xrange(N):
		save_name = save_base + '_' + param_name + '_test_'+str(i)
		mean_maps = fn(input_fname, epochs=epochs, param_name = param_name, param = param_list[i],save_name = save_name)
		if error:
			#we load and process the sal maps
			salmaps = load(sal_map_fname)
			salmaps = salmaps[:,:,:,0]
			shape = salmaps.shape
			salmaps = np.reshape(salmaps,(shape[0], shape[1], shape[2]))
			if error_list:
				err, errlist = get_errors(mean_maps, salmaps, error, error_list, save_name='save_base' + '_' + str(param) + '_'+str(param_list[i]) + '_errors')
			if not error_list:
				err = get_errors(mean_maps, salmaps, error, error_list, save_name='save_base' + '_' + str(param) + '_'+str(param_list[i]) + '_errors')


def multi_hyperparam_grid_search(param_names, param_lists, input_fname, save_bases, epochs=100, error=True, error_list = False, sal_map_fname = 'testsaliences_combined', fn=run_colour_split_experiments_images_from_file):
	N = len(param_names)
	assert N == len(param_lists) == len(save_bases), "each hyperparam must have param list and save base"
	for i in xrange(N):
		hyperparam_grid_search(param_names[i], param_lists[i], input_fname, save_bases[i], epochs=epochs, error=error, error_list=error_list, sal_map_fname = sal_map_fname, fn=fn)

def try_gestalt_model_for_normal(fname,epochs=100, both=True):
	imgs = load_array(fname)
	imgs = imgs.astype('float32')/255.
	red, green, blue = split_dataset_by_colour(imgs)
	redtrain, redtest = split_first_test_train(red)
	greentrain, greentest = split_first_test_train(green)
	shape = redtrain.shape

	model = SimpleConvDropoutBatchNorm((shape[1], shape[2], shape[3]))
	model.compile(optimizer='sgd',loss='mse')
	callbacks = build_callbacks("./")
	his=model.fit(redtrain, greentrain, epochs=epochs, batch_size=128, shuffle=True, validation_data=(redtest, greentest), callbacks=callbacks)

	if both:
		model2 = SimpleConvDropoutBatchNorm((shape[1], shape[2], shape[3]))
		model2.compile(optimizer='sgd', loss='mse')
		his2 = model2.fit(greentrain, redtrain, epochs=epochs, batch_size=128, shuffle=True, validation_data=(greentest, redtest), callbacks=callbacks)

	print "MODEL FITTED"

	preds1 = model.predict(redtest)
	history = serialize_class_object(his)
	res = [history, preds1, redtest, greentest]
	save_array(res, "STANDARD_WITH_GESTALT_AUTOENCODER_MODEL_1")

	if both:
		preds2 = model2.predict(greentest)
		history2 = serialize_class_object(his2)
		res2 = [history2, preds2, greentest, redtest]
		save_array(res2, "STANDARD_WITH_GESTALT_AUTOENCODER_MODEL_2")


def get_error_maps(preds, test):
	assert preds.shape ==  test.shape
	res = []
	for i in xrange(len(preds)):
		res.append(np.absolute(preds[i] - test[i]))
	res = np.array(res)
	return res


def plot_error_maps_saliences_from_preds(preds_fname, sal_fname, N = 20, save=True, save_name="gestalt/STANDARD_WITH_GESTALT_ERROR_MAPS"):
	history1, preds1, redtest, greentest = load_array(preds_fname + "_1")
	history2, preds2, greentest, redtest = load_array(preds_fname + "_2")
	errmaps1 = get_error_maps(preds1, redtest)
	errmaps2 = get_error_maps(preds2, greentest)
	errmaps = np.concatenate((errmaps1, errmaps2), axis=0)
	original_images = np.concatenate((redtest, greentest), axis=0)
	preds = np.concatenate((preds1, preds2), axis=0)
	# now we need to load the actual salince maps
	sal_maps = load_array(sal_fname)[:,:,:,0]
	_, sal_maps = split_first_test_train(sal_maps)
	sh = sal_maps.shape
	sal_maps = np.reshape(sal_maps, (sh[0], sh[1], sh[2], 1))
	print sal_maps.shape
	print preds1.shape
	print preds2.shape
	assert sal_maps.shape == preds1.shape == preds2.shape, 'all potential images must be same shape!'
	if save:
		save_array(errmaps, save_name)

	if N == -1:
		N = len(preds)
	#now we do the plotting
	for i in xrange(N):
		imgs = (original_images[i],preds[i], errmaps[i], sal_maps[i])
		titles = ('Original Image','Predicted Image', 'Error Map', 'Ground-truth salience map')
		compare_images(imgs, titles)

	return [errmaps, original_images]
	
	
	
	
def main():
	fname = ''
	save_name = 'test_results'
	epochs = 50
	if len(sys.argv) >=2:
		fname = sys.argv[1]
	if len(sys.argv)>=3:
		save_name = sys.argv[2]
	if len(sys.argv)>=4:
		epochs = int(sys.argv[3])
	if len(sys.argv) <=1:
		raise ValueError('Need to input a filename for the data when running the model')

	run_all_colour_split_experiments_images_from_file(fname, epochs=epochs, test_all=True, save_name=save_name)





if __name__ == '__main__':
	#run_colour_experiments(epochs=1, save=False)
	#run_spatial_frequency_split_experiments(epochs=1, save=False)
	#run_half_split_experiments(epochs=1, save=False)
	#run_benchmark_image_set_experiments(20)
	#run_colour_experiments(5, save=False, test_up_to=10)
	#run_colour_split_experiments_images_from_file('testimages_combined', epochs=50)
	#compare_error_map_to_salience_map('BenchmarkIMAGES_images', 'BenchmarkIMAGES_output')
	#compare_error_map_to_salience_map('testimages_combined_imgs_preds_errmaps', 'testsaliences_combined', gauss=True)

	#compare_mean_map_to_salience_map('benchmark_red_green_error_maps', 'testsaliences_combined', gauss=True)

	#run_colour_split_experiments_images_from_file('testimages_combined', epochs=50, test_all=True, save_name="all_errmaps")
	#run_spatial_frequency_split_experiments_images_from_file('benchmark_images_spatial_frequency_split', epochs=50,test_all=True, save_name='spfreq_errmaps')
	#run_half_split_experiments()

	
	#run_all_colour_split_experiments_images_from_file('testimages_combined', epochs=50, test_all=True, save_name="all_errmaps_all_colour_combinations")
	main()