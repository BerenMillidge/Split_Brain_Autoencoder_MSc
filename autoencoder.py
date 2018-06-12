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
import keras

# define reasonable defaults
batch_size = 25
dropout = 0.3
verbose = True
architecture = None
activation = 'relu'
padding = 'same'

epochs = 20

lrate = 0.001
decay = 1e-6
momentum = 0.9
nesterov = True
shuffle = True
loss = 'binary_crossentropy'

gestalt = False

optimizer = optimizers.SGD(lr =lrate, decay=decay, momentum = momentum, nesterov = nesterov)

default_callbacks = [keras.callbacks.TerminateOnNaN()]


class Hemisphere(object):
	
	def __init__(self,input_data, output_data,test_input, test_output, batch_size = batch_size, dropout = dropout, verbose = verbose, architecture = architecture, activation = activation, padding = padding, optimizer = optimizer, epochs = epochs, loss=loss):

		#init our paramaters
		self.input_data = input_data
		self.output_data = output_data
		self.test_input = test_input
		self.test_output = test_output
		self.batch_size = batch_size
		self.dropout = dropout
		self.verbose = verbose
		self.architecture = architecture
		self.activation = activation
		self.padding = padding
		self.optimizer = optimizer
		self.epochs = epochs
		self.loss = loss

		self.shape = input_data.shape
		#assert self.shape == output_data.shape, 'input and output data do not have the same shape'

		if architecture is not None:
			# initialise and set it up, but we've not impleneted it yet
			raise NotImplementedError
			pass
		if architecture is None:
				
			if verbose:
				print self.shape
			
			input_img = Input(shape=(self.shape[1], self.shape[2], self.shape[3]))

			x = Conv2D(16, (3, 3), activation=self.activation, padding=self.padding)(input_img)
			if verbose:
				print x.shape
			#x = MaxPooling2D((2, 2), padding='same')(x)
			#print x.shape
			x = Conv2D(8, (3, 3), activation=self.activation, padding=self.padding)(x)
			if verbose:
				print x.shape
			#x = MaxPooling2D((2, 2), padding='same')(x)
			#print x.shape
			x = Conv2D(8, (3, 3), activation=self.activation, padding=self.padding)(x)
			if verbose:
				print x.shape
			encoded = MaxPooling2D((2, 2), padding='same')(x)
			if verbose:
				print encoded.shape
				print "  "

			# at this point the representation is (4, 4, 8) i.e. 128-dimensional

			x = Conv2D(8, (3, 3), activation=self.activation, padding=self.padding)(encoded)
			if verbose:
				print x.shape
			x = UpSampling2D((2, 2))(x)
			if verbose:
				print x.shape
			x = Conv2D(8, (3, 3), activation=self.activation, padding=self.padding)(x)
			if verbose:
				print x.shape
			#x = UpSampling2D((2, 2))(x)
			#print x.shape
			x = Conv2D(16, (3, 3), activation=self.activation, padding=self.padding)(x)
			if verbose:
				print x.shape
			#x = UpSampling2D((2, 2))(x)
			#print x.shape
			decoded = Conv2D(1, (3, 3), activation='sigmoid', padding=self.padding)(x)
			if verbose:
				print decoded.shape
			
			if gestalt:
				#sh = self.output_data.shape
				#final = Dense(20, activation='sigmoid')(decoded)
				#final = Reshape((sh[1], sh[2], sh[3]))(decoded)
				print decoded.shape
				dec = decoded.shape
				print dec
				final = Permute((1,3,2))(decoded)
				print final.shape
				final = Dense(20, activation='sigmoid')(final)
				print final.shape
				final = Permute((1,3,2))(final)
				print final.shape
				#if verbose:
				#	print final.shape

			self.model = Model(input_img, decoded)
			if gestalt:
				self.model = Model(input_img, final)
			print self.model.summary()
			self.model.compile(optimizer = self.optimizer, loss = self.loss)


	def train(self, epochs = None, shuffle=True, callbacks = None, get_weights=False):
		if epochs is None:
			epochs = self.epochs
		print "Model training:"
		history = self.model.fit(self.input_data, self.output_data, epochs=epochs, shuffle = shuffle, callbacks = callbacks)
		print "Training complete"
		if get_weights:
			weights, biases= self.model.layers[-2].get_weights()
			print weights
			print biases
			return (history, weights, biases)
		return history

	def predict(self, test_data = None):
		if test_data is not None:
			return self.model.predict(test_data)
		if test_data is None:
			return self.model.predict(self.test_input)

	def plot_results(self, preds = None, inputs=None, N = 10, start = 0):
		if preds is None:
			preds = self.predict()
		if inputs is None:
			inputs = self.test_output

		print preds.shape
		shape = (preds.shape[1],preds.shape[2])
			
		fig = plt.figure(figsize=(20,4))
		for i in range(N):
			#display original
			ax = plt.subplot(2,N,i+1)
			plt.imshow(inputs[start + i].reshape(shape))
			plt.gray()
			plt.title('original')
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)

			#display reconstructoin
			ax = plt.subplot(2, N, i+1+N)
			plt.imshow(preds[start + i].reshape(shape))
			plt.gray()
			plt.title('reconstruction')
			ax.get_xaxis().set_visible(False)
			ax.get_yaxis().set_visible(False)
	
		plt.show()
		return fig
		
	
	def get_error_maps(self, input_data = None, predictions= None, return_preds = False):
		if input_data is None:
			input_data = self.test_input
		if predictions is None:
			predictions = self.predict(test_data = input_data)
		maps = np.absolute(predictions - self.test_output)
		assert input_data.shape == predictions.shape, 'predictoins and input data must have same dimensions'
		shape = predictions.shape

		if return_preds:
			return predictions, np.reshape(maps,(shape[0], shape[1], shape[2]))
		return np.reshape(maps, (shape[0], shape[1], shape[2]))

	def plot_error_maps(self, error_maps = None, N = 10, original_images = None, predictions = None):
		if error_maps is None:
			error_maps = get_error_maps()
		if original_images is None:
			shape = self.test_input.shape
			original_images = np.reshape(self.test_output, (shape[0], shape[1], shape[2]))
		
		if predictions is None:
			for i in xrange(N):
				compare_two_images(original_images[i], error_maps[i], 'Original', 'Error Map')
		if predictions is not None:
			for i in xrange(N):
				imgs = (original_images[i], predictions[i], error_maps[i])
				titles = ('Original','Prediction','Error Map')
				compare_images(imgs, titles)


	def generate_mean_maps(self, error_maps1, error_maps2, N = -1):
		n = len(error_maps1)
		assert len(error_maps2) != n, 'different numbers of maps in each'
		if N == -1:
			N = n
		mean_maps = []
		for i in xrange(N):
			mean_maps.append(mean_map(error_maps1[i], error_maps2[i]))
		mean_maps = np.array(mean_maps)
		return mean_maps

	def plot_mean_error_maps(self,mean_maps, N = 10):
		if N == -1:
			N = len(mean_maps)
		self.plot_error_maps(mean_maps, N)
		

		
