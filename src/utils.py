from __future__ import division
import numpy as np
import math
import h5py

def one_hot_encoding_batch_per_point(y, num_classes):
		y_one_hot = np.zeros((y.shape[0], y.shape[1], num_classes))
		for i in xrange(y.shape[0]):
			for j in xrange(y.shape[1]):
					y_one_hot[i, j, y[i][j]] = 1
		return y_one_hot

def one_hot_encoding_batch(y, num_classes):
		y_one_hot = np.zeros((y.shape[0], num_classes))
		for i in xrange(y.shape[0]):
				y_one_hot[i, y[i]] = 1
		return y_one_hot

def one_hot_encoding(y, num_classes):
		y_one_hot = np.zeros((num_classes))
		y_one_hot[y] = 1
		return y_one_hot

