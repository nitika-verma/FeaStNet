from __future__ import division
import tensorflow as tf
import numpy as np
import math
import time
import h5py
import argparse
from model import *
from utils import * 

random_seed = 0
np.random.seed(random_seed)

sess = tf.InteractiveSession()

parser = argparse.ArgumentParser()
parser.add_argument('--architecture', type=int, default=0)
parser.add_argument('--dataset_path')
parser.add_argument('--results_path')
parser.add_argument('--num_iterations', type=int, default=50000)
parser.add_argument('--num_input_channels', type=int, default=3)
parser.add_argument('--learning_rate', type=float, default=0.001)
parser.add_argument('--num_points', type=int)
parser.add_argument('--num_classes', type=int)

FLAGS = parser.parse_args()

ARCHITECTURE = FLAGS.architecture
DATASET_PATH = FLAGS.dataset_path
RESULTS_PATH = FLAGS.results_path
NUM_ITERATIONS = FLAGS.num_iterations
NUM_INPUT_CHANNELS = FLAGS.num_input_channels
LEARNING_RATE = FLAGS.learning_rate
NUM_POINTS = FLAGS.num_points
NUM_CLASSES = FLAGS.num_classes

if not os.path.exists(RESULTS_PATH):
		os.makedirs(RESULTS_PATH)


"""
Load dataset 
x (train_data) of size [batch_size, num_points, in_channels] : in_channels can be x,y,z coordinates or any other descriptor
adj (adj_input) of size [batch_size, num_points, K] : This is a list of indices of neigbors of each vertex. (Index starting with 1)
										  K is the maximum neighborhood size. If a vertex has less than K neighbors, the remaining list is filled with 0.

"""

x = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_POINTS, IN_CHANNELS])
adj = tf.placeholder(tf.int32, shape=[BATCH_SIZE, NUM_POINTS, K])
y_ = tf.placeholder(tf.float32, shape=[BATCH_SIZE, NUM_POINTS, NUM_CLASSES])

y_conv = get_model(x, adj, NUM_CLASSES, ARCHITECTURE)

batch = tf.Variable(0, trainable=False)

# Standard classification loss
cross_entropy = tf.reduce_mean(tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv), axis=1))

train_step = tf.train.AdamOptimizer(LEARNING_RATE).minimize(cross_entropy, global_step=batch)
correct_prediction = tf.equal(tf.argmax(y_conv,2), tf.argmax(y_,2))
predictions = tf.argmax(y_conv, 2)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())

ckpt = tf.train.get_checkpoint_state(os.path.dirname(RESULTS_PATH))
if ckpt and ckpt.model_checkpoint_path:
		saver.restore(sess, ckpt.model_checkpoint_path)
		write_logs("Checkpoint restored\n")

# Train for the dataset

for iter in range(NUM_ITERATIONS):
	i = train_shuffle[iter%(len(train_data))]
	input = train_data[i]
	if iter%1000 == 0:
		train_accuracy = accuracy.eval(feed_dict={x:input, adj:adj_input, y_: label})
		write_logs("Iteration %d, training accuracy %g\n"%(iter, train_accuracy))
	train_step.run(feed_dict={x:input, adj:train_adj, y_: label})

