# CNN Version 1

import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

# Working of our algorithm is as follows:
# Conv1_layer -> Conv2_layer -> Flatten_layer -> FullyConnected_layer -> FullyConnected_layer (With 10 Classes)

# Reading handwritten digits from MNIST dataset
data = input_data.read_data_sets('data/MNIST/', one_hot = True)
# The informations about image dataset
img_size = 28
img_size_flat = 28 * 28
img_shape = (img_size, img_size)
num_channels = 1
classes = 10
train_batch_size = 50

learning_rate = 1e-4

# Convolutional Layer 1
filter1_size = 5
number_of_filter1 = 32

# Convolutional Layer 2
filter2_size = 5
number_of_filter2 = 64

# Fully Connected Layer 
fc_size = 1024

# No of training steps
STEPS = 1001


# Defining the arrays 
Hidden_Weights = []
Hidden_Bias = []

# Function for defining weights
def new_weights(shape):
	initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
	Hidden_Weights.append(initial)
	return initial


# Function for defining biases
def new_bias(length):
	initial = tf.Variable(tf.constant(0.1, shape=[length]))
	Hidden_Bias.append(initial)
	return initial


# Function to create the convolution layer with/without max-pooling
def new_conv_layer(input, num_input_channels, filter_size, num_filters, use_pooling=True):
	shape = [filter_size, filter_size, num_input_channels, num_filters]
	weights = new_weights(shape = shape)
	biases = new_bias(length = num_filters)

	# tf.nn.conv2d needs a 4D input
	layer = tf.nn.conv2d(input = input, filter= weights, strides=[1,1,1,1], padding='SAME')
	layer += biases
	if use_pooling:
		layer = tf.nn.max_pool(value = layer, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
	# relu activation function converts all negatives to zero
	layer = tf.nn.relu(layer)
	return layer, weights

# After all convolutions, we need to flatten the layer
def flatten_layer(layer):
	layer_shape = layer.get_shape()
	num_features = layer_shape[1:4].num_elements()
	layer_flat = tf.reshape(layer, [-1, num_features])
	return layer_flat, num_features

# Fully connected layer
def new_fc_layer(input, num_inputs, num_outputs, use_relu=True, dropout=False, keep_prob=1.0):
	weights = new_weights(shape=[num_inputs, num_outputs])
	biases = new_bias(length= num_outputs)
	layer = tf.matmul(input, weights) + biases
	if use_relu:
		layer = tf.nn.relu(layer)
	if dropout:
		layer = tf.nn.dropout(layer, keep_prob)
	return layer


# The placeholder to hold the X and Y values while training
x = tf.placeholder(tf.float32, shape=[None, img_size_flat], name='x')
y_true = tf.placeholder(tf.float32, shape=[None, 10], name='y_true')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true_cls = tf.argmax(y_true, dimension=1)

# The beginning of the process
layer_conv1, weights_conv1 = new_conv_layer(input = x_image, num_input_channels= num_channels, filter_size = filter1_size, num_filters = number_of_filter1, use_pooling=True)
layer_conv2, weights_conv2 = new_conv_layer(input = layer_conv1, num_input_channels= number_of_filter1, filter_size = filter2_size, num_filters = number_of_filter2, use_pooling=True)
layer_flat, num_features = flatten_layer(layer_conv2)
layer_fc1 = new_fc_layer(layer_flat, num_features, fc_size, True, True, 0.5)
layer_fc2 = new_fc_layer(layer_fc1, fc_size, classes, False, False)

# Finally Softmax function
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)

def Custom_Optimizer(learning_rate):
    train_list = []
    Custom_Hidden_Weights = []
    Custom_Hidden_Bias = []

    for i in xrange(len(Hidden_Weights)):
        weight = Hidden_Weights[i]
        bias = Hidden_Bias[i]
        ## Gradient Descent update
        weight_update = tf.gradients(cross_entropy, weight)[0] + 0*weight
        bias_update   = tf.gradients(cross_entropy,   bias)[0] + 0*bias

        # Generate the updated variables
        train_list.append((weight_update, weight))
        train_list.append((bias_update, bias))

    return tf.train.AdamOptimizer(learning_rate).apply_gradients(train_list)

# Cost function calculation and optimization function
cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_true)
cost = tf.reduce_mean(cross_entropy)
#optimizer = tf.train.AdamOptimizer(learning_rate=1e-4).minimize(cost)
optimizer = Custom_Optimizer(learning_rate)

# Checking for the right predictions
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# TF Session initiation
session = tf.Session()
session.run(tf.global_variables_initializer())

# Counter for total number of iterations performed so far.
total_iterations = 0

# The trainer function to iterate the training process to learn further
def optimize(num_iterations):
    for i in range(num_iterations):
        x_batch, y_true_batch = data.train.next_batch(train_batch_size)
        feed_dict_train = {x: x_batch, y_true: y_true_batch}
        session.run(optimizer, feed_dict=feed_dict_train)

        # Print status every 100 iterations.
        if i % 100 == 0:
            # Calculate the accuracy on the training-set.
            acc = session.run(accuracy, feed_dict=feed_dict_train)

            # Message for printing.
            print "Step ",i+1,': ', acc*100

    
optimize(STEPS)

print("\nTraining complete!")
print("Test Accuracy: ", session.run(accuracy, feed_dict={x: data.test.images, y_true: data.test.labels})*100)


# test_number = data.train._images[2]
# test_number = test_number.reshape([1,784])
# print np.argmax(data.train._labels[2])
# print session.run(y_pred_cls, {x: test_number})