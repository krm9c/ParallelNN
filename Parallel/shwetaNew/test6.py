# CNN Version 2

import tensorflow as tf
import numpy as np
import gzip, cPickle, random

sess = tf.InteractiveSession()

# Getting the data
path_here = '/media/krm9c/My Book/DataFolder/'

datasets = ['arcene', 'cifar10', 'cifar100', 'gas', 'gisette', 'madelon',\
'mnist', 'notmnist', 'rolling', 'sensorless', 'shuttle', 'dexter']

datasetName = 'mnist'

def load_data(datasetName):
    print datasetName
    f = gzip.open(path_here+datasetName+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]

    print X_train.shape, y_train.shape, X_test.shape, y_test.shape

    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_data(datasetName)


# The information about image dataset
img_size = X_train.shape[1]
img_size_flat = X_train.shape[1]*X_train.shape[2]*X_train.shape[3]
img_shape = (img_size, img_size)
num_channels = X_train.shape[3]
classes = y_train.shape[1]
train_batch_size = 50

learning_rate = 0.001

# Convolutional Layer 1
filter1_size = 5
number_of_filter1 = 32

# Convolutional Layer 2
filter2_size = 5
number_of_filter2 = 64

# Fully Connected Layer 
fc_size = 1024

# No of training steps
steps = 1001

X_train = X_train.reshape((X_train.shape[0],(X_train.shape[1]*X_train.shape[2]*X_train.shape[3])))
X_test = X_test.reshape((X_test.shape[0],(X_test.shape[1]*X_test.shape[2]*X_test.shape[3])))

print ("Reshaping", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

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
y_ = tf.placeholder(tf.float32, shape=[None, classes], name='y_')

x_image = tf.reshape(x, [-1, img_size, img_size, num_channels])
y_true_cls = tf.argmax(y_, dimension=1)

# Defining the layers of the CNN Network
layer_conv1, weights_conv1 = new_conv_layer(input = x_image, num_input_channels= num_channels, filter_size = filter1_size, num_filters = number_of_filter1, use_pooling=True)
layer_conv2, weights_conv2 = new_conv_layer(input = layer_conv1, num_input_channels= number_of_filter1, filter_size = filter2_size, num_filters = number_of_filter2, use_pooling=True)
layer_flat, num_features = flatten_layer(layer_conv2)
layer_fc1 = new_fc_layer(layer_flat, num_features, fc_size, True, True, 0.5)
layer_fc2 = new_fc_layer(layer_fc1, fc_size, classes, False, False)

# Finally Softmax function
y_pred = tf.nn.softmax(layer_fc2)
y_pred_cls = tf.argmax(y_pred, dimension=1)


# Cost function calculation and optimization function
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=layer_fc2,labels=y_))

def Custom_Optimizer(learning_rate):
    train_list = []

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


# Getting the Optimizer
optimizer = "Custom"

optDict = {'Adam':tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy),\
'Adagrad':tf.train.AdagradOptimizer(learning_rate).minimize(cross_entropy),\
'GradientDescent': tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy),\
'Custom': Custom_Optimizer(learning_rate)}

print optimizer
print optDict[optimizer]

train_step = optDict[optimizer]

# Evaluating the model
# Checking for the right predictions
correct_prediction = tf.equal(y_pred_cls, y_true_cls)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Initializing all the variables in the session
sess.run(tf.global_variables_initializer())


# Training
for k in xrange(steps):
    x_batch = []
    y_batch = []
    arr = random.sample(range(0, len(X_train)), train_batch_size)
    for idx in arr:
        x_batch.append(X_train[idx])
        y_batch.append(y_train[idx])
    train_step.run(feed_dict={x: x_batch, y_: y_batch})
    if k%50 == 0:
        print "Step ", k
        print("Loss", cross_entropy.eval(feed_dict={x: x_batch, y_: y_batch}))
        print("Train Accuracy", accuracy.eval(feed_dict={x: X_train, y_: y_train})*100)
        print("Test  Accuracy", accuracy.eval(feed_dict={x: X_test, y_: y_test})*100)


print("Final Accuracy", accuracy.eval(feed_dict={x: X_test, y_: y_test})*100)





