# Single Layer Network

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import os, cPickle
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

# creating nodes for the computation graph
# placeholders is a promise that a value will be delivered later
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# defining weights and biases
W = tf.Variable(tf.zeros([784,10]))
b = tf.Variable(tf.zeros([10]))

# initializing all the variables in the session
sess.run(tf.global_variables_initializer())

# single neural layer
y = tf.matmul(x,W) + b

# loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


# optimization
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

# _ holds the result of the last executed expression(/statement)
for _ in range(100):
  batch = mnist.train.next_batch(100)
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})

# Evaluating the model
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))


accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})*100)

