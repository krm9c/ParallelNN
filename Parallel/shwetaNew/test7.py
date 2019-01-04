import tensorflow as tf
import numpy as np
import cPickle, os, os.path, time

''' x = tf.Variable([1.0, 2.0])

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
v = sess.run(x)    
print v # will show you your variable.
 '''

def read_file(file_path):
    while not os.path.exists(file_path):
        time.sleep(0.5)
    if os.path.isfile(file_path):
        with open(file_path, "rb") as input_file:
            batch_data = cPickle.load(input_file)
    return batch_data


def write_file(file_name, data):
    #lock.acquire()
    with open(file_name, "wb") as output_file:
        cPickle.dump(data, output_file)
    #lock.release()

def xavier(fan_in, fan_out):
    # use 4 for sigmoid, 1 for tanh activation
    low = -1 * np.sqrt(1.0 / (fan_in + fan_out))
    high = 1 * np.sqrt(1.0 / (fan_in + fan_out))
    return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)

def weight_variable(shape, trainable, name):
    initial = xavier(shape[0], shape[1])
    return tf.Variable(initial, trainable=trainable, name=name)

def bias_variable(shape, trainable, name):
    initial = tf.random_normal(shape, trainable, stddev=1)
    return tf.Variable(initial, trainable=trainable, name=name)

def my_func(data):
    data = tf.convert_to_tensor(data, dtype=tf.float32)
    data = tf.matmul(data, tf.transpose(data))
    return data
    


v = weight_variable([10,5], False, 'weight')
w = weight_variable([2,5], False, 'weight')
b = bias_variable([5], False, 'bias')

print v

init = tf.global_variables_initializer()

#b = tf.placeholder(tf.float32, shape=[None, 5])

sess = tf.Session()
sess.run(init)

a2 = sess.run(w)
a3 = sess.run(b)

print "a2:", a2
print "a3:", a3

Weights = []

Weights.append(a2)
Weights.append(a3)

write_file('abcd1',Weights)
#write_file('abcd2',a2)

data = read_file('abcd1')
#data2 = read_file('abcd2')
print "Read from file"
print "data: ", data
print "w: ", data[0]
print "b: ", data[1]
#print "data2: ", data2

#value1 = my_func(data)
#value2 = my_func(data2)

#print "value1:", sess.run(value1)
#print "value2:", sess.run(value2)
