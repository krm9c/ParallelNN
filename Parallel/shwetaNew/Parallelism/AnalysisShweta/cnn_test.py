import sys

sys.dont_write_bytecode = True

# Testing of Generalized Convolutional Neural Network

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
import gzip, cPickle, random, time
import cnn_class as NN_class
from cnn_load_data import load_data
import csv


_start_time = time.time()

def time_start():
    global _start_time 
    _start_time = time.time()

def time_end():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    print "\n" 
    print('Time Taken: {} hour : {} min : {} sec'.format(t_hour,t_min,t_sec))
    print "\n"



def perturbData(X, m, n):
    return (X+np.random.normal(1, 1, size=[m, n]))
    #return (X+(10*np.random.uniform(-4,4,size=[m,n])))

def get_session_config():
    num_cores = 12
    num_CPU = 1
    num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=False,
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})
    return config

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]



dataset_path = '/usr/local/home/krm9c/shwetaNew/data/'

X_train, y_train, X_test, y_test, img_size, num_channels = load_data('mnist', dataset_path)

x_ = []
y_ = []
arr = random.sample(range(0, len(X_train)), 1000)
X_train = X_train[arr, :]
y_train = y_train[arr,:]

arr = random.sample(range(0, len(X_test)), 1000)
X_test = X_test[arr, :]
y_test = y_test[arr,:]
print(X_train.shape, y_train.shape)


# Model parameters
img_size_flat = X_train.shape[1]
img_shape = img_size
classes = y_train.shape[1]

steps = 50
batch_size = 100
display_step = 5
optimizer = 'Adam'
activation = 'relu'

num_conv_layers = 2
num_fc_layers = 2

# Convolutional Layer 1
filter1_size = 5
number_of_filter1 = 16

# Convolutional Layer 2
filter2_size = 5
number_of_filter2 = 32

# Max Pooling shape
pool_shape = [2,2]

# Fully Connected Layer 
fc_size = 1024

filter_size = []
filter_size.append(filter1_size)
filter_size.append(filter2_size)

num_of_filters = []
num_of_filters.append(num_channels)
num_of_filters.append(number_of_filter1)
num_of_filters.append(number_of_filter2)

# Define model
model = NN_class.learners(config=get_session_config())
model = model.init_NN_custom(classes, num_conv_layers, num_fc_layers, img_size_flat, \
    img_size, num_channels, filter_size, num_of_filters, pool_shape, fc_size, batch_size,\
    activation=activation, optimizer=optimizer)

lr = 0.001
z_updates = 100
rho = 1.5
lr_dat = 0.1
l2_norm = 0.00001

acc_test = np.zeros((steps, 1))
acc_train = np.zeros((steps, 1))
cost_M1 = np.zeros((steps, 1))
cost_Z = np.zeros((steps, 1))
cost_Total = np.zeros((steps, 1))


time_start()
# Training
result_file_name = 'cnn_2.csv'
print result_file_name
with open('cnn_Results/'+result_file_name, 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['Optimizer', optimizer])
    writer.writerow(['Activation', activation])
    writer.writerow(['lr_w', lr])
    writer.writerow(['lr_z', lr_dat])
    writer.writerow(['Z updates', z_updates])
    writer.writerow(['rho', rho])
    writer.writerow(['l2_norm', l2_norm])
    writer.writerow(['Training_steps', steps])
    writer.writerow(['Cost Z', 'Cost T', 'Cost Total', 'Train Accuracy', 'Test Accuracy'])
    for j in range(steps):
        lr   = 0.99*lr
        rho = 0.99*rho
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            x_batch, y_batch = batch
            lr_dat = 0.1
            for m in xrange(z_updates):
                lr_dat = 0.99*lr_dat
                _ = model.sess.run([model.Trainer["Zis_op_dist"]],\
                    feed_dict={model.Deep['Conv_Layer_00']: x_batch, model.classifier['Target']: \
                    y_batch, model.classifier["learning_rate"]:lr_dat, model.classifier["rho"]:rho, \
                    model.classifier["l2_norm"]: l2_norm})

            model.sess.run([model.Trainer["Weight_op_dist"]],\
            feed_dict={model.Deep['Conv_Layer_00']: x_batch, model.classifier['Target']: \
            y_batch, model.classifier["learning_rate"]:lr, model.classifier["rho"]:rho, \
            model.classifier["l2_norm"]: l2_norm})


        if j%1 == 0:        
            print "Step", j
            #X_test_perturbed = perturbData(X_test, X_test.shape[0], X_test.shape[1])

            acc_test[j] = model.sess.run([model.Evaluation['accuracy']], \
            feed_dict={model.Deep['Conv_Layer_00']: X_test, model.classifier['Target']:\
            y_test, model.classifier["learning_rate"]:lr})
            
            acc_train[j] = model.sess.run([ model.Evaluation['accuracy']],\
            feed_dict={model.Deep['Conv_Layer_00']: X_train, model.classifier['Target']:\
            y_train, model.classifier["learning_rate"]:lr})

            cost_Total[j] = model.sess.run([model.classifier["Overall_cost_dist"]],\
            feed_dict={model.Deep['Conv_Layer_00']: x_batch, model.classifier['Target']:\
            y_batch, model.classifier["learning_rate"]:lr, model.classifier["rho"]:rho, \
            model.classifier["l2_norm"]: l2_norm})
                    
            cost_M1[j]  = model.sess.run([ model.classifier["Cost_M1_Dist"] ],\
            feed_dict={model.Deep['Conv_Layer_00']: x_batch, model.classifier['Target']:\
            y_batch, model.classifier["learning_rate"]:lr, model.classifier["rho"]:rho, \
            model.classifier["l2_norm"]: l2_norm})   
            
            cost_Z[j]  = model.sess.run([ model.classifier["Total_Z_cost"] ],\
            feed_dict={model.Deep['Conv_Layer_00']: x_batch, model.classifier['Target']:\
            y_batch, model.classifier["learning_rate"]:lr, model.classifier["rho"]:rho, \
            model.classifier["l2_norm"]: l2_norm})

            # Print all the outputs
            print("Total Cost:", cost_Total[j][0], "Z Cost:", cost_Z[j][0], "T Cost:", cost_M1[j][0])
            print("Train Acc:", acc_train[j][0]*100, "Test Acc:", acc_test[j][0]*100)

            data = [cost_Z[j][0], cost_M1[j][0], cost_Total[j][0], acc_train[j][0]*100, acc_test[j][0]*100]
            writer.writerow(data)

            if max(acc_train) > 0.99:
                break

        

time_end()