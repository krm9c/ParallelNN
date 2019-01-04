import sys

sys.dont_write_bytecode = True

import tensorflow as tf
import multiprocessing
from multiprocessing import Process, Value, Array, Lock, Manager, Pool
import os, os.path, time
import time, random, cPickle, shutil
import numpy as np
from NNprocess2 import NNProcess
from load_data import load_data
import fcntl
import collections

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


def write_file(write_dir_path, file_name, data, pr_id):
    if not os.path.exists(write_dir_path):
        os.makedirs(write_dir_path)
    file_path = write_dir_path+file_name
    try:
        with open(file_path, "wb") as output_file:
            #print "Pr:", pr_id, "Writing file: ", file_path
            fcntl.flock(output_file, fcntl.LOCK_EX)
            cPickle.dump(data, output_file)
            fcntl.flock(output_file, fcntl.LOCK_UN)
    except Exception as e:
        #print "Writing file ERROR !!!!!!: ", file_path
        write_file(write_dir_path, file_name, data, pr_id)


# config for prediction and evaluation only
def get_session_config(num_cores):
    num_CPU = 1
    num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
    inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
    device_count={'CPU': num_CPU, 'GPU': num_GPU})

    return config

#dir_path = '/usr/local/home/krm9c/shwetaNew/Parallelism/Weights/'
dir_path = '/media/krm9c/My Book/ShwetaExt/Weights_1/'



def iterate_minibatches(inputs, targets, batchsize, shuffle=True):
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


def get_batches(process_id, nr_processes, steps, X_train, y_train, batch_size):
    print("Process " + str(process_id) + " starting...")
    print("Process " + str(nr_processes+2) + " starting...")
    for j in range(1,steps+1):
        x_batch = []
        y_batch = []
        arr = random.sample(range(0, len(X_train)), batch_size)
        for idx in arr:
            x_batch.append(X_train[idx])
            y_batch.append(y_train[idx])
        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)
        #print ("Process:", process_id, "Step:", j)
        x_file_dir = dir_path+'Layer_'+str(process_id)+'/'
        x_file_path  = 'z_'+str(process_id)+'_'+str(j)
        write_file(x_file_dir, x_file_path, x_batch, process_id)
        #print ("Process:", nr_processes+2, "Step:", j)
        y_file_dir = dir_path+'Layer_'+str(nr_processes+2)+'/'
        y_file_path  = 't_'+str(nr_processes+2)+'_'+str(j)
        write_file(y_file_dir, y_file_path, y_batch, nr_processes+2)
    
    print("Process " + str(process_id) + " finished.")
    print("Process " + str(nr_processes+2) + " finished.")

dataset_path = '/usr/local/home/krm9c/shwetaNew/data/'

def main():
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)
    
    #Parameters
    X_train, y_train, X_test, y_test = load_data('mnist', dataset_path)
    ''' X_train = X_train[:100]
    y_train = y_train[:100]
    X_test = X_test[:100]
    y_test = y_test[:100] '''

    depth = [784, 100, 100, 100, 10]
    classes = 10
    lr_z = 0.01
    lr_w = 0.001
    lambda_value = 0.8
    batch_size = 64
    steps = 10001
    z_update_steps = 10
    w_update_steps = 5
    nr_processes = 3

    op = 'Adam'
    act = tf.nn.tanh


    d = Manager().dict()

    for i in range(1,steps+1):
        d[i] = 0
    

    #print d


    processes = []

    nn_process_0 = Process(target=get_batches, args=(0, nr_processes, steps, X_train, y_train, batch_size,))
    processes.append(nn_process_0)


    for i in range(1, nr_processes+1):
        nn_process = NNProcess(nr_processes, i, depth, classes, lr_z, \
            lr_w, lambda_value, op, batch_size, steps, z_update_steps, w_update_steps, d)
        nn_process.set_train_val(X_train, y_train, X_test, y_test)
        processes.append(nn_process)

    #print len(processes)

    #nn_process_0.start()
    time_start()
    
    for nn_process in processes:
        nn_process.start()
        
    for nn_process in processes:
        nn_process.join()


    #test(nr_processes,steps)

    #tf_session = tf.Session(config=get_session_config(12))

    #print "Train Accuracy and Test Accuracy: "
    #print d

    time_end()
    
main()
