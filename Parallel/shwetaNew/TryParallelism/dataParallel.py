import multiprocessing
import subprocess
import os
import tensorflow as tf
import gzip, cPickle
from multiprocessing import Process, Value, Array, Lock, Manager, Pool
from NNprocess import NNProcess
import os.path
import time, random
import numpy as np
import shutil

def read_file(file_path):
    while not os.path.exists(file_path):
        time.sleep(0.5)
    if os.path.isfile(file_path):
        with open(file_path, "rb") as input_file:
            batch_data = cPickle.load(input_file)
    return batch_data

def write_file(write_dir_path, file_name, data, lock):
    #lock.acquire()
    if not os.path.exists(write_dir_path):
        os.makedirs(write_dir_path)
    file_path = write_dir_path+file_name
    #print "File written: ", file_path
    with open(file_path, "wb") as output_file:
        cPickle.dump(data, output_file)
    #lock.release()

# Getting the data
path_here = '/usr/local/home/krm9c/shwetaNew/data/'

datasets = ['arcene', 'cifar10', 'cifar100', 'gas', 'gisette', 'madelon',\
'mnist', 'notmnist', 'rolling', 'sensorless', 'SVHN']

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

    try:
        if X_train.shape[2]:
            X_train = X_train.reshape((X_train.shape[0],(X_train.shape[1]*X_train.shape[2]*X_train.shape[3])))
            X_test = X_test.reshape((X_test.shape[0],(X_test.shape[1]*X_test.shape[2]*X_test.shape[3])))
    except IndexError:
        pass

    print X_train.shape, y_train.shape, X_test.shape, y_test.shape

    return X_train, y_train, X_test, y_test


def info(title):
    print title
    print 'module name:', __name__
    if hasattr(os, 'getppid'):
        print 'parent process:', os.getppid()
    print 'process id:', os.getpid()
    print("\n")


#Parameters
depth = [784, 100]
classes = 10
lr = 0.001

batch_size = 64
steps = 50
op = 'Adam'
act = tf.nn.relu


# config for prediction and evaluation only
def get_session_config(num_cores):
    num_CPU = 1
    num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})

    return config

dir_path = '/usr/local/home/krm9c/shwetaNew/TryParallelism/Weights/'

def get_batches(process_id, steps, X_train, y_train, batch_size):
    for j in range(steps):
        print ("Process:" , process_id, "Step:", j)
        x_batch = []
        y_batch = []
        arr = random.sample(range(0, len(X_train)), batch_size)
        for idx in arr:
            x_batch.append(X_train[idx])
            y_batch.append(y_train[idx])
        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)
        lock = Lock()
        write_file_dir = dir_path+'Layer_'+str(process_id)+'/'
        write_file_path  = 'data_'+str(process_id)+'_'+str(j)
        write_file(write_file_dir, write_file_path, x_batch, lock)
        y_file_path  = 'label_'+str(process_id)+'_'+str(j)
        write_file(write_file_dir, y_file_path, y_batch, lock)
        

def main():

    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    X_train, y_train, X_test, y_test = load_data('mnist')
    X_train = X_train[:10000]
    y_train = y_train[:10000]
    X_test = X_test[:5000]
    y_test = y_test[:5000]

    nn_queue = Manager().Queue()

    d = Manager().dict()
    
    processes = []

    process_0 = Process(target=get_batches, args=(0, steps, X_train, y_train, batch_size,))
    
    processes.append(process_0)

    nr_processes = 10

    for i in range(1, nr_processes+1):
        nn_process = NNProcess(i,depth,classes,lr,op,batch_size, steps, d, nn_queue)
        nn_process.set_train_val(X_train, y_train, X_test, y_test)
        processes.append(nn_process)

    #print len(processes)

    for nn_process in processes:
        nn_process.start()
        #print nn_process.process_id, nn_process.is_alive()
    
    for nn_process in processes:
        nn_process.join()

    tf_session = tf.Session(config=get_session_config(12))

    print "Train Accuracy and Test Accuracy: "
    print d
    
main()
