import multiprocessing
import subprocess
import os
import tensorflow as tf
import gzip, cPickle
from multiprocessing import Process, Value, Array, Lock, Manager, Pool
from ShwetaNNprocess import NNProcess

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
depth = [784]
classes = 10
lr = 0.001

batch_size = 32
steps = 5
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


def main(nr_processes= int):
    X_train, y_train, X_test, y_test = load_data('mnist')
    X_train = X_train[:5000]
    y_train = y_train[:5000]
    X_test = X_test[:1000]
    y_test = y_test[:1000]

    nn_queue = Manager().Queue()

    d = Manager().dict()
    
    processes = []

    for i in range(0, nr_processes):
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


main(nr_processes=1)