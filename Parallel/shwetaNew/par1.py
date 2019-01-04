import pandas as pd
from multiprocessing import Manager
import tensorflow as tf
from keras import backend as K

from train_val_set import TrainValSet
from nn_process import NNProcess

path_here = '/usr/local/home/krm9c/shwetaNew/data/'

datasets = ['arcene', 'cifar10', 'cifar100', 'gas', 'gisette', 'madelon',\
'mnist', 'notmnist', 'rolling', 'sensorless', 'SVHN']

#datasetName = 'mnist'

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




def load_train_val_test_datasets():
    df_train = pd.read_csv('/usr/local/home/krm9c/shwetaNew/mnist-'+ "train.csv", header=None)
    df_val = pd.read_csv('/usr/local/home/krm9c/shwetaNew/mnist-'+"validation.csv", header=None)
    df_test = pd.read_csv('/usr/local/home/krm9c/shwetaNew/mnist-'+"test.csv", header=None)

    return df_train, df_val, df_test


# config for prediction and evaluation only
def get_session_config(num_cores):
    num_CPU = 1
    num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=True,
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})

    return config


def train_test(nr_nets= int, nr_processes= int):
    df_train, df_val, df_test = load_train_val_test_datasets()
    train_val_set = TrainValSet(df_train, df_val)
    #X_train, y_train, X_test, y_test = load_data('mnist')
    nets_per_proc = int(nr_nets/nr_processes)

    nn_queue = Manager().Queue()

    processes = []

    for i in range(0, nr_processes):
        nn_process = NNProcess(i, nets_per_proc, nn_queue)
        nn_process.set_train_val(train_val_set)
        processes.append(nn_process)

    for nn_process in processes:
        nn_process.start()
        nn_process.run()

    for nn_process in processes:
        nn_process.join()

    tf_session = tf.Session(config=get_session_config(4))
    K.set_session(tf_session)

    # ...
    # load neural nets from files
    # do predictions


train_test(6,3)