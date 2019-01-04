## Lets go through data-sets one by one
import os, sys
sys.path.append('../CommonLibrariesDissertation')
path_here = '../data/'
from sklearn.model_selection import train_test_split
from Data_import import *

# 1-- Mnist
def Mnist_pickle():
    from Data_import import *
    print("Mnist")
    X_train, X_test, y_train, y_test = Data_MNIST()
    X_train = X_train.reshape(-1, 28*28);
    X_test = X_test.reshape(-1, 28*28);
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))
    # Open a gzip and save data
    f = gzip.open('mnist.pkl.gz','wb')
    dataset = [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

import numpy as np

# 2-mnist
def not_mnist_pickle():
    p = [5, 6, 7, 8]
    string = ['notmnist']
    for i,s in enumerate(string):
        f = gzip.open(path_here+string[i]+'.pkl.gz','rb')
        dataset = cPickle.load(f)
        X_train = dataset[0]
        X_test  = dataset[1]
        y_train = dataset[2]
        y_test  = dataset[3]
        print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))


import gzip, cPickle
from six.moves import range
def not_mnist_pickle_temp():
    pickle_file = 'notMNIST.pickle'
    with open(pickle_file, 'rb') as f:
      save = cPickle.load(f)
      train_dataset = save['train_dataset']
      train_labels = save['train_labels']
      valid_dataset = save['valid_dataset']
      valid_labels = save['valid_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']
      del save  # hint to help gc free up memor

      X_train = train_dataset.reshape(-1,784)
      x_test = test_dataset.reshape(-1,784)
      x_valid = valid_dataset.reshape(-1,784)
      X_test = np.concatenate((x_test, x_valid))
      y_train = train_labels
      y_test = np.concatenate((test_labels,valid_labels))

      print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))
      # Open a gzip and save data
      f = gzip.open('notmnist.pkl.gz','wb')
      dataset = [X_train, X_test, y_train, y_test]
      cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
      f.close()

## 3 -- CIFAR-10 data-set
def data_cifar_10():
    from Data_import import *

    X_train, X_test, y_train, y_test = CIFAR_10()

    print("X_train", X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))

    X_train = X_train.reshape(-1,32*32*3)

    X_test  = X_test.reshape(-1,32*32*3)

    f = gzip.open("cifar10.pkl.gz",'wb')

    dataset =  [X_train, X_test, y_train, y_test]

    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)

    f.close()

# 4 -- Rolling Element
def Roll():

    X, y = Bearing_Samples("RollingElement/",10000,1)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.33)

    f = gzip.open("rolling.pkl.gz",'wb')

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))

    dataset =  [X_train, X_test, y_train, y_test]

    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)

    f.close()

def  dataimport(filename):
	return np.loadtxt(filename)

def arcene():
	X_train = dataimport("Arcene/arcene_train.data")
	X_test  = dataimport("Arcene/arcene_valid.data")
	y_train = dataimport("Arcene/arcene_train.labels")
	y_test  = dataimport("Arcene/arcene_valid.labels")

	print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

	# X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.33)
	f = gzip.open("arcene.pkl.gz",'wb')
	# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
	dataset =  [X_train, X_test, y_train, y_test]
	cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
	f.close()

def data_sensorless():
    X_train, y_train = sensorless("Sensorless/Sensorless.scale.tr")
    X_test, y_test   = sensorless("Sensorless/Sensorless.scale")
    f = gzip.open("sensorless.pkl.gz",'wb')
    y_test = y_test-1
    y_train = y_train-1
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def gisette():
    X_train, y_train = sensorless("Gisette/gisette_scale")

    X_test, y_test   = sensorless("Gisette/gisette_scale.t")



    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.33)

    f = gzip.open("gisette.pkl.gz",'wb')
    for i, element in enumerate(y_train):
        if element <0:
            y_train[i] = y_train[i]+1
    for i, element in enumerate(y_test):
        if element <0:
            y_test[i] = y_test[i]+1

    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))


    dataset =  [X_train, X_test, y_train, y_test]

    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)

    f.close()


def dexter():

    X_train = dataimport("Dexter/dexter_train.data")

    X_test  = dataimport("Dexter/dexter_valid.data")

    y_train = dataimport("Dexter/dexter_train.labels")
    y_test  = dataimport("Dexter/dexter_valid.labels")

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))

    f = gzip.open("dexter.pkl.gz",'wb')

    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))
    dataset =  [X_train, X_test, y_train, y_test]

    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)

    f.close()

def madelon():
    X_train = dataimport("Madelon/madelon_train.data")
    X_test  = dataimport("Madelon/madelon_valid.data")
    y_train = dataimport("Madelon/madelon_train.labels")
    y_test  = dataimport("Madelon/madelon_valid.labels")

    for i, element in enumerate(y_train):
        if element <0:
            y_train[i] = y_train[i]+1
    for i, element in enumerate(y_test):
        if element <0:
            y_test[i] = y_test[i]+1
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))
    # X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.33)
    f = gzip.open("madelon.pkl.gz",'wb')
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


def gas_array():
    X, y = DataImport(8)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.33)

    f = gzip.open("gas.pkl.gz",'wb')

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))
    dataset =  [X_train, X_test, y_train, y_test]

    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)

    f.close()


## 3 -- CIFAR-100 data-set
def data_cifar_100():

    from Data_import import *

    X_train, X_test, y_train, y_test = CIFAR100()

    X_train = X_train.reshape(-1,32*32*3)

    X_test  = X_test.reshape(-1,32*32*3)

    f = gzip.open("cifar100.pkl.gz",'wb')

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))

    dataset =  [X_train, X_test, y_train, y_test]

    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)

    f.close()


def shuttle():
    X_train, y_train = sensorless("Shuttle/shuttle.scale.tr")
    X_test, y_test   = sensorless("Shuttle/shuttle.scale.t")
    f = gzip.open("shuttle.pkl.gz",'wb')
    y_test = y_test-1
    y_train = y_train-1
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))
    # print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def dam_prop():
    train = dataimport("CMAPS/madelon_train.data")
    

def load_data():
    p = [5, 6, 7, 8]
    string = ['arcene', 'cifar10', 'gas', 'gisette', 'madelon', 'mnist', 'cifar100', 'notmnist', 'rolling', 'sensorless', 'shuttle']
    for i,s in enumerate(string):
        f = gzip.open(path_here+string[i]+'.pkl.gz','rb')
        dataset = cPickle.load(f)
        X_train = dataset[0]
        X_test  = dataset[1]
        y_train = dataset[2]
        y_test  = dataset[3]
        print X_train.shape, y_train.shape, X_test.shape, y_test.shape



###########################################################
# 1 -- MNIST
# Mnist_pickle()

# 2-- NOTMNIST
# not_mnist_pickle()

# 3-- CIFAR10
# data_cifar_10()

# 4-- SENSORLESS
# data_sensorless()

# 5 --Gisette
# gisette()

# 6 -- shuttle
# shuttle()

# 7 --
# gas_array()

# 8 -- Rolling
# Roll()

# 9 -- CIFAR-100
# data_cifar_100()

# 10 -- Madelon
# madelon()

# 11 -- arcene
# arcene()

# load_data()
