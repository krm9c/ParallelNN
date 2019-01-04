###########################################################
## Lets go through data-sets one by one
import os, sys
import numpy as np
sys.path.append('../CommonLibrariesDissertation')
path_here = '../data/'
from sklearn.model_selection import train_test_split
from Data_import import *
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import gzip, cPickle
from six.moves import range
##########################################################







###########################################################
# 1-- Mnist
def Mnist_pickle():
    print("MNIST")
    from Data_import import *
    X_train, X_test, y_train, y_test = Data_MNIST()
    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)
    X_train = X_train.reshape([X_train.shape[0],28, 28, 1])
    X_test  = X_test.reshape( [X_test.shape[0], 28, 28, 1])
    # Open a gzip and save data
    f = gzip.open('mnist.pkl.gz','wb')
    dataset = [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    f.close()


###########################################################
def not_mnist_pickle():
    pickle_file = 'notMNIST.pickle'
    print("NotMNIST")
    with open(pickle_file, 'rb') as f:
      save = cPickle.load(f)
      train_dataset = save['train_dataset']
      train_labels = save['train_labels']
      valid_dataset = save['valid_dataset']
      valid_labels = save['valid_labels']
      test_dataset = save['test_dataset']
      test_labels = save['test_labels']

      del save  # hint to help gc free up memor

      scaler  = MinMaxScaler(feature_range=(0, 1))
      X_train = train_dataset.reshape(-1,784)
      x_test = test_dataset.reshape(-1,784)
      x_valid = valid_dataset.reshape(-1,784)
      X_test = np.concatenate((x_test, x_valid))
      y_train = train_labels
      y_test = np.concatenate((test_labels,valid_labels))
      X_train = scaler.fit_transform(X_train)
      X_test  = scaler.fit_transform(X_test)
      X_train = X_train.reshape([X_train.shape[0],28, 28, 1])
      X_test  = X_test.reshape( [X_test.shape[0], 28, 28, 1])
      onehot_encoder = OneHotEncoder(sparse=False)
      y_train = y_train.reshape(len(y_train), 1)
      y_test  = y_test.reshape(len(y_test), 1)
      y_train = onehot_encoder.fit_transform(y_train)
      y_test  = onehot_encoder.fit_transform(y_test)

      # Open a gzip and save data
      f = gzip.open('notmnist.pkl.gz','wb')
      dataset = [X_train, X_test, y_train, y_test]
      cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
      print X_train.shape, y_train.shape, X_test.shape, y_test.shape
      f.close()


###########################################################
## 3 -- CIFAR-10 data-set
def data_cifar_10():
    print("CiFar10")
    from Data_import import *
    X_train, X_test, y_train, y_test = CIFAR_10()
    X_train = X_train.reshape(-1,32*32*3)
    X_test  = X_test.reshape(-1,32*32*3)
    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(len(y_train), 1)
    y_test  = y_test.reshape(len(y_test), 1)
    y_train = onehot_encoder.fit_transform(y_train)
    y_test  = onehot_encoder.fit_transform(y_test)
    X_train = X_train.reshape([X_train.shape[0], 32, 32, 3])
    X_test  = X_test.reshape( [X_test.shape[0],  32, 32, 3])

    f = gzip.open("cifar10.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    f.close()

###########################################################
# 4 -- Rolling Element
def Roll():
    print("Roll")
    X, y = DataImport(22, classes=4, file=0, sample_size = 100000, features = 200)
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.33)
    f = gzip.open("rolling.pkl.gz",'wb')
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(len(y_train), 1)
    y_test  = y_test.reshape(len(y_test), 1)
    y_train = onehot_encoder.fit_transform(y_train)
    y_test  = onehot_encoder.fit_transform(y_test)
    dataset = [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    f.close()


###########################################################
def arcene():
    print("Arcene")
    X_train = dataimport("Arcene/arcene_train.data")
    X_test  = dataimport("Arcene/arcene_valid.data")
    y_train = dataimport("Arcene/arcene_train.labels")+1
    y_test  = dataimport("Arcene/arcene_valid.labels")+1
    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(len(y_train), 1)
    y_test  = y_test.reshape(len(y_test), 1)
    y_train = onehot_encoder.fit_transform(y_train)
    y_test  = onehot_encoder.fit_transform(y_test)

    f = gzip.open("arcene.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    f.close()

###########################################################
def data_sensorless():
    print("Sensorless")
    X_train, y_train = sensorless("Sensorless/Sensorless.scale.tr")
    X_test, y_test   = sensorless("Sensorless/Sensorless.scale")
    f = gzip.open("sensorless.pkl.gz",'wb')
    y_test = y_test-1
    y_train = y_train-1
    
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(len(y_train), 1)
    y_test  = y_test.reshape(len(y_test), 1)
    y_train = onehot_encoder.fit_transform(y_train)
    y_test  = onehot_encoder.fit_transform(y_test)


    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    f.close()

###########################################################
def gisette():
    print("Gisette")
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
    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)

    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(len(y_train), 1)
    y_test  = y_test.reshape(len(y_test), 1)
    y_train = onehot_encoder.fit_transform(y_train)
    y_test  = onehot_encoder.fit_transform(y_test)
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)

    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    f.close()

###########################################################
def dexter():
    print("Dexter")
    X_train = dataimport("Dexter/dexter_train.data")
    X_test  = dataimport("Dexter/dexter_valid.data")
    y_train = dataimport("Dexter/dexter_train.labels")
    y_test  = dataimport("Dexter/dexter_valid.labels")
    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(len(y_train), 1)
    y_test  = y_test.reshape(len(y_test), 1)
    y_train = onehot_encoder.fit_transform(y_train)
    y_test  = onehot_encoder.fit_transform(y_test)
    f = gzip.open("dexter.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    f.close()


###########################################################
def madelon():
    print("Madelon")
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

    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)

    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(len(y_train), 1)
    y_test  = y_test.reshape(len(y_test), 1)
    y_train = onehot_encoder.fit_transform(y_train)
    y_test  = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    f = gzip.open("madelon.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    f.close()

###########################################################
def gas_array():
    print("Gas Array")
    X, y = DataImport(8)

    scaler  = MinMaxScaler(feature_range=(0, 1))
    X = scaler.fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.33)


    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(len(y_train), 1)
    y_test  = y_test.reshape(len(y_test), 1)
    y_train = onehot_encoder.fit_transform(y_train)
    y_test  = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    f = gzip.open("gas.pkl.gz",'wb')

    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    f.close()

###########################################################
## 3 -- CIFAR-100 data-set
def data_cifar_100():
    print("Cifar100")
    from Data_import import *
    X_train, X_test, y_train, y_test = CIFAR100()
    X_train = X_train.reshape(-1,32*32*3)
    X_test  = X_test.reshape(-1,32*32*3)
    
    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(len(y_train), 1)
    y_test  = y_test.reshape(len(y_test), 1)
    y_train = onehot_encoder.fit_transform(y_train)
    y_test  = onehot_encoder.fit_transform(y_test)
    X_train = X_train.reshape([X_train.shape[0], 32, 32, 3])
    X_test  = X_test.reshape( [X_test.shape[0],  32, 32, 3])

    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
    f = gzip.open("cifar100.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    f.close()

###########################################################
def shuttle():
    print("Shuttle")
    X_train, y_train = sensorless("Shuttle/shuttle.scale.tr")
    X_test, y_test   = sensorless("Shuttle/shuttle.scale.t")
    f = gzip.open("shuttle.pkl.gz",'wb')
    y_test = y_test-1
    y_train = y_train-1
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape, min(y_test), max(y_test))

    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)

    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(len(y_train), 1)
    y_test  = y_test.reshape(len(y_test), 1)
    y_train = onehot_encoder.fit_transform(y_train)
    y_test  = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)


    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()


###########################################################
## 3 --SVHN
def SVHN():
    print("SVHN")
    from Data_import import *
    X_train, y_train = sensorless("SVHN/SVHN.scale.t")
    X_test, y_test   = sensorless("SVHN/SVHN.scale.t")
    X_train = X_train.reshape(-1,32*32*3)
    X_test  = X_test.reshape(-1,32*32*3)
    
    scaler  = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.fit_transform(X_test)
    onehot_encoder = OneHotEncoder(sparse=False)
    y_train = y_train.reshape(len(y_train), 1)
    y_test  = y_test.reshape(len(y_test), 1)
    y_train = onehot_encoder.fit_transform(y_train)
    y_test  = onehot_encoder.fit_transform(y_test)
    X_train = X_train.reshape([X_train.shape[0], 32, 32, 3])
    X_test  = X_test.reshape( [X_test.shape[0],  32, 32, 3])
    # print("SVHN")
    
    f = gzip.open("SVHN.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    print X_train.shape, y_train.shape, X_test.shape, y_test.shape
    f.close()
    
###########################################################
def load_data():
    p = [5, 6, 7, 8]
    string = ['arcene', 'cifar10', 'gas', 'gisette', 'madelon', 'mnist', 'cifar100', 'notmnist', 'rolling', 'sensorless']
    for i,s in enumerate(string):
        f = gzip.open(path_here+string[i]+'.pkl.gz','rb')
        dataset = cPickle.load(f)
        X_train = dataset[0]
        X_test  = dataset[1]
        y_train = dataset[2]
        y_test  = dataset[3]
        print X_train.shape, y_train.shape, X_test.shape, y_test.shape


###########################################################
def  CSVimport(filename):
	return np.loadtxt(filename)


############################################################
# # 1 -- Rolling
# Roll()
# # 2 -- MNIST
# Mnist_pickle()
# # 3-- NOTMNIST
# not_mnist_pickle()
# # 4-- CIFAR10
# data_cifar_10()
# # 5-- SENSORLESS
# data_sensorless()
# # 6 --Gisette
# gisette()
# # 7 -- Madelon
# madelon()
# # 8 -- shuttle
# # shuttle()
# # 9 -- CIFAR-100
# data_cifar_100()
# # 10 -- Gas Sensor Array
# gas_array()
# # 11 -- arcene
# arcene()
# # 12 -- SVHN
# SVHN()
###########################################################




###########################################################
# Finally test all the data-sets
load_data()
###########################################################

