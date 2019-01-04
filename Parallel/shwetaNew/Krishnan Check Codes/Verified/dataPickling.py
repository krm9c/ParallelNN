# Pickling the data

import tensorflow as tf
import numpy as np
import gzip, cPickle, random
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from Data_import import *

sess = tf.InteractiveSession()

# Getting the data
path_here = '/usr/local/home/krm9c/shwetaNew/data/'

def Mnist_pickle():
    print("Mnist")
    X_train, X_test, y_train, y_test = Data_MNIST()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    try:
        if X_train.shape[3]:
            pass
    except IndexError:
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    try:
        if y_train.shape[1]:
            pass
    except IndexError:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)

    if y_train.shape[1] == 1:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    # Open a gzip and save data
    f = gzip.open(path_here+'mnist.pkl.gz','wb')
    dataset = [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#Mnist_pickle()

def data_cifar_10():
    print("Cifar10")
    X_train, X_test, y_train, y_test = CIFAR_10()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    try:
        if X_train.shape[3]:
            pass
    except IndexError:
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    try:
        if y_train.shape[1]:
            pass
    except IndexError:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)

    if y_train.shape[1] == 1:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    f = gzip.open(path_here+"cifar10.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#data_cifar_10()

def Roll():
    print("Roll")

    X, y = Bearing_Samples(path_here+"RollingElement/",10000,1)

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.33)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
   
    try:
        if y_train.shape[1]:
            pass
    except IndexError:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)

    if y_train.shape[1] == 1:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    f = gzip.open(path_here+"rolling.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#Roll()

def  dataimport(filename):
	return np.loadtxt(filename)

def arcene():
    print("Arcene")
    X_train = dataimport(path_here+"Arcene/arcene_train.data")
    X_test  = dataimport(path_here+"Arcene/arcene_valid.data")
    y_train = dataimport(path_here+"Arcene/arcene_train.labels")
    y_test  = dataimport(path_here+"Arcene/arcene_valid.labels")
    print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

    ''' unique_elements, counts_elements = np.unique(y_train, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements))) '''

    print y_train[:10]
    print y_test[:10]

    y_train[y_train < 0] = 0

    y_test[y_test < 0] = 0

    print y_train[:10]
    print y_test[:10]

    try:
        if y_train.shape[1]:
            pass
    except IndexError:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)

    if y_train.shape[1] == 1:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    f = gzip.open(path_here+"arcene.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#arcene()

def data_sensorless():
    X_train, y_train = sensorless(path_here+"Sensorless/Sensorless.scale.tr")
    X_test, y_test   = sensorless(path_here+"Sensorless/Sensorless.scale")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    try:
        if y_train.shape[1]:
            pass
    except IndexError:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)

    if y_train.shape[1] == 1:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    f = gzip.open(path_here+"sensorless.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#data_sensorless()

def gisette():
    X_train, y_train = sensorless(path_here+"Gisette/gisette_scale")
    X_test, y_test   = sensorless(path_here+"Gisette/gisette_scale.t")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    ''' print y_train[:10]
    unique_elements, counts_elements = np.unique(y_train, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements)))

    print y_train[:10]
    print y_test[:10] '''

    y_train[y_train < 0] = 0

    y_test[y_test < 0] = 0

    ''' print y_train[:10]
    print y_test[:10] '''

    try:
        if y_train.shape[1]:
            pass
    except IndexError:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)

    if y_train.shape[1] == 1:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    f = gzip.open(path_here+"gisette.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#gisette()

def madelon():
    X_train = dataimport(path_here+"Madelon/madelon_train.data")
    X_test  = dataimport(path_here+"Madelon/madelon_valid.data")
    y_train = dataimport(path_here+"Madelon/madelon_train.labels")
    y_test  = dataimport(path_here+"Madelon/madelon_valid.labels")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    ''' print y_train[:10]
    unique_elements, counts_elements = np.unique(y_train, return_counts=True)
    print("Frequency of unique values of the said array:")
    print(np.asarray((unique_elements, counts_elements))) '''

    #print y_train[:10]
    #print y_test[:10]

    y_train[y_train < 0] = 0

    y_test[y_test < 0] = 0

    #print y_train[:10]
    #print y_test[:10]

    try:
        if y_train.shape[1]:
            pass
    except IndexError:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)

    if y_train.shape[1] == 1:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    f = gzip.open(path_here+"madelon.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#madelon()

def gas_array():
    X, y = DataImport(8)
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size =0.33)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    try:
        if y_train.shape[1]:
            pass
    except IndexError:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)

    if y_train.shape[1] == 1:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    f = gzip.open(path_here+"gas.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#gas_array()

def data_cifar_100():
    X_train, X_test, y_train, y_test = CIFAR100()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    try:
        if X_train.shape[3]:
            pass
    except IndexError:
        X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
        X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
    try:
        if y_train.shape[1]:
            pass
    except IndexError:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)

    if y_train.shape[1] == 1:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    f = gzip.open(path_here+"cifar100.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    
#data_cifar_100()

def shuttle():
    X_train, y_train = sensorless(path_here+"Shuttle/shuttle.scale.tr")
    X_test, y_test   = sensorless(path_here+"Shuttle/shuttle.scale.t")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    try:
        if y_train.shape[1]:
            pass
    except IndexError:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)

    if y_train.shape[1] == 1:
        onehot_encoder = OneHotEncoder(sparse=False)
        y_train = y_train.reshape(len(y_train), 1)
        y_test = y_test.reshape(len(y_test), 1)
        y_train = onehot_encoder.fit_transform(y_train)
        y_test = onehot_encoder.fit_transform(y_test)
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

    f = gzip.open(path_here+"shuttle.pkl.gz",'wb')
    dataset =  [X_train, X_test, y_train, y_test]
    cPickle.dump(dataset,f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

#shuttle()

def not_mnist_pickle():
    pickle_file = path_here+ 'notMNIST.pickle'
    with open(pickle_file, 'rb') as f:
        save = cPickle.load(f)
        train_dataset = save['train_dataset']
        train_labels = save['train_labels']
        valid_dataset = save['valid_dataset']
        valid_labels = save['valid_labels']
        test_dataset = save['test_dataset']
        test_labels = save['test_labels']
        del save  # hint to help gc free up memor

        #print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        X_train = train_dataset
        x_test = test_dataset
        x_valid = valid_dataset
        X_test = np.concatenate((x_test, x_valid))
        y_train = train_labels
        y_test = np.concatenate((test_labels,valid_labels))

        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        try:
            if X_train.shape[3]:
                pass
        except IndexError:
            X_train = X_train.reshape((X_train.shape[0],X_train.shape[1],X_train.shape[2],1))
            X_test = X_test.reshape((X_test.shape[0],X_test.shape[1],X_test.shape[2],1))
            print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

        try:
            if y_train.shape[1]:
                pass
        except IndexError:
            onehot_encoder = OneHotEncoder(sparse=False)
            y_train = y_train.reshape(len(y_train), 1)
            y_test = y_test.reshape(len(y_test), 1)
            y_train = onehot_encoder.fit_transform(y_train)
            y_test = onehot_encoder.fit_transform(y_test)

        if y_train.shape[1] == 1:
            onehot_encoder = OneHotEncoder(sparse=False)
            y_train = y_train.reshape(len(y_train), 1)
            y_test = y_test.reshape(len(y_test), 1)
            y_train = onehot_encoder.fit_transform(y_train)
            y_test = onehot_encoder.fit_transform(y_test)
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
        # Open a gzip and save data
        f = gzip.open(path_here+'notmnist.pkl.gz','wb')
        dataset = [X_train, X_test, y_train, y_test]
        cPickle.dump(dataset, f, protocol=cPickle.HIGHEST_PROTOCOL)
        f.close()
        print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)


#not_mnist_pickle()

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

    ''' try:
        if X_train.shape[2]:
            print X_train.shape, y_train.shape, X_test.shape, y_test.shape
            X_train = X_train.reshape((X_train.shape[0],(X_train.shape[1]*X_train.shape[2]*X_train.shape[3])))
            X_test = X_test.reshape((X_test.shape[0],(X_test.shape[1]*X_test.shape[2]*X_test.shape[3])))
    except IndexError:
        pass

    print X_train.shape, y_train.shape, X_test.shape, y_test.shape '''

    return X_train, y_train, X_test, y_test

#X_train, y_train, X_test, y_test = load_data(datasetName)


for d in datasets:
    X_train, y_train, X_test, y_test = load_data(d)