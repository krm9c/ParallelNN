import gzip, cPickle
import os

# Getting the data

def load_data(datasetName, dataset_path):
    print datasetName
    f = gzip.open(dataset_path+datasetName+'.pkl.gz','rb')
    dataset = cPickle.load(f)
    X_train = dataset[0]
    X_test  = dataset[1]
    y_train = dataset[2]
    y_test  = dataset[3]

    print X_train.shape, y_train.shape, X_test.shape, y_test.shape

    num_channels = 1
    img_size = []
    try:
        if X_train.shape[2]:
            img_size = (X_train.shape[1],X_train.shape[2])
            num_channels = X_train.shape[3]
            X_train = X_train.reshape((X_train.shape[0],(X_train.shape[1]*X_train.shape[2]*X_train.shape[3])))
            X_test = X_test.reshape((X_test.shape[0],(X_test.shape[1]*X_test.shape[2]*X_test.shape[3])))
    except IndexError:
        pass

    print X_train.shape, y_train.shape, X_test.shape, y_test.shape

    return X_train, y_train, X_test, y_test, img_size, num_channels
