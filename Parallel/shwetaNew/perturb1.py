## Final Version
# Generalized Feed Forward Neural Network
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
import gzip, cPickle, random
import fnn1 as NN_class

# Getting the data
path_here = '/media/krm9c/My Book/DataFolder/'

datasets = ['arcene', 'cifar10', 'cifar100', 'gas', 'gisette', 'madelon',\
'mnist', 'notmnist', 'rolling', 'sensorless', 'shuttle']

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

X_train, y_train, X_test, y_test = load_data(datasetName)

# Give file name to get parameters
fileName = "para.txt"

# Getting the parameters
mydict   = {'num_NNlayers':'',\
'num_features':'', 'hidden_nodes':'',\
'num_labels':'', 'learning_rate':'', 'steps':'',\
'display_step':'', 'batch_size':'', 'optimizer':'', 'activation':''}

def getParas(fileName):
    dictNew = {}
    with open(fileName, 'r') as f:
        for line in f:
            for i in mydict.keys():
                if i in line:
                    dictNew[i] = line.split(':')[1].split('\n')[0].strip()
                    break;
    f.close()
    for i in dictNew.keys():
        if i is "learning_rate":
            dictNew[i] = float(dictNew[i])
        elif i is "optimizer" or i is "activation":
            pass
        else:
            dictNew[i] = int(dictNew[i])
    dictNew['num_features']  = X_train.shape[1]
    dictNew['num_labels'] = y_train.shape[1]
    return dictNew

paras = getParas(fileName)
print paras

# Activation Function
def act_ftn(name):
    if(name == "tanh"):
        return(tf.nn.tanh)
    elif(name == "relu"):
        return(tf.nn.relu)
    elif(name == 'sigmoid'):
        return(tf.nn.sigmoid)
    else:
        print("no activation")

def perturbData(X, m, n):
    return X+np.random.uniform(0,1,size=[m,n])

# Define Model parameters
depth = []
classes = paras['num_labels']
lr = paras['learning_rate']
depth.append(paras['num_features'])
#print depth
if [paras['num_NNlayers'] > 2]:
    for i in range(paras['num_NNlayers']-2):
        depth.append(paras['hidden_nodes'])

print depth
batch_size = paras['batch_size']
back_range = 1
op = paras['optimizer']
act = act_ftn(paras['activation'])


# Define model
model = NN_class.learners()
model = model.init_NN_custom(classes, lr, depth, act, batch_size, back_range, optimizer=op)


# Training
for j in range(paras['steps']):
    
    for k in xrange(100):
        x_batch = []
        y_batch = []
        arr = random.sample(range(0, len(X_train)), paras['batch_size'])
        for idx in arr:
            x_batch.append(X_train[idx])
            y_batch.append(y_train[idx])
        x_batch = np.asarray(x_batch)
        y_batch = np.asarray(y_batch)
        
        x_batch_perturbed = perturbData(x_batch, x_batch.shape[0], x_batch.shape[1])
        model.sess.run([model.Trainer["Weight_op"]],\
        feed_dict={model.Deep['FL_layer_10']: x_batch_perturbed, model.classifier['Target']: \
        y_batch, model.classifier["learning_rate"]:lr})

    if j%paras['display_step'] == 0:        
        print "Step", j
        X_test_perturbed = perturbData(X_test, X_test.shape[0], X_test.shape[1])
        acc_test = model.sess.run([model.Evaluation['accuracy']], \
        feed_dict={model.Deep['FL_layer_10']: X_test_perturbed, model.classifier['Target']:\
        y_test, model.classifier["learning_rate"]:lr})

        acc_train = model.sess.run([ model.Evaluation['accuracy']],\
        feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']:\
            y_train})

        cost_error = model.sess.run([model.classifier["Overall cost"]],\
        feed_dict={model.Deep['FL_layer_10']: x_batch_perturbed, model.classifier['Target']:\
        y_batch})

        # Print all the outputs
        print("Loss: ", cost_error[0])
        print("Train Acc:", acc_train[0]*100, "Test Acc:", acc_test[0]*100)


print("Final Test Accuracy", model.sess.run([ model.Evaluation['accuracy']], \
        feed_dict={model.Deep['FL_layer_10']: X_test_perturbed, model.classifier['Target']: \
        y_test})[0]*100)
