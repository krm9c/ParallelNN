## Final Version
# Generalized Feed Forward Neural Network

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import gzip, cPickle, random
from sklearn.preprocessing import OneHotEncoder

sess = tf.InteractiveSession()

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

    #print X_train.shape, y_train.shape, X_test.shape, y_test.shape

    try:
        if X_train.shape[2]:
            X_train = X_train.reshape((X_train.shape[0],(X_train.shape[1]*X_train.shape[2]*X_train.shape[3])))
            X_test = X_test.reshape((X_test.shape[0],(X_test.shape[1]*X_test.shape[2]*X_test.shape[3])))
    except IndexError:
        pass

    #print X_train.shape, y_train.shape, X_test.shape, y_test.shape

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
#print paras

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
    #return (X+np.random.normal(0, 1, size=[m, n]))
    return (X+np.random.uniform(-1,1,size=[m,n]))


# creating nodes for the computation graph
# placeholders is a promise that a value will be delivered later
x  = tf.placeholder(tf.float32, shape=[None, paras['num_features']])
y_ = tf.placeholder(tf.float32, shape=[None, paras['num_labels']])


# Defining the arrays 
Hidden_Weights = []
Hidden_Bias = []
hidden_layer =[]

if paras['num_NNlayers'] == 1:
    # For Single Layer
    W = tf.get_variable(shape=[paras['num_features'], paras['num_labels']],initializer=tf.contrib.layers.xavier_initializer(), name='Weight_Layer_0')
    b = tf.Variable(tf.zeros([paras['num_labels']]), name='Bias_Layer_0')
    Hidden_Weights.append(W)
    Hidden_Bias.append(b)

else:
    # For Multi Layer
    Wi = tf.get_variable(shape=[paras['num_features'], paras['hidden_nodes']],initializer=tf.contrib.layers.xavier_initializer(), name='Weight_Layer_0')
    bi = tf.Variable(tf.zeros([paras['hidden_nodes']]), name='Bias_Layer_0')
    Wo = tf.get_variable(shape=[paras['hidden_nodes'], paras['num_labels']],initializer=tf.contrib.layers.xavier_initializer(), name='Weight_Layer_' + str(paras['num_NNlayers']-1))
    bo = tf.Variable(tf.zeros([paras['num_labels']]), name='Bias_Layer_'+str(paras['num_NNlayers']-1))
    Hidden_Weights.append(Wi)
    Hidden_Bias.append(bi)
    for i in range(paras['num_NNlayers']-2):
        W2 = tf.get_variable(shape=[paras['hidden_nodes'], paras['hidden_nodes']],initializer=tf.contrib.layers.xavier_initializer(), name='Weight_Layer_' + str(i+1))
        b2 = tf.Variable(tf.zeros([paras['hidden_nodes']]), name='Bias_Layer_' + str(i+1))
        Hidden_Weights.append(W2)
        Hidden_Bias.append(b2)
    Hidden_Weights.append(Wo)
    Hidden_Bias.append(bo)


# Neural Network
def neural_net(data):
    hidden_layer.append(data)
    for i in range(paras['num_NNlayers']-1):  
        hidden_layer.append( tf.nn.relu(tf.add(tf.matmul(hidden_layer[i], Hidden_Weights[i]) , Hidden_Bias[i]) ) )
    if paras['num_NNlayers'] == 1:
        output_layer = tf.add(tf.matmul(data,W),b)
    else:
        output_layer = tf.add(tf.matmul(hidden_layer[len(hidden_layer)-1],Wo), bo)
    return output_layer

# Model Scores
y = neural_net(x)

# Loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))


def Custom_Optimizer(learning_rate, optimizer):
    optDict = {'Adam':tf.train.AdamOptimizer,\
    'Adagrad':tf.train.AdagradOptimizer,\
    'GradientDescent': tf.train.GradientDescentOptimizer}

    opt = optDict[optimizer]
    
    train_list = []

    for i in xrange(len(Hidden_Weights)):
        weight = Hidden_Weights[i]
        bias = Hidden_Bias[i]
        ## Gradient Descent update
        weight_update = tf.gradients(cross_entropy, weight)[0] + 0.0001*weight
        bias_update   = tf.gradients(cross_entropy,   bias)[0] + 0.0001*bias

        # Generate the updated variables
        train_list.append((weight_update, weight))
        train_list.append((bias_update, bias))

    return tf.train.AdamOptimizer(learning_rate).apply_gradients(train_list)


# Evaluating the model
train_step = Custom_Optimizer(paras['learning_rate'], paras['optimizer'])
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y),1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

# Initializing all the variables in the session
sess.run(tf.global_variables_initializer())


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
        train_step.run(feed_dict={x: x_batch, y_: y_batch})
    if j%paras['display_step'] == 0:        
        print "Step", j
        print("Loss", cross_entropy.eval(feed_dict={x: x_batch, y_: y_batch}))
        print("Train Accuracy", accuracy.eval(feed_dict={x: X_train, y_: y_train})*100)
        #print("Test  Accuracy", accuracy.eval(feed_dict={x: X_test, y_: y_test})*100)
        X_test_perturbed = perturbData(X_test, X_test.shape[0], X_test.shape[1])
        print("Test  Accuracy", accuracy.eval(feed_dict={x: X_test_perturbed, y_: y_test})*100)

#print("Final Accuracy", accuracy.eval(feed_dict={x: X_test, y_: y_test})*100)
print("Final Accuracy", accuracy.eval(feed_dict={x: X_test_perturbed, y_: y_test})*100)






