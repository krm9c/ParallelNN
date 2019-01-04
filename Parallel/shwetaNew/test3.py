# Generalized Feed Forward Neural Network
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
sess = tf.InteractiveSession()

# creating nodes for the computation graph
# placeholders is a promise that a value will be delivered later
x  = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# Give file name to get parameters
fileName = "para.txt"

# Getting the parameters
mydict   = {'num_NNlayers':'',\
'num_features':'', 'hidden_nodes':'',\
'num_labels':'', 'learning_rate':'', 'steps':'',\
'display_step':'', 'batch_size':'', 'optimizer':''}

def getParas(fileName):
    dictNew = {}
    with open(fileName, 'r') as f:
        for line in f:
            for i in mydict.keys():
                if i in line:
                    #print i
                    dictNew[i] = line.split(':')[1].split('\n')[0].strip()
                    #print dictNew[i]
                    break;
    f.close()
    for i in dictNew.keys():
        if i is "learning_rate":
            dictNew[i] = float(dictNew[i])
        elif i is "optimizer":
            pass
        else:
            dictNew[i] = int(dictNew[i])
    #print dictNew
    return dictNew


paras = getParas(fileName)
print(paras['learning_rate'])
#print paras


# Defining the arrays 
Hidden_Weights = []
Hidden_Bias = []
hidden_layer =[]

# Defining weights and biases
# For Single Layer
W = tf.Variable(np.random.rand(paras['num_features'], paras['num_labels']).astype(np.float32))
b = tf.Variable(np.random.rand(paras['num_labels']).astype(np.float32))

# For Multi Layer
Wi = tf.Variable(np.random.rand(paras['num_features'], paras['hidden_nodes']).astype(np.float32))
bi = tf.Variable(np.random.rand(paras['hidden_nodes']).astype(np.float32))
Wo = tf.Variable(np.random.rand(paras['hidden_nodes'], paras['num_labels']).astype(np.float32))
bo = tf.Variable(np.random.rand(paras['num_labels']).astype(np.float32))

Hidden_Weights.append(Wi)
Hidden_Bias.append(bi)

for i in range(paras['num_NNlayers']-2):
    W2 = tf.Variable(np.random.rand(paras['hidden_nodes'], paras['hidden_nodes'])\
    .astype(np.float32))
    b2 = tf.Variable(np.random.rand(paras['hidden_nodes']).astype(np.float32))
    Hidden_Weights.append(W2)
    Hidden_Bias.append(b2)

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

# Getting the Optimizer
optimizer = paras['optimizer']

optDict = {'Adam':tf.train.AdamOptimizer,\
'Adagrad':tf.train.AdagradOptimizer,\
'GradientDescent': tf.train.GradientDescentOptimizer}

print optimizer
#print optDict[optimizer]

train_step = optDict[optimizer](paras['learning_rate']).minimize(cross_entropy)

# Evaluating the model
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y),1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

# Initializing all the variables in the session
sess.run(tf.global_variables_initializer())

# Training
for j in range(paras['steps']):
    print "Step", j+1
    for k in xrange(100):
        batch = mnist.train.next_batch(paras['batch_size'])
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    print("Loss", cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]}))
    print("Train Accuracy", accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels})*100)
    print("Test  Accuracy", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})*100)


print("Final Accuracy", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})*100)





