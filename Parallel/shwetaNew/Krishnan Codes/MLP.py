
# Deep Learning Simulations
# Author : Krishnan Raghavan
# Date: Dec 25, 2016
#######################################################################################
# Define all the libraries
import os, sys, random, time, tflearn
import numpy as np
from   sklearn import preprocessing
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tqdm import tqdm

###################################################################################
# Setup some parameters for the analysis
# The NN parameters
Train_batch_size = 100
Train_Glob_Iterations = 50

####################################################################################
# Helper Function for the weight and the bias variable
# Weight
def xavier(fan_in, fan_out):
    low = -4*np.sqrt(6.0/(fan_in + fan_out)) # use 4 for sigmoid, 1 for tanh activation
    high = 4*np.sqrt(6.0/(fan_in + fan_out))
    return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)

def weight_variable(shape, trainable, name):
  initial = xavier(shape[0], shape[1])
  return tf.Variable(initial, trainable = trainable, name = name)

# Bias function
def bias_variable(shape, trainable, name):
  initial = tf.random_normal(shape, trainable, stddev =1)
  return tf.Variable(initial, trainable = trainable, name = name)

#  Summaries for the variables
def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram_1', var)

# Class
class Agent():
    def __init__(self):
		self.classifier = {}
		self.Deep = {}
		self.Trainer = {}
		self.Summaries = {}
		self.sess = tf.InteractiveSession()

    ###################################################################################
    # Function for defining every NN
    def nn_layer(self, input_tensor, input_dim, output_dim, act, trainability, key):
        with tf.name_scope(key):
            with tf.name_scope('weights'+key):
                self.classifier['Weight'+key] = weight_variable([input_dim, output_dim], trainable = trainability, name = 'Weight'+key)
                variable_summaries(self.classifier['Weight'+key])
            with tf.name_scope('bias'+key):
                self.classifier['Bias'+key] = bias_variable([output_dim], trainable = trainability, name = 'Bias'+key)
                variable_summaries(self.classifier['Weight'+key])
            with tf.name_scope('Wx_plus_b'+key):
                preactivate = tf.matmul(input_tensor, self.classifier['Weight'+key]) + self.classifier['Bias'+key]
                tf.summary.histogram('pre_activations', preactivate)
            activations = act(preactivate, name='activation'+key)
            tf.summary.histogram('activations', activations)
        return activations


    ############################################################# tf.random_normal(shape, stddev=1)######################
    # Initialization for the default graph and the corresponding NN.
    def init_NN(self, classes, lr, Layers, act_function):
        Keys = []

        with tf.name_scope("FLearners"):
            self.Deep['FL_layer0'] = tf.placeholder(tf.float32, shape=[None, Layers[0]])

            for i in range(1,len(Layers)):
                self.Deep['FL_layer'+str(i)] = self.nn_layer(self.Deep['FL_layer'+str(i-1)], Layers[i-1],\
                Layers[i], act= act_function, trainability = True, key = 'FL_layer'+str(i))

                Keys.append(self.classifier['Weight'+'FL_layer'+str(i)])
                Keys.append(self.classifier['Bias'+'FL_layer'+str(i)])



        with tf.name_scope("Targets"):
            self.classifier['Target'] = tf.placeholder(tf.float32, shape=[None, classes])


        with tf.name_scope("Classifier"):
            self.classifier['class'] = self.nn_layer( self.Deep['FL_layer'+str(len(Layers)-1)],\
            Layers[len(Layers)-1], classes, act=tf.identity, trainability =  True, key = 'class')

            tf.summary.histogram('Output', self.classifier['class'])

            Keys.append(self.classifier['Weightclass'])
            Keys.append(self.classifier['Biasclass'])

        with tf.name_scope("Trainer"):

            Error_Loss  =  tf.nn.softmax_cross_entropy_with_logits(logits = \
            self.classifier['class'], \
            labels = self.classifier['Target'], name='Cost')
            Reg =  tf.add_n([ tf.nn.l2_loss(v) for v in Keys ]) *0.001

            tf.summary.scalar('LearningRate', lr)

            # Learner
            self.Trainer["cost_NN"] =  tf.reduce_mean(Error_Loss+ Reg)
            tf.summary.scalar('Cost_NN', self.Trainer["cost_NN"])

            self.Trainer['Optimizer_NN']    =  tf.train.AdamOptimizer(lr)
            self.Trainer["TrainStep_NN"]    =  self.Trainer['Optimizer_NN'].minimize(self.Trainer["cost_NN"],var_list = Keys)
            # self.Trainer["grads"] = tf.gradients(self.Trainer["cost_NN"], Keys)
            # self.Trainer["grad_placeholder"] = [(tf.placeholder("float", shape=grad[1].get_shape()), grad[1]) for grad in self.Trainer["grads"] ]
            # self.Trainer["apply_placeholder_op"] = self.Trainer['Optimizer_NN'].apply_gradients(self.Trainer["grad_placeholder"])


            with tf.name_scope('Evaluation'):
                with tf.name_scope('CorrectPrediction'):
                    self.Trainer['correct_prediction'] = tf.equal(tf.argmax(self.classifier['class'],1),\
                    tf.argmax(self.classifier['Target'],1))
                with tf.name_scope('Accuracy'):
                    self.Trainer['accuracy'] = tf.reduce_mean(tf.cast(self.Trainer['correct_prediction'], tf.float32))
                with tf.name_scope('Prob'):
                    self.Trainer['prob'] = tf.cast( tf.nn.softmax(self.classifier['class']), tf.float32 )
                tf.summary.scalar('Accuracy', self.Trainer['accuracy'])
                tf.summary.histogram('Prob', self.Trainer['prob'])


		self.Summaries['merged'] = tf.summary.merge_all()
		self.Summaries['train_writer'] = tf.summary.FileWriter('../train', self.sess.graph)
		self.Summaries['test_writer'] = tf.summary.FileWriter('../test')
		self.sess.run(tf.global_variables_initializer())
		return self


###################################################################################
# Let us import some dataset
# Lets do the analysis
def Analyse():
    import gc
    # Lets start with creating a model and then train batch wise.
    model = Agent()
    model = model.init_NN(10, 0.001, [784, 100, 100, 100], tf.nn.relu)
    print model.Trainer
    # ####################################################################################
    ## Start the learning Procedure Now
    acc = 0.0
    # Declare a saver
    try:
        t = xrange(Train_Glob_Iterations)
        for i in t:
            for k in xrange((mnist.train.num_examples/Train_batch_size)):
                batch_xs, batch_ys  = mnist.train.next_batch(Train_batch_size)
                summary, _  = model.sess.run([model.Summaries['merged'], model.Trainer['TrainStep_NN']],\
             feed_dict ={ model.Deep['FL_layer0'] : batch_xs, model.classifier['Target']: batch_ys })
                model.Summaries['train_writer'].add_summary(summary, i)
            if i % 1 == 0:
                summary, a  = model.sess.run( [model.Summaries['merged'], model.Trainer['accuracy']], feed_dict={ model.Deep['FL_layer0'] : \
                mnist.test.images, model.classifier['Target'] :  mnist.test.labels})
                model.Summaries['test_writer'].add_summary(summary, i)
                print "Iteration", i, "Accuracy", a
            if a > 0.99:
                summary, pr  = model.sess.run( [ model.Summaries['merged'], model.Trainer['prob'] ], \
                feed_dict ={ model.Deep['FL_layer0'] :  mnist.test.images, model.classifier['Target'] :  mnist.test.labels } )
                model.Summaries['test_writer'].add_summary(summary, i)
                break
    except Exception as e:
        print e
        print "I found an exception"
        tf.reset_default_graph()
        del model
        gc.collect()
        return 0
    tf.reset_default_graph()
    del model
    gc.collect()
    return 0


# DataSeT
print "---MNIST---"
### Import all the libraries required by us
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
Analyse()
