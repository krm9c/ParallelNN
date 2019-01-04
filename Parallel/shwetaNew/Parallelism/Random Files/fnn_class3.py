
# Class for Generalized Feed Forward Neural Network

import tensorflow as tf
import numpy as np
import random
import operator

# Xavier Initialization
def xavier(fan_in, fan_out):
    # use 4 for sigmoid, 1 for tanh activation
    low = -1 * np.sqrt(1.0 / (fan_in + fan_out))
    high = 1 * np.sqrt(1.0 / (fan_in + fan_out))
    return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)

# Weights Initialization
def weight_variable(shape, trainable, name):
    initial = xavier(shape[0], shape[1])
    return tf.Variable(initial, trainable=trainable, name=name)

# Z's Initialization
def z_variable(shape, trainable, name):
    initial = xavier(shape[0], shape[1])
    return tf.Variable(initial, trainable=trainable, name=name)

# Bias Initialization
def bias_variable(shape, trainable, name):
    initial = tf.random_normal(shape, trainable, stddev=1)
    return tf.Variable(initial, trainable=trainable, name=name)

def get_optimizer_name(optimizer):
    if optimizer == 'Adam':
        opt = tf.train.AdamOptimizer
    elif optimizer == 'Adadelta':
        opt = tf.train.AdadeltaOptimizer
    elif optimizer == 'Adagrad':
        opt = tf.train.AdagradOptimizer
    elif optimizer == 'GradientDescent':
        opt = tf.train.GradientDescentOptimizer
    elif optimizer == 'RMSProp':
        opt = tf.train.RMSPropOptimizer
    return opt


# The main Class
class learners():
    def __init__(self,config):
        self.classifier = {}
        self.Deep = {}
        self.Trainer = {}
        self.Evaluation = {}
        self.keys = []
        self.sess = tf.Session(config=config)

        # Extra Parameters
        self.Weights =[]
        self.Z = []
        self.pred = []
        self.targ = []

    def compute_cost_function(self, lambda_value, act): 
        first_term = lambda_value*(tf.nn.l2_loss(tf.subtract(self.classifier['z_next'], act(tf.matmul(self.classifier['z_self'], self.classifier['w_next'])))))
        second_term = lambda_value*(tf.nn.l2_loss(tf.subtract(self.classifier['z_self'], act(tf.matmul(self.classifier['z_prev'], self.classifier['w_self'])))))
        self.classifier['Local Cost H'] = tf.add(first_term, second_term)
        return self.classifier['Local Cost H']

    def penultimate_cost_function(self, lambda_value, act):
        z_next = tf.nn.softmax(tf.matmul(self.classifier['z_self'], self.classifier['w_next'])) 
        first_term = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    logits=z_next, labels=self.classifier['t_next']))
        #first_term = tf.nn.l2_loss(tf.subtract(self.classifier['t_next'], z_next))
        second_term = lambda_value*(tf.nn.l2_loss(tf.subtract(self.classifier['z_self'], \
            act(tf.matmul(self.classifier['z_prev'], self.classifier['w_self'])))))
        self.classifier['Local Cost H'] = tf.add(first_term, second_term)
        return self.classifier['Local Cost H']

    def last_cost_function(self):
        z_self = tf.nn.softmax(tf.matmul(self.classifier['z_prev'], self.classifier['w_self']))
        self.classifier['z_self'] = z_self
        #self.classifier['Local Cost H'] = tf.nn.l2_loss(tf.subtract(self.classifier['t_next'], z_self))
        self.classifier['Local Cost H'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits=z_self, labels=self.classifier['t_next'], name='Error_Cost'))
        return self.classifier['Local Cost H']

    def cost_function(self, pr_id, nr_processes, lambda_value, act=tf.nn.tanh):
        if pr_id == nr_processes:
            self.classifier['Local Cost H'] = self.last_cost_function()
        elif pr_id == nr_processes-1:
            self.classifier['Local Cost H'] = self.penultimate_cost_function(lambda_value, act)
        else:
            self.classifier['Local Cost H'] = self.compute_cost_function(lambda_value, act)

        return self.classifier['Local Cost H']

    def compute_z_error(self):
        mul = tf.nn.tanh(tf.matmul(self.classifier['z_prev'], self.classifier['w_self']))
        self.classifier['z cost'] = tf.nn.l2_loss(tf.subtract(self.classifier['z_self'],mul))
        return self.classifier['z cost']

    def Z_Custom_Optimizer(self, lr_z, optimizer):
        #opt = get_optimizer_name(optimizer) 
        train_list = []
        z_update = tf.gradients(self.classifier['Local Cost H'], self.classifier['z_self'])[0]
        train_list.append((z_update, self.classifier['z_self']))
    
        return tf.train.AdamOptimizer(lr_z).apply_gradients(train_list), self.classifier['Local Cost H']

    def W_Custom_Optimizer(self, lr_w, optimizer):
        #opt = get_optimizer_name(optimizer)
        train_list = []
        w_update = tf.gradients(self.classifier['Local Cost H'], self.classifier['w_self'])[0]   + 0.0001*self.classifier['w_self']
        train_list.append((w_update, self.classifier['w_self']))
        return tf.train.AdamOptimizer(lr_w).apply_gradients(train_list), self.classifier['Local Cost H'], tf.nn.l2_loss(w_update)

    def testing(self, input_data):
        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
        self.classifier['z_self_test'] = tf.nn.tanh(tf.matmul(input_data, self.classifier['w_self']))

    def evaluation(self):
        #self.classifier['class'] = self.pred
        #self.classifier['Target'] = self.targ
        self.classifier['class'] = self.classifier['z_self_test']
        self.classifier['Target'] = self.classifier['target_data']
        self.Evaluation['correct_prediction'] = tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class']),1 ), \
            tf.argmax(self.classifier['Target'], 1))
        self.Evaluation['accuracy'] = tf.reduce_mean(tf.cast(self.Evaluation['correct_prediction'], tf.float32))
        return self.Evaluation['accuracy']

    def last_testing(self, input_data, target_data):
        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
        target_data = tf.convert_to_tensor(target_data, dtype=tf.float32)
        self.classifier['z_self_test'] = tf.nn.softmax(tf.matmul(input_data, self.classifier['w_self']))
        #self.pred.append(self.classifier['z_self_test'])
        #self.targ.append(target_data)
        self.classifier['target_data'] = target_data
        
         
    # Function for defining the network
    def Def_Network(self, classes, Layers, batch_size, act_function):
        with tf.name_scope("Trainer_Network"):
            self.classifier['w_self'] = weight_variable(
                [Layers[0], Layers[1]], trainable=False, name='w_self')
            self.classifier['z_self'] = z_variable(
                [batch_size, Layers[1]], trainable=False, name='z_self')
            self.classifier['b_self'] = bias_variable(
                [Layers[1]], trainable=False, name='b_self')
            
            input_model  = self.Deep['FL_layer_10'] 
            preactivate = tf.matmul(input_model, self.classifier['w_self']) + self.classifier['b_self']
            self.Deep['FL_layer_11'] = act_function(preactivate)

            self.Weights.append((self.classifier['w_self'], self.classifier['b_self']))
            self.Z.append(self.classifier['z_self'])

        with tf.name_scope("Classifier"):
            self.classifier['class'] = self.Deep['FL_layer_11']
        
    def init_NN_custom(self, pr_id, nr_processes, classes, Layers, depth, batch_size, lambda_value, act_function=tf.nn.relu,
                        par='GDR', optimizer="Adam"):
            print "aaaa", pr_id, Layers, depth
            with tf.name_scope("PlaceHolders"):  

                # Setup the placeholders
                self.classifier['Target'] = tf.placeholder(
                    tf.float32, shape=[None, Layers[1]])
                self.classifier['t_next'] = tf.placeholder(
                    tf.float32, shape=[None, classes])
                self.classifier['z_next'] = tf.placeholder(
                    tf.float32, shape=[None, depth[pr_id+1]])
                self.classifier['z_prev'] = tf.placeholder(
                    tf.float32, shape=[None, depth[pr_id-1]])
                self.classifier['w_next'] = tf.placeholder(
                    tf.float32, shape=[None, depth[pr_id+1]])
                self.classifier["learning_rate"] = tf.placeholder(
                    tf.float32, [], name='learning_rate')
                self.Deep['FL_layer_10'] = tf.placeholder(
                    tf.float32, shape=[None, Layers[0]])

                # The main network
                #self.Def_Network(classes, Layers, batch_size, act_function)

                self.classifier['w_self'] = weight_variable(
                    [Layers[0], Layers[1]], trainable=False, name='w_self')
                if not pr_id == nr_processes:
                    self.classifier['z_self'] = z_variable(
                        [batch_size, Layers[1]], trainable=False, name='z_self')
                    self.classifier['z error'] = self.compute_z_error()


            with tf.name_scope("Trainer"):
                # The overall cost function
                self.classifier['Local Cost H'] = self.cost_function(pr_id, nr_processes, lambda_value)
                

            # The final optimization
            with tf.name_scope("Trainers"): 
                # Call the other optimizer
                self.Trainer['Weight_op'], self.classifier['Cost'], self.classifier['w update'] = self.W_Custom_Optimizer(self.classifier["learning_rate"], optimizer)
                
                if not pr_id == nr_processes:
                    self.Trainer['Z_op'], self.classifier['Cost'] = self.Z_Custom_Optimizer(self.classifier["learning_rate"], optimizer)
                
            ''' # The evaluation section of the methodology
            with tf.name_scope('Evaluation'):
                self.Evaluation['correct_prediction'] = \
                        tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class']),1 ) , 
                                tf.argmax(self.classifier['Target'], 1))
                self.Evaluation['accuracy'] = tf.reduce_mean(
                        tf.cast(self.Evaluation['correct_prediction'], tf.float32)) '''
            
            self.sess.run(tf.global_variables_initializer())
            return self