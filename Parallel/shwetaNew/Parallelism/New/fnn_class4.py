
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
        self.Trainer = {}
        self.sess = tf.Session(config=config)

    def compute_cost_function(self, act): 
        first_term = self.classifier['lambda_value']*(tf.nn.l2_loss(tf.subtract(self.classifier['z_next'], \
            act(tf.add(tf.matmul(self.classifier['z_self'], self.classifier['w_next']),self.classifier['b_next'])))))
        second_term = self.classifier['lambda_value']*(tf.nn.l2_loss(tf.subtract(self.classifier['z_self'], \
            act(tf.add(tf.matmul(self.classifier['z_prev'], self.classifier['w_self']),self.classifier['b_self'])))))

        return tf.add(first_term, second_term)

    def last_cost_function(self, act):
        z_next = tf.nn.softmax(tf.add(tf.matmul(self.classifier['z_self'], self.classifier['w_next']),self.classifier['b_next']))
        first_term = tf.nn.l2_loss(tf.subtract(self.classifier['t_next'], z_next))

        # z_next = tf.add(tf.matmul(self.classifier['z_self'], self.classifier['w_next']),self.classifier['b_next'])
        # first_term = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=z_next, labels=self.classifier['t_next'], name='Error_Cost')) 
        
        second_term = self.classifier['lambda_value']*(tf.nn.l2_loss(tf.subtract(self.classifier['z_self'], \
            act(tf.add(tf.matmul(self.classifier['z_prev'], self.classifier['w_self']),self.classifier['b_self'])))))

        return tf.add(first_term, second_term)

    def cost_function(self, pr_id, nr_processes, act=tf.nn.tanh):
        if pr_id == nr_processes:
            self.classifier['Local_Cost_H'] = self.last_cost_function(act)
        else:
            self.classifier['Local_Cost_H'] = self.compute_cost_function(act)

        return self.classifier['Local_Cost_H']


    def compute_z_error(self):
        mul = tf.nn.tanh(tf.add(tf.matmul(self.classifier['z_prev'], self.classifier['w_self']), self.classifier['b_self']))
        return tf.nn.l2_loss(tf.subtract(self.classifier['z_self'],mul))


    def Z_Custom_Optimizer(self, lr_z):
        train_list = []
        z_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['z_self'])[0]
        train_list.append((z_update, self.classifier['z_self']))
        return tf.train.AdamOptimizer(lr_z).apply_gradients(train_list)

    def self_W_Custom_Optimizer(self, lr_w):
        train_list = []
        w_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['w_self'])[0]   
        b_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['b_self'])[0]  
        train_list.append((w_update, self.classifier['w_self']))
        train_list.append((b_update, self.classifier['b_self']))
        return tf.train.AdamOptimizer(lr_w).apply_gradients(train_list)

    def next_W_Custom_Optimizer(self, lr_w):
        train_list = []
        w_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['w_next'])[0]   
        b_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['b_next'])[0]  
        train_list.append((w_update, self.classifier['w_next']))
        train_list.append((b_update, self.classifier['b_next']))
        return tf.train.AdamOptimizer(lr_w).apply_gradients(train_list)

        
    def init_NN_custom(self, pr_id, nr_processes, classes, depth, batch_size):
        with tf.name_scope("PlaceHolders"):  
            # Setup the placeholders
            self.classifier['t_next'] = tf.placeholder(
                tf.float32, shape=[None, classes])
            self.classifier['z_next'] = tf.placeholder(
                tf.float32, shape=[None, depth[pr_id+1]])
            self.classifier['z_prev'] = tf.placeholder(
                tf.float32, shape=[None, depth[pr_id-1]])
            self.classifier['learning_rate'] = tf.placeholder(
                tf.float32, [], name='learning_rate')
            self.classifier['lambda_value'] = tf.placeholder(
                tf.float32, [], name='lambda_value')


            self.classifier['w_next'] = weight_variable(
                    [depth[pr_id], depth[pr_id+1]], trainable=False, name='w_next')
            self.classifier['b_next'] = bias_variable(
                    [depth[pr_id+1]], trainable=False, name='b_next')

            if not pr_id == 1:
                self.classifier['w_self'] = tf.placeholder(
                tf.float32, shape=[None, depth[pr_id]])
                self.classifier['b_self'] = tf.placeholder(
                    tf.float32, shape=[depth[pr_id]])

            elif pr_id == 1:
                self.classifier['w_self'] = weight_variable(
                    [depth[pr_id-1], depth[pr_id]], trainable=False, name='w_self')
                self.classifier['b_self'] = bias_variable(
                    [depth[pr_id]], trainable=False, name='b_self')

            
            
            self.classifier['z_self'] = z_variable(
                [batch_size, depth[pr_id]], trainable=False, name='z_self')
            self.classifier['z_error'] = self.compute_z_error()

            self.classifier['Local_Cost_H'] = self.cost_function(pr_id, nr_processes)

        
        # The final optimization
        with tf.name_scope("Trainers"):
            # Call the other optimizer

            if pr_id == 1:
                self.Trainer['self_Weight_op']  =\
                    self.self_W_Custom_Optimizer(self.classifier['learning_rate'])

            self.Trainer['next_Weight_op']  =\
                self.next_W_Custom_Optimizer(self.classifier['learning_rate']) 
            
            self.Trainer['Z_op'] = \
                self.Z_Custom_Optimizer(self.classifier['learning_rate'])

        self.sess.run(tf.global_variables_initializer())
        return self