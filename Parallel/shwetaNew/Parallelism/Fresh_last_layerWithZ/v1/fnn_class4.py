
# Class for Generalized Feed Forward Neural Network
import tensorflow as tf
import numpy as np
import random
import operator

# Xavier Initialization
def xavier(fan_in, fan_out):
    # use 4 for sigmoid, 1 for tanh activation
    low = -1* np.sqrt(1.0 / (fan_in + fan_out))
    high = 1* np.sqrt(1.0 / (fan_in + fan_out))
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
        return(tf.train.AdamOptimizer)
    elif optimizer == 'Adadelta':
        return(tf.train.AdadeltaOptimizer)
    elif optimizer == 'Adagrad':
        return(tf.train.AdagradOptimizer)
    elif optimizer == 'GradientDescent':
        return(tf.train.GradientDescentOptimizer)
    elif optimizer == '':
        return(tf.train.RMSPropOptimizer)

def act_ftn(name):
    if(name == 'tanh'):
        return(tf.nn.tanh)
    elif(name == 'relu'):
        return(tf.nn.relu)
    elif(name == 'sigmoid'):
        return(tf.nn.sigmoid)


# The main Class
class learners():
    def __init__(self,config):
        self.classifier = {}
        self.Trainer = {}
        self.Weights = []
        self.Zs =[]
        self.sess = tf.Session(config=config)

    def compute_cost_function(self, act): 
        first_term = self.classifier['lambda_value']*(tf.nn.l2_loss(tf.subtract(self.classifier['z_next'], \
            act(tf.add(tf.matmul(self.classifier['z_self'], self.classifier['w_next']),self.classifier['b_next'])))))
        second_term = self.classifier['lambda_value']*(tf.nn.l2_loss(tf.subtract(self.classifier['z_self'], \
            act(tf.add(tf.matmul(self.classifier['z_prev'], self.classifier['w_self']),self.classifier['b_self'])))))
        return tf.add(first_term, second_term)
 
    def compute_z_error(self):
        mul = tf.nn.tanh(tf.add(tf.matmul(self.classifier['z_prev'], self.classifier['w_self']), self.classifier['b_self']))
        return tf.nn.l2_loss(tf.subtract(self.classifier['z_self'],mul))

#################################################################################################
    def cost_function(self, pr_id, nr_processes, activation, optimizer, n_layers):
        if pr_id == 1:
            return self.first_cost_function(activation, optimizer, n_layers)
        elif pr_id == nr_processes:
            return self.last_cost_function(activation, optimizer, n_layers)

    def last_cost_function(self, activation, optimizer, n_layers):
        act = act_ftn(activation)
        opt = get_optimizer_name(optimizer)
        zFinal = tf.nn.softmax(tf.matmul(self.classifier['z4'], self.classifier['w5'] ) + self.classifier['b5'])

        self.classifier['Local_Cost_H'] =  tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = zFinal,labels=self.classifier['t_next'], name='Error_Cost'))  +\
        0.5*self.classifier['lambda_value']*tf.losses.mean_squared_error (self.classifier['z4_hat'], self.classifier['z4'])
        #List for updating the Ws and the Zs
        self.classifier['train_list_W'] =[]
        self.classifier['train_list_Z'] =[] 
        for (number, (weight, bias)) in enumerate(self.Weights):
            w_update = tf.gradients(self.classifier['Local_Cost_H'], weight)[0]   
            b_update = tf.gradients(self.classifier['Local_Cost_H'], bias)[0]   
            self.classifier['train_list_W'].append((w_update,weight))
            self.classifier['train_list_W'].append((b_update, bias))

        for (number, z) in enumerate(self.Zs):
            z_update = tf.gradients(self.classifier['Local_Cost_H'],z)[0]
            self.classifier['train_list_Z'].append((z_update, z))
        return self.classifier['Local_Cost_H'], opt(self.classifier['learning_rate']).apply_gradients(self.classifier['train_list_W']), \
        opt(self.classifier['learning_rate']).apply_gradients(self.classifier['train_list_Z'])

    def first_cost_function(self, activation, optimizer, n_layers):
        act = act_ftn(activation)
        opt = get_optimizer_name(optimizer)
        self.classifier['z4'] = act(tf.matmul(self.classifier['z3_hat'], self.classifier['w4'] ) + self.classifier['b4'] )
        z5 = tf.nn.softmax( tf.matmul( self.classifier['z4_hat'], self.classifier['w5'] ) + self.classifier['b5'] ) 

        self.classifier['Local_Cost_H'] =  self.classifier['lambda_value']*tf.losses.mean_squared_error(self.classifier['z3_hat'], self.classifier['z3'])\
        +0.5*self.classifier['lambda_value']*tf.losses.mean_squared_error(self.classifier['z4_hat'], self.classifier['z4'])

        #List for updating the Ws and the Zs
        self.classifier['train_list_W'] =[]
        self.classifier['train_list_Z'] =[] 
        for (number, (weight, bias)) in enumerate(self.Weights):
            w_update = tf.gradients(self.classifier['Local_Cost_H'], weight)[0]   
            b_update = tf.gradients(self.classifier['Local_Cost_H'], bias)[0]   
            self.classifier['train_list_W'].append((w_update,weight))
            self.classifier['train_list_W'].append((b_update, bias))
        for (number, z) in enumerate(self.Zs):
            z_update = tf.gradients(self.classifier['Local_Cost_H'],z)[0]
            self.classifier['train_list_Z'].append((z_update, z))
        return self.classifier['Local_Cost_H'], \
            opt(self.classifier['learning_rate']).apply_gradients(self.classifier['train_list_W']), \
            opt(self.classifier['learning_rate']).apply_gradients(self.classifier['train_list_Z'])
            
#################################################################################################
    def init_NN_custom(self, pr_id, nr_processes, classes, depth, batch_size, activation='tanh', optimizer='Adam'):
        with tf.name_scope("Trainers"):  
            act = act_ftn(activation)
            if pr_id == 1:
                self.classifier['t_next'] = tf.placeholder(tf.float32, shape=[None, classes])
                # Z0 (Data Input)
                self.classifier['z0'] = tf.placeholder(tf.float32, shape=[None, depth[0]])
                self.classifier['learning_rate'] = tf.placeholder(tf.float32, [], name='learning_rate')
                self.classifier['lambda_value'] = tf.placeholder(tf.float32, [], name='lambda_value')
        
                ## The W1 and z1
                self.classifier['w1'] = weight_variable([depth[0], depth[1]], trainable=False, name='w1')
                self.classifier['b1'] = bias_variable([depth[1]], trainable=False, name='b1')
                self.classifier['z1'] = act(tf.matmul(self.classifier['z0'], self.classifier['w1'] ) + self.classifier['b1'] )
                
                ## The W2 and Z2
                self.classifier['w2'] = weight_variable([depth[1], depth[2]], trainable=False, name='w2')
                self.classifier['b2'] = bias_variable([depth[2]], trainable=False, name='b2')
                self.classifier['z2'] = act(tf.matmul(self.classifier['z1'], self.classifier['w2'] ) + self.classifier['b2'] )
                
                ## The W3 and Z3
                self.classifier['w3'] = weight_variable([depth[2], depth[3]], trainable=False, name='w3')
                self.classifier['b3'] = bias_variable([depth[3]], trainable=False, name='b3')
                self.classifier['z3'] = act(tf.matmul(self.classifier['z2'], self.classifier['w3'] ) + self.classifier['b3'] )
                self.classifier['z3_hat'] = z_variable([batch_size, depth[3]], trainable=False, name='z3_hat')
                
                # (Weights of next Process)
                n_layers = 3
                self.classifier['w4'] = tf.placeholder(tf.float32, shape=[ depth[3], depth[4]])
                self.classifier['b4'] = tf.placeholder(tf.float32, shape=[depth[4]])
                self.classifier['w5'] = tf.placeholder(tf.float32, shape=[ depth[4], depth[5]])
                self.classifier['b5'] = tf.placeholder(tf.float32, shape=[depth[5]])
                self.classifier['z4_hat'] = tf.placeholder(tf.float32, shape=[None, depth[4]])

                ## The final variable updates 
                self.Weights.append((self.classifier['w1'], self.classifier['b1']))
                self.Weights.append((self.classifier['w2'], self.classifier['b2']))
                self.Weights.append((self.classifier['w3'], self.classifier['b3']))
                self.Zs.append(self.classifier['z3_hat'])
                # The cost and variables
                self.classifier['cost'+str(pr_id)], self.Trainer['Weight_op'], self.Trainer['Z_op']   = self.cost_function(pr_id, nr_processes, activation, optimizer, n_layers)

            elif pr_id == nr_processes:          
                # Setup the placeholders
                self.classifier['t_next'] = tf.placeholder(tf.float32, shape=[None, classes])
                # Z3_hat (Coming from previous process)
                self.classifier['z3_hat'] = tf.placeholder(tf.float32, shape=[None, depth[3]])
                self.classifier['learning_rate'] = tf.placeholder(tf.float32, [], name='learning_rate')
                self.classifier['lambda_value'] = tf.placeholder(tf.float32, [], name='lambda_value')

                ## The W, b definitions for the last layer 
                self.classifier['w4'] = weight_variable([depth[3], depth[4]], trainable=False, name='w4')
                self.classifier['b4'] = bias_variable([depth[4]], trainable=False, name='b4')
                self.classifier['z4'] = act(tf.matmul(self.classifier['z3_hat'], self.classifier['w4'] ) + self.classifier['b4'] )
                self.classifier['z4_hat'] = z_variable([batch_size, depth[4]], trainable=False, name='z4_hat')
                
                n_layers = 3
                ### The W, b definitions
                self.classifier['w5'] = weight_variable([depth[4], depth[5]], trainable=False, name='w5')
                self.classifier['b5'] = bias_variable([depth[5]], trainable=False, name='b5')
                # The cost and variables
                self.Weights.append((self.classifier['w4'], self.classifier['b4']))
                self.Weights.append((self.classifier['w5'], self.classifier['b5']))
                self.Zs.append(self.classifier['z4_hat'])


                # self.classifier['cost'+str(pr_id)], self.Trainer['Weight_op']\
                # = self.cost_function(pr_id, nr_processes, activation, optimizer, n_layers)

                self.classifier['cost'+str(pr_id)], self.Trainer['Weight_op'], self.Trainer['Z_op']\
                = self.cost_function(pr_id, nr_processes, activation, optimizer, n_layers)
     

        self.sess.run(tf.global_variables_initializer())
        return self