
# Class for Generalized Feed Forward Neural Network
import tensorflow as tf
import numpy as np
import random
import operator

# Xavier Initialization
def xavier(fan_in, fan_out):
    # use 4 for sigmoid, 1 for tanh activation
    low = -0.1* np.sqrt(2.0 / (fan_in + fan_out))
    high = 0.1* np.sqrt(2.0 / (fan_in + fan_out))
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
    elif optimizer == 'RMSProp':
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


    def first_cost_function(self, activation, optimizer):
        act = act_ftn(activation)
        opt = get_optimizer_name(optimizer)

        # Z Calculation
        z_next1 = act(tf.matmul(self.classifier['z_self'], self.classifier['w_next'] ) + self.classifier['b_next'] )
        z_next2 = act(tf.matmul(self.classifier['z_1'], self.classifier['w_self'] ) + self.classifier['b_self'] )
        
        
        self.classifier['Local_Cost_H'] = 0.5*self.classifier['lambda_value']*tf.nn.l2_loss(tf.subtract(self.classifier['z_next'], z_next1) ) \
        + self.classifier['lambda_value']*tf.nn.l2_loss(tf.subtract(self.classifier['z_self'], z_next2) )
        
        # Update the Ws and the Zs
        self.classifier['train_list_W'] =[]

        w_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['w_prev'])[0]   
        b_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['b_prev'])[0]   
        self.classifier['train_list_W'].append((w_update, self.classifier['w_prev']))
        self.classifier['train_list_W'].append((b_update, self.classifier['b_prev']))

        w_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['w_self'])[0]   
        b_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['b_self'])[0]   
        self.classifier['train_list_W'].append((w_update, self.classifier['w_self']))
        self.classifier['train_list_W'].append((b_update, self.classifier['b_self']))


        self.classifier['train_list_Z'] =[] 
        z_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['z_self'])[0]
        self.classifier['train_list_Z'].append((z_update, self.classifier['z_self']))

        return self.classifier['Local_Cost_H'], \
            opt(self.classifier['learning_rate']).apply_gradients(self.classifier['train_list_W']), \
            opt(self.classifier['learning_rate']).apply_gradients(self.classifier['train_list_Z'])


    def middle_cost_function(self, activation, optimizer):
        act = act_ftn(activation)
        opt = get_optimizer_name(optimizer)

        # Z Calculation
        z_next = act(tf.matmul(self.classifier['z_prev'], self.classifier['w_self'] ) + self.classifier['b_self'] )
        zFinal  = tf.add(tf.matmul( self.classifier['z_self'], self.classifier['w_next']),self.classifier['b_next'])
        
        
        self.classifier['Local_Cost_H']= 0.5*self.classifier['lambda_value']*tf.nn.l2_loss(tf.subtract(self.classifier['z_self'], z_next)) \
        + tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            logits = zFinal, labels=self.classifier['t_next'], name='Error_Cost'))
            
        
        # Update the Ws and the Zs
        self.classifier['train_list_W'] =[]

        w_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['w_self'])[0]   
        b_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['b_self'])[0]   
        self.classifier['train_list_W'].append((w_update, self.classifier['w_self']))
        self.classifier['train_list_W'].append((b_update, self.classifier['b_self']))

        self.classifier['train_list_Z'] =[]

        z_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['z_self'])[0]
        self.classifier['train_list_Z'].append((z_update, self.classifier['z_self']))

        return self.classifier['Local_Cost_H'], \
            opt(self.classifier['learning_rate']).apply_gradients(self.classifier['train_list_W']), \
            opt(self.classifier['learning_rate']).apply_gradients(self.classifier['train_list_Z'])

    
    def last_cost_function(self, activation, optimizer):
        act = act_ftn(activation)
        opt = get_optimizer_name(optimizer)

        z_next = tf.add(tf.matmul(self.classifier['z_prev'], self.classifier['w_self']),self.classifier['b_self'])
        # first_term = tf.nn.l2_loss(tf.subtract(self.classifier['t_next'], z_next))
        # self.classifier['Local_Cost_H'] = first_term

        self.classifier['Local_Cost_H'] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits = z_next, labels=self.classifier['t_next'], name='Error_Cost'))

        # Update the Ws and the Zs
        self.classifier['train_list_W'] =[]    
        w_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['w_self'])[0]   
        b_update = tf.gradients(self.classifier['Local_Cost_H'], self.classifier['b_self'])[0]   
        self.classifier['train_list_W'].append((w_update, self.classifier['w_self']))
        self.classifier['train_list_W'].append((b_update, self.classifier['b_self']))

        return self.classifier['Local_Cost_H'], \
            opt(self.classifier['learning_rate']).apply_gradients(self.classifier['train_list_W'])


    def cost_function(self, pr_id, nr_processes, activation, optimizer):
        if pr_id == 2:
            return self.first_cost_function(activation, optimizer)
        elif pr_id == nr_processes+1:
            return self.last_cost_function(activation, optimizer)
        else:
            return self.middle_cost_function(activation, optimizer)

        
    def init_NN_custom(self, pr_id, nr_processes, classes, depth, batch_size, activation='tanh', optimizer='Adam'):
        with tf.name_scope("Trainers"):  

            act = act_ftn(activation)

            print("The process id", pr_id)

            # First Process
            if pr_id == 2:
                self.classifier['learning_rate'] = tf.placeholder(tf.float32, [], name='learning_rate')
                self.classifier['lambda_value'] = tf.placeholder(tf.float32, [], name='lambda_value')
                
                # Z0 (Data Input)
                self.classifier['z_prev'] = tf.placeholder(tf.float32, shape=[None, depth[0]])

                ## W1
                self.classifier['w_prev'] = weight_variable([depth[0], depth[1]], trainable=False, name='w_prev')
                self.classifier['b_prev'] = bias_variable([depth[1]], trainable=False, name='b_prev')

                ## W2 (Self)
                self.classifier['w_self'] = weight_variable([depth[pr_id-1], depth[pr_id]], trainable=False, name='w_self')
                self.classifier['b_self'] = bias_variable([depth[pr_id]], trainable=False, name='b_self')

                # W3 (Weight of next Process)
                self.classifier['w_next'] = tf.placeholder(tf.float32, shape=[None, depth[pr_id+1]])
                self.classifier['b_next'] = tf.placeholder(tf.float32, shape=[depth[pr_id+1]])
                
                # Z1 (Assigning value from Z0)
                self.classifier['z_1'] = act(tf.matmul(self.classifier['z_prev'], self.classifier['w_prev'] ) + self.classifier['b_prev'])
                
                # Z2 (Self)
                self.classifier['z_self'] = z_variable([batch_size, depth[pr_id]], trainable=False, name='z_self')

                # Z3 (Z from Middle process)
                self.classifier['z_next'] = tf.placeholder(tf.float32, shape=[None, depth[pr_id+1]])
                
                # The cost and variables
                self.classifier['cost'+str(pr_id)], self.Trainer['Weight_op'], self.Trainer['Z_op']   = self.cost_function(pr_id, nr_processes, activation, optimizer)

            
            # Last Process
            elif pr_id == nr_processes+1: 
                self.classifier['learning_rate'] = tf.placeholder(tf.float32, [], name='learning_rate')
                self.classifier['lambda_value'] = tf.placeholder(tf.float32, [], name='lambda_value')

                self.classifier['t_next'] = tf.placeholder(tf.float32, shape=[None, classes])

                # Z3 (Coming from middle process)
                self.classifier['z_prev'] = tf.placeholder(tf.float32, shape=[None, depth[pr_id-1]])

                ## W4 and b4 (Self)
                self.classifier['w_self'] = weight_variable([depth[pr_id-1], depth[pr_id]], trainable=False, name='w_self')
                self.classifier['b_self'] = bias_variable([depth[pr_id]], trainable=False, name='b_self')

                # The cost and variables
                self.classifier['cost'+str(pr_id)], self.Trainer['Weight_op'] = self.cost_function(pr_id, nr_processes, activation, optimizer)     


            # Middle Process
            else:
                self.classifier['learning_rate'] = tf.placeholder(tf.float32, [], name='learning_rate')
                self.classifier['lambda_value'] = tf.placeholder(tf.float32, [], name='lambda_value')

                self.classifier['t_next'] = tf.placeholder(tf.float32, shape=[None, classes])

                # Z2 (Coming from first process)
                self.classifier['z_prev'] = tf.placeholder(tf.float32, shape=[None, depth[pr_id-1]])

                # Z3 (Self)
                self.classifier['z_self'] = z_variable([batch_size, depth[pr_id]], trainable=False, name='z_self')

                ## W3 and b3 (Self) 
                self.classifier['w_self'] = weight_variable([depth[pr_id-1], depth[pr_id]], trainable=False, name='w_self')
                self.classifier['b_self'] = bias_variable([depth[pr_id]], trainable=False, name='b_self')

                # W4 (Weight of last Process)
                self.classifier['w_next'] = tf.placeholder(tf.float32, shape=[None, depth[pr_id+1]])
                self.classifier['b_next'] = tf.placeholder(tf.float32, shape=[depth[pr_id+1]])

                # The cost and variables
                self.classifier['cost'+str(pr_id)], self.Trainer['Weight_op'], self.Trainer['Z_op']   = self.cost_function(pr_id, nr_processes, activation, optimizer)


        self.sess.run(tf.global_variables_initializer())
        return self