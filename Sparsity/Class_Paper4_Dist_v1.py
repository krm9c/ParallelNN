# The distributed learning paradigm
# Author : Krishnan Raghavan
# Date: June 22nd, 2018

#######################################################################################
# Define all the libraries
import random
import numpy as np
import tensorflow as tf
import operator
from functools import reduce


####################################################################################
# Helper Function for the weight and the bias variable initializations
# Weight
####################################################################################
def sample_Z(m, n, kappa):
    # return(np.random.uniform(-kappa, kappa, size=[m, n]))
    return(np.random.normal(kappa, kappa, size=[m, n]))


############################################################################################
def xavier(fan_in, fan_out):
   # use 4 for sigmoid, 1 for tanh activation
   low = -1 * np.sqrt(1.0 / (fan_in + fan_out))
   high = 1 * np.sqrt(1.0 / (fan_in + fan_out))
   return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)


############################################################################################
def weight_variable(shape, trainable, name):
   initial = xavier(shape[0], shape[1])
   return tf.Variable(initial, trainable=trainable, name=name)

def lambda_variable(shape, trainable, name):
    initial = tf.random_uniform([1, 1], minval=0, maxval=0.1, dtype=tf.float32)
    return tf.Variable(initial, trainable=trainable, name=name)
#############################################################################################
# Bias function
def bias_variable(shape, trainable, name):
   initial = tf.random_normal(shape, trainable, stddev=1)
   return tf.Variable(initial, trainable=trainable, name=name)


#############################################################################################
#  Summaries for the variables
def variable_summaries(var, key):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries' + key):
        mean = tf.reduce_mean(var)
        with tf.name_scope('stddev' + key):
           stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))


#############################################################################################
def init_ftn(name, num_input, num_output, runiform_range):
    if(name == "normal"):
        return(tf.truncated_normal([num_input, num_output]))
    elif(name == "uniform"):
        return(tf.random_uniform([num_input, num_output], minval=-runiform_range, maxval=runiform_range))
    else:
        print("not normal or uniform")


#############################################################################################
# The main Class
class learners():
    def __init__(self):
        self.classifier = {}
        self.Deep = {}
        self.Trainer = {}
        self.Evaluation = {}
        self.keys = []
        self.sess = tf.Session()

        # Extra Parameters
        self.Layer =[]
        self.Weights =[]
        self.Zs =[]
        self.Cost =[]
        self.Noise_List =[]
        self.Lambda_List =[]
        self.Lambda_Cost =[]
#############################################################################################
    # Function for defining every NN
    def nn_layer(self, input_tensor, input_dim, output_dim, act,
                 trainability, key, act_par="tf.nn.tanh"):
        with tf.name_scope(key):
            with tf.name_scope('weights' + key):
                self.classifier['Weight' + key] = weight_variable(
                    [input_dim, output_dim], trainable=trainability, name='Weight' + key)
            with tf.name_scope('bias' + key):
                self.classifier['Bias' + key] = bias_variable(
                    [output_dim], trainable=trainability, name='Bias' + key)
            with tf.name_scope('Wx_plus_b' + key):
                preactivate = tf.matmul(
                    input_tensor, self.classifier['Weight' + key]) + self.classifier['Bias' + key]
            return act(preactivate, name='activation' + key)



# The old optimizer
###################################################################################
    def Dist_optimizer(self, lr):
        train_list_weight = []
        train_list_Zis = []
        print("length", len(self.Weights), len(self.Zs) )
        for (i,(weight, bias)) in enumerate(self.Weights):
            cost = self.classifier["Overall_cost_dist"]
            # Zs update
            if i <(len(self.Weights)-1):
                Z_i = self.Zs[i]
                z_update = tf.gradients(cost,Z_i)[0] 
                train_list_Zis.append((z_update, Z_i))
            # Weight and bias updates
            weight_update = tf.gradients(cost,weight)[0]+ 0.0001*weight
            bias_update   = tf.gradients(cost,bias)[0]  + 0.0001*bias

            # Generate the updated variables
            train_list_weight.append((weight_update, weight))
            train_list_weight.append((bias_update, bias))      
              
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list_weight),\
        tf.train.AdamOptimizer(lr).apply_gradients(train_list_Zis)

    # Function for optimizing the noise
    def Lambda_optimizer_dist(self, lr):
        train_list =[]

        for (i,weight) in enumerate(self.Lambda_List):
            weight_update = -1*tf.gradients(self.classifier["Overall_cost_dist"], weight)[0] 
            # #Generate the updated variables
            train_list.append((weight_update, weight))

        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)



# The gradient descent optimizer
#############################################################################################
    # Function for optimizing the noise
    def Lambda_optimizer(self, lr):
        train_list =[]

        for (i,weight) in enumerate(self.Lambda_List):
            weight_update = -1*tf.sign(tf.gradients(self.classifier["Overall_cost"], weight)[0])
            # #Generate the updated variables
            train_list.append((weight_update, weight))

        return tf.train.AdamOptimizer(lr).apply_gradients(train_list)

###################################################################################
    def Grad_optimizer(self, lr):
        train_list_weight = []
        for (i,(weight, bias)) in enumerate(self.Weights):
            # Weight and bias updates
            weight_update = tf.gradients(self.classifier["Overall_cost"],weight)[0]+ 0.0001*weight
            bias_update   = tf.gradients(self.classifier["Overall_cost"],bias)[0]  + 0.0001*bias

            # Generate the updated variables
            train_list_weight.append((weight_update, weight))
            train_list_weight.append((bias_update, bias))      
              
        return tf.train.AdamOptimizer(lr).apply_gradients(train_list_weight)

# The network definitions
#############################################################################################
    def Def_Network(self, classes, Layers, act_function, batch_size, gamma, back_range = 1):
        with tf.name_scope("Trainer_Network"):            
            # Defining the weight and bias variable for the first layer
            i = 1
            self.classifier['WeightFL_layer_11'] = weight_variable([Layers[0], Layers[1]], trainable=False, name='WeightFL_layer_11')
            self.classifier['BiasFL_layer_11'] = bias_variable([Layers[1]], trainable=False, name='BiasFL_layer_11')
            self.Deep['FL_layer_11'] = act_function(tf.matmul(self.Deep['FL_layer_10'], self.classifier['WeightFL_layer_11']) + self.classifier['BiasFL_layer_11'])

            # Append the weights and the bias into the training list
            self.Weights.append((self.classifier['WeightFL_layer_11'], self.classifier['BiasFL_layer_11']))
            self.classifier['lambda'+str(i)] = lambda_variable([1,1], trainable=False, name='Z'+str(i))

            temp_cost = gamma*self.classifier['lambda'+str(i)]*tf.norm(self.classifier['WeightFL_layer_11'], ord = 1) \
            - tf.losses.mean_squared_error(self.classifier['lambda'+str(i)],0*self.classifier['lambda'+str(i)])
            self.Lambda_Cost.append( temp_cost)
            
            self.Lambda_List.append((self.classifier['lambda'+str(i)]))


            # The rest of the layers.
            for i in range(2, len(Layers)):
                key = 'FL_layer_1' + str(i)
                self.Deep['FL_layer_1' + str(i)] = self.nn_layer(self.Deep['FL_layer_1' + str(i-1)], Layers[i-1], Layers[i], act=act_function, trainability=False, key=key)
                # weights for each layer is append to the training list                
                self.Weights.append((self.classifier['Weight'+key], self.classifier['Bias'+key]))   
                self.classifier['lambda'+str(i)] = lambda_variable([1,1], trainable=False, name='Z'+str(i))

                temp_cost = gamma*self.classifier['lambda'+str(i)]*tf.norm(self.classifier['Weight'+key], ord = 1) \
                - tf.losses.mean_squared_error(self.classifier['lambda'+str(i)], 0*self.classifier['lambda'+str(i)])
                self.Lambda_Cost.append( temp_cost )
                self.Lambda_List.append((self.classifier['lambda'+str(i)]))

        # The final classifier
        with tf.name_scope("Classifier"):
            self.classifier['class_Noise'] = self.nn_layer(self.Deep['FL_layer_1' + str(len(Layers) - 1)],
            Layers[len(Layers)-1], classes, act=tf.identity, trainability=False, key='class')
            # weights for the final layer is appended on to the training list
            self.classifier['lambdaclass'] = lambda_variable([1,1], trainable=False, name='Z'+str(i))
            self.Weights.append((self.classifier['Weight'+'class'], self.classifier['Bias'+'class']))
            
            
            temp_cost = gamma*self.classifier['lambdaclass']*tf.norm(self.classifier['Weightclass'], ord = 1) \
            - tf.losses.mean_squared_error(self.classifier['lambdaclass'],0*self.classifier['lambdaclass'])
            self.Lambda_Cost.append(temp_cost)
            self.Lambda_List.append((self.classifier['lambdaclass']))

################################################################################################################           
    def New_Formulation(self, classes, Layers, act_function, batch_size, gamma, back_range = 1, rho = 0.8):
        #Finally setup all the Z's Due to the new formulation
        #The first layer
        i = 1 
        key = 'FL_layer_1' + str(i)
        self.classifier['Zs' + key] =  act_function(tf.matmul(self.Deep['FL_layer_10'], self.classifier['Weight'+key])+self.classifier['Bias'+key])

        # The second onwards the final layer
        for i in range(2, len(Layers)):
            key = 'FL_layer_1' + str(i)
            self.classifier['Zs' + key] = weight_variable([batch_size, Layers[i]], trainable=False, name='Z'+str(i))
            self.Zs.append(self.classifier['Zs' + key])

            # Cost for each layer Zs, that are used in optimization
            key_prev = 'FL_layer_1' + str(i-1)

            ####Layer Cost
            Temp_Cost =  rho*tf.losses.mean_squared_error( self.classifier['Zs' + key], \
            act_function(tf.matmul(self.classifier['Zs' + key_prev], self.classifier['Weight'+key]) + self.classifier['Bias'+key]) ) 
            self.Cost.append(Temp_Cost)

        # The classifier and the Z for the final classifier
        self.classifier['Zsclass'] = weight_variable([batch_size, classes], trainable=False, name= 'Z'+str((len(Layers)-1)) )
        self.Zs.append(self.classifier['Zsclass'])
        key = 'Zsclass'
        key_prev = 'FL_layer_1' + str(len(Layers)-1)

        # Final layer Cost
        Temp_Cost = rho*tf.losses.mean_squared_error( self.classifier[key], tf.matmul( self.classifier['Zs'+key_prev],\
        self.classifier['Weightclass']) + self.classifier['Biasclass'] )  
        self.Cost.append(Temp_Cost)


 ############################################################################################################################
    def init_NN_custom(self, classes, lr, Layers, act_function, batch_size, gamma,
                        back_range= 1,  par='GDR', act_par="tf.nn.tanh", rho = 0.8):
################################################################## Initial Definitions
            with tf.name_scope("PlaceHolders"):  
                #### Setup the placeholders        
                # Label placeholder
                self.classifier['Target'] = tf.placeholder(
                tf.float32, shape=[None, classes])
                # Place holder for the learning rate
                self.classifier["learning_rate"] = tf.placeholder(
                tf.float32, [], name='learning_rate')
                # Input placeholder
                self.Deep['FL_layer_10'] = tf.placeholder(
                tf.float32, shape=[None, Layers[0]])

##################################################################################### The network
                # The main network
                self.Def_Network(classes, Layers, act_function, batch_size, gamma, back_range)
                self.New_Formulation(classes, Layers, act_function, batch_size, gamma, back_range, rho)

############################################################## Design the trainer for the  network
            with tf.name_scope("Trainer"):
            #The overall cost function
                # self.classifier["Overall_cost_dist"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.classifier['Zsclass'], labels=self.classifier['Target'], name='Error_Cost')) \
                # + tf.add_n(self.Cost) + 0.001*tf.add_n(self.Lambda_Cost)
                
                self.classifier["Overall_cost"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.classifier['class_Noise'], labels=self.classifier['Target'], name='Error_Cost')) \
                + tf.add_n(self.Lambda_Cost) 

                # The Regular optimizer
                self.Trainer["Grad_op"]   = self.Grad_optimizer(self.classifier["learning_rate"]) 
                self.Trainer["Lambda_op"] = self.Lambda_optimizer(self.classifier["learning_rate"])  

                # # Distributed Op
                # self.Trainer["Weight_op_dist"], self.Trainer["Zis_op_dist"] = self.Dist_optimizer(self.classifier["learning_rate"]) 
                # self.Trainer["Lambda_op_dist"] = self.Lambda_optimizer_dist(self.classifier["learning_rate"]) 

        ############## The evaluation section of the methodology
            with tf.name_scope('Evaluation'):
                self.Evaluation['correct_prediction'] = \
                        tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class_Noise']),1 ) , 
                                tf.argmax(self.classifier['Target'], 1))
                self.Evaluation['accuracy'] = tf.reduce_mean(
                        tf.cast(self.Evaluation['correct_prediction'], tf.float32))
            self.sess.run(tf.global_variables_initializer())
            return self




