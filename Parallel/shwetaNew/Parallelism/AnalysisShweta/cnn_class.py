
# Class for Generalized Convolutional Neural Network
import tensorflow as tf
import numpy as np
import random
import operator

# Function for defining weights
def new_weights(shape, trainable, name):
    initial = tf.Variable(tf.truncated_normal(shape, stddev=0.1), trainable=trainable, name=name)
    return initial

# Function for defining biases
def new_bias(length, trainable, name):
    initial = tf.Variable(tf.constant(0.1, shape=[length]), trainable=trainable, name=name)
    return initial

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
    def __init__(self, config):
        self.classifier = {}
        self.Deep = {}
        self.Trainer = {}
        self.Evaluation = {}
        self.keys = []
        self.sess = tf.Session(config=config)

        # Extra Parameters
        self.Ind_Cost = {}
        self.Weights =[]
        self.Zs =[]
        self.Cost =[]

    def new_conv_layer(self,input, key, pool_shape, use_pooling=True):
        with tf.name_scope(key):
            # tf.nn.conv2d needs a 4D input
            layer = tf.nn.conv2d(input = input, filter= self.classifier['Weight' + key], strides=[1,1,1,1], padding='SAME')
            layer += self.classifier['Bias' + key]
            if use_pooling:
                ksize = [1, pool_shape[0], pool_shape[1], 1]
                layer = tf.nn.max_pool(value = layer, ksize=ksize, strides=[1,2,2,1], padding='SAME')
            # relu activation function converts all negatives to zero
            layer = tf.nn.relu(layer)
            return layer

    # After all convolutions, we need to flatten the layer
    def flatten_layer(self,layer):
        layer_shape = layer.get_shape()
        num_features = layer_shape[1:4].num_elements()
        layer_flat = tf.reshape(layer, [-1, num_features])
        return layer_flat, num_features

    # Fully connected layer
    def new_fc_layer(self, input, key, use_relu=True, dropout=False, keep_prob=1.0):
        with tf.name_scope(key):
            layer = tf.matmul(input, self.classifier['Weight' + key]) + self.classifier['Bias' + key]
            if use_relu:
                layer = tf.nn.relu(layer)
            if dropout:
                layer = tf.nn.dropout(layer, keep_prob)
            return layer

    def Custom_Optimizer(self, lr, optimizer):
        opt = get_optimizer_name(optimizer)
        
        train_list = []
        for i in xrange(len(self.Weights)):
            weight, bias = self.Weights[i]
            ## Gradient Descent update
            weight_update = tf.gradients(self.classifier['Cost_M1_Dist'], weight)[0]   + 0.0001*weight
            bias_update   = tf.gradients(self.classifier['Cost_M1_Dist'],   bias)[0]   + 0.0001*bias

            # Generate the updated variables
            train_list.append((weight_update, weight))
            train_list.append((bias_update, bias))
        return opt(lr).apply_gradients(train_list)

    def Dist_optimizer(self, lr, optimizer):
        opt = get_optimizer_name(optimizer)

        train_list_weight = []
        train_list_Zis = []
        print("length cost", len(self.Cost), "weights", len(self.Weights), "Zs", len(self.Zs) )
        T_Cost = tf.add_n(self.Cost)
        
        for (i,z) in enumerate(self.Zs):
            if i==0:
                # cost = self.Cost[i] + self.Cost[i+1]
                cost = T_Cost 
                weight1, bias1 = self.Weights[i]
                weight2, bias2 = self.Weights[i+1]
                # Weight and bias updates
                weight_update = tf.gradients(cost,weight1)[0]+ self.classifier["l2_norm"]*weight1
                bias_update   = tf.gradients(cost,bias1)[0]  + self.classifier["l2_norm"]*bias1
                train_list_weight.append((weight_update, weight1))
                train_list_weight.append((bias_update, bias1))   

                # Weight and bias updates
                weight_update = tf.gradients(cost,weight2)[0]+ self.classifier["l2_norm"]*weight2
                bias_update   = tf.gradients(cost,bias2)[0]  + self.classifier["l2_norm"]*bias2
                train_list_weight.append((weight_update, weight2))
                train_list_weight.append((bias_update, bias2))  
            else:
                # cost = self.Cost[i]+ self.Cost[i+1] 
                cost = T_Cost
                weight, bias = self.Weights[i+1]
                # Weight and bias updates
                weight_update = tf.gradients(cost,weight)[0]+ self.classifier["l2_norm"]*weight
                bias_update   = tf.gradients(cost,bias)[0]  + self.classifier["l2_norm"]*bias
                train_list_weight.append((weight_update, weight))
                train_list_weight.append((bias_update, bias))   
            
            z_update = tf.gradients(cost,z)[0] 
            train_list_Zis.append((z_update,z))
        
        return opt(lr).apply_gradients(train_list_weight),\
            opt(lr).apply_gradients(train_list_Zis)



    def Def_Network(self, classes, num_conv_layers, num_fc_layers, img_size, num_channels, \
        filter_size, num_of_filters, pool_shape, fc_size, batch_size, activation):

        act_function = act_ftn(activation)

        with tf.name_scope("Trainer_Network"):
            #inp_fc_layer = []
            self.Deep['Conv_Layer_0'] = tf.reshape(self.Deep['Conv_Layer_00'], [-1, img_size[0], img_size[1], num_channels])
            
            for i in range(1, num_conv_layers+1):
                key = 'Conv_Layer_'+ str(i)
                num_input_channels= num_of_filters[i-1]
                filter = filter_size[i-1]
                num_filters = num_of_filters[i]
                shape = [filter, filter, num_input_channels, num_filters]
                self.classifier['Weight' + key] = new_weights(
                    shape = shape, trainable=False, name='Weight'+key)
                self.classifier['Bias' + key] = new_bias(
                    length = num_filters, trainable=False, name='Bias'+key)
                self.Deep['Conv_Layer_'+ str(i)] = self.new_conv_layer(input = self.Deep['Conv_Layer_'+str(i-1)], \
                    key=key, pool_shape=pool_shape, use_pooling=True)
                self.Weights.append((self.classifier['Weight' + key], self.classifier['Bias' + key]))
            
        
            self.Deep['Conv_Layer_'+ str(i)], num_features = self.flatten_layer(self.Deep['Conv_Layer_'+ str(i)])
            

            ''' i = i +1
            inp_fc_layer[0] = num_features
            for k in range(1, len(num_fc_layers)):
                inp_fc_layer[k] = fc_size
            inp_fc_layer.append(classes)

            temp = i
            for i in range(temp, (len(num_fc_layers)+temp)):
                key = 'Conv_Layer_'+ str(i)
                self.classifier['Weight' + key] = new_weights(
                    shape=[inp_fc_layer.append[i-temp], inp_fc_layer.append[i-temp+1]], trainable=False, name='Weight'+key)
                self.classifier['Bias' + key] = new_bias(
                    length= inp_fc_layer.append[i-temp+1], trainable=False, name='Bias'+key)
                self.Deep['Conv_Layer_'+ str(i)] = model.new_fc_layer(self.Deep['Conv_Layer_'+ str(i-1)], \
                key=key, use_relu=True, dropout=True, keep_prob=0.5) 
                self.Weights.append((self.classifier['Weight'+key], self.classifier['Bias'+key]))
                print key '''
            
            i = i +1
            key = 'Conv_Layer_'+ str(i)
            self.classifier['Weight' + key] = new_weights(
                    shape=[num_features, fc_size], trainable=False, name='Weight'+key)
            self.classifier['Bias' + key] = new_bias(
                    length= fc_size, trainable=False, name='Bias'+key)
            self.Deep['Conv_Layer_'+str(i)] = self.new_fc_layer(self.Deep['Conv_Layer_'+str(i-1)], \
                key=key,use_relu=True, dropout=True, keep_prob=0.5)
            self.Weights.append((self.classifier['Weight'+key], self.classifier['Bias'+key]))

            
            i = i+1
            key = 'Conv_Layer_'+ str(i)
            self.classifier['Weight' + key] = new_weights(
                    shape=[fc_size, classes], trainable=False, name='Weight'+key)
            self.classifier['Bias' + key] = new_bias(
                    length= classes, trainable=False, name='Bias'+key)
            self.Deep['Conv_Layer_'+str(i)] = self.new_fc_layer(self.Deep['Conv_Layer_'+str(i-1)], \
                key=key, use_relu=False, dropout=False)
            self.Weights.append((self.classifier['Weight'+key], self.classifier['Bias'+key]))
            
                
        with tf.name_scope("Classifier"):
            self.classifier['class'] = self.Deep['Conv_Layer_'+str(i)]


        # Finally setup all the Z's Due to the new formulation
        # The first layer
        i = 1 
        input_data = self.Deep['Conv_Layer_0']
        key = 'Conv_Layer_' + str(i)
        self.classifier['Zs' + key] =  self.new_conv_layer(input = input_data, \
            key=key, pool_shape=pool_shape, use_pooling=True)
        self.classifier['Zs' + key]= tf.reshape(self.classifier['Zs' + key], \
            [batch_size,img_size[0]/pow(2,i),img_size[1]/pow(2,i),num_of_filters[1]])

        
        for i in range(2, num_conv_layers+1):
            key = 'Conv_Layer_' + str(i)            
            shape = [batch_size, img_size[0]/pow(2,i), img_size[0]/pow(2,i), num_of_filters[i]]
            self.classifier['Zs' + key] = new_weights(shape = shape, trainable=False, name='Z'+str(i))          
            self.Zs.append(self.classifier['Zs' + key])
            key_prev = 'Conv_Layer_' + str(i-1)
            z_prev = self.new_conv_layer(input = self.classifier['Zs'+key_prev], \
                key=key, pool_shape=pool_shape, use_pooling=True)
            self.Ind_Cost[key] =  tf.losses.mean_squared_error(self.classifier['Zs' +key], z_prev) 
            self.Cost.append(self.Ind_Cost[key])

        
        z_ = self.flatten_layer(self.classifier['Zs'+key])[0]
            
        
        i = i + 1
        key = 'Conv_Layer_'+ str(i)
        key_prev = 'Conv_Layer_' + str(i-1)
        self.classifier['Zs' + key] = new_weights(shape=[batch_size, fc_size], trainable=False, name='Z'+str(i))
        self.Zs.append(self.classifier['Zs' + key])
        z_prev = self.new_fc_layer(z_, key=key,use_relu=True, dropout=False, keep_prob=0)
        self.Ind_Cost[key] = tf.losses.mean_squared_error(self.classifier['Zs' +key], z_prev) 
        self.Cost.append(self.Ind_Cost[key])


        i = i + 1
        key = 'Conv_Layer_'+ str(i)
        key_prev = 'Conv_Layer_' + str(i-1)
        self.classifier['Zsclass'] = new_weights(shape=[batch_size, classes], trainable=False, name='Z'+str(i))
        self.Zs.append(self.classifier['Zsclass'])
        z_prev = self.new_fc_layer(self.classifier['Zs'+key_prev], key=key, use_relu=False, dropout=False)
        self.Ind_Cost[key] = tf.losses.mean_squared_error(self.classifier['Zsclass'], z_prev)
        self.Cost.append(self.Ind_Cost[key])
        
        
    def init_NN_custom(self, classes, num_conv_layers, num_fc_layers, img_size_flat, img_size, num_channels, \
        filter_size, num_of_filters, pool_shape, fc_size, batch_size, activation='relu', optimizer="Adam"):
            with tf.name_scope("PlaceHolders"):  
                self.classifier['Target'] = tf.placeholder(
                    tf.float32, shape=[None, classes])
                self.classifier["learning_rate"] = tf.placeholder(
                    tf.float32, [], name='learning_rate')
                self.classifier["rho"] = tf.placeholder(
                    tf.float32, [], name='rho')
                self.classifier["l2_norm"] = tf.placeholder(
                    tf.float32, [], name='l2_norm')
                self.Deep['Conv_Layer_00'] = tf.placeholder(
                    tf.float32, shape=[None, img_size_flat])

                # The main network
                self.Def_Network(classes, num_conv_layers, num_fc_layers, img_size, num_channels, \
                filter_size, num_of_filters, pool_shape, fc_size, batch_size, activation)

                # with tf.name_scope("Trainer"):
                    # The overall cost function
                    # self.classifier["Overall cost"] = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                    #     logits=self.classifier['class'], labels=self.classifier['Target'], name='Error_Cost'))
                    # self.Trainer["Weight_op"] = self.Custom_Optimizer(self.classifier["learning_rate"], optimizer)     

                # The final optimization
                with tf.name_scope("Trainers"): 
                    # Call the other optimizer
                    self.classifier["Cost_M1_Dist"]=tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                        logits=self.classifier['Zsclass'], labels=self.classifier['Target'], name='Error_Cost'))
                    self.Cost.append(self.classifier["Cost_M1_Dist"]) 

                    self.classifier["Total_Z_cost"] = self.classifier["rho"]*tf.add_n(self.Cost) 
                    # The final cost function
                    self.classifier["Overall_cost_dist"]   = (self.classifier["Cost_M1_Dist"] + self.classifier["Total_Z_cost"])
                    # Distributed Op
                    self.Trainer["Weight_op_dist"], self.Trainer["Zis_op_dist"] = self.Dist_optimizer(self.classifier["learning_rate"], optimizer) 
                    
                # The evaluation section of the methodology
                with tf.name_scope('Evaluation'):
                    self.Evaluation['correct_prediction'] = \
                            tf.equal(tf.argmax(tf.nn.softmax(self.classifier['class']),1 ) , 
                                    tf.argmax(self.classifier['Target'], 1))
                    self.Evaluation['accuracy'] = tf.reduce_mean(
                            tf.cast(self.Evaluation['correct_prediction'], tf.float32))

            self.sess.run(tf.global_variables_initializer())
            return self