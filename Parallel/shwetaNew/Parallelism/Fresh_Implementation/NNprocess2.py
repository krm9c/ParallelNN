import tensorflow as tf
import multiprocessing
from multiprocessing import Process, Queue, Manager, Lock
import fnn_class4 as NN_class
import numpy as np
import random, cPickle, time, os, os.path
import fcntl


def read_file(file_path, pr_id):
    while not os.path.exists(file_path):
        #print ("Pr:", pr_id, "Waiting for file: ", file_path)
        time.sleep(1.5)
    if os.path.isfile(file_path):
        try:
            with open(file_path, "rb") as input_file:
                batch_data = cPickle.load(input_file)
                #print "Pr:", pr_id, "Reading file: ", file_path
                return batch_data
        except Exception as e:
            #print "Reading file ERROR !!!!!!: ", file_path
            read_file(file_path, pr_id)

def read_file_nextLayer(file_path, counter, pr_id):
    c = counter
    if counter == 0:
        batch_data = read_file(file_path+str(0), pr_id)
        return batch_data
    while counter>=0:
        new_file_path = file_path + str(counter)
        if os.path.isfile(new_file_path):
            try:
                with open(new_file_path, "rb") as input_file:
                    batch_data = cPickle.load(input_file)
                    #print "Pr:", pr_id, "Reading file forward: ", new_file_path
                    return batch_data
            except Exception as e:
                #print "Reading file forward ERROR !!!!!!: ", new_file_path
                read_file_nextLayer(file_path, c, pr_id)
        else:
            counter = counter-1
       
    
def write_file(write_dir_path, file_name, data, pr_id):
    if not os.path.exists(write_dir_path):
        os.makedirs(write_dir_path)
    file_path = write_dir_path+file_name
    try:
        with open(file_path, "wb") as output_file:
            #print "Pr:", pr_id, "Writing file: ", file_path
            fcntl.flock(output_file, fcntl.LOCK_EX)
            cPickle.dump(data, output_file)
            fcntl.flock(output_file, fcntl.LOCK_UN)
    except Exception as e:
        #print "Writing file ERROR !!!!!!: ", file_path
        write_file(write_dir_path, file_name, data, pr_id)


#dir_path = '/usr/local/home/krm9c/shwetaNew/Parallelism/Weights/'
dir_path = '/media/krm9c/My Book/ShwetaExt/Weights_1/'


# Main class for the process
class NNProcess(Process):
    def __init__(self, nr_processes, process_id, depth, classes, lr_z, lr_w, lambda_value, \
        optimizer, batch_size, steps, z_update_steps, w_update_steps, d):
        super(NNProcess, self).__init__()
        self.nr_processes = nr_processes
        self.process_id = process_id
        self.neural_nets = []
        self.model = None
        self.depth = depth
        self.classes = classes
        self.lr_z = lr_z
        self.lr_w = lr_w
        self.lambda_value = lambda_value
        self.batch_size = batch_size
        self.steps = steps
        self.z_update_steps = z_update_steps
        self.w_update_steps = w_update_steps
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.acc_test = None
        self.acc_train = None
        self.d = d

    def set_train_val(self, X_train, y_train, X_test, y_test):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def get_session_config(self):
        num_cores = 12
        num_CPU = 1
        num_GPU = 0

        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                inter_op_parallelism_threads=num_cores, allow_soft_placement=False,
                                device_count={'CPU': num_CPU, 'GPU': num_GPU})
        return config

    def sum_network_cost(self, step, cost, lock):
        lock.acquire()
        value = self.d[step]
        value = value + cost
        self.d[step] = value
        lock.release()

    def all_network_cost(self, step, cost, lock):
        lock.acquire()
        print "I am updating for process:", self.process_id, "Cost:", round(cost,4)
        cost = round(cost,4)
        self.d[step][self.process_id] = cost
        print self.d
        lock.release()

    def run(self):
        print("Process " + str(self.process_id) + " Starting...")
        with tf.Session(graph=tf.Graph(), config=self.get_session_config()) as sess:
            self.init_nets()
            print("Process " + str(self.process_id) + " Training Starting.....")
            self.train()           
            print("Process " + str(self.process_id) + " Training Finished.")
        print("Process " + str(self.process_id) + " Finished.")
        

    def init_nets(self):
        self.model = NN_class.learners(config=self.get_session_config())
        self.model = self.model.init_NN_custom(self.process_id, \
        self.nr_processes, self.classes, self.depth, self.batch_size)

        if self.process_id == 1:
            w_self_file_dir = dir_path+'Layer_'+str(self.process_id)+'/'
            Weights = []
            w_self_file_path  = 'w_'+str(self.process_id)+'_'+'0'
            weight = self.model.sess.run(self.model.classifier['w_self'])
            bias = self.model.sess.run(self.model.classifier['b_self'])
            Weights.append(weight)
            Weights.append(bias)
            write_file(w_self_file_dir, w_self_file_path, Weights, self.process_id)

            w_next_file_dir = dir_path+'Layer_'+str(self.process_id+1)+'/'
            Weights = []
            w_next_file_path  = 'w_'+str(self.process_id+1)+'_'+'0'
            weight = self.model.sess.run(self.model.classifier['w_next'])
            bias = self.model.sess.run(self.model.classifier['b_next'])
            Weights.append(weight)
            Weights.append(bias)
            write_file(w_next_file_dir, w_next_file_path, Weights, self.process_id)

            z_file_dir = dir_path+'Layer_'+str(self.process_id)+'/'
            z_file_path  = 'z_'+str(self.process_id)+'_'+'0'
            write_file(z_file_dir, z_file_path, self.model.sess.run(self.model.classifier['z_self']), self.process_id)

        elif  self.process_id == self.nr_processes:
            w_self_file_dir = dir_path+'Layer_'+str(self.process_id+1)+'/'
            Weights = []
            w_self_file_path  = 'w_'+str(self.process_id+1)+'_'+'0'
            weight = self.model.sess.run(self.model.classifier['w_self'])
            bias = self.model.sess.run(self.model.classifier['b_self'])
            Weights.append(weight)
            Weights.append(bias)
            write_file(w_self_file_dir, w_self_file_path, Weights, self.process_id)


    def train(self):
        lock = Lock()
        for j in range(1,self.steps+1):
            if j%5 == 0:
                self.lr_w = 0.99*self.lr_w
                self.lr_z = self.lr_z 
                self.lambda_value = self.lambda_value
            # First Layer
            if self.process_id == 1:
                read_z_path_prev  = dir_path+'Layer_'+str(self.process_id-1)+'/z_'+str(self.process_id-1)+'_'+str(j)
                z_prev = read_file(read_z_path_prev, self.process_id)
                if 'NoneType' in str(type(z_prev)):
                    while 'NoneType' in str(type(z_prev)):
                        z_prev = read_file(read_z_path_prev, self.process_id)

                # # W3
                read_w_path_self  = dir_path+'Layer_'+str(self.process_id+2)+'/w_'+str(self.process_id+2)+'_'+str(j-1)
                weight = read_file(read_w_path_self, self.process_id)
                if 'NoneType' in str(type(weight)):
                    while 'NoneType' in str(type(weight)):
                        weight = read_file(read_w_path_self, self.process_id)
                w_lay_n = weight[0]
                b_lay_n = weight[1]

                # T
                read_t_path_next  = dir_path+'Layer_'+str(self.nr_processes+1)+'/t_'+str(self.nr_processes+1)+'_'+str(j)
                t_next = read_file(read_t_path_next, self.nr_processes+1)
                if 'NoneType' in str(type(t_next)):
                    while 'NoneType' in str(type(t_next)):
                        t_next = read_file(read_t_path_next, self.nr_processes+1)

                self.lr_z = 0.1
                for m in range(1, self.z_update_steps+1):
                    self.lr_z = 0.99*self.lr_z 
                    _, z_cost = self.model.sess.run([self.model.Trainer['Z_op'] ,\
                    self.model.classifier['cost'+str(self.process_id)]],\
                    feed_dict={self.model.classifier['learning_rate']: self.lr_z, \
                    self.model.classifier['lambda_value']: self.lambda_value, \
                    self.model.classifier['z_prev']: z_prev,\
                    self.model.classifier['w_lay_next']:  w_lay_n,\
                    self.model.classifier['b_lay_next']:  b_lay_n,\
                    self.model.classifier['t_next']:      t_next})

                _,w_cost =self.model.sess.run([self.model.Trainer['Weight_op'],\
                self.model.classifier['cost'+str(self.process_id)]],\
                feed_dict={self.model.classifier['learning_rate']: self.lr_w, \
                self.model.classifier['lambda_value']: self.lambda_value, \
                self.model.classifier['z_prev']:  z_prev,\
                self.model.classifier['w_lay_next']: w_lay_n,\
                self.model.classifier['b_lay_next']:  b_lay_n,\
                self.model.classifier['t_next'] : t_next})    

                # self.lr_z = 0.1
                # for m in range(1, self.z_update_steps+1):
                #     self.lr_z = 0.99*self.lr_z 
                #     _, z_cost = self.model.sess.run([self.model.Trainer['Z_op'] ,\
                #     self.model.classifier['cost'+str(self.process_id)]],\
                #     feed_dict={self.model.classifier['learning_rate']: self.lr_z, \
                #     self.model.classifier['lambda_value']: self.lambda_value, \
                #     self.model.classifier['z_prev']: z_prev,\
                #     self.model.classifier['t_next']: t_next})

                # _,w_cost =self.model.sess.run([self.model.Trainer['Weight_op'],\
                # self.model.classifier['cost'+str(self.process_id)]],\
                # feed_dict={self.model.classifier['learning_rate']: self.lr_w, \
                # self.model.classifier['lambda_value']: self.lambda_value, \
                # self.model.classifier['z_prev']:  z_prev,\
                # self.model.classifier['t_next'] : t_next})    
                # self.sum_network_cost(j, w_cost, lock)

                ## The z1
                z_file_dir = dir_path+'Layer_'+str(self.process_id)+'/'
                z_file_path  = 'z_'+str(self.process_id)+'_'+str(j)
                write_file(z_file_dir, z_file_path, self.model.sess.run(self.model.classifier['z_self']), self.process_id)

                # The W1  
                w_self_file_dir = dir_path+'Layer_'+str(self.process_id)+'/'
                Weights = []
                w_self_file_path  = 'w_'+str(self.process_id)+'_'+str(j+1)
                weight = self.model.sess.run(self.model.classifier['w_self'])
                bias = self.model.sess.run(self.model.classifier['b_self'])
                Weights.append(weight)
                Weights.append(bias)
                write_file(w_self_file_dir, w_self_file_path, Weights, self.process_id)
            
                ### The W2
                w_next_file_dir = dir_path+'Layer_'+str(self.process_id+1)+'/'
                Weights = []
                w_next_file_path  = 'w_'+str(self.process_id+1)+'_'+str(j+1)
                weight = self.model.sess.run(self.model.classifier['w_next'])
                bias = self.model.sess.run(self.model.classifier['b_next'])
                Weights.append(weight)
                Weights.append(bias)
                write_file(w_next_file_dir, w_next_file_path, Weights, self.process_id)

                # # # The W3
                # w_self_file_dir = dir_path+'Layer_'+str(self.process_id+2)+'/'
                # Weights = []
                # w_self_file_path  = 'w_'+str(self.process_id+2)+'_'+str(j)
                # weight = self.model.sess.run(self.model.classifier['w_lay_next'])
                # bias = self.model.sess.run(self.model.classifier['b_lay_next'])
                # Weights.append(weight)
                # Weights.append(bias)
                # write_file(w_self_file_dir, w_self_file_path, Weights, self.process_id)


                if (j-1)%50 == 0 and not (j-1) == 0:
                    train_acc, cost_func = self.test(self.X_train, self.y_train, j-1)
                    print ("Step:", j-1, "Total Cost:", w_cost, "Acc:", train_acc, "Softmax:", cost_func) 

            # Top Layer
            elif self.process_id == self.nr_processes:
                read_t_path_next  = dir_path+'Layer_'+str(self.nr_processes+1)+'/t_'+str(self.nr_processes+1)+'_'+str(j)
                read_z_path_prev  = dir_path+'Layer_'+str(self.nr_processes-1)+'/z_'+str(self.nr_processes-1)+'_'+str(j)
                t_next = read_file(read_t_path_next, self.process_id)
                if 'NoneType' in str(type(t_next)):
                    while 'NoneType' in str(type(t_next)):
                        t_next = read_file(read_t_path_next, self.process_id)

                
                z_prev = read_file(read_z_path_prev, self.process_id-1)
                if 'NoneType' in str(type(z_prev)):
                    while 'NoneType' in str(type(z_prev)):
                        z_prev = read_file(read_z_path_prev, self.process_id)
                
                
                _, w_cost = self.model.sess.run([self.model.Trainer['Weight_op'], \
                self.model.classifier['cost'+str(self.process_id)]],\
                feed_dict={self.model.classifier['learning_rate']: self.lr_w, \
                self.model.classifier['lambda_value']: self.lambda_value, \
                self.model.classifier['t_next']: t_next, \
                self.model.classifier['z_prev']: z_prev})

                self.sum_network_cost(j, w_cost, lock)

                # The W3
                w_self_file_dir = dir_path+'Layer_'+str(self.process_id+1)+'/'
                Weights = []
                w_self_file_path  = 'w_'+str(self.process_id+1)+'_'+str(j)
                weight = self.model.sess.run(self.model.classifier['w_self'])
                bias = self.model.sess.run(self.model.classifier['b_self'])
                Weights.append(weight)
                Weights.append(bias)
                write_file(w_self_file_dir, w_self_file_path, Weights, self.process_id)

                
                # if (j-1)%5 == 0 and not (j-1) == 0:
                #     train_acc, cost_func = self.test(self.X_train, self.y_train, j-1)
                #     print ("Step:", j-1,  "Acc:", train_acc, "Softmax:", cost_func)         
                

            # # First Layer
            # elif self.process_id == 1:
            #     #read_z_path_next  = dir_path+'Layer_'+str(self.process_id+1)+'/z_'+str(self.process_id+1)+'_'
            #     #z_next = read_file_nextLayer(read_z_path_next, j-1, self.process_id)

                
            #     self.lr_z = 0.2
            #     for m in range(1, 10*self.z_update_steps+1):
            #         self.lr_z = 0.99*self.lr_z 
            #         update, z_cost, z_error = self.model.sess.run([self.model.Trainer['Z_op'],\
            #         self.model.classifier['Local_Cost_H'], self.model.classifier['z_error']],\
            #         feed_dict={self.model.classifier['learning_rate']: self.lr_z, \
            #         self.model.classifier['lambda_value']: self.lambda_value, \
            #         self.model.classifier['z_next']: z_next, \
            #         self.model.classifier['z_prev']: z_prev})
            #         print("Error at the end  of the first layer ", z_error)

            #     print("Error at the end  of the first layer ", z_error)
            #     update, w_cost = self.model.sess.run([self.model.Trainer['self_Weight_op'],\
            #     self.model.classifier['Local_Cost_H']],\
            #     feed_dict={self.model.classifier['learning_rate']: self.lr_w, \
            #     self.model.classifier['lambda_value']: self.lambda_value, \
            #     self.model.classifier['z_prev']: z_prev})    

            #     update, w_cost = self.model.sess.run([self.model.Trainer['next_Weight_op'],\
            #     self.model.classifier['Local_Cost_H']],\
            #     feed_dict={self.model.classifier['learning_rate']: self.lr_w, \
            #     self.model.classifier['lambda_value']: self.lambda_value, \
            #     self.model.classifier['z_prev']: z_prev})    

            #     self.sum_network_cost(j, w_cost, lock)

            #     # print ("Process:", self.process_id, "Step:", j, "W Cost:", w_cost, "Z Error:", z_error)
            #     #print "Process:", self.process_id, "Step:", j, "Dict:", self.d

            # # All other Layers
            # else:
            #     read_z_path_next  = dir_path+'Layer_'+str(self.process_id+1)+'/z_'+str(self.process_id+1)+'_'+str(j-1)
            #     z_next = read_file(read_z_path_next, self.process_id)
            #     if 'NoneType' in str(type(z_next)):
            #         while 'NoneType' in str(type(z_next)):
            #             z_next = read_file(read_z_path_next, self.process_id)

            #     read_w_path_self  = dir_path+'Layer_'+str(self.process_id)+'/w_'+str(self.process_id)+'_'+str(j)
            #     weight = read_file(read_w_path_self, self.process_id)
            #     if 'NoneType' in str(type(weight)):
            #         while 'NoneType' in str(type(weight)):
            #             weight = read_file(read_w_path_self, self.process_id)
            #     w_self = weight[0]
            #     b_self = weight[1]

            #     if 'NoneType' in str(type(w_self)) or 'NoneType' in str(type(b_self)):
            #         while 'NoneType' in str(type(w_self)) or 'NoneType' in str(type(b_self)):
            #             weight = read_file(read_w_path_self, self.process_id)
            #             w_self = weight[0]
            #             b_self = weight[1]
                
            #     self.lr_z = 0.1
            #     for m in range(1, self.z_update_steps+1):
            #         # self.lr_z = 0.99*self.lr_z 
            #         update, z_cost = self.model.sess.run([self.model.Trainer['Z_op'], \
            #         self.model.classifier['Local_Cost_H']],\
            #         feed_dict={self.model.classifier['learning_rate']: self.lr_z, \
            #         self.model.classifier['lambda_value']: self.lambda_value, \
            #         self.model.classifier['z_next']: z_next, \
            #         self.model.classifier['w_self']: w_self, \
            #         self.model.classifier['b_self']: b_self, \
            #         self.model.classifier['z_prev']: z_prev})
            #     print ("Process:", self.process_id, "Step:", j, m, "Z Cost:", z_cost)

            #     # for temp in range(1,self.w_update_steps+1):
            #     update, w_cost = self.model.sess.run([self.model.Trainer['next_Weight_op'], \
            #     self.model.classifier['Local_Cost_H']],\
            #     feed_dict={self.model.classifier['learning_rate']: self.lr_w, \
            #     self.model.classifier['lambda_value']: self.lambda_value, \
            #     self.model.classifier['z_next']: z_next, \
            #     self.model.classifier['w_self']: w_self, \
            #     self.model.classifier['b_self']: b_self, \
            #     self.model.classifier['z_prev']: z_prev})
            #     # print ("Process:", self.process_id, "Step:", j,temp, "W Cost:", w_cost)

            #     self.sum_network_cost(j, w_cost, lock)

            #     # print ("Process:", self.process_id, "Step:", j, "W Cost:", w_cost)

            #     #print "Process:", self.process_id, "Step:", j, "Dict:", self.d
            
            # z_file_dir = dir_path+'Layer_'+str(self.process_id)+'/'
            # z_file_path  = 'z_'+str(self.process_id)+'_'+str(j)
            # write_file(z_file_dir, z_file_path, self.model.sess.run(self.model.classifier['z_self']), self.process_id)

            # if self.process_id == 1:
            #     w_self_file_dir = dir_path+'Layer_'+str(self.process_id)+'/'
            #     Weights = []
            #     w_self_file_path  = 'w_'+str(self.process_id)+'_'+str(j)
            #     weight = self.model.sess.run(self.model.classifier['w_self'])
            #     bias = self.model.sess.run(self.model.classifier['b_self'])
            #     Weights.append(weight)
            #     Weights.append(bias)
            #     write_file(w_self_file_dir, w_self_file_path, Weights, self.process_id)
            
            
            # w_next_file_dir = dir_path+'Layer_'+str(self.process_id+1)+'/'
            # Weights = []
            # w_next_file_path  = 'w_'+str(self.process_id+1)+'_'+str(j)
            # weight = self.model.sess.run(self.model.classifier['w_next'])
            # bias = self.model.sess.run(self.model.classifier['b_next'])
            # Weights.append(weight)
            # Weights.append(bias)
            # write_file(w_next_file_dir, w_next_file_path, Weights, self.process_id)


    def test(self, input_data, target_data, num):
        #print "Testing !!"
        Layers_num = self.nr_processes+1
        #dir_path = '/media/krm9c/My Book/ShwetaExt/Weights_1/'
        input_data = tf.convert_to_tensor(input_data, dtype=tf.float32)
        target_data = tf.convert_to_tensor(target_data, dtype=tf.float32)
        layer_input  = []
        layer_input.append(input_data)
        weight_layer = []
        bias_layer   = []

        for p_id  in range(1,Layers_num):
            read_weight  = dir_path+'Layer_'+str(p_id)+'/w_'+str(p_id)+'_'+str(num)
            data = read_file(read_weight, p_id)
            if 'NoneType' in str(type(data)):
                while 'NoneType' in str(type(data)):
                    data = read_file(read_weight, p_id)
            weight = data[0]
            bias   = data[1]
            if 'NoneType' in str(type(weight)) or 'NoneType' in str(type(bias)):
                while 'NoneType' in str(type(weight)) or 'NoneType' in str(type(bias)):
                    data = read_file(read_weight, p_id)
                    weight = data[0]
                    bias = data[1]
            weight_layer.append(tf.convert_to_tensor(weight, dtype=tf.float32))
            bias_layer.append(tf.convert_to_tensor(bias, dtype=tf.float32))
            layer_input.append(tf.nn.tanh(tf.add(tf.matmul(layer_input[p_id-1], weight_layer[p_id-1]), bias_layer[p_id-1])))

        read_weight  = dir_path+'Layer_'+str(Layers_num)+'/w_'+str(Layers_num)+'_'+str(num)
        data = read_file(read_weight, Layers_num)
        if 'NoneType' in str(type(data)):
            while 'NoneType' in str(type(data)):
                data = read_file(read_weight, Layers_num)
        weight = data[0]
        bias = data[1]
        if 'NoneType' in str(type(weight)) or 'NoneType' in str(type(bias)):
            while 'NoneType' in str(type(weight)) or 'NoneType' in str(type(bias)):
                data = read_file(read_weight, Layers_num)
                weight = data[0]
                bias = data[1]
       
        weight = tf.convert_to_tensor(weight, dtype=tf.float32)
        bias = tf.convert_to_tensor(bias, dtype=tf.float32)
        output_data = tf.add(tf.matmul(layer_input[len(layer_input)-1], weight), bias)

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=tf.nn.softmax(output_data), labels=target_data, name='Error_Cost'))

        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(output_data),1 ), \
                tf.argmax(target_data, 1))

        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return self.model.sess.run(acc)*100, self.model.sess.run(cost)