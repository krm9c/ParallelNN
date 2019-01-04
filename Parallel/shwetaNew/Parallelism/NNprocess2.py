import tensorflow as tf
import multiprocessing
from multiprocessing import Process, Queue, Manager, Lock
import fnn_class4 as NN_class
import numpy as np
import random, cPickle, time, os, os.path
import fcntl


def read_file(file_path):
    while not os.path.exists(file_path):
        #print ("Waiting for file: ", file_path)
        time.sleep(2)
    if os.path.isfile(file_path):
        try:
            with open(file_path, "rb") as input_file:
                batch_data = cPickle.load(input_file)
                #return batch_data
                if 'NoneType' in str(type(batch_data)):
                    read_file(file_path)                
                else:
                    return batch_data
        except Exception as e:
            #print "Reading file ERROR !!!!!!: ", file_path
            read_file(file_path)

def read_file_nextLayer(file_path, counter):
    c = counter
    if counter == 0:
        batch_data = read_file(file_path+str(0))
        return batch_data
    while counter>=0:
        new_file_path = file_path + str(counter)
        if os.path.isfile(new_file_path):
            try:
                with open(new_file_path, "rb") as input_file:
                    batch_data = cPickle.load(input_file)
                    #return batch_data
                    if 'NoneType' in str(type(batch_data)):
                        read_file_nextLayer(file_path, c)               
                    else:
                        return batch_data
            except Exception as e:
                #print "Reading file forward ERROR !!!!!!: ", new_file_path
                read_file_nextLayer(file_path, c)
        else:
            counter = counter-1
       
    
def write_file(write_dir_path, file_name, data):
    if not os.path.exists(write_dir_path):
        os.makedirs(write_dir_path)
    file_path = write_dir_path+file_name
    try:
        with open(file_path, "wb") as output_file:
            fcntl.flock(output_file, fcntl.LOCK_EX)
            cPickle.dump(data, output_file)
            fcntl.flock(output_file, fcntl.LOCK_UN)
    except Exception as e:
        print "Writing file ERROR !!!!!!: ", file_path
        write_file(write_dir_path, file_name, data)

def xavier(fan_in, fan_out):
    # use 4 for sigmoid, 1 for tanh activation
    low = -1 * np.sqrt(1.0 / (fan_in + fan_out))
    high = 1 * np.sqrt(1.0 / (fan_in + fan_out))
    return tf.random_uniform([fan_in, fan_out], minval=low, maxval=high, dtype=tf.float32)

def z_variable(shape, trainable, name):
    initial = xavier(shape[0], shape[1])
    return tf.Variable(initial, trainable=trainable, name=name)

#dir_path = '/usr/local/home/krm9c/shwetaNew/Parallelism/Weights/'
dir_path = '/media/krm9c/My Book/ShwetaExt/Weights_1/'


# Main class for the process
class NNProcess(Process):
    def __init__(self, nr_processes, process_id, depth, classes, lr_z, lr_w, lambda_value, \
        optimizer, batch_size, steps, z_update_steps, d):
        super(NNProcess, self).__init__()
        self.nr_processes = nr_processes
        self.process_id = process_id
        self.neural_nets = []
        self.model = None
        self.depth = depth
        self.Layers = []
        self.classes = classes
        self.lr_z = lr_z
        self.lr_w = lr_w
        self.lambda_value = lambda_value
        self.batch_size = batch_size
        self.steps = steps
        self.z_update_steps = z_update_steps
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

    def run(self):
        print("Process " + str(self.process_id) + " Starting...")
        with tf.Session(graph=tf.Graph(), config=self.get_session_config()) as sess:
            self.init_nets()
            print("Process " + str(self.process_id) + " Training Starting.....")
            self.train()           
            print("Process " + str(self.process_id) + " Training Finished.")
        print("Process " + str(self.process_id) + " Finished.")
        


    def init_nets(self):
        self.Layers = [self.depth[self.process_id-1], self.depth[self.process_id]]
        self.model = NN_class.learners(config=self.get_session_config())
        self.model = self.model.init_NN_custom(self.process_id, self.nr_processes, self.classes, self.Layers,\
            self.depth, self.batch_size)
        para_file_dir = dir_path+'Layer_'+str(self.process_id)+'/'

        if not self.process_id == self.nr_processes:
            z_file_path  = 'z_'+str(self.process_id)+'_'+'0'
            write_file(para_file_dir, z_file_path, self.model.sess.run(self.model.classifier['z_self']))
        
        Weights = []
        w_file_path  = 'w_'+str(self.process_id)+'_'+'0'
        weight = self.model.sess.run(self.model.classifier['w_self'])
        bias = self.model.sess.run(self.model.classifier['b_self'])
        Weights.append(weight)
        Weights.append(bias)
        write_file(para_file_dir, w_file_path, Weights)



    def train(self):
        optimizer = 'Adam'
        lock = Lock()
        for j in range(1,self.steps+1):
            if j%10 == 0:
                self.lambda_value = 1.01*self.lambda_value
            if j%100 == 0:
                self.lr_w = 0.99*self.lr_w
                self.lr_z = 0.99*self.lr_z
            if j%1000 == 0:
                self.lambda_value = 1
            #print ("Process:", self.process_id, "Step:", j, "Lambda:", self.lambda_value)
            
            read_z_path_prev  = dir_path+'Layer_'+str(self.process_id-1)+'/z_'+str(self.process_id-1)+'_'+str(j)
            z_prev = read_file(read_z_path_prev)
            if 'NoneType' in str(type(z_prev)):
                while 'NoneType' in str(type(z_prev)):
                    z_prev = read_file(read_z_path_prev)


            ## Top Layer
            if self.process_id == self.nr_processes:
                read_t_path_next  = dir_path+'Layer_'+\
                str(self.process_id+1)+'/t_'+str(self.process_id+1)+'_'+str(j)
                t_next = read_file(read_t_path_next)
                if 'NoneType' in str(type(t_next)):
                    while 'NoneType' in str(type(t_next)):
                        t_next = read_file(read_t_path_next)

                for temp in range(1,5):
                    update, w_cost, w_update = self.model.sess.run([self.model.Trainer['Weight_op'],\
                    self.model.classifier['Cost'], self.model.classifier['w update']],\
                    feed_dict={self.model.classifier['learning_rate']: self.lr_w, \
                    self.model.classifier['lambda_value']: self.lambda_value, \
                    self.model.classifier['t_next']: t_next, \
                    self.model.classifier['z_prev']: z_prev})

                

                self.sum_network_cost(j, w_cost, lock)
                if (j-1)%1000 == 0 and not j-1 == 0:
                    print ("Process: top", self.process_id, "Step:", j, "Cost:", w_cost)
                    train_acc, cost_func = self.test(self.X_train, self.y_train, j-1)
                    print "j:", j-1, "Total:", self.d[j-1], "Acc:", train_acc, "Softmax:", cost_func
                elif (j-1)%10 == 0 and not j-1 == 0:
                    print "j:", j-1, "Total:", self.d[j-1]
            # Penultimate Layer
            elif self.process_id == self.nr_processes-1:
                read_t_path_next  = dir_path+'Layer_'+str(self.process_id+2)+'/t_'+str(self.process_id+2)+'_'+str(j)
                t_next = read_file(read_t_path_next)
                if 'NoneType' in str(type(t_next)):
                    while 'NoneType' in str(type(t_next)):
                        t_next = read_file(read_t_path_next)

                read_w_path_next  = dir_path+'Layer_'+str(self.process_id+1)+'/w_'+str(self.process_id+1)+'_'
                weight = read_file_nextLayer(read_w_path_next, j-1)
                if 'NoneType' in str(type(weight)):
                    while 'NoneType' in str(type(weight)):
                        weight = read_file_nextLayer(read_w_path_next, j-1)
                w_next = weight[0]
                b_next = weight[1]

                if 'NoneType' in str(type(w_next)) or 'NoneType' in str(type(b_next)):
                    while 'NoneType' in str(type(w_next)) or 'NoneType' in str(type(b_next)):
                        weight = read_file_nextLayer(read_w_path_next, j-1)
                        w_next = weight[0]
                        b_next = weight[1]

                for m in range(1, self.z_update_steps+1):
                    update, z_cost, z_error = self.model.sess.run([self.model.Trainer['Z_op'],\
                    self.model.classifier['Cost'], self.model.classifier['z error']],\
                    feed_dict={self.model.classifier['learning_rate']: self.lr_z, \
                    self.model.classifier['lambda_value']: self.lambda_value, \
                    self.model.classifier['t_next']: t_next, \
                    self.model.classifier['w_next']: w_next, \
                    self.model.classifier['b_next']: b_next, \
                    self.model.classifier['z_prev']: z_prev})
                    update, w_cost, w_update = self.model.sess.run([self.model.Trainer['Weight_op'], self.model.classifier['Cost'], self.model.classifier['w update']],\
                        feed_dict={self.model.classifier['learning_rate']: self.lr_w, \
                        self.model.classifier['lambda_value']: self.lambda_value, \
                        self.model.classifier['t_next']: t_next, \
                        self.model.classifier['w_next']: w_next, \
                        self.model.classifier['b_next']: b_next, \
                        self.model.classifier['z_prev']: z_prev})
                # print ("Process: penultimate ", self.process_id, "Step:", j, "Cost:", w_cost, "Weight:", w_update)
                self.sum_network_cost(j, w_cost, lock)

            # All the other Layers
            else:
                read_z_path_next  = dir_path+'Layer_'+str(self.process_id+1)+'/z_'+str(self.process_id+1)+'_'
                z_next = read_file_nextLayer(read_z_path_next, j-1)
                if 'NoneType' in str(type(z_next)):
                    while 'NoneType' in str(type(z_next)):
                        z_next = read_file_nextLayer(read_z_path_next, j-1)
                
                read_w_path_next  = dir_path+'Layer_'+str(self.process_id+1)+'/w_'+str(self.process_id+1)+'_'
                weight = read_file_nextLayer(read_w_path_next, j-1)
                if 'NoneType' in str(type(weight)):
                    while 'NoneType' in str(type(weight)):
                        weight = read_file_nextLayer(read_w_path_next, j-1)
                
                w_next = weight[0]
                b_next = weight[1]
                
                if 'NoneType' in str(type(w_next)) or 'NoneType' in str(type(b_next)):
                    while 'NoneType' in str(type(w_next)) or 'NoneType' in str(type(b_next)):
                        weight = read_file_nextLayer(read_w_path_next, j-1)
                        w_next = weight[0]
                        b_next = weight[1]

                for m in range(1, self.z_update_steps+1):
                    update, z_cost, z_error = self.model.sess.run([self.model.Trainer['Z_op'],\
                    self.model.classifier['Cost'], self.model.classifier['z error']],\
                        feed_dict={self.model.classifier['learning_rate']: self.lr_z, \
                        self.model.classifier['lambda_value']: self.lambda_value, \
                        self.model.classifier['w_next']: w_next, \
                        self.model.classifier['b_next']: b_next, \
                        self.model.classifier['z_next']: z_next, \
                        self.model.classifier['z_prev']: z_prev})
                    _, w_cost, w_update = self.model.sess.run([self.model.Trainer['Weight_op'], self.model.classifier['Cost'], self.model.classifier['w update']],\
                        feed_dict={self.model.classifier['learning_rate']: self.lr_w, \
                        self.model.classifier['lambda_value']: self.lambda_value, \
                        self.model.classifier['w_next']: w_next, \
                        self.model.classifier['b_next']: b_next, \
                        self.model.classifier['z_next']: z_next, \
                        self.model.classifier['z_prev']: z_prev})    
                    
                # print ("Process: other layers", self.process_id, "Step:", j, "Cost:", w_cost, "Weight:", w_update)

                self.sum_network_cost(j, w_cost, lock)
            
            para_file_dir = dir_path+'Layer_'+str(self.process_id)+'/'
            if not self.process_id == self.nr_processes:
                z_file_path  = 'z_'+str(self.process_id)+'_'+str(j)
                write_file(para_file_dir, z_file_path, self.model.sess.run(self.model.classifier['z_self']))
            
            Weights = []
            w_file_path  = 'w_'+str(self.process_id)+'_'+str(j)
            weight = self.model.sess.run(self.model.classifier['w_self'])
            bias = self.model.sess.run(self.model.classifier['b_self'])
            Weights.append(weight)
            Weights.append(bias)
            write_file(para_file_dir, w_file_path, Weights)

        
        para_file_dir = dir_path+'Layer_'+str(self.process_id)+'/'
        #z_file_path  = 'final_z_'+str(self.process_id)
        #write_file(para_file_dir, z_file_path, self.model.sess.run(self.model.classifier['z_self']), lock)

        Weights = []
        w_file_path  = 'final_w_'+str(self.process_id)
        weight = self.model.sess.run(self.model.classifier['w_self'])
        bias = self.model.sess.run(self.model.classifier['b_self'])
        Weights.append(weight)
        Weights.append(bias)
        write_file(para_file_dir, w_file_path, Weights)



    def test(self, input_data, target_data, num):
        nr_processes = self.nr_processes
        dir_path = '/media/krm9c/My Book/ShwetaExt/Weights_1/'
        for p_id in range(1,nr_processes):
            read_weight  = dir_path+'Layer_'+str(p_id)+'/w_'+str(p_id)+'_'+str(num)
            #read_weight  = dir_path+'Layer_'+str(process_id)+'/final_w_'+str(process_id)
            data = read_file(read_weight)
            weight = data[0]
            bias = data[1]
            weight = tf.convert_to_tensor(weight, dtype=tf.float32)
            bias = tf.convert_to_tensor(bias, dtype=tf.float32)
            #print weight
            input_data = tf.nn.tanh(tf.add(tf.matmul(input_data, weight), bias))
            #print input_data

        #read_weight  = dir_path+'Layer_'+str(nr_processes)+'/final_w_'+str(nr_processes)
        read_weight  = dir_path+'Layer_'+str(nr_processes)+'/w_'+str(nr_processes)+'_'+str(num)
        data = read_file(read_weight)
        weight = data[0]
        bias = data[1]
        weight = tf.convert_to_tensor(weight, dtype=tf.float32)
        bias = tf.convert_to_tensor(bias, dtype=tf.float32)
        output_data = tf.nn.softmax(tf.add(tf.matmul(input_data, weight), bias))

        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
                logits=output_data, labels=target_data, name='Error_Cost'))

        correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(output_data),1 ), \
                tf.argmax(target_data, 1))

        acc = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        return self.model.sess.run(acc)*100, self.model.sess.run(cost)