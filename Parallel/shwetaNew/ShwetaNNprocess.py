import multiprocessing
from multiprocessing import Process, Queue, Manager
import tensorflow as tf
import fnn_class2 as NN_class
import numpy as np
import random


class NNProcess(Process):
    def __init__(self, process_id, depth, classes, lr, optimizer, batch_size, steps, d, ret_queue= Queue):
        super(NNProcess, self).__init__()
        self.process_id = process_id
        self.neural_nets = []
        self.ret_queue = ret_queue
        self.model = None
        self.depth = depth
        self.classes = classes
        self.learning_rate = lr
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.steps = steps
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

    def run(self):
        print("Process " + str(self.process_id) + " starting...")

        with tf.Session(graph=tf.Graph(), config=self.get_session_config()) as session:
            self.init_nets()
            self.train()           
        print("Process " + str(self.process_id) + " finished.")
        print ("Process " + str(self.process_id) + " Train Acc: " + str(self.acc_train[0]*100) + " Test Acc: " + str(self.acc_test[0]*100) )
        self.d[self.process_id] = (self.acc_train[0]*100, self.acc_test[0]*100)

    def init_nets(self):
        self.model = NN_class.learners(config=self.get_session_config())
        self.model = self.model.init_NN_custom(self.classes, self.learning_rate, self.depth,\
            optimizer=self.optimizer) 
        self.neural_nets.append(self.model)

    def train(self):
        for j in range(self.steps):
            for k in xrange(100):
                x_batch = []
                y_batch = []
                arr = random.sample(range(0, len(self.X_train)), self.batch_size)
                for idx in arr:
                    x_batch.append(self.X_train[idx])
                    y_batch.append(self.y_train[idx])
                x_batch = np.asarray(x_batch)
                y_batch = np.asarray(y_batch)
                out = self.model.sess.run([self.model.Trainer["Weight_op"]],\
                feed_dict={self.model.Deep['FL_layer_10']: x_batch, self.model.classifier['Target']: y_batch,\
                self.model.classifier['learning_rate']: self.learning_rate})
        
            print "Step" + str(j), out
               
        self.acc_train = self.model.sess.run([ self.model.Evaluation['accuracy']],\
            feed_dict={self.model.Deep['FL_layer_10']: self.X_train, self.model.classifier['Target']:\
            self.y_train, self.model.classifier["learning_rate"]:self.learning_rate})
        
        self.acc_test = self.model.sess.run([self.model.Evaluation['accuracy']], \
            feed_dict={self.model.Deep['FL_layer_10']: self.X_test, self.model.classifier['Target']:\
            self.y_test, self.model.classifier["learning_rate"]:self.learning_rate})
        
        


    