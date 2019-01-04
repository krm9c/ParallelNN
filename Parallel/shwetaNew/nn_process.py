from keras.layers import Dense
from keras.models import Sequential
import multiprocessing
from multiprocessing import Process, Queue
import tensorflow as tf

from train_val_set import TrainValSet


class NNProcess(Process):
    def __init__(self, process_id= int, nr_nets= int, ret_queue= Queue):
        super(NNProcess, self).__init__()
        self.process_id = process_id
        self.neural_nets = []
        self.train_val_set = None
        self.nr_nets = nr_nets
        self.ret_queue = ret_queue

    def set_train_val(self, train_val_set= TrainValSet):
        self.train_val_set = train_val_set

    def get_session_config(self):
        num_cores = 1
        num_CPU = 1
        num_GPU = 0

        config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                                inter_op_parallelism_threads=num_cores, allow_soft_placement=False,
                                device_count={'CPU': num_CPU, 'GPU': num_GPU})

        return config

    def run(self):
        print("process " + str(self.process_id) + " starting...")

        with tf.Session(graph=tf.Graph(), config=self.get_session_config()) as session:
            self.init_nets()
            self.compile()
            self.fit_nets(self.train_val_set)
            for i in range(0, self.nr_nets):
                file_name = self.neural_nets[i].name + "_" + str(i) + ".pickle"
                self.neural_nets[i].save(file_name)
                self.ret_queue.put(file_name)
        print("process " + str(self.process_id) + " finished.")

    def compile(self):
        for neural_net in self.neural_nets:
            neural_net.compile(loss='categorical_crossentropy',
                          optimizer='sgd',
                          metrics=['accuracy'])

    def init_nets(self):
        for i in range(0, self.nr_nets):
            model = Sequential()
            model.add(Dense(units=64, activation='relu', input_dim=100))
            model.add(Dense(units=10, activation='softmax'))
            self.neural_nets.append(model)

    def fit_nets(self, train_val_set= TrainValSet):
        for i in range(0, self.nr_nets):
            self.neural_nets[i].fit(steps_per_epoch=1)