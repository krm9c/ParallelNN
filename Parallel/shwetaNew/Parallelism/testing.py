import tensorflow as tf
import os, os.path
import time, random, cPickle
import numpy as np
from load_data import load_data

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

#dir_path = '/usr/local/home/krm9c/shwetaNew/Parallelism/Weights/'

dir_path = '/media/krm9c/My Book/ShwetaExt/Weights/'

#dir_path = '/usr/local/home/krm9c/Desktop/shweta/Parallelism/Weights/'


def read_file(file_path):
    while not os.path.exists(file_path):
        #print ("Waiting for file: ", file_path)
        time.sleep(1)
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
            print "Reading file ERROR !!!!!!: ", file_path
            read_file(file_path)

def test(input_data, target_data, nr_processes, num):
    for process_id in range(1,nr_processes):
        read_weight  = dir_path+'Layer_'+str(process_id)+'/w_'+str(process_id)+'_'+str(num)
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
    
    return sess.run(acc)*100, cost


def main():
    dataset_path = '/usr/local/home/krm9c/shwetaNew/data/'
    #dataset_path = '/usr/local/home/krm9c/Desktop/shweta/Parallelism/'

    X_train, y_train, X_test, y_test = load_data('mnist', dataset_path)
    ''' X_train = X_train[:100]
    y_train = y_train[:100]
    X_test = X_test[:100]
    y_test = y_test[:100] '''

    nr_processes = 4
    steps = 100
    display_step = 20

    for i in range(1, steps+1):
        if i%display_step == 0:
            train_acc, cost = test(X_train, y_train, nr_processes, i)
            print "Step:", i, "Training Accuracy:", train_acc, "Cost:", cost
    
    test_acc, cost = test(X_test, y_test, nr_processes, i)
    print "Testing Accuracy:", test_acc


        
main()