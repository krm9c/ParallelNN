import sys
sys.dont_write_bytecode = True

# Testing of Generalized Feed Forward Neural Network

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import tensorflow as tf
import numpy as np
import gzip, cPickle, random, time
import fnn_class as NN_class
from fnn_load_data import load_data
import csv


_start_time = time.time()

def time_start():
    global _start_time 
    _start_time = time.time()

def time_end():
    t_sec = round(time.time() - _start_time)
    (t_min, t_sec) = divmod(t_sec,60)
    (t_hour,t_min) = divmod(t_min,60)
    print "\n" 
    print('Time Taken: {} hour : {} min : {} sec'.format(t_hour,t_min,t_sec))
    print "\n"



def perturbData(X, m, n):
    return (X+np.random.normal(1, 1, size=[m, n]))
    #return (X+(10*np.random.uniform(-4,4,size=[m,n])))

def get_session_config():
    num_cores = 12
    num_CPU = 1
    num_GPU = 0

    config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
                            inter_op_parallelism_threads=num_cores, allow_soft_placement=False,
                            device_count={'CPU': num_CPU, 'GPU': num_GPU})
    return config

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    assert inputs.shape[0] == targets.shape[0]
    if shuffle:
        indices = np.arange(inputs.shape[0])
        np.random.shuffle(indices)
    for start_idx in range(0, inputs.shape[0] - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]




dataset_path = '/usr/local/home/krm9c/shwetaNew/data/'
datasetName = 'rolling'
X_train, y_train, X_test, y_test = load_data(datasetName, dataset_path)



''' x_ = []
y_ = []
arr = random.sample(range(0, len(X_train)), 1000)
X_train = X_train[arr, :]
y_train = y_train[arr,:]

arr = random.sample(range(0, len(X_test)), 1000)
X_test = X_test[arr, :]
y_test = y_test[arr,:]
print(X_train.shape, y_train.shape) '''



def Z_Training():
    Testing_Accuracy = []
    Training_Accuracy = []
    for s in range(12,13):
        depth = []
        depth.append(X_train.shape[1])
        L = [50 for i in xrange(1)]
        depth.extend(L)
        classes = y_train.shape[1]
        batch_size = 256
        activation = 'tanh'
        optimizer = 'Adam'

        # Define model
        model = NN_class.learners(config=get_session_config())
        model = model.init_NN_custom(classes, depth, batch_size, activation=activation, optimizer=optimizer)

        steps = 30

        acc_test = np.zeros((steps, 1))
        acc_train = np.zeros((steps, 1))
        cost_M1 = np.zeros((steps, 1))
        cost_Z = np.zeros((steps, 1))
        cost_Total = np.zeros((steps, 1))
        weights_update = np.zeros((steps, len(depth)))
        grads_update = np.zeros((steps, len(depth)))

        z_updates = 100
        rho = 10
        lr = 0.001
        lr_z = 0.8
        l2_norm = 0.00001

        paras_dict = {
        'Dataset': datasetName,
        'Hidden_Layers': L,
        'Optimizer': optimizer,
        'Activation': activation,
        'lr_w': lr,
        'lr_z': lr_z,
        'Z updates': z_updates,
        'rho': rho,
        'l2_norm': l2_norm,
        'Training_steps': steps,
        'Batch_size': batch_size
        }

        time_start()
        # Training
        result_file_name = 'exp6_z_'+datasetName+'_'+str(s)
        print result_file_name
        print paras_dict
        with open('fnn_Results/Exp6/'+result_file_name+'.csv', 'wb') as csvfile:
            writer = csv.writer(csvfile, delimiter=',')
            writer.writerow(['Dataset', datasetName])
            writer.writerow(['Optimizer', optimizer])
            writer.writerow(['Activation', activation])
            writer.writerow(['lr_w', lr])
            writer.writerow(['lr_z', lr_z])
            writer.writerow(['Z updates', z_updates])
            writer.writerow(['rho', rho])
            writer.writerow(['l2_norm', l2_norm])
            writer.writerow(['Training_steps', steps])
            writer.writerow(['Batch_size', batch_size])
            writer.writerow(['Hidden_Layers', L])
            writer.writerow(['Cost Z', 'Cost T', 'Cost Total', 'Train Accuracy', 'Test Accuracy', 'W_Update'])
            for j in range(steps):
                lr   = 0.95*lr
                rho = 0.99*rho
                grads = []
                weights = []
                #c = 0
                for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                    x_batch, y_batch = batch
                    lr_dat = lr_z
                    for m in xrange(z_updates):
                        lr_dat = 0.99*lr_dat
                        _ = model.sess.run([model.Trainer["Zis_op_dist"]],\
                            feed_dict={model.Deep['FL_layer_10']: x_batch, model.classifier['Target']: \
                            y_batch, model.classifier["learning_rate"]:lr_dat, model.classifier["rho"]:rho, \
                            model.classifier["l2_norm"]: l2_norm})

                    _, w_up = model.sess.run([model.Trainer["Weight_op_dist"], model.classifier['w_update']],\
                    feed_dict={model.Deep['FL_layer_10']: x_batch, model.classifier['Target']: \
                    y_batch, model.classifier["learning_rate"]:lr, model.classifier["rho"]:rho, \
                    model.classifier["l2_norm"]: l2_norm})
                    #print c, w_up
                    #c = c+1

                for m in w_up:
                    grads.append(m[0])
                    weights.append(m[1])
                
                if j%1 == 0:        
                    print "Step", j

                    weights_update[j] = weights
                    grads_update[j] = grads
                    
                    #X_test_perturbed = perturbData(X_test, X_test.shape[0], X_test.shape[1])

                    acc_test[j] = model.sess.run([model.Evaluation['accuracy']], \
                    feed_dict={model.Deep['FL_layer_10']: X_test, model.classifier['Target']:\
                    y_test, model.classifier["learning_rate"]:lr})
                    
                    acc_train[j] = model.sess.run([ model.Evaluation['accuracy']],\
                    feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']:\
                    y_train, model.classifier["learning_rate"]:lr})

                    cost_Total[j] = model.sess.run([model.classifier["Overall_cost_dist"]],\
                    feed_dict={model.Deep['FL_layer_10']: x_batch, model.classifier['Target']:\
                    y_batch, model.classifier["learning_rate"]:lr, model.classifier["rho"]:rho, \
                    model.classifier["l2_norm"]: l2_norm})
                            
                    cost_M1[j]  = model.sess.run([ model.classifier["Cost_M1_Dist"] ],\
                    feed_dict={model.Deep['FL_layer_10']: x_batch, model.classifier['Target']:\
                    y_batch, model.classifier["learning_rate"]:lr, model.classifier["rho"]:rho, \
                    model.classifier["l2_norm"]: l2_norm})   
                    
                    cost_Z[j]  = model.sess.run([ model.classifier["Total_Z_cost"] ],\
                    feed_dict={model.Deep['FL_layer_10']: x_batch, model.classifier['Target']:\
                    y_batch, model.classifier["learning_rate"]:lr, model.classifier["rho"]:rho, \
                    model.classifier["l2_norm"]: l2_norm})

                    # Print all the outputs
                    print(result_file_name)
                    print("Total Cost:", cost_Total[j][0], "Z Cost:", cost_Z[j][0], "T Cost:", cost_M1[j][0])
                    print("W_updates:", grads)
                    print("Train Acc:", acc_train[j][0]*100, "Test Acc:", acc_test[j][0]*100)

                    data = [cost_Z[j][0], cost_M1[j][0], cost_Total[j][0], acc_train[j][0]*100, acc_test[j][0]*100, grads]
                    writer.writerow(data)

                    if max(acc_train) > 0.99:
                        break
        
        np.savetxt('fnn_Results/Exp6/'+result_file_name+'_weights', weights_update, delimiter=',')
        np.savetxt('fnn_Results/Exp6/'+result_file_name+'_grads', grads_update, delimiter=',')
        #Testing_Accuracy.append(acc_test[j][0]*100)
        #Training_Accuracy.append(acc_train[j][0]*100)
        time_end()

    
    ''' print (datasetName, iter)
    print "\n"
    print "Training:", Training_Accuracy
    print "\n"
    print "Testing:", Testing_Accuracy
    print "\n" '''

    ''' for idx in Training_Accuracy:
        #print idx
        sum=0
        for b in idx:
            sum=sum+b
        avg = sum/len(idx)
        final_acc.append(avg)

    print final_acc '''

    # print datasetName
    # print "\n"
    # print "Training:", Training_Accuracy
    # print "\n"
    # print "Testing:", Testing_Accuracy
    
    # sum=0
    # for t in Testing_Accuracy:
    #     sum = sum+t
    # test_avg = sum/len(Testing_Accuracy)
    # std_test = np.std(Testing_Accuracy)

    # sum2=0
    # for e in Training_Accuracy:
    #     sum2=sum2+e
    # train_avg = sum2/len(Training_Accuracy)
    # std_train = np.std(Training_Accuracy)

    # gen_error = train_avg-test_avg

    # print paras_dict
    # print "\n"
    # print ("Dataset:", datasetName)
    # print ("Avg Training Accuracy:", train_avg, "Std Deviation:", std_train)
    # print ("Avg Testing Accuracy:", test_avg, "Std Deviation:", std_test)
    # print ("Generalization Error:", gen_error)

    


def Non_Z_Training():
    Final_Test = {}
    Final_Train = {}
    for iter in range(1,11):
        Testing_Accuracy = []
        Training_Accuracy = []
        for s in range(1,11):
            depth = []
            depth.append(X_train.shape[1])
            L = [50 for i in xrange(s)]
            depth.extend(L)
            classes = 10
            batch_size = 128
            activation = 'tanh'
            optimizer = 'Adam'

            # Define model
            model = NN_class.learners(config=get_session_config())
            model = model.init_NN_custom(classes, depth, batch_size, activation=activation, optimizer=optimizer)

            steps = 30

            acc_test = np.zeros((steps, 1))
            acc_train = np.zeros((steps, 1))
            cost_M1 = np.zeros((steps, 1))
            cost_Z = np.zeros((steps, 1))
            cost_Total = np.zeros((steps, 1))
            weights_update = np.zeros((steps, len(depth)))
            grads_update = np.zeros((steps, len(depth)))

            lr = 0.0001
            l2_norm = 0.00001

            paras_dict = {
            'Dataset': datasetName,
            'Hidden_Layers': L,
            'Optimizer': optimizer,
            'Activation': activation,
            'lr_w': lr,
            'l2_norm': l2_norm,
            'Training_steps': steps,
            'Batch_size': batch_size
            }

            time_start()
            # Training
            result_file_name = 'exp7_'+str(iter)+'_non_z_'+activation+'_Layers_'+str(s)
            print result_file_name
            print paras_dict
            with open('fnn_Results/Exp7/'+result_file_name+'.csv', 'wb') as csvfile:
                writer = csv.writer(csvfile, delimiter=',')
                writer.writerow(['Dataset', datasetName])
                writer.writerow(['Optimizer', optimizer])
                writer.writerow(['Activation', activation])
                writer.writerow(['lr_w', lr])
                writer.writerow(['l2_norm', l2_norm])
                writer.writerow(['Training_steps', steps])
                writer.writerow(['Batch_size', batch_size])
                writer.writerow(['Hidden_Layers', L])
                writer.writerow(['Cost Total', 'Train Accuracy', 'Test Accuracy', 'W_Update'])
                for j in range(steps):
                    lr   = 0.95*lr
                    grads = []
                    weights = []
                    for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
                        x_batch, y_batch = batch
                        _, w_up = model.sess.run([model.Trainer["Grad_op"], model.classifier['non_z_w_update']],\
                        feed_dict={model.Deep['FL_layer_10']: x_batch, model.classifier['Target']: \
                        y_batch, model.classifier["learning_rate"]:lr, model.classifier["l2_norm"]: l2_norm})
                        #print c, w_up
                        
                    for m in w_up:
                        grads.append(m[0])
                        weights.append(m[1])
                    
                    if j%1 == 0:        
                        print "Step", j

                        weights_update[j] = weights
                        grads_update[j] = grads
                        
                        #X_test_perturbed = perturbData(X_test, X_test.shape[0], X_test.shape[1])

                        acc_test[j] = model.sess.run([model.Evaluation['non_z_accuracy']], \
                        feed_dict={model.Deep['FL_layer_10']: X_test, model.classifier['Target']:\
                        y_test, model.classifier["learning_rate"]:lr})
                        
                        acc_train[j] = model.sess.run([ model.Evaluation['non_z_accuracy']],\
                        feed_dict={model.Deep['FL_layer_10']: X_train, model.classifier['Target']:\
                        y_train, model.classifier["learning_rate"]:lr})

                        cost_Total[j] = model.sess.run([model.classifier["Overall_cost"]],\
                        feed_dict={model.Deep['FL_layer_10']: x_batch, model.classifier['Target']:\
                        y_batch, model.classifier["learning_rate"]:lr, model.classifier["l2_norm"]: l2_norm})
                                

                        # Print all the outputs
                        print("Total Cost:", cost_Total[j][0])
                        print("W_updates:", grads)
                        print("Train Acc:", acc_train[j][0]*100, "Test Acc:", acc_test[j][0]*100)

                        data = [cost_Total[j][0], acc_train[j][0]*100, acc_test[j][0]*100, grads]
                        writer.writerow(data)

                        if max(acc_train) > 0.99:
                            break

            np.savetxt('fnn_Results/Exp7/'+result_file_name+'_weights', weights_update, delimiter=',')
            np.savetxt('fnn_Results/Exp7/'+result_file_name+'_grads', grads_update, delimiter=',')
            Testing_Accuracy.append(acc_test[j][0]*100)
            Training_Accuracy.append(acc_train[j][0]*100)
            time_end()

        print "Training:", Training_Accuracy
        print "Testing:", Testing_Accuracy
        print "\n"

        Final_Test[iter] = Testing_Accuracy
        Final_Train[iter] = Training_Accuracy

    print (datasetName, "NON-Z")
    print "\n"

    print "Final Training: ", Final_Train
    print "\n"
    print "Final Testing: ", Final_Test
    print "\n"




#Non_Z_Training()

Z_Training()