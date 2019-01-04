# Five-Layer Neural Network

from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

sess = tf.InteractiveSession()

# creating nodes for the computation graph
# placeholders is a promise that a value will be delivered later
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

# variables
num_features = 784
hidden_nodes = 100
num_labels = 10
steps = 500
batch_size= 64
learning_rate = 0.001

# Defining the arrays 
Hidden_Weights = []
Hidden_Bias = []
hidden_layer =[]

# defining weights and biases
W = tf.Variable(np.random.rand(num_features, hidden_nodes).astype(np.float32))
b = tf.Variable(np.random.rand(hidden_nodes).astype(np.float32))
Wo = tf.Variable(np.random.rand(hidden_nodes, num_labels).astype(np.float32))
bo = tf.Variable(np.random.rand(num_labels).astype(np.float32))

Hidden_Weights.append(W)
Hidden_Bias.append(b)

for i in range(3):
    W2 = tf.Variable(np.random.rand(hidden_nodes, hidden_nodes).astype(np.float32))
    b2 = tf.Variable(np.random.rand(hidden_nodes).astype(np.float32))
    Hidden_Weights.append(W2)
    Hidden_Bias.append(b2)


# Five-Layer Network
def five_layer_network(data):
    hidden_layer.append(data)
    for i in range(4):  
        hidden_layer.append( tf.nn.relu(tf.add(tf.matmul(hidden_layer[i], Hidden_Weights[i]) , Hidden_Bias[i]) ) )
    output_layer = tf.add(tf.matmul(hidden_layer[len(hidden_layer)-1],Wo), bo)
    return output_layer

# Three-Layer Network
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.nn.relu( tf.add(tf.matmul(x, Hidden_Weights[0]),Hidden_Bias[0]))
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, Hidden_Weights[1]), Hidden_Bias[1]))
    # Output fully connected layer with a neuron for each class
    layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, Hidden_Weights[2]), Hidden_Bias[2]))
    # Output fully connected layer with a neuron for each class
    layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, Hidden_Weights[3]), Hidden_Bias[3]))
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_4, Wo) + bo
    return out_layer


y = five_layer_network(x)
# print y
# Model Scores
# y = neural_net(x)

# loss function
cross_entropy = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

# Evaluating the model
correct_prediction = tf.equal(tf.argmax(tf.nn.softmax(y),1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")



# initializing all the variables in the session
sess.run(tf.global_variables_initializer())

# Training
for j in range(steps):
    print "Step", j+1
    for k in xrange(100):
        batch = mnist.train.next_batch(batch_size)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})
    print("Loss", cross_entropy.eval(feed_dict={x: batch[0], y_: batch[1]}))
    print("Train Accuracy", accuracy.eval(feed_dict={x: mnist.train.images, y_: mnist.train.labels})*100)
    print("Test  Accuracy", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})*100)

    
print("Final Accuracy", accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels})*100)



