from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from load_features import *

# Training Parameters
learning_rate = 0.001
training_steps = 1000
display_step = 200

# Network Parameters
num_input = 25088 # data input (img shape: 28*28)
timesteps = 75 # timesteps
num_hidden = 128 # hidden layer num of features
num_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, timesteps, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}


def RNN(x, weights, biases):
    x = tf.unstack(x, timesteps, 1)								# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
    lstm_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)			# Define a lstm cell with tensorflow
    outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)		# Get lstm cell output

    return tf.matmul(outputs[-1], weights['out']) + biases['out']		# Linear activation, using rnn inner loop last output

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

# Define loss and optimizer
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


training_set = training_set()
#training_set.load_set()
#training_set.save_set()
training_set.reload_set()
        
        


# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for step in range(1, training_steps+1):
        batch_x, batch_y = training_set.next_batch()
        

        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

        
        
