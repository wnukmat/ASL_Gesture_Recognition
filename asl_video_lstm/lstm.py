from __future__ import print_function

import tensorflow as tf
from tensorflow.contrib import rnn
from load_features import *
import matplotlib.pyplot as plt

class lstm():

	def __init__(self):
		# Training Parameters
		self.learning_rate = 0.001
		self.training_epocs = 1000
		self.display_epoc = 1
		self.training_error = []
		self.validation_error = []
		self.load_data = True
		
		# Network Parameters
		self.num_input = 25088 # data input (7x7x512 feature vector extracted via VGG, flattened)
		self.timesteps = 115 # timesteps
		self.num_hidden = 25 # hidden layer num of features
		self.num_classes = 10 # Total classes 

		# tf Graph input
		self.X = tf.placeholder("float", [None, self.timesteps, self.num_input])
		self.Y = tf.placeholder("float", [None, self.num_classes])

		
	def _define(self):
		# Define weights
		self.weights = {
		    'out': tf.Variable(tf.random_normal([self.num_hidden, self.num_classes]))
		}
		self.biases = {
		    'out': tf.Variable(tf.random_normal([self.num_classes]))
		}


	def RNN(self):
	    self.x = tf.unstack(self.X, self.timesteps, 1)								# Unstack to get a list of 'timesteps' tensors of shape (batch_size, n_input)
	    lstm_cell = rnn.BasicLSTMCell(self.num_hidden, forget_bias=1.0, reuse=tf.get_variable_scope().reuse)			# Define a lstm cell with tensorflow
	    outputs, states = rnn.static_rnn(lstm_cell, self.x, dtype=tf.float32)		# Get lstm cell output

	    return tf.matmul(outputs[-1], self.weights['out']) + self.biases['out']		# Linear activation, using rnn inner loop last output

		
	def Setup(self):
		self._define()
		self.logits = self.RNN()
		self.prediction = tf.nn.softmax(self.logits)

		# Define loss and optimizer
		self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
		    logits=self.logits, labels=self.Y))
		optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
		self.train_op = optimizer.minimize(self.loss_op)

		# Evaluate model (with test logits, for dropout to be disabled)
		correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.Y, 1))
		self.accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

		# Initialize the variables (i.e. assign their default value)
		self.init = tf.global_variables_initializer()


		self.training_set = training_set()
		if self.load_data == True:
			self.training_set.load_set()
			self.training_set.save_set()
		else:
			self.training_set.reload_set()
		   
		   

	def train(self):
		# Start training
		with tf.Session() as sess:

		    # Run the initializer
		    sess.run(self.init)

		    for epoc in range(1, self.training_epocs+1):
			   batch_x, batch_y = self.training_set.next_batch()
			   

			   # Run optimization op (backprop)
			   sess.run(self.train_op, feed_dict={self.X: batch_x, self.Y: batch_y})
			   if epoc % self.display_epoc == 0 or epoc == 1:
				  # Calculate batch loss and accuracy
				  loss, acc = sess.run([self.loss_op, self.accuracy], feed_dict={self.X: batch_x,
				                                                       self.Y: batch_y})
				  print("Step " + str(epoc) + ", Minibatch Loss= " + \
				        "{:.4f}".format(loss) + ", Training Accuracy= " + \
				        "{:.3f}".format(acc))
				        
				  self.training_error.append(1-acc)
				        
				  validation_x, validation_y = self.training_set.validation_batch()
				  vacc = sess.run(self.accuracy, feed_dict={self.X: validation_x, self.Y: validation_y})
				  print("Testing Accuracy: ", vacc)
				      
				  self.validation_error.append(1-vacc)
				  
		    print("Optimization Finished!")


	def plot(self):
		plt.plot(self.training_error)
		plt.plot(self.validation_error)
		plt.legend(['training error', 'validation error'])
		plt.title('Training vs Validation Error')
		plt.show()



if __name__=='__main__':
	lstm = lstm()
	lstm.Setup()
	lstm.train()
	
