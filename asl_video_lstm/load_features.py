import cv2, os
import numpy as np
import tensorflow as tf
import os, shutil

class training_set():
	'''
	Loads training set from positive_examples, negative_examples folders
	creates class attributes for ease of use in batch training
	'''
	
	def __init__(self):
		self.class_id = {}
		self.class_num = 0
		self.winsize = 75
		self.batch_size = 132
		self.batch = 0
		self.samples = []
		self.labels = []
		
	def _augment(self, sample):
		examples = [] 
		for i in range(0, sample.shape[0] - self.winsize, 10):
			examples.append(sample[i:i+self.winsize,:])
			
		return np.array(examples)
		
	def load_set(self, training_dir = None):
		training_dir = '/home/matthew/Desktop/Gesture/data_features/'
		
		
		for subdir in os.walk(training_dir):
			print 'subdir: ', subdir[0][44:], 'class num: ', self.class_num
			
			self.class_id[subdir[0][44:]] = self.class_num
			self.class_num += 1
			
			for features in os.listdir(subdir[0]):	
				label = [0]*10		
				if features.endswith(".npy"):						
					example = np.load(subdir[0] + '/' + features)		#load feature set
					example = example.reshape(example.shape[0], -1)		#flatten
					augmented_samples = self._augment(example)				#augment with sliding window
					for j in range (augmented_samples.shape[0]):
						#print 'np.squeeze(augmented_samples[j,:,:]): ', np.squeeze(augmented_samples[j,:,:]).shape
						self.samples.append(list(np.squeeze(augmented_samples[j,:,:])))
						label[self.class_num-2] = 1
						self.labels.append(label)


		self.samples = np.array(self.samples)
		self.labels = np.array(self.labels)
		# Randomize Order for training
		shuffle = np.random.permutation(len(self.samples))
		self.samples = self.samples[shuffle,:,:]
		self.labels = self.labels[shuffle,:] 
		
		print 'samples: ', self.samples.shape
		print 'labels: ', self.labels.shape
		print 'Data Set Loaded'

	def save_set(self):
		np.save('samples', self.samples)
		np.save('labels', self.labels)
		print 'Saving Set Complete'
		
		
	def reload_set(self):
		self.samples = np.load('samples.npy')
		self.labels = np.load('labels.npy')
		print 'Reloading Set Complete'
				
	def next_batch(self):
		start = self.batch_size*self.batch
		end = start + self.batch_size

		if(len(self.samples) > end):
			sample_batch = self.samples[start:end]
			label_batch = self.labels[start:end]
			self.batch = self.batch + 1
		else:
			sample_batch = self.samples[start:-1]
			label_batch = self.labels[start:-1,:]
			self.batch = 0

		return sample_batch, label_batch

if __name__ == '__main__':
	testset = training_set()
	testset.load_set()








