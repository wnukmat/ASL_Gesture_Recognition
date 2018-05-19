################################################################################
# Title: ASL_Class.py
#
# Author: Mansur Amin
#
# Description: functions to be used in the ASL training and testing phase
#
# Current Status: Contains all functions needed for program to run
################################################################################

#####################################################################
# libraries
#####################################################################
import numpy as np
import matplotlib.pyplot as plt

#####################################################################
# def read_data(path) :
# Reads data specidfed by path and stores into numpy array, Splits data nad lbl
# path = location of where data is stored
#####################################################################
def read_data(path) :
   num_lines = sum(1 for line in open(path)) # get number of lines in file
   myfile = open(path,'r')
   j = 0
   read_lines = np.zeros((num_lines,785))    # expects a 25x25 pixel row vector
   for line in myfile :
      if(j >= 1) :
         read_lines[j,:] = np.asarray(line.split(','))
      j = j+1
   myfile.close()
   lables = read_lines[:,0]
   read_lines = np.delete(read_lines,0,1)
   return read_lines,lables.astype(np.int16) # .reshape(len(lables),1)

#####################################################################
# def vis_data(dat_vec,dat_lbl) :
# Visualize data.
# dat_vec = vector representing data
# dat_lbl = label representing data
#####################################################################
def vis_data(dat_vec,dat_lbl) :
   lbl_map = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']
   dat_mat = dat_vec.reshape(28,28)
   plt.title('Image Label = ' + str(dat_lbl) + ' = ' + lbl_map[dat_lbl])
   plt.imshow(dat_mat)
    
#####################################################################
# def normalize_MNIST_images(x) :
# Normalize data in range of -1 to 1
# x = vector representing data
#####################################################################
def normalize_MNIST_images(x,max_x, diff) : 
    x = x.astype(np.float64)
    x = max_x*(x-np.min(x))/(np.max(x)-np.min(x))-diff
    print("Normalize_MNIST_images", x.shape)
    return x
