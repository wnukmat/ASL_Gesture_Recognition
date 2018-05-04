################################################################################
# Title: top level file for ASL net 
#
# Author: Mansur Amin
# 
# Description: Top leve file for ASL NN. 
# 
################################################################################

# libraries
import numpy as np
import os 

path = 'C:\\Users\\maamin\\Desktop\\ECE_285_ML_DL\\ASL_Gesture_Recognition\\data\\sign_mnist_train.csv'

num_lines = sum(1 for line in open(path))


myfile = open(path,'r')
j = 0
read_lines = np.zeros((num_lines,785))
for line in myfile :
    if(j == 1) :
        read_lines[j,:] = line.split(',')
        j = j+1
myfile.close()

