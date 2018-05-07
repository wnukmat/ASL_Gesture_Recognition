################################################################################
# Title: top level file for ASL net
#
# Author: Mansur Amin
#
# Description: Top leve file for ASL NN.
#
# Current Status: Calls all functions and imports all files needed to run
# training net.
################################################################################

# libraries
import numpy as np
import matplotlib.pyplot as plt
import ASL_Class as asl

#####################################################################
# Main Program
#####################################################################

####### Read training data

# Windows Path
# path = 'C:\\Users\\maamin\\Desktop\\ECE_285_ML_DL\\data\\sign_mnist_train.csv'

# Mac path
# path = '/Users/mansuramin/Desktop/ECE_285_ML/data/sign_mnist_train.csv'

# Linux Server Path
path = '/datasets/home/56/256/maamin/data/sign_mnist_train.csv'


trn_dat_vec,trn_lbl = asl.read_data(path)
indx = 10240
asl.vis_data(trn_dat_vec[indx,:],trn_lbl[indx])
