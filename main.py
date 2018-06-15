# -*- coding: utf-8 -*-
"""
ASL Dynamics
Main Script to run all 3 models in the ASL project

Contributors: 
Juan Camilo Castillo - castillojuancamilo@gmail.com
Mansur Amin          - maamin@eng.ucsd.edu
"""
# =============================================================================
#  Project Imports
# =============================================================================
import numpy as np
import cv2
import os
import pandas as pd
import time 
import matplotlib.pylab as plt
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import BatchNormalization
from keras.applications.vgg16 import preprocess_input
from keras.layers import GlobalAveragePooling2D, Dense, Dropout, Flatten, Input, Conv2D, multiply, LocallyConnected2D, Lambda
from keras.models import Model
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import DirectoryIterator
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from asl_img_cnn.cnn_asl_mnist import main as cnn_main

# import sys
# sys.path.append('/asl_img_cnn/cnn_asl_mnist')
# from cnn_asl_mnist import main as cnn_main


# =============================================================================
# FUNCTIONS
# =============================================================================

# =============================================================================
# TrimVideo: takes a video and creates a set of two video with removed annotations

# INPUTS:  videoSource - path to source of video
# newName1 - new video file name of the first output video
# newName2 - new video files name of the second output video
# destinationFolder - desired destination path
# =============================================================================
def TrimVideo(videoSource,newName1,newName2,destinationFolder):
    #get video source object
    cap = cv2.VideoCapture(videoSource)
    Annotationoffset = 20 #offset to get rid of the words at the top

    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    frameShape1 = (int(w),int(h/2) - Annotationoffset )
    frameShape2 = (int(w),int(h/2) - Annotationoffset - 1)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P','G')
    # codec used for WINDOWS
    out1Name = os.path.join(destinationFolder,newName1) 
    out2Name = os.path.join(destinationFolder,newName2) 
    #our 2 video writers
    out1 = cv2.VideoWriter(out1Name,fourcc, 30, frameShape1)
    out2 = cv2.VideoWriter(out2Name,fourcc, 30, frameShape2)

    if(cap.isOpened() and out1.isOpened() and out2.isOpened()):
        while(True):
            ret, frame = cap.read()
            if(not ret):
                #no more frames so lets exit the loop
                break
    
            #now lets trim our video frame
            FrameHalfway = int(frame.shape[0]/2)
            tophalfFrame = frame[Annotationoffset:FrameHalfway,:,:] 
            bottomhalfFrame = frame[FrameHalfway + Annotationoffset:-1,:,:]
    #        cv2.imshow('Top half',tophalfFrame)
    #        cv2.waitKey(25)
            #Write the results
            out1.write(tophalfFrame)
            out2.write(bottomhalfFrame)
        #release everything
        cap.release()
        out1.release()
        out2.release()
    else:
        print('Error reading/writing videos in TrimVideo')
        
# =============================================================================
# cleanData: takes videos in the data folder and creates a a new data folder 
# with trimmed videos
        
# INPUTS:  
# dataFolderbase_str - path of base folder where data folder is in
# dataFoldersource_str - name of data folder
# dataFoldercleaned_str - name of to-be created data cleaned folder
# =============================================================================      
def cleanData(dataFolderbase_str,dataFoldersource_str,dataFoldercleaned_str):
    print('...Beginning raw data cleaning...')
    dataFolder_str = os.path.join(dataFolderbase_str,dataFoldersource_str)
    # first lets check to make sure that we can find the data folder
    try:											#check for data directory and create if needed
        os.stat(dataFolder_str)
        datafolderFound = True
    except:
        print('data/ directory could not be found!')
        datafolderFound = False
    
    
    # now lets makes the new clean data directory 
    if not os.path.exists(dataFoldercleaned_str):
        os.makedirs(dataFoldercleaned_str)
    else:
        print('Data Cleaned folder already created, data assumed to be cleaned')
        print('...exiting')
        return

    # now lets go into each folder
    if(datafolderFound): 
        allWords = os.listdir(dataFolder_str)
        for wordfolder in allWords:
            fullwordFolder = os.path.join(dataFolder_str,wordfolder)
            videoNames =  os.listdir(fullwordFolder) # all the video names in the word folder
            numVideos = len(videoNames)
            newVideoNames = [str(x) + '.avi' for x in range(2*numVideos)]    
            destinationFolder = os.path.join(dataFoldercleaned_str,wordfolder)
            try:
                os.stat(destinationFolder)
            except:
                os.mkdir(destinationFolder)
    
            for videoname in videoNames:
                newName1= newVideoNames.pop(0)
                newName2 = newVideoNames.pop(0)
                videoSource = os.path.join(fullwordFolder,videoname)
                TrimVideo(videoSource,newName1,newName2,destinationFolder)
            print('Word {} processed'.format(wordfolder))
    print('... Data cleaning complete')

# =============================================================================
# getImagesFromVideoFile : takes in the absoulte path to a vdieo source and 
# returns all the associated frames of the video resized to (224,224,3)
    
# INPUTS:
# videoName - string of absolute path of video files
    
# OUTPUTS:
# vid_Frames - list of all the frames from the video file resized to (224,224) 
            #and caseted to np.float64
# size - tuple of the size of the original frame (height,width)
# numFrames - total number of frames originally found in the video
# =============================================================================
def getImagesFromVideoFile(videoName):
    #get video source object
    cap = cv2.VideoCapture(videoName)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    numFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (h,w)
    vid_Frames = [] #initialize
    
    if(cap.isOpened()):
        while(True):
            #read the captured video
            ret, frame = cap.read()
            
            if(not ret):
                #no more frames so lets exit the loop
                break
            im = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            #now lets just cast it and fix the channels
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB).astype(np.float64)
            vid_Frames.append(im)
    else:
        print('Something wrong with VideoCapture')
    #release the object
    cap.release()
    return vid_Frames,size,numFrames    
# =============================================================================
# getFeatures: takes videos in the cleaned data folder and creates and stores 
# numpy array features to be used in the LSTM
    
# INPUTS:  
# dataFolderbase_str - path of base folder where data folder is in
# dataFoldercleaned_str - name of data folder with cleaned features
# dataFolderfeatures_str - name of to-be created data feature folders that will 
    #store the numpy arrays
# ============================================================================= 
def getFeatures(dataFolderbase_str,dataFoldercleaned_str,dataFolderfeatures_str,numClasses):
    print('... Beginning Feature Extraction...')
    # first lets look at the data
    dataFolder = dataFoldercleaned_str
    wordLists = os.listdir(dataFolder)
    
    #lets create the destination directory
    baseDirectory = dataFolderfeatures_str
    if not os.path.exists(baseDirectory):
        os.makedirs(baseDirectory)
    else:
        print('Feature Directory already created and feature assumed to be extracted')
        print('...exiting')
        return

    #populate the data frame with words
    ASL_df = pd.DataFrame(wordLists,columns = ['Word']);
    #populate with full paths
    ASL_df['Full Path'] = ASL_df['Word'].apply(lambda x: os.path.join(dataFolder,x))
    #populate with # samples
    ASL_df['# samples'] = ASL_df['Full Path'].apply(lambda x: len(os.listdir(x)))
    #sort them by largest or smallest
    ASL_df = ASL_df.sort_values(by=['# samples'],ascending = False)
                                    
    # now lets build the model 
    FeatureExtractor = Sequential()
    conv_base = VGG16(weights='imagenet', include_top=False)
    FeatureExtractor.add(conv_base)
    
    print('\nModel Summary of Feature Extractor for LSTM:')
    FeatureExtractor.summary()
    print('')
    NumClasses = numClasses
    for index, row in ASL_df.head(NumClasses).iterrows():
        print('Processing Class {}'.format(row['Word']))
        #get the full paths of the videos
        videoNameSources = [os.path.join(row['Full Path'],x) for x in os.listdir(row['Full Path'])]
        for i, videoName in enumerate(videoNameSources):
            vid_Frames,size,numFrames = getImagesFromVideoFile(videoName)
            destDirectory = os.path.join(baseDirectory,row['Word'] + '_'+str(i))
            #now lets extract the features in batch
            x = np.array(vid_Frames) # create the mult-dim array
            x = preprocess_input(x) # process input to work with the VGG network
            y = FeatureExtractor.predict(x) # extract Features
            
            
            # now lets save the features to the destination directory 
            np.save(destDirectory, y)

    print('...Feature extraction complete')
    
# =============================================================================
# ImagesFromVideoFile : takes in video file and converts it to images and saved 
# in folder with class name
    
# INPUTS: 
# videoName - absolute path to the video file to be converted
# desDirectory - absolute path to the destination directory where the image will reside
# wordsampleIndex_int - integer used to name the pictures for namessake
# =============================================================================
def ImagesFromVideoFile(videoName,desDirectory,wordsampleIndex_int):
    #get video source object
    cap = cv2.VideoCapture(videoName)
    w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    numFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    size = (h,w)    
    count = 0
    if(cap.isOpened()):
        while(True):
            #read the captured video
            ret, frame = cap.read()
            
            if(not ret):
                #no more frames so lets exit the loop
                break
            im = cv2.resize(frame, (224, 224), interpolation = cv2.INTER_AREA)
            #now lets just cast it and fix the channels
#             im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            filename = os.path.join(desDirectory,'pic_{0}_{1}.png'.format(wordsampleIndex_int,count))

            retval = cv2.imwrite(filename, im)
            count += 1
    else:
        print('Something wrong with VideoCapture')
    #release the object
    cap.release()
    
# =============================================================================
# createImageDirectories: will gather the videos of interest and creates folders 
# with the videos as images for an image classification model
    
# INPUTS: 
# dataFoldercleaned_str - name of data folder with cleaned features
# dataFolderImages_str - name of data folder where the image classes will be stored
# =============================================================================
def createImageDirectories(dataFoldercleaned_str,dataFolderImages_str,numClasses):
    print('... Beginning the creation of Image Directories...')

    # first lets look at the data
    dataFolder = dataFoldercleaned_str
    wordLists = os.listdir(dataFolder)
    
    #populate the data frame with words
    ASL_df = pd.DataFrame(wordLists,columns = ['Word']);
    #populate with full paths
    ASL_df['Full Path'] = ASL_df['Word'].apply(lambda x: os.path.join(dataFolder,x))
    #populate with # samples
    ASL_df['# samples'] = ASL_df['Full Path'].apply(lambda x: len(os.listdir(x)))
    
    #sort them by largest or smallest
    ASL_df = ASL_df.sort_values(by=['# samples'],ascending = False)
    NumClasses = numClasses
    baseDirectory = dataFolderImages_str
    #lets create the destination directory
    if not os.path.exists(baseDirectory):
        os.makedirs(baseDirectory)
    else: 
        print('Image Directories already created, images assumed to have been created')
        print('...exiting')
        return
        
    for index, row in ASL_df.head(NumClasses).iterrows():
        #get the full paths of the videos
        videoNameSources = [os.path.join(row['Full Path'],x) for x in os.listdir(row['Full Path'])]
        destDirectory = os.path.join(baseDirectory,row['Word'])
        if not os.path.exists(destDirectory):
            os.makedirs(destDirectory)
        for i, videoName in enumerate(videoNameSources):
            ImagesFromVideoFile(videoName,destDirectory,i)
        print('Class : {} Images have been created'.format(row['Word']))
    print('...Image directories creation complete')
    
    
def train_test_foldSplit(baseDirectory,ValDirectory,testSize = .2):
    trainSize = 1-testSize
    for className in os.listdir(baseDirectory):
        valClassFolder = os.path.join(ValDirectory,className)
        baseClassFolder = os.path.join(baseDirectory,className)
        
        if not os.path.exists(valClassFolder):
            os.makedirs(valClassFolder)
        
        picNames = os.listdir(baseClassFolder)
        numSampled = int(len(picNames)*testSize)
        picNames_sampled = random.sample(picNames,numSampled)
        filePaths_lst = [os.path.join(baseClassFolder,x) for x in picNames_sampled ]
        filePaths_val_lst = [os.path.join(valClassFolder,x) for x in picNames_sampled]
        
        for filebase,fileval in zip(filePaths_lst,filePaths_val_lst):
            os.rename(filebase,fileval)
        
        print('Finished Splitting class {}'.format(className))
    print('Done')

def createModel():
    # lets first create an input layer, based on the input image size
    inputSize = (256,256,3)
    
    in_lay = Input(inputSize)
    # load the VGG network
    conv_base = VGG16(input_shape =  inputSize, include_top = False, weights = 'imagenet')
    conv_base.trainable = False # freeze the layers
    numFeatureMaps = conv_base.get_output_shape_at(0)[-1] # going to need shape for attention mapping later
    
    # lets start building the network
    pt_features = conv_base(in_lay)
    bn_features = BatchNormalization(name = 'OutputConvBase')(pt_features)
    
    # Now lets build our attention mapping mechanism
    attn_layer = Conv2D(64, kernel_size = (1,1), padding = 'same', activation = 'relu')(bn_features)
    attn_layer = Conv2D(16, kernel_size = (1,1), padding = 'same', activation = 'relu')(attn_layer)
     # let's lower the dimension to have single feature map, resemebles local receptive field of visual cortex!
    attn_layer = LocallyConnected2D(1,kernel_size = (1,1),padding = 'valid', activation = 'sigmoid')(attn_layer)
    
    # lets bump th enumber of features back up to match the output of our convbase
    up_c2_w = np.ones((1, 1, 1, numFeatureMaps))
    up_c2 = Conv2D(numFeatureMaps, kernel_size = (1,1), padding = 'same',activation = 'linear',name = 'AttentionLayer', use_bias = False, weights = [up_c2_w])
    up_c2.trainable = False
    attn_layer = up_c2(attn_layer)
    
    # lets point feature map multiply our attention layer to the output of our convbase
    mask_features = multiply([attn_layer, bn_features])
    gap_features = GlobalAveragePooling2D()(mask_features) # pass them through GAP to find which feature maps are most important
    # gap_mask = GlobalAveragePooling2D()(attn_layer) # pass our attenttino layer through gap to find associated weights of attn layer
    
    # to account for missing values from the attention model
    # gap = Lambda(lambda x: x[0]/x[1], name = 'RescaleGAP')([gap_features, gap_mask])
    gap_dr = Dropout(0.5)(gap_features)
    FC = Dense(1024, activation = 'relu')(gap_dr)
    dr_steps = Dropout(0.25)(FC)
    out_layer = Dense(10, activation = 'softmax')(dr_steps) 
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    
    ASLModel = Model(inputs = [in_lay], outputs = [out_layer])
    ASLModel.compile(optimizer = adam, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])
    
    print('CNN Video Model Summary:')
    ASLModel.summary()
    return ASLModel
                                    
def runCNNVideoModel(dataFolderImages_str):
    baseDirectory = dataFolderImages_str
    ValDirectory = 'data_Images_val'
    if not os.path.exists(ValDirectory):
        print('Splitting Data into training and validation')
        os.makedirs(ValDirectory)
        train_test_foldSplit(baseDirectory,ValDirectory,testSize = .2)
    
    
    print('...Creating CNN Video Model...')
    ASLModel = createModel()
    
    # lets load the image generator, and set some of our parameters ~ data augmentation
    print('...Creating Data Generator...')
    myImageGenerator = ImageDataGenerator(samplewise_center=False, samplewise_std_normalization=False, horizontal_flip = False, 
                             vertical_flip = False, height_shift_range = 0.15, width_shift_range = 0.15, 
                             rotation_range = 5, shear_range = 0.01,fill_mode = 'nearest',zoom_range=0.25,
                             preprocessing_function = preprocess_input)

    baseDirectory = dataFolderImages_str
    baseValDirectory = ValDirectory
    trainGen = myImageGenerator.flow_from_directory(baseDirectory, class_mode = 'sparse',color_mode = 'rgb',batch_size = 32)
    valGen = myImageGenerator.flow_from_directory(baseValDirectory, class_mode = 'sparse',color_mode = 'rgb',batch_size = 32)
    
    # lets create our callback lists
    #filename for our weights
    weight_filename="{}_weights.best.hdf5".format('ASLModel')
    #create a checkpoint to save the weights for every epoch based on the validation loss
    checkpoint = ModelCheckpoint(weight_filename, monitor='val_loss', verbose=1, 
                                 save_best_only=True, mode='min', save_weights_only = True)
    #let reduce the learining rate if learning stagnates based on validation loss
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=5, min_lr=0.0001)
    #early stopping condition
    early = EarlyStopping(monitor="val_loss", mode="min", patience=5)
    callbacks_list = [checkpoint, early, reduceLROnPlat]
    
    # training
    MaxEpochs = 30
    print('')
    print('...Beginning Training Model...')
    trainingHistory = ASLModel.fit_generator(trainGen, validation_data = valGen, epochs = MaxEpochs, callbacks = callbacks_list)
    
    print('...Showing Loss and Accuracy...')
    loss = trainingHistory.history['loss']
    lossval = trainingHistory.history['val_loss']
    
    plt.plot(loss,label = 'Training')
    plt.plot(lossval,label = 'Validation')
    plt.title('Loss')
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()
    
    acc = trainingHistory.history['acc']
    accval = trainingHistory.history['val_acc']
    plt.plot(acc,label = 'Training')
    plt.plot(accval,label = 'Validation')
    plt.title('Accuracy')
    plt.legend()
    plt.xlabel('Epoch')
    plt.show()
    
    print('..Showing Attention Maps for random sample...')
    testX, testY = next(trainGen)

    # lets load the best model found from training
    weight_filename="{}_weights.best.hdf5".format('ASLModel')
    ASLModel.load_weights(weight_filename)
    
    for attn_layer in ASLModel.layers:
        c_shape = attn_layer.get_output_shape_at(0)
        if len(c_shape)==4:
            if c_shape[-1]==1:
                break
                
    import keras.backend as K
    rand_idx = np.random.choice(range(len(testX)), size = 2)
    attn_func = K.function(inputs = [ASLModel.get_input_at(0), K.learning_phase()],
               outputs = [attn_layer.get_output_at(0)]
              )
    fig, m_axs = plt.subplots(len(rand_idx), 2, figsize = (8, 4*len(rand_idx)))
    [c_ax.axis('off') for c_ax in m_axs.flatten()]
    for c_idx, (img_ax, attn_ax) in zip(rand_idx, m_axs):
        cur_img = testX[c_idx:(c_idx+1)]
        attn_img = attn_func([cur_img, 0])[0]
        img_ax.imshow(cur_img[0,:,:,0])
        attn_ax.imshow(attn_img[0, :, :, 0], cmap = 'jet',
                       vmin = 0, vmax = 1, 
                       interpolation = 'lanczos')
        real_class = testY[c_idx]
        img_ax.set_title('Frame Sample Image\nTrue Class:{}'.format(real_class))
        pred_class = np.argmax(ASLModel.predict(cur_img))
        attn_ax.set_title('Attention Map\nPred Class:{}'.format(pred_class) )
        
    print('...CNN Video Model Complete')
        
                                    
        
# =============================================================================
#  main 
# =============================================================================
def main():
    dataFolderbase = '/datasets/ee285s-public/KaggleASL'
    dataFoldersource = 'data'
    dataFoldercleaned = 'data_cleaned'
    
    # DATA PROCESSING
    #first we'll take the data and clean it. i.e. remove the annotations and 
    #split the videos in half
    cleanData(dataFolderbase,dataFoldersource,dataFoldercleaned)
    print('')
    
    # now let's extract the features to create a set of feautres to be used by 
    # the LSTM model
    dataFolderfeatures = 'data_features'
    numClasses = 10
    getFeatures(dataFolderbase,dataFoldercleaned,dataFolderfeatures,numClasses)
    print('')
    
    #lets create the image directories to be used for the CNN model
    dataFolderImages = 'data_Images'
    createImageDirectories(dataFoldercleaned,dataFolderImages,numClasses)
    print('')
    
    #MODEL TRAINING/RESULTS
    # CNN Video model
#     runCNNVideoModel(dataFolderImages)
    

    #LSTM Video model
    #TODO-------

    #CNN Image Model
    cnn_main()
    
    
    
if __name__ == '__main__':
    main()
