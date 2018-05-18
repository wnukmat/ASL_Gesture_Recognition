import  os
import cv2 

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
        print('Error reading/writing videos')

        

if __name__ == '__main__':
    # first lets check to make sure that we can find the data folder
    try:											#check for data directory and create if needed
        os.stat('data/')
        datafolderFound = True
    except:
        print('data/ directory could not be found!')
        datafolderFound = False
    
    
    # now lets makes the new clean data directory 
    try:											#check for data directory and create if needed
        os.stat('data_cleaned/')
    except:
        os.mkdir('data_cleaned/')
    
    # now lets go into each folder
    if(datafolderFound): 
        allWords = os.listdir('data')
        for wordfolder in allWords:
            fullwordFolder = os.path.join('data',wordfolder)
            videoNames =  os.listdir(fullwordFolder) # all the video names in the word folder
            numVideos = len(videoNames)
            newVideoNames = [str(x) + '.avi' for x in range(2*numVideos)]    
            destinationFolder = os.path.join('data_cleaned',wordfolder)
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
                
        
        
        