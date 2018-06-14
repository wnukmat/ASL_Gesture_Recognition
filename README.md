# ASL_Gesture_Recognition

## Getting Started

```
git clone https://github.com/wnukmat/ASL_Gesture_Recognition
```

### Dependancies

pip install numpy
pip install matplotlib.pyplot


### Dataset

```
https://www.kaggle.com/datamunge/sign-language-mnist/data
https://www.bu.edu/av/asllrp/dai-asllvd.html
```

## Authors

* **Matthew Wnuk**   - *LSTM ASL Videos* - [wnukmat](https://github.com/wnukmat)
* **Juan Castillo**  - *RNN ASL Videos* - [juancastillo](https://github.com/wnukmat)
* **Mansur Amin**    - *CNN ASL Images* - [mansuramin](https://github.com/mansuramin)



## Description 

## Requirments 

## Code Organization

main.py                                       -- Extracts Data and Trains all Models

asl_img_cnn/cnn_asl_mnist.ipynb               -- Trains and Test using Kaggel Image Dataset
asl_img_cnn/cnn_asl_mnist.py                  -- .py version of cnn_asl_mnist.ipynb 

asl_video_lstm/Create_Image_Directories.ipynb -- Creates images directories to be used for single frame classifier
asl_video_lstm/clean_data.py                  -- Crops the videos and produces two new videos
asl_video_lstm/getFeatures.ipynb              -- Creates and saves features for LSTM model
asl_video_lstm/get_data.py                    -- Download data from Online 
asl_video_lstm/load_features.py               -- Loads features from directories to be used for LSTM training 
asl_video_lstm/lstm.py                        -- Trains LSTM
asl_video_lstm/test.xlsx                      -- Excel file pointing to online videos and used by get_data.py

