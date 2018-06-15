# ASL_Gesture_Recognition

## Getting Started

```
git clone https://github.com/wnukmat/ASL_Gesture_Recognition
```

### Dataset

```
https://www.kaggle.com/datamunge/sign-language-mnist/data <br/>
https://www.bu.edu/av/asllrp/dai-asllvd.html <br/>
```

## Authors

* **Mansur Amin**    - *CNN Images*               - [mansuramin](https://github.com/mansuramin) <br/>
* **Juan Castillo**  - *Transfer LEarning Videos* - [juancastillo](https://github.com/camiloj4) <br/>
* **Matthew Wnuk**   - *LSTM Videos*              - [wnukmat](https://github.com/wnukmat) <br/>


## Description
In this project we propose using deep learning to implement a system that can identify/classify both static images and gestures of mono-morphemic signs from videos. We will be using 2 strategies that leverage the advances in deep learning .The first strategy employs the use of a single Convolutional Neural Network that will classify letters of the alphabet and will be known as the ASL Image Model. The second strategy employs the use of transfer learning, where we attempt at using two network architectures to classify ASL videos, this model will be known as ASL Video Models. The first being a CNN paired with a attention mapping architecture, and the second being a CNN paired with an Recurrent Neural Network (RNN).

## Requirments 
Install packages as follows: <br/>

pip install os <br/>
pip install cv2 <br/>
pip install time <br/>
pip install numpy <br/>
pip install pandas <br/>
pip install matplotlib.pyplot <br/>

## Code Organization

main.py                                       -- Extracts Data and Trains all Models <br/>

asl_img_cnn/cnn_asl_mnist.ipynb               -- Trains and Test using Kaggel Image Dataset <br/>
asl_img_cnn/cnn_asl_mnist.py                  -- .py version of cnn_asl_mnist.ipynb <br/>
 
asl_video_lstm/Create_Image_Directories.ipynb -- Creates images directories to be used for single frame classifier <br/>
asl_video_lstm/clean_data.py                  -- Crops the videos and produces two new videos <br/>
asl_video_lstm/getFeatures.ipynb              -- Creates and saves features for LSTM model <br/>
asl_video_lstm/get_data.py                    -- Download data from Online <br/>
asl_video_lstm/load_features.py               -- Loads features from directories to be used for LSTM training <br/> 
asl_video_lstm/lstm.py                        -- Trains LSTM <br/>
asl_video_lstm/test.xlsx                      -- Excel file pointing to online videos and used by get_data.py <br/>

