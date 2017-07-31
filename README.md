# Behavioral Cloning
This is the Behavioral Cloning Project for Self Driving Car Engineer Program at Udacity.

## FILES
### Training Program
**model.py**  
This is the program to train the neural network. This neural networks contains the following parts:
- Convolution + MaxPool x 2
- Flattening x 1
- Fully Connected Layer x 3

Time to run in AWS EC2 Server : **12 hours - NO GPU USAGE**

### Drive Program
**drive.py**  
This program comunicates with the Simulator to indicate the steering the car needs to use
To use this program run the following line  
``python drive.py model.h5``

### Write Up Report
**writeup_report**  
This document explains how the program behaves,
results and furthers improvements.

### Network File
**model.h5**  
This file contains all the information about the Neural Network.

### Video File
**video.mp4**  
This file contains a video of the result of the neural network.

### Video Generator
**video.py**  
This is a python program to generate a video. To di this run the following lines:
1. Save the pictures  
``python drive.py model.h5 run1``
2. Create the video file  
``python video.py run1``


## TRAINING ON AWS
If you want to train the Neural Network in AWS follow the lines above:

1. Clone the project  
``git clone https://github.com/rbtluisenriquerbt/behavioralcloning.git``
2. Download Anaconda  
``curl -O https://repo.continuum.io/archive/Anaconda3-4.2.0-Linux-x86_64.sh
``  
3. Install Anaconda  
``bash Anaconda3-4.2.0-Linux-x86_64.sh``
4. Create a Python3 Environment  
``conda create --name py3 python=3``
5. Activate the Python3 Environment  
``source activate py3``
6. Install OpenCV  
``conda install -c menpo opencv``
7. Install Scikit Library  
``conda install scikit-learn``
8. Install Keras  
``conda install -c conda-forge keras``
9. Train the network  
``python model.py``  

> NOTE 1  
The repository has already a set of images and a CSV for training purpose.  
> NOTE 2  
Please submit an issue if you find any problem.
