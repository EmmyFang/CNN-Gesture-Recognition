# CNN-Gesture-Recognition
Note: This is a course project of MIE324 at University of Toronto (year 2018). If you are a student of this course and find this project is similar to one of the projects you got, please contact me and I will make this repository private. Thank you!

This project builds and trains several convolutional neural nets aiming to classify the 26 letters of English alphabet drawn by moving smartphones (recorded by the accelerometer and gyroscope of the phone). 


### Data 
All students enrolled in MIE324 course moved their smartphones to draw the 26 letters of English alphabet for five times each. The motions (gestures) were recorded by the accelerometer and gyroscope of the phone. 
The acceleration and angular velocity measured along the x, y, z axes sampled with a frequency of 50 Hz (100 samples in 2 seconds). 


## main.py
run main.py to load data and start training the classifier 

## Requirements
1. python3.6
2. numpy, matplotlib, pytorch, pandas, scikit-learn, scipy 


### csv2numpy.py 
It walks through the directory that stores the data and parse the data into the sensor values array (instances.npy) and the label for each instance (label.npy)

### visualize_data.py
It helps to visualize the sensor values of the 6 channels of a gesture instance. 

### bin_data.py
It finds the average sensor values over both time and gesture instances for a given gesture and plot the mean and standard deviation of each channel for that gesture. 

### normalize_data.py
It performs the ‘local’ normalization of each gesture instance. 

### dataset.py
data loader class


### model.py
the model class 
