# Gesture Recognition

This is a project which can recognize hand gestures.

The key parts in this project is:

1. Detect human skin from the video frames.
2. Draw the contour of hands.
3. Extract the features of the contours by using Fourier Descriptor and create datasets for training.
4. Train the datasets with DNNClassifier in Tensorflow.

## Installation:

### Systems Tested:

- Fedora 25

### Requirements:

- tensorflow 1.0
- numpy
- pandas
- openCV

## Start:

    $ python gestion_recognition
