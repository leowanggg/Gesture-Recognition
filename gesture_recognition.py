# This file is for gesture recognition.
#
# I use the DNNClassifier trained in the
# tensor_gesture.py
#
# The way to recognize gestures:
# First, show your gestures which recorded in
# the training datasets. When the contour of your
# gesture is completed and clear, press button 'c'
# to classify.
# The result is shown in console.
# ===============================================

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import cv2
import cmath
import tensorflow as tf
import pandas as pd

# Define the length of fourier descriptor
K = 10

########## Methods for gesture recognition ##########
# Find the longest contour in frames
def findLongestContour(contours):
    if len(contours) != 0:
        index_longest_contour = max(
            enumerate(contours), key=lambda tup: len(tup[1]))[0]
    else:
        index_longest_contour = -1
    return index_longest_contour

# Convert coordinate list into complex
def convert2Complex(contour, N):
    complex_contour = []
    for i in xrange(0, N):
        z = complex(contour[i][0][0], contour[i][0][1])
        complex_contour.append(z)
    return complex_contour

# Convert coordinate list into integer
def convert2Int(rebuilt_complex_contour):
    rebuilt_contour = []
    for i in xrange(0, len(rebuilt_complex_contour)):
        z = [[rebuilt_complex_contour[i].real, rebuilt_complex_contour[i].imag]]
        rebuilt_contour.append(z)
    return rebuilt_contour

# Calculate the average of the complex coordinate list of the contour
def calAvg(complex_contour, N):
    real, imag = 0, 0
    for i in xrange(0, N):
        real += complex_contour[i].real
        imag += complex_contour[i].imag
    z_sum = complex(real, imag)
    avg_complex_contour = z_sum / float(N)
    return avg_complex_contour

# Fourier decriptor
def fourierDescriptor(contour):
    N = len(contour)
    complex_contour = convert2Complex(contour, N)
    avg_complex_contour = calAvg(complex_contour, N)
    fourier_series = []
    for k in xrange(-K, K + 1):
        a = complex(0, 0)
        for m in xrange(0, N):
            a += (complex_contour[m] - avg_complex_contour) * \
                np.exp(complex(0, -2 * np.pi * k * m / N))
        a /= N
        fourier_series.append(a)
    return N, avg_complex_contour, fourier_series

# Normalisation
def normalisation(fourier_series):
    # Method classsic
    # Direction invariant
    for k in xrange(0, K):
        temp = fourier_series[k]
        fourier_series[k] = fourier_series[2 * K - k]
        fourier_series[2 * K - k] = temp
    # Size invariant
    for k in xrange(-K, K + 1):
        fourier_series[k + K] /= abs(fourier_series[K + 1])
    # Method improved
    N = len(fourier_series)
    phi = cmath.phase(fourier_series[(N - 1) / 2 - 1]
                      * fourier_series[(N - 1) / 2 + 1]) / 2.0
    for k in xrange(-K, K + 1):
        fourier_series[k + K] *= np.exp(complex(0, -phi))
    theta = cmath.phase(fourier_series[(N - 1) / 2 + 1])
    for k in xrange(-K, K + 1):
        fourier_series[k + K] *= np.exp(complex(0, -theta * k))
    return fourier_series

# Rebuild the contour by using fourier descriptor
def rebuildContour(fourier_series, avg_complex_contour, N):
    rebuilt_complex_contour = []
    for m in xrange(0, N):
        z = complex(0, 0)
        for k in xrange(-K, K + 1):
            z += fourier_series[k + K] * \
                np.exp(complex(0, 2 * np.pi * k * m / N))
        z += avg_complex_contour
        rebuilt_complex_contour.append(z)
    rebuilt_contour = convert2Int(rebuilt_complex_contour)
    return rebuilt_contour

# Calculate modules of fourier series
def calModule(fourier_series):
    for k in xrange(-K, K+1):
        fourier_series[k+K] = abs(fourier_series[k+K])
    return fourier_series


########## Initialize DNN Classifier from tensorflow ##########
FEATURES = ["-10", "-9", "-8", "-7", "-6", "-5", "-4", "-3", "-2", "-1", "0", "1", "2",
            "3", "4", "5", "6", "7", "8", "9", "10"]
# Build the input_fn.
def input_fn(data_set):
    data_set = pd.DataFrame(data_set, index=[0])
    feature_cols = {k: tf.constant(data_set[k].values, shape=[data_set[k].size, 1])
                    for k in FEATURES}
    return feature_cols

def buildDNN():
    # Specify that all features have real-value datasets.
    feature_cols = [tf.contrib.layers.real_valued_column(k)
                    for k in FEATURES]

    # Build 3 layer DNN with 10, 20, 10 units respectively.
    classifier = tf.contrib.learn.DNNClassifier(feature_columns=feature_cols,
                                                hidden_units=[10, 20, 10],
                                                n_classes=5,
                                                model_dir="gesture_model")
    return classifier

########## Start recognition ##########
# Initialize DNN Classifier
DNNClassifier = buildDNN()

# Initialize video capture
cap = cv2.VideoCapture(0)
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        ########## Choose the color space for human skin detecting ##########
        # Convert the frame from BGR to YCrCb
        frame_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(frame_YCrCb, (0, 135, 75), (80, 175, 125))

        # Convert the frame from BGR to HSV
        # frame_HSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # skin_mask = cv2.inRange(frame_HSV, (0, 58, 40), (35, 174, 255))

        ########## Apply a series of erosions and dilations to the mask ##########
        # Use an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel)
        skin_mask = cv2.dilate(skin_mask, kernel)
        # Blur the mask to help remove noise
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)
        # Apply the mask to the frame
        # skin = cv2.bitwise_and(frame, frame, mask = skin_mask)

        ########## Draw the contour ##########
        # Find the contours in the frame
        _, contours, _ = cv2.findContours(
            skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Find the longest contour
        index_longest_contour = findLongestContour(contours)
        # Create a totally black frame
        frame_black_original = np.zeros(frame.shape, dtype="uint8")
        # frame_black_rebuilt = np.zeros(frame.shape, dtype="uint8")
        # Draw the largest contour in the black frame
        cv2.drawContours(frame_black_original, contours,
                         index_longest_contour, (255, 255, 255), 3)

        ########## Use fourier descriptor to extract the features of the contour ##########
        # Calculate the fourier descriptor of the contour
        N, avg_complex_contour, fourier_series = fourierDescriptor(
            contours[index_longest_contour])
        # Normalise the fourier_series
        fourier_series = normalisation(fourier_series)

        ########## Rebuild the contour by the fourier series ##########
        # Rebuild the contour and add it into contours
        # rebuilt_contour = rebuildContour(
        #     fourier_series, avg_complex_contour, N)
        # Draw the rebuilt contour in the black frame
        # contours = [np.array(rebuilt_contour, dtype=np.int32)]
        # cv2.drawContours(frame_black_rebuilt, contours, 0, (255, 255, 255), 3)

        ########## Show the contour ##########
        cv2.imshow("images", np.hstack([frame_black_original]))

        ########## Classify the contour ##########
        # Calculate the modules of fourier series
        fourier_series = calModule(fourier_series)
        # Classify contour
        key_value = cv2.waitKey(1)
        if key_value == ord('c'):
            prediction_dict = dict(zip(FEATURES, fourier_series))
            y = list(DNNClassifier.predict(input_fn=lambda: input_fn(prediction_dict)))
            y[0] += 1
            print('The number you show is: {}'.format(str(y)))
        elif key_value == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
