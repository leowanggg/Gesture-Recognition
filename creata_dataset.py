# This file is for making datasets for gesture training.
#
# 5 different gestures are used for training and recognition.
# Each gesture is sampled 100 times in the traning dataset, and
# 20 times in the test dataset.
#
# The way to make a traning dataset:
# first, prepare your first gesture in front of the web cam.
# second, when the contour of your gesture is completed
# and clear, press button 1 in your keyboard until the number
# in your console counts to 100.
# Then, you can finish your next 4 different gestures with buttons
# 2,3,4,5 till the number counts to 500.
# ================================================================


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import
import numpy as np
import cv2
import csv
import cmath

# Define the length of fourier descriptor
K = 10

# Create csv file
csvfile = open('gesture_data_train.csv', 'wb')
writer = csv.writer(csvfile, delimiter=',',
                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
writer.writerow([500, 21, '-10', '-9', '-8', '-7', '-6', '-5', '-4', '-3', '-2', '-1', '0',
                '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'number'])

########## Methods for Fourier Descriptor ##########
# Find the longest contour in frames
def findLongestContour(contours):
    if len(contours) != 0:
        index_longest_contour = max(enumerate(contours), key = lambda tup: len(tup[1]))[0]
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
    avg_complex_contour = z_sum/float(N)
    return avg_complex_contour


# Fourier decriptor
def fourierDescriptor(contour):
    N = len(contour)
    complex_contour = convert2Complex(contour, N)
    avg_complex_contour = calAvg(complex_contour, N)
    fourier_series = []
    for k in xrange(-K, K+1):
        a = complex(0, 0)
        for m in xrange(0, N):
            a += (complex_contour[m]-avg_complex_contour)*np.exp(complex(0, -2*np.pi*k*m/N))
        a /= N
        fourier_series.append(a)
    return N, avg_complex_contour, fourier_series

# Normalisation
def normalisation(fourier_series):
    # Method classsic
    # Direction invariant
    for k in xrange(0, K):
        temp = fourier_series[k]
        fourier_series[k] = fourier_series[2*K-k]
        fourier_series[2*K-k] = temp
    # Size invariant
    for k in xrange(-K, K+1):
        fourier_series[k+K] /= abs(fourier_series[K+1])
    # Method improved
    N = len(fourier_series)
    phi = cmath.phase(fourier_series[(N-1)/2-1] * fourier_series[(N-1)/2+1])/2.0
    for k in xrange(-K, K+1):
        fourier_series[k+K] *= np.exp(complex(0, -phi))
    theta = cmath.phase(fourier_series[(N-1)/2+1])
    for k in xrange(-K, K+1):
        fourier_series[k+K] *= np.exp(complex(0, -theta*k))
    return fourier_series

# Calculate modules of fourier series
def calModule(fourier_series):
    for k in xrange(-K, K+1):
        fourier_series[k+K] = abs(fourier_series[k+K])
    return fourier_series

# Initiate video capture
cap = cv2.VideoCapture(0)
i = 0
while(cap.isOpened()):
    # Capture frame-by-frame
    ret, frame = cap.read()
    if ret:
        ########## Choose the color space for human skin detecting ##########
        # Convert the frame from BGR to YCrCb
        frame_YCrCb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        skin_mask = cv2.inRange(frame_YCrCb, (0, 135, 75), (80, 175, 125)

        ########## Apply a series of erosions and dilations to the mask ##########
        # Using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        skin_mask = cv2.erode(skin_mask, kernel)
        skin_mask = cv2.dilate(skin_mask, kernel)
        # Blur the mask to help remove noise
        skin_mask = cv2.GaussianBlur(skin_mask, (3, 3), 0)

        ########## Draw the contour ##########
        # Find the contours in the frame
        _, contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        # Find the longest contour
        index_longest_contour = findLongestContour(contours)
        # Create a totally black frame
        frame_black_original = np.zeros(frame.shape, dtype="uint8")
        # Draw the largest contour in the black frame
        cv2.drawContours(frame_black_original, contours, index_longest_contour, (255, 255, 255), 3)

        ########## Use fourier descriptor to extract the features of the contour ##########
        # Calculate the fourier descriptor of the contour
        N, avg_complex_contour, fourier_series = fourierDescriptor(contours[index_longest_contour])
        # Normlise
        fourier_series = normalisation(fourier_series)
        # Calcule the modules of fourier series
        fourier_series = calModule(fourier_series)
        cv2.imshow("images", np.hstack([frame_black_original]))
        # Create datasets
        key_value = cv2.waitKey(1)
        if i < 100 and key_value == ord('1'):
            i += 1
            fourier_series.append('0')
            writer.writerow(fourier_series)

        elif i < 200 and key_value == ord('2'):
            i += 1
            fourier_series.append('1')
            writer.writerow(fourier_series)

        elif i < 300 and key_value == ord('3'):
            i += 1
            fourier_series.append('2')
            writer.writerow(fourier_series)

        elif i < 400 and key_value == ord('4'):
            i += 1
            fourier_series.append('3')
            writer.writerow(fourier_series)

        elif i < 500 and key_value == ord('5'):
            i += 1
            fourier_series.append('4')
            writer.writerow(fourier_series)
        elif cv2.key_value == ord('q'):
            break

        print("Counting: ", i)

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
