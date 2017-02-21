# This file is for shuffle the records in training
# and test datasets.

# Because the datasets are created in order which
# is not good for training and test. So this file
# is needed to shuffle the datasets
# ================================================


import csv
import random

with open("gesture_data_train.csv", "rb") as f:
    rows = list(csv.reader(f, delimiter=","))

random.shuffle(rows)

with open("gesture_data_train.csv", "wb") as f:
    csv.writer(f, delimiter=",").writerows(rows)

with open("gesture_data_test.csv", "rb") as f:
    rows = list(csv.reader(f, delimiter=","))

random.shuffle(rows)

with open("gesture_data_test.csv", "wb") as f:
    csv.writer(f, delimiter=",").writerows(rows)
