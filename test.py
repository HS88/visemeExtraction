from FeatureExtractor import FeatureExtractor
import MSA

import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time


start_time = time.time()

label_dict_train = {}
label_dict_test = {}

# Read training labels
train_label_file = open(os.path.join('label_data', 'train_complete_file.txt'), 'r')
for line in train_label_file.readlines():
    line_input = line.split(" ")
    label_dict_train[line_input[0]] = line_input[2].rstrip()
train_label_file.close()

# Read test labels
test_label_file = open(os.path.join('label_data', 'test_complete_file.txt'), 'r')
for line in test_label_file.readlines():
    line_input = line.split(" ")
    label_dict_test[line_input[0]] = line_input[2].rstrip()
test_label_file.close()

# Read training files
files = os.listdir('train_data')
files = files
training_labels = []
training_features = np.empty((len(files), 60))
for i, file_name in enumerate(files):
    # Extract lip as 60x80 image
    try:
        file = os.path.join('train_data', file_name)
        fe = FeatureExtractor.from_image(file)
        fe.face_detect(draw=True)
        fe.landmark_detect()
        fe.crop_lips()
    # print("File {} lips extracted.".format(i))

    # Convert lip image to 60-dimensional feature vector
        if fe.lips is None:
            print("File {} skipped, lips could not be found.".format(i))
            continue
        filters = MSA.multiscale_full(fe.lips[0].flatten('F'))
        differences = filters[1:] - filters[:-1]
        training_features[i] = np.sum(differences, 1)

        training_labels.append(label_dict_train[file_name])
    # print("File {} complete.".format(i))
    except:
        continue

# Crop training features, in case any faces could not be detected
training_features = training_features[:len(training_labels), ]
print("------{} seconds elapsed for feature extraction------".format(time.time()-start_time))

# Dump features from training data
pickle.dump(training_features, open('training_features.pk', 'wb'))


# Perform PCA on (normalized) training data
scaler = StandardScaler()
scaler.fit(training_features)
training_normalized = scaler.transform(training_features)
pca = PCA(0.95)
pca.fit(training_normalized)

pickle.dump(pca, open('pca.pk', 'wb'))

training_transformed = pca.transform(training_normalized)

pickle.dump(scaler, open('scaler.pk', 'wb'))


# Train logistic regression
logistic_regression = LogisticRegression(solver='lbfgs')
logistic_regression.fit(training_transformed, training_labels)
training_predictions = logistic_regression.predict(training_transformed)

pickle.dump( logistic_regression, open('logistic_regression.pk', 'wb'))


# Read test files
files = os.listdir('test_data')

test_labels = []
test_features = np.empty((len(files), 60))
for i, file in enumerate(files):
    # Get label for file
    try:
        file_name = file
    # Extract lip as 60x80 image
        file = os.path.join('test_data', file)
        fe = FeatureExtractor.from_image(file)
        fe.face_detect()
        fe.landmark_detect()
        fe.crop_lips()
        # print("File {} lips extracted.".format(i))

        # Convert lip image to 60-dimensional feature vector
        if fe.lips is None:
            print("File {} skipped, lips could not be found.".format(i))
            continue
        filters = MSA.multiscale_full(fe.lips[0].flatten('F'))
        differences = filters[1:] - filters[:-1]
        test_features[i] = np.sum(differences, 1)
        test_labels.append(label_dict_test[file_name])
    except:
        continue
        # print("File {} complete.".format(i))

# Crop training features, in case any faces could not be detected
test_features = test_features[:len(test_labels), ]

# Apply PCA mapping on (normalized) test data

test_normalized = scaler.transform(test_features)
test_transformed = pca.transform(test_normalized)

# Test logistic regression
testing_predictions = logistic_regression.predict(test_transformed)
total = 0
correct = 0
for i in range(len(testing_predictions)):
    if (testing_predictions[i] == test_labels[i]):
        correct = correct + 1
    total = total + 1
print(total)
print(correct)
