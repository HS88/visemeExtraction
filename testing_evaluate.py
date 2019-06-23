from FeatureExtractor import FeatureExtractor
import MSA

import numpy as np
import os
import pickle
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import time

pca = pickle.load(open('pca.pk','rb'))
scaler = pickle.load(open('scaler.pk', 'rb'))
logistic_regression = pickle.load( open( "logistic_regression.pk", "rb" ) )


label_dict_test = {}
files = os.listdir('test_data')

test_label_file = open(os.path.join('label_data', 'test_complete_file.txt'), 'r')
for line in test_label_file.readlines():
    line_input = line.split(" ")
    label_dict_test[line_input[0]] = line_input[2].rstrip()
test_label_file.close()

test_labels = []
test_features = np.empty((len(files), 60))
for i, file in enumerate(files):
    # Get label for file
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


