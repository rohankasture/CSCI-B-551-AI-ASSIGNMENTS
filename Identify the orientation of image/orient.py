#!/usr/bin/python
from __future__ import division

from StringIO import StringIO

import numpy as np
from collections import Counter
from scipy.special import expit
import multiprocessing as mp
import pickle
import sys


####################################################
#############       KNN       ######################
####################################################

def test_func(test_image, image_id, the_label, num, kp, train_orient, train_features):
    index_of_k = np.argsort(np.sum(
        (1 - np.divide((np.minimum(train_features, test_image) + 1), (np.maximum(train_features, test_image) + 1))),
        axis=1))[:kp]
    k_labels = [train_orient[j] for j in index_of_k]
    res = str(num) + ":" + str(image_id) + ":" + the_label + ":" + str(Counter(k_labels).most_common(1)[0][0])
    # print "Label: ", the_label, " pred: ", str(Counter(k_labels).most_common(1)[0][0])
    return res


def train_knn(img_data):
    knn_model_file = open(model_file, "w")
    # Read the training data from the train file.
    for line in open(img_data, "r"):
        knn_model_file.write(line)
    knn_model_file.close()


def test_knn(img_data, model_file):
    # Output result file
    knn_outputfile = open("output.txt", "w")

    # Setup the basic data structures. I am using three seperate lists to hold image id, orientation and the feature data.
    photo_id_array = list()
    train_orient = list()
    train_features = []

    # List to store the test data
    test_features = list()
    test_labels = list()
    test_photo_id = list()

    # List to try different values of K.
    k_values = [55]

    # List to store the final results.
    final_result = []

    for line in open(model_file, "r"):
        temp_data_holder = line.split(" ")
        photo_id_array.append(temp_data_holder[0].strip())
        train_orient.append(temp_data_holder[1].strip())
        train_features.append(temp_data_holder[2:])
        # break

    # Slight speedup by np arrays
    train_features = np.asarray(train_features, dtype=float)

    # Read the test data here and store it in a list.
    for line in open(img_data, "r"):
        if line != "":
            test_features.append(line.split()[2:])
            test_labels.append(line.split()[1])
            test_photo_id.append(line.split()[0])

    test_features = np.asarray(test_features, dtype=float)

    # Setting up the processor here (this is only valid for the burrow system and so
    # if testing on another system, please change this number to 75% of the total number of cores.)
    pool = mp.Pool(processes=2)

    for k in k_values:
        correct = 0
        output = [pool.apply_async(test_func, args=(
            test_features[num], test_photo_id[num], test_labels[num], num, k, train_orient, train_features))
                  for num in range(0, len(test_features))]
        results = [p.get() for p in output]

        for elem in results:
            if elem.split(":")[2] == elem.split(":")[3]:
                correct = correct + 1
            knn_outputfile.write(elem.split(":")[1] + " " + elem.split(":")[3] + "\n")
        final_result.append("Prediction Accuracy is: " + str((correct / float(len(test_features)) * 100)) + " %")
    print " ".join(final_result)
    knn_outputfile.close()


####################################################
#############       NNET       #####################
####################################################


# Setup the basic data structures here, I am using three seperate lists to hold filename, orientation and the feature data for the training data set.
nnet_photo_id_array = list()
nnet_train_orient = list()
nnet_train_features = []

# Random weight parameters
a = np.random.uniform(low=-1, high=1, size=(192, 6))
b = np.random.uniform(low=-1, high=1, size=(6, 6))
g = np.random.uniform(low=-1, high=1, size=(6, 4))

weight_list = [a, b, g]
learned_weights = []

c1 = np.random.uniform(0, 1)
c2 = np.random.uniform(0, 1)
c3 = np.random.uniform(0, 1)
c4 = np.random.uniform(0, 1)
constant = (c1, c2, c3)
learned_weights = list()


def find_output_err(final_out, target):
    return target - final_out


# Test module for Nnet
def do_test(input, learned_weights):
    temp_list = list()
    temp_list.append(np.asarray(expit(np.dot(input, learned_weights[0])), dtype=float))

    for layer in range(1, len(learned_weights)):
        temp_list.append(np.asarray(expit(np.dot(temp_list[layer - 1], learned_weights[layer])), dtype=float))

    a1 = temp_list[len(temp_list) - 1]

    r = a1.argmax(axis=0)

    if r == 0:
        return 0
    if r == 1:
        return 90
    if r == 2:
        return 180
    if r == 3:
        return 270


def train_nnet(model_file):
    for i in range(0, len(nnet_train_features)):

        activation_list = list()
        activation_list.append(
            np.asarray(expit(np.dot(nnet_train_features[i], weight_list[0]) + constant[0]), dtype=float))

        for j in range(0, 2):

            for layer in range(1, len(weight_list)):
                activation_list.append(
                    np.asarray(expit(np.dot(activation_list[layer - 1], weight_list[layer]) + constant[layer]),
                               dtype=float))

        # First level backpropogation (from output to first hidden layer)
        final_prediction = activation_list[2]
        temp = find_output_err(final_prediction, np.asarray([0, 0.25, 0.35, 0.45], dtype=float))
        temp1 = final_prediction.dot(1 - final_prediction)
        u = -temp.dot(temp1)
        w = np.asmatrix(activation_list[1], dtype=float)
        y = (0.05) * np.transpose(w) * u
        weight_list[2] = np.asarray(weight_list[2] - y, dtype=float)

        # Second level backprop, (from 2nd hidden layer to first hiddedn layer)
        new_out = np.asarray(expit(np.dot(weight_list[2], activation_list[2]) + constant[1]), dtype=float)
        temp = find_output_err(activation_list[1], new_out)
        temp1 = activation_list[1].dot(1 - activation_list[1])
        u = -temp * temp1
        w = np.asmatrix(activation_list[1], dtype=float)
        y = (0.05) * np.transpose(w) * u
        weight_list[1] = np.asarray(weight_list[1] - y, dtype=float)

        # 3rd back prop
        new_out = np.asarray(expit(np.dot(weight_list[1], activation_list[1]) + constant[0]), dtype=float)
        temp = find_output_err(activation_list[0], new_out)
        temp1 = activation_list[0].dot(1 - activation_list[0])
        u = -temp.dot(temp1)
        w = np.asmatrix(nnet_train_features[i], dtype=float)
        y = (0.05) * np.transpose(w) * u
        weight_list[0] = np.asarray(weight_list[0] - y, dtype=float)

    with open(model_file, "wb") as we:  # Pickling
        pickle.dump(weight_list, we)

    # print "1 ",weight_list[0].shape


def test_nnet(img_data):
    correct = 0
    test_image = list()
    test_labels = list()
    test_features = list()

    nnet_output = open("output.txt", "w")

    with open(model_file, "rb") as we:  # Unpickling
        learned_weights = pickle.load(we)

    for line in open(img_data, "r"):
        test_features.append(line.split()[2:])
        test_labels.append(line.split()[1])
        test_image.append(line.split()[0])

    test_features = np.asarray(test_features, dtype=float)
    test_labels = np.asarray(test_labels)

    for i in range(0, len(test_features)):
        test_features[i] = test_features[i] / 255
        label = do_test(test_features[i], learned_weights)
        if label == int(test_labels[i].strip()):
            correct = correct + 1
        nnet_output.write(str(test_image[i]) + " " + str(label) + "\n")

    nnet_output.close()

    print "Prediction Accuracy is: ", correct / (float(len(test_features))) * 100, "%"


####################################################
#############       ADABOOST       #################
####################################################
import pickle
import math


class Img:
    def __init__(self, img_data):
        self.name = img_data[0]
        self.label = int(img_data[1])
        self.pixels = map(int, img_data[2:])
        self.wgt = 1.0

    def __repr__(self):
        return '{}_{}'.format(self.name, self.label)


blue_filter = {
    'h': {0: 1, 1: 1, 6: -1, 7: -1, 8: 1, 9: 1, 14: -1, 15: -1, 16: 1, 17: 1, 22: -1, 23: -1, 24: 1, 25: 1, 30: -1,
          31: -1, 32: 1, 33: 1, 38: -1, 39: -1, 40: 1, 41: 1, 46: -1, 47: -1, 48: 1, 49: 1, 54: -1, 55: -1, 56: 1,
          57: 1, 62: -1, 63: -1}, 'colors': 'b',
    'v': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 48: -1,
          49: -1, 50: -1, 51: -1, 52: -1, 53: -1, 54: -1, 55: -1, 56: -1, 57: -1, 58: -1, 59: -1, 60: -1, 61: -1,
          62: -1, 63: -1}}

light_filter = {
    'h': {0: 1, 1: 1, 2: 1, 5: -1, 6: -1, 7: -1, 8: 1, 9: 1, 10: 1, 13: -1, 14: -1, 15: -1, 16: 1, 17: 1, 18: 1, 21: -1,
          22: -1, 23: -1, 24: 1, 25: 1, 26: 1, 29: -1, 30: -1, 31: -1, 32: 1, 33: 1, 34: 1, 37: -1, 38: -1, 39: -1,
          40: 1, 41: 1, 42: 1, 45: -1, 46: -1, 47: -1, 48: 1, 49: 1, 50: 1, 53: -1, 54: -1, 55: -1, 56: 1, 57: 1, 58: 1,
          61: -1, 62: -1, 63: -1}, 'colors': 'rgb',
    'v': {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1, 10: 1, 11: 1, 12: 1, 13: 1, 14: 1, 15: 1, 16: 1,
          17: 1, 18: 1, 19: 1, 20: 1, 21: 1, 22: 1, 23: 1, 40: -1, 41: -1, 42: -1, 43: -1, 44: -1, 45: -1, 46: -1,
          47: -1, 48: -1, 49: -1, 50: -1, 51: -1, 52: -1, 53: -1, 54: -1, 55: -1, 56: -1, 57: -1, 58: -1, 59: -1,
          60: -1, 61: -1, 62: -1, 63: -1}}

train_pile = []
features = []
classifiers = {}
fs = []
alphas = []


def classify_filter(img, f):
    score_v = 0.0
    sum_v = 0.0
    score_h = 0.0
    sum_h = 0.0

    for cell, sign in f['v'].iteritems():
        if 'r' in f['colors']:
            score_v += img.pixels[3 * cell] * sign
            sum_v += img.pixels[3 * cell]
        if 'g' in f['colors']:
            score_v += img.pixels[3 * cell + 1] * sign
            sum_v += img.pixels[3 * cell + 1]
        if 'b' in f['colors']:
            score_v += img.pixels[3 * cell + 2] * sign
            sum_v += img.pixels[3 * cell + 2]

    for cell, sign in f['h'].iteritems():
        if 'r' in f['colors']:
            score_h += img.pixels[3 * cell] * sign
            sum_h += img.pixels[3 * cell]
        if 'g' in f['colors']:
            score_h += img.pixels[3 * cell + 1] * sign
            sum_h += img.pixels[3 * cell + 1]
        if 'b' in f['colors']:
            score_h += img.pixels[3 * cell + 2] * sign
            sum_h += img.pixels[3 * cell + 2]

    p_vert = abs(score_v / sum_v)
    p_horz = abs(score_h / sum_h)

    if p_vert >= p_horz:
        if score_v >= 0:
            return 0
        else:
            return 180
    else:
        if score_h >= 0:
            return 270
        else:
            return 90


def classify(img, f):
    global classifiers

    if f == 'blue_filter':
        return classify_filter(img, blue_filter)
    if f == 'light_filter':
        return classify_filter(img, light_filter)

    px1 = f[0]
    px2 = f[1]

    return max(classifiers[px1][px2], key=classifiers[px1][px2].get) if img.pixels[px1] < img.pixels[px2] \
        else min(classifiers[px1][px2], key=classifiers[px1][px2].get)


def generate_classifiers():
    global classifiers, features, train_pile
    for px1 in range(0, 192):
        for px2 in range(0, 192):
            if px1 < px2:
                features.append([px1, px2])

    # initializing classifiers for every pixel pair
    for px1_px2 in features:
        px1 = px1_px2[0]
        px2 = px1_px2[1]
        if px1 not in classifiers:
            classifiers[px1] = {}
        if px1 < px2:
            classifiers[px1][px2] = {0: 0, 90: 0, 180: 0, 270: 0}

    # looking for classifiers
    for img in train_pile:
        for pair in features:
            px1 = pair[0]
            px2 = pair[1]
            if img.pixels[px1] < img.pixels[px2]:
                classifiers[px1][px2][img.label] += 1

    features.append('blue_filter')
    features.append('light_filter')


# https://web.stanford.edu/~hastie/Papers/samme.pdf
def run_adaboost():
    global features, train_pile, fs, alphas

    T = 20
    for t in range(0, T):
        # normalize weights
        total_wgt = sum(img.wgt for img in train_pile)
        for img in train_pile:
            img.wgt = img.wgt / total_wgt

        least_e = float('inf')
        f_least_e = None

        for f in features:
            e = 0.0
            for img in train_pile:
                if classify(img, f) != img.label:
                    e += img.wgt
            if e < least_e:
                least_e = e
                f_least_e = f

        alpha = math.log((1 - least_e) * 3 / least_e)

        for img in train_pile:
            if img.label != classify(img, f_least_e):
                img.wgt = img.wgt * math.exp(alpha)

        fs.append(f_least_e)
        alphas.append(alpha)


def train_adaboost(train_file, model_txt):
    global classifiers, features, train_pile, fs, alphas
    train_data = open(train_file, "r")
    for line in train_data:
        img_data = line.strip().split()
        img = Img(img_data)
        train_pile.append(img)
    train_data.close()

    generate_classifiers()
    run_adaboost()

    # writing the classifiers into adaboost_model.txt
    model = {'classifiers': classifiers, 'fs': fs, 'alphas': alphas}
    model_file = open(model_txt, "wb")
    pickle.dump(model, model_file)
    model_file.close()


def test_adaboost(test_file, model_txt):
    global classifiers, fs, alphas
    model_file = open(model_txt, "rb")
    model = pickle.load(model_file)
    classifiers = model['classifiers']
    fs = model['fs']
    alphas = model['alphas']

    test_data = open(test_file, "r")
    test_pile = []

    for line in test_data:
        img_data = line.strip().split()
        img = Img(img_data)
        test_pile.append(img)

    output_str = StringIO()
    correct = 0
    for img in test_pile:
        label_scores = {0: 0.0, 90: 0.0, 180: 0.0, 270: 0.0}

        for alpha, f in zip(alphas, fs):
            label_scores[classify(img, f)] += alpha

        predicted_label = max(label_scores, key=label_scores.get)
        output_str.write('{} {}\n'.format(img.name, predicted_label))
        if predicted_label == img.label:
            correct += 1

    with open('output.txt', 'w') as f:
        f.write(output_str.getvalue())
    print 'Accuracy: {} %'.format(correct * 100.0 / len(test_pile))


mode = sys.argv[1]  # train or test
img_data = sys.argv[2]  # train-img-data or test-img-data
model_file = sys.argv[3]  # model file to read or write
model = sys.argv[4]  # which model?

if mode == 'train':
    if model == 'nearest':
        train_knn(img_data)
    elif model == 'adaboost':
        train_adaboost(img_data, model_file)
    elif model == 'nnet':
        train_nnet(model_file)
    elif model == 'best':
        train_knn(img_data)
    else:
        print "Invalid model name. Possible values are 'nearest','adaboost','nnet' and 'best'"
elif mode == 'test':
    if model == 'nearest':
        test_knn(img_data, model_file)
    elif model == 'adaboost':
        test_adaboost(img_data, model_file)
    elif model == 'nnet':
        test_nnet(img_data)
    elif model == 'best':
        test_knn(img_data, model_file)
    else:
        print "Invalid model name. Possible values are 'nearest','adaboost','nnet' and 'best'"
else:
    print "Invalid mode name. Possible values are 'train' and 'test'"
