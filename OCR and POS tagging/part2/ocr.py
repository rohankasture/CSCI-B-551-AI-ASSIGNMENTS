#!/usr/bin/python
#
# ./ocr.py : Perform optical character recognition, usage:
#     ./ocr.py train-image-file.png train-text.txt test-image-file.png
# 
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Oct 2017)


#####
####Implementation Details####
#####

# Training data:
# 1) The courier-train.png image is used the training data for calculating the probability distribution for the
# pixels in the image.
# 2) The bc.train file (sample of the corpus as a approximate language model) is used for the calculation of the prior probabilities
# and transition probabilities. These are used in the Naive Bayes, Variable Elimination and Viterbi implementation.

# Approach:
# 1. For Simplified we have used Bayes law for each character (letter) in a sentence and predict the letter
#    by taking the maximum value of the probability.

# 2. For predicting the letters (using the Viterbi process) the following steps are implemented:
# Ingredients:

# 1) The Initialization Probabilities: The Model is started using a set of initialization probabilities(Probability of a particular character occuring).
# The initialization probabilities are set to be uniform.

# 2) The transition probabilities: The transition probabilities specify the chances to move from one state to the other state in the model.
# These are calculated from the training data (from the bc.train file of the 1st question) as P [ S(i+1) | S(i) ] where S = a valid
# letter in the train data.

# 3) The Emission Probabilities: The emission probabilities are calculated using a simple naive bayes approach.(p(pixel|leter)).
# The emisison probabilities are learned from the courier-train.png file. Each pixel will have a probabiltiy of emission from a
# given character.

# 4) The Dynamic Programming Approach. At each state of the HMM, the probabilities are calculated by multiplying three terms,

# A) Transition probability from the previous letter to current letter P(Si/Si-1): How often a letter is followed
# by other leters in the training file (bc.train).
# B) Emission probability of the current letter P(Wi/Si): Learned through the Naive Bayes.
# C) Probability calculated so far P(i-1)

# The Backtracking process
# The sequence of letters (in the test data) is returned by following the max probability path from last state to first state
# of the HMM.

# 3.The Variable Elimination Algorithm: The Variable elimination process is implemented using a forward and backward algorithm.
# References: https://www.cs.cmu.edu/~epxing/Class/10708-14/scribe_notes/scribe_note_lecture4.pdf

from PIL import Image, ImageDraw, ImageFont
import sys
import math

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25


def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    # print im.size
    # print int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [["".join(['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH)]) for y in
                    range(0, CHARACTER_HEIGHT)], ]
    return result


def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}


#####
# main program
(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
test_letters = load_letters(test_img_fname)
N = len(test_letters)


## Below is just some sample code to show you how the functions above work.
# You can delete them and put your own code here!


# Each training letter is now stored as a list of characters, where black
#  dots are represented by *'s and white dots are spaces. For example,
#  here's what "a" looks like:
# print "\n".join([r for r in train_letters['a']])

# Same with test letters. Here's what the third letter of the test data
#  looks like:
# print "\n".join([r for r in test_letters[2]])

def get_emission_p(img, char):
    template = train_letters[char]
    p = 1.0
    img_blk = 0.0
    img_wht = 0.0
    for r in range(3, len(img) - 1):
        for c in range(0, len(img[0])):
            if img[r][c] == '*':
                img_blk += 1
            else:
                img_wht += 1

    if img_blk == 0:
        factor = 999
    else:
        factor = (img_wht / img_blk)

    # ignoring the first row and the last row since its all blank for all the characters
    for r in range(3, len(img) - 1):
        for c in range(0, len(img[0])):
            if img[r][c] == '*' and template[r][c] == '*':
                p *= factor
            elif img[r][c] == ' ' and template[r][c] == ' ':
                pass
            elif img[r][c] == ' ' and template[r][c] == '*':
                p *= 1 / math.sqrt(factor)
            else:
                p *= 1 / factor
    return p


def get_emission_p_ve(img, char):
    template = train_letters[char]
    count = 0.0

    img_blk = 0.0
    img_wht = 0.0
    for r in range(3, len(img) - 1):
        for c in range(0, len(img[0])):
            if img[r][c] == '*':
                img_blk += 1
            else:
                img_wht += 1

    if img_blk == 0:
        factor = 999
    else:
        factor = (img_wht / img_blk)

    # ignoring first 3 rows and the last row since its all blank for all the characters
    for r in range(3, len(img) - 1):
        for c in range(0, len(img[0])):
            if img[r][c] == '*' and template[r][c] == '*':
                count += factor
            elif img[r][c] == ' ' and template[r][c] == ' ':
                count += 1
            elif img[r][c] == ' ' and template[r][c] == '*':
                count += 1 / math.sqrt(factor)
            else:
                count += 1 / factor
    return count


def simplified():
    seq = []
    for i in range(0, N):
        img = test_letters[i]
        p_img_matches_chars = {}
        for char in train_letters.keys():
            p_img_matches_chars[char] = get_emission_p(img, char)
        char_max_p = max(p_img_matches_chars, key=p_img_matches_chars.get)
        seq.append(char_max_p)
    print ' Simple: ' + ''.join([char for char in seq])


p_l1 = {}  # Probability of any letter to be first character
for char in list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'):
    p_l1[char] = 1.0 / 52
for char in list("0123456789(),.-!?\"' "):
    p_l1[char] = 0.0

# transition probabilities from li to li+1 P(li+1|li)
p_l_l1 = {}

Letters = "abcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
for c in Letters:
    p_l_l1[c] = {}
    for c1 in Letters:
        p_l_l1[c][c1] = 0.0


def read_data(fname):
    exemplars = []
    file = open(fname, 'r')
    for line in file:
        data = tuple([w.lower() for w in line.split()])
        exemplars += [(data[0::2])]
    file.close()
    return exemplars


data = read_data(train_txt_fname)

for list_str in data:
    line = ' '.join(word for word in list_str)
    n = len(line)
    for i in range(0, n - 1):
        char = line[i]
        char_next = line[i + 1]
        if char in Letters and char_next in Letters:
            p_l_l1[char][char_next] += 1

for c in p_l_l1.keys():
    sum_c = sum(p_l_l1[c].values())
    if sum_c == 0:
        continue
    for c1 in p_l_l1[c]:
        p_l_l1[c][c1] = p_l_l1[c][c1] / sum_c


def hmm_ve():
    alpha = {}
    beta = {}
    alpha[-1] = {}
    beta[N] = {}
    for char in train_letters.keys():
        alpha[-1][char] = 1.0
        beta[N][char] = 1.0

    # calculate alpha for i=0 to i=N-2. Leave the last img i=N-1
    for i in range(0, N - 1):
        img = test_letters[i]
        alpha[i] = {}
        for next_char in train_letters.keys():
            sum_p = 0.0
            for char in train_letters.keys():
                if i is 0:
                    a = p_l1
                else:
                    a = alpha[i - 1]
                ep = get_emission_p_ve(img, char)
                tp = p_l_l1[char.lower()][next_char.lower()]
                sum_p += a[char] * ep * tp
            alpha[i][next_char] = sum_p

    # calculate beta for i=0 to i=N-2. Leave the first img i=0
    for i in range(N - 1, 0, -1):
        img = test_letters[i]
        beta[i] = {}
        for char_prev in train_letters.keys():
            sum_p = 0.0
            for char in train_letters.keys():
                b = beta[i + 1]
                ep = get_emission_p_ve(img, char)
                tp = p_l_l1[char_prev.lower()][char.lower()]
                sum_p += b[char] * ep * tp
            beta[i][char_prev] = sum_p

    seq = []
    for i in range(0, N):
        img = test_letters[i]
        max = -1.0
        char_maxp = ' '
        for char in train_letters.keys():
            if i == 0:
                p = p_l1[char] * get_emission_p(img, char) * alpha[i - 1][char] * beta[i + 1][char]
            else:
                p = get_emission_p(img, char) * alpha[i - 1][char] * beta[i + 1][char]
            if p > max:
                max = p
                char_maxp = char
        seq.append(char_maxp)
    print ' HMM VE: ' + ''.join(char for char in seq)


def hmm_viterbi():
    space = []
    for i in range(0, N):
        img = test_letters[i]
        col = {}
        for char in train_letters.keys():
            node = Node(char)
            if i is 0:
                node.nlog_p = nlog(p_l1[char] * get_emission_p(img, char))
            else:
                for char_prev in train_letters.keys():
                    prev_node = space[i - 1][char_prev]
                    node.set_if_max_prob(prev_node, prev_node.nlog_p + \
                                         nlog(p_l_l1[char_prev.lower()][char.lower()] * get_emission_p(img, char)))

            col[char] = node
        space.append(col)

    last_col = space[N - 1]
    min_logp = +float('inf')
    min_logp_node = Node

    for char, node in last_col.iteritems():
        if node.nlog_p < min_logp:
            min_logp = node.nlog_p
            min_logp_node = node

    seq = []
    curr = min_logp_node
    while curr is not None:
        seq.append(curr.char)
        curr = curr.prev

    seq.reverse()
    print 'HMM MAP: ' + ''.join(char for char in seq)


def nlog(x):
    if x == 0.0:
        return +float('inf')
    else:
        return -1.0 * math.log(x)


class Node:
    def __init__(self, char):
        self.char = char
        self.nlog_p = +float("inf")
        self.prev = None

    def set_if_max_prob(self, node, other):
        # since we are storing negative log for the probability, we need to minimize
        if other < self.nlog_p:
            self.nlog_p = other
            self.prev = node


simplified()
hmm_viterbi()
print 'This might take a while!'
hmm_ve()
