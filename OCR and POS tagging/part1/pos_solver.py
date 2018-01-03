###################################
# CS B551 Fall 2017, Assignment #3
#
# Your names and user ids: hrakholi, rkasture
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
#Training data:
#Approach:
# 1. For Simplified we have used Bayes law for each part of speech for a particular word in a sentence and found the final part of
#    speech for that word by taking the maximum value of the probability
# 2. For predicting the labels (using the Viterbi process) the following steps are implemented:
#Ingredients:
#1) The Initialization Probabilities: The Model is started using a set of initialization probabilities(Probability of a particular part of speech occuring at first word)
#2) The transition probabilities: The transition probabilities specify the chances to move from one part of speech to the other state in the model.
#   These are calculated from the training data (from the bc.train file of the 1st question) as P [ S(i+1) | S(i) ] where S = part of speech in sentence
# 3) The Emission Probabilities: The emission probabilities are calculated using a simple naive bayes approach.(p(wi|si))
#4) The Dynamic Programming Approach. At each state of the HMM, the probabilities are calculated by multiplying three terms,
#A) Transition probability from the previous pos to current pos P(Si/Si-1)
#B) Emission probability of the current pos P(Wi/Si)
#C) Probability from calculated so far P(i-1)
# The sequence of part of speech is returned by following the max probability path from last state to first state

#3.The Variable Elimination Algorithm: The Variable elimination process is implemented using a forward and backward algorithm.
#References: https://www.cs.cmu.edu/~epxing/Class/10708-14/scribe_notes/scribe_note_lecture4.pdf
#Train data: bc.train
#Test data: bc.test
#==> So far scored 2000 sentences with 29442 words.
#                   Words correct:     Sentences correct:
#   0. Ground truth:      100.00%              100.00%
#   1. Simplified:        93.92%               47.45%
#       2. HMM VE:        95.08%               54.35%
#       3. HMM MAP:       95.06%               54.45%
####

import random
import math
import copy

# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#

POS = ["x", "adp", "det", "noun", "adj", "verb", ".", "conj", "prt", "adv", "num", "pron"]
s_w1 = {}  # Probability of any POS for the first word P(S|W1)
s_s1 = {}  # transition probabilities from Si to Si+1 P(Si+1|Si)
w_s = {}  # emission probabilities P(Wi|Si)
total_words = 0  # Total no. of words
count_pos = {}  # count of occurrences of each POS


def get_emission_prob(word, pos):
    # return w_s[pos][word] if word in w_s[pos] else 1.0 / total_words
    return w_s[pos][word] if word in w_s[pos] else 1e-10


def nlog(x):
    if x == 0.0:
        return +float('inf')
    else:
        return -1.0 * math.log(x)


class Node:
    def __init__(self, pos):
        self.pos = pos
        self.nlog_p = +float("inf")
        self.prev = None

    def set_if_max_prob(self, node, other):
        # since we are storing negative log for the probability, we need to minimize
        if other < self.nlog_p:
            self.nlog_p = other
            self.prev = node

    def __str__(self):
        return '%.2f' % self.nlog_p

    def __repr__(self):
        return '%.2f' % self.nlog_p


class Solver:
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling
    def posterior(self, sentence, label):
        x = 1.0
        random_prob = 0.0000001
        for i in range(0,len(label)):
                if i == 0:
                    x *= w_s[label[i]].get(sentence[i],random_prob) * s_w1[label[i]]
                else:
                    x *= w_s[label[i]].get(sentence[i],random_prob)* s_s1[label[i-1]].get(label[i],random_prob)
        return math.log(x)

    # Do the training!
    # Here data is list of 2 tuples
    # 1st tuple contains all the words of a sentence and 2nd tuple contains all the POS tags of the words
    # W represents observed words in sentence and S represents POS tags
    def train(self, data):

        # Calculate P(S1)

        for s in POS:
            s_w1[s] = 0.0
            s_s1[s] = {}
            for s1 in POS:
                s_s1[s][s1] = 0.0
            w_s[s] = {}

        # to calculate the count where POS occurs as the first word of sentence
        for line in data:
            s_w1[line[1][0]] += 1.0

        tot_words = sum(s_w1.values())

        for s in s_w1.keys():
            s_w1[s] = s_w1[s] / tot_words

        # Calculate P(Si+1|Si)
        for line in data:
            for i in range(1, len(line[1])):
                s_s1[line[1][i - 1]][line[1][i]] += 1.0

        # Here keys will be all POS
        for s in s_s1.keys():
            key_sum = sum(s_s1[s].values())
            for s1 in s_s1[s]:  # It will consider all combinations like [noun,verb],[noun,adj],[noun,adv]
                s_s1[s][s1] = (s_s1[s][s1]) / key_sum  # All POS that come after a specific POS

        # Calculate P(Wi|Si)
        # w_s contains all keys as POS and words occuring as that POS
        for line in data:
            for i in range(0, len(line[0])):
                if line[0][i] not in w_s[line[1][i]]:  # line[0][i] will be word and line[1][i] will be its POS tag
                    w_s[line[1][i]][line[0][i]] = 1.0
                else:
                    w_s[line[1][i]][line[0][i]] += 1.0

        for pos in POS:
            pos_sum = sum(w_s[pos].values())
            count_pos[pos] = 1.0 if pos_sum == 0.0 else pos_sum

        global total_words
        total_words = sum([sum(w_s[row].values()) for row in w_s])
        # Here keys will be all POS
        for key in w_s.keys():
            key_sum = sum(w_s[key].values())
            for pos in w_s[key]:  # It will consider all combinations like [noun,verb],[noun,adj],[noun,adv]
                w_s[key][pos] = w_s[key][pos] / key_sum  # All POS that come after a specific POS

    # Calculate probability of POS given word
    # p(s|w) = (p(w|s)*p(s))/ p(w)
    def get_word_tag(self, word):
        s_w = []
        for parts_of_speech in POS:
            w_s = get_emission_prob(word, parts_of_speech)
            s = (count_pos[parts_of_speech] * 1.0) / total_words
            s_w.append([parts_of_speech, (w_s * s)])
        max_row = 0
        max_probability = s_w[0][1]
        for row in range(0, len(s_w)):
            if s_w[row][1] > max_probability:
                max_probability = s_w[row][1]
                max_row = row
        return s_w[max_row]

    # Functions for each algorithm.
    #
    def simplified(self, sentence):
        result = []
        for i in range(0, len(sentence)):
            pos, prob = self.get_word_tag(sentence[i])
            result.append(pos)
        return result

    # estimating sequence of parts of speech using forward backward variable elimination on HMM
    # based on this source: https://www.cs.cmu.edu/~epxing/Class/10708-14/scribe_notes/scribe_note_lecture4.pdf
    def hmm_ve(self, sentence):
        N = len(sentence)
        alpha = {}
        beta = {}
        alpha[-1] = {}
        beta[N] = {}
        for pos in POS:
            alpha[-1][pos] = 1.0
            beta[N][pos] = 1.0

        # calculate alpha for i=0 to i=N-2. Leave the last word i=N-1
        for i in range(0, N - 1):
            word = sentence[i]
            alpha[i] = {}
            for s_next in POS:
                sum_s = 0.0
                for s in POS:
                    if i is 0:
                        a = s_w1
                    else:
                        a = alpha[i - 1]
                    ep = get_emission_prob(word, s)
                    tp = s_s1[s][s_next]
                    sum_s += a[s] * ep * tp
                alpha[i][s_next] = sum_s

        # calculate beta for i=0 to i=N-2. Leave the first word i=0
        for i in range(N - 1, 0, -1):
            word = sentence[i]
            beta[i] = {}
            for s_prev in POS:
                sum_s = 0.0
                for s in POS:
                    b = beta[i + 1]
                    ep = get_emission_prob(word, s)
                    tp = s_s1[s_prev][s]
                    sum_s += b[s] * ep * tp
                beta[i][s_prev] = sum_s

        seq = []
        for i in range(0, N):
            word = sentence[i]
            max = -1.0
            s_maxp = "noun"
            for s in POS:
                if i == 0:
                    p = s_w1[s] * get_emission_prob(word, s) * alpha[i - 1][s] * beta[i + 1][s]
                else:
                    p = get_emission_prob(word, s) * alpha[i - 1][s] * beta[i + 1][s]
                if p > max:
                    max = p
                    s_maxp = s
            seq.append(s_maxp)
        return seq

    def hmm_viterbi(self, sentence):
        N = len(sentence)
        space = []
        for i in range(0, N):
            word = sentence[i]
            col = {}
            for s in POS:
                node = Node(s)
                if i is 0:
                    node.nlog_p = nlog(s_w1[s] * get_emission_prob(word, s))
                else:
                    for s_minus_1 in POS:
                        prev_node = space[i - 1][s_minus_1]
                        node.set_if_max_prob(prev_node, prev_node.nlog_p + \
                                             nlog(s_s1[s_minus_1][s] * get_emission_prob(word, s)))

                col[s] = node
            space.append(col)

        last_col = space[N - 1]
        min_logp = +float('inf')
        min_logp_node = Node

        for s, node in last_col.iteritems():
            if node.nlog_p < min_logp:
                min_logp = node.nlog_p
                min_logp_node = node

        seq = []
        curr = min_logp_node
        while curr is not None:
            seq.append(curr.pos)
            curr = curr.prev

        seq.reverse()
        return seq

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, algo, sentence):
        if algo == "Simplified":
            return self.simplified(sentence)
        elif algo == "HMM VE":
            return self.hmm_ve(sentence)
        elif algo == "HMM MAP":
            return self.hmm_viterbi(sentence)
        else:
            print "Unknown algo!"
