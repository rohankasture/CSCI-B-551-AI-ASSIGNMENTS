#!/usr/bin/python
import sys
from collections import defaultdict
from collections import Counter
import math
import re
import numpy as np

#####################General execution flow of the program and description of the algorithm###############
# The following steps are performed in the respective order with the help of functions for each step.
#
# 1) Read the training and testing data from the file on the disk. Perform data cleaning while reading the data. Detailed explanation is given before the function.
#
# 2) Calculate the basic elements for probability determination.
#         2. a) How many words are present in a given  city.
#         2. b) How many city in the training data. Each city is divided by the total number of cities to find the
#               probability of occurence of that city.
#         2.c) To find the probability of the word given the city, occurrence of the given word is divided by the total number of words in that
#                 city (from step 2.a) plus the total number of unique words in the training data.
#
#  3) The terms from 2.b and 2.c are used as follows in the bayes formula for prediction the posterior probability of the city given the word,
#
#         Formula:  P(C|w) =  P(w|C) * P(C) / P(w),  C = city, w = word (from Baye's law)
#
#         Where, P(w|C) = from step 2.c above
#                P(C) = from step 2.b above
#                Since, we don't need the denominator because the probability of the words is constant for the given training dataset. So, we are
#                only maximising over the numerator for all the tweets and then seelcting the maximum from that.

#Design Decisions:

# 1) Using Laplacian Smoothing: When a word from a tweet in the test data set is absent in the train data set (for a given city).
#     it's probability will be calculated by using 1 in the numerator.' Alternatively, I assume that every word is present in the every city
#     at-least once. This is equivalen to uniform distribution of the words across the different cities.
#Note: If I ignore the laplacian smoothing, the prediction accuracy decreases by around 10-12 %. So, I am keeping the smoothing in place.

#2) Ignoring the special characters/words. (Effect on the memory and execution speed)
#Since, I need to keep track of all the words belonging to a given city (in the training data set) so removing words
# optimises the size of the data structure. This also means, that I have relatively less words to iterate while making the prediciton. This
# keeps the size of the data manageable and the execution speed reasonable.

# References:
# While implementing the naive baye's algorithm, I have referred the following resources.
# https://web.stanford.edu/class/cs124/lec/naivebayes.pdf
# http://ieeexplore.ieee.org/document/4223081/
#https://nlp.stanford.edu/IR-book/html/htmledition/naive-bayes-text-classification-1.html
##############################################################################################################

#Declaring the required data structures below, the details of important to this implementation are given below:
#prior_labels  will have the probability of each city in the format: city1: number, city2:number ...
#training_data_words contains all the unique words form the training data. It's a set to prevent repetetion of words.
#cond_prob_table will contain words for each city in the format city1: word1,word2 ....

prior_labels = []
training_data_words = set([])
cond_prob_table = defaultdict(list)
feature_in_label= {}
feature_count_in_label = {}
final_list = [[]]
test_data = []
out_data = []
total_records = 0

#Getting the input from the command line
train_file_path =  sys.argv[1]
test_file_path = sys.argv[2]
out_file_path = sys.argv[3]

out_file = open(out_file_path,"w")

#Compiling the regular expression for removing special characters and setting a static set of stop words.
remove_special = re.compile("[^a-zA-Z]")
stopwords = set(['all', 'just', 'being', 'over', 'both', 'through', 'yourselves','its', 'before', 'herself', 'had', 'should', 'to', 'only', 'under', 'ours', 'has', 'do', 'them', 'his', 'very', 'they', 'not', 'during', 'now', 'him', 'nor', 'did', 'this', 'she', 'each', 'further', 'where', 'few', 'because', 'doing', 'some', 'are', 'our', 'ourselves', 'out', 'what', 'for', 'while', 'does', 'above', 'between', 't', 'be', 'we', 'who', 'were', 'here', 'hers', 'by', 'on', 'about', 'of', 'against', 's', 'or', 'own', 'into', 'yourself', 'down', 'your', 'from', 'her', 'their', 'there', 'been', 'whom', 'too', 'themselves', 'was', 'until', 'more', 'himself', 'that', 'but', 'don', 'with', 'than', 'those', 'he', 'me', 'myself', 'these', 'up', 'will', 'below', 'can', 'theirs', 'my', 'and', 'then', 'is', 'am', 'it', 'an', 'as', 'itself', 'at', 'have', 'in', 'any', 'if', 'again', 'no', 'when', 'same', 'how', 'other', 'which', 'you', 'after', 'most', 'such', 'why', 'a', 'off', 'i', 'yours', 'so', 'the', 'having', 'once','','a','about','above','across','after','afterwards','again','against','all','almost','alone','along','already','also','although','always','am','among','amongst','amoungst','amount','an','and','another','any','anyhow','anyone','anything','anyway','anywhere','are','around','as','at','back','be','became','a','about','above','across','after','afterwards','again','against','all','almost','alone','along','already','also','although','always','am','among','amongst','amoungst','amount','an','and','another','any','anyhow','anyone','anything','anyway','anywhere','are','around','as','at','back','be','became','because','become','becomes','becoming','been','before','beforehand','behind','being','below','beside','besides','between','beyond','bill','both','bottom','but','by','call','can','cannot','cant','co','computer','con','could','couldnt','cry','de','describe','detail','do','done','down','due','during','each','eg','eight','either','eleven','else','elsewhere','empty','enough','etc','even','ever','every','everyone','everything','everywhere','except','few','fifteen','fify','fill','find','fire','first','five','for','former','formerly','forty','found','four','from','front','full','further','get','give','go','had','has','hasnt','have','he','hence','her','here','hereafter','hereby','herein','hereupon','hers','herse"','him','himse"','his','how','however','hundred','i','ie','if','in','inc','indeed','interest','into','is','it','its','itse"','keep','last','latter','latterly','least','less','ltd','made','many','may','me','meanwhile','might','mill','mine','more','moreover','most','mostly','move','much','must','my','myse"','name','namely','neither','never','nevertheless','next','nine','no','nobody','none','noone','nor','not','nothing','now','nowhere','of','off','often','on','once','one','only','onto','or','other','others','otherwise','our','ours','ourselves','out','over','own','part','per','perhaps','please','put','rather','re','same','see','seem','seemed','seeming','seems','serious','several','she','should','show','side','since','sincere','six','sixty','so','some','somehow','someone','something','sometime','sometimes','somewhere','still','such','system','take','ten','than','that','the','their','them','themselves','then','thence','there','thereafter','thereby','therefore','therein','thereupon','these','they','thick','thin','third','this','those','though','three','through','throughout','thru','thus','to','together','too','top','toward','towards','twelve','twenty','two','un','under','until','up','upon','us','very','via','was','we','well','were','what','whatever','when','whence','whenever','where','whereafter','whereas','whereby','wherein','whereupon','wherever','whether','which','while','whither','who','whoever','whole','whom','whose','why','will','with','within','without','would','yet','you','your','yours','yourself','yourselves'])

#Reading the training data below. The words present in stopwords set are ignored and special characters are ignored too using the regular expression above.
train_data = open(train_file_path,"r")
for line in train_data:
    temp = line.split(" ")
    if temp[0] != "":
        prior_labels.append(temp[0].strip())
        for word in temp[1:]:
            word = remove_special.sub("", word.lower().strip())
            if word not in stopwords:
                cond_prob_table[temp[0]].append(word)
            training_data_words.add(word)
    total_records = total_records + 1

prior_labels = Counter(prior_labels)
prior_labels = {e:prior_labels[e]/float(sum(prior_labels.values())) for e in prior_labels.keys() if e != ""}

#Reading in the test data.
testdata = open(test_file_path,"r")
for line in testdata:
    out_data.append(line.strip())
    temp = line.split(" ")
    temp1 = [w for w in temp]
    if temp1 != ['']:
        test_data.append(temp1[0:1]+[remove_special.sub("",word.lower().strip()) for word in temp1[1:] if word.lower() not in stopwords])

#Now initializing the conditional probability table for storing all the words for a given city. The format is explained above in the declaration section.
for labels in cond_prob_table.keys():
    s= ""
    feature_in_label[labels] = Counter(cond_prob_table[labels]).keys()
    feature_count_in_label[labels] = Counter(cond_prob_table[labels]).values()
    temp = feature_count_in_label[labels]
    #arr = np.array(temp)
    t = np.argsort(temp)[-5:][::-1]
    for elem in t:
        s = s +" "+ feature_in_label[labels][elem]
    print labels, ": ", s

#print training_data_words
#print prior_labels
#print cond_prob_table
#print test_data
#print feature_in_label
#print feature_count_in_label

total_word_count = len(training_data_words)

#Prediction: for each tweet in the test data, weight contributed by all the word in that tweet towards each city is calculated.
# The weight is then multiplied by the probability of occurrence of each city to find the posterior probability of that tweet belonging to a each city.
# Out of all the calculated probabilities, the city with the maximum probability is assigned to the tweet.
for record in range(len(test_data)-1):
    test_words = test_data[record]
    max_prob_list = {}

    for labels in feature_in_label.keys():

        word_count = len(cond_prob_table[labels])
        words_in_current_label = feature_in_label[labels]
        temp_prob = 0
        total_prob = 0

        for word in test_words[1:]:
            current_word_count = 0
            if word in words_in_current_label and word !='':
                word_index = words_in_current_label.index(word)
                current_word_count = feature_count_in_label[labels][word_index]
                word_prob = current_word_count+1/float(word_count + total_word_count)
            else:
                word_prob = 1 / float(word_count + total_word_count)

            word_prob = math.log10(word_prob)
            temp_prob = temp_prob + word_prob

        total_prob = math.log10(prior_labels[labels]) + temp_prob
        max_prob_list[labels] = total_prob

    c = max(max_prob_list, key=max_prob_list.get)
    final_list.append([" ".join(x for x in test_words[1:]) , test_words[0], c, max_prob_list[c]])
    out_file.write(c+" "+out_data[record]+"\n")

correct = 0
final_list = [elem for elem in final_list if elem != []]

#Calculating the accuracy as the percentage fraction of correct prediction out of total tweets in the test data.
for elem in final_list:
    if elem != [] and elem[0] != "":
        if elem[1] == elem[2]:
            correct = correct + 1

print "Prediction accuracy is: ",correct*100/float(len(final_list)), " length is: ", len(final_list)
out_file.close()
train_data.close()
testdata.close()