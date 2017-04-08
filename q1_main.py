from __future__ import division
from NaiveBayes import NB
import numpy as np

"""
Script for testing NaiveBayes.py
"""

data = []
with open("trainingData.txt") as file:
    for line in file:
        data.append(line.split())
train_data = np.array(data,int)

data = []
with open("testingData.txt") as file:
    for line in file:
        data.append(line.split())
test_data = np.array(data,int)

nb = NB(train_data)
tp = 0
tn = 0
fp = 0
fn = 0
for row_idx in range(len(test_data[:, 0])):
    cls_zero_probability, cls_one_probability = nb.estimate_class(test_data[row_idx,1:])
    if cls_zero_probability > cls_one_probability:
        estimated_class = 0
        if test_data[row_idx,0] == 0:
            tn += 1
        else:
            fn += 1
    else:
        estimated_class =1
        if test_data[row_idx, 0] == 0:
            fp += 1
        else:
            tp += 1

print "accuracy: %s " % ((tn+tp)/100)
recall = tp / (tp+fn)
precision = tp / (tp+fp)
print "f1-score: %s " % ((2*recall*precision) / (recall+precision))
