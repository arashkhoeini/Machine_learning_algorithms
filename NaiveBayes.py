from __future__ import division
import numpy as np


class NB(object):
    """
    Class that contains Naive Bayes algorithm.
    This class needs to get data as a Numpy array.
    First column is considered to be sample class name. And class values are 0 or 1
    """

    def __init__(self, data):
        self.data = data
        self.feature_values = self._find_feature_values()
        self.p_table = self._build_probability_table()  # p_table[class][feature_name][feature_value]

    #This method finds all possible values for each feature and returns a list of list
    def _find_feature_values(self):
        feature_values = []
        for f in range(1, len(self.data[0, :])):
            values = set(self.data[:, f])
            feature_values.append(list(values))
        return feature_values
    #Most important part of this algorithm. This methods calculate all probabilities and store it on a
    #3d list and returns it. The 3d array arguments order will be like this: p_table[class][feature_name][feature_value]
    def _build_probability_table(self):
        self.cls_ones_count = np.sum(self.data[:, 0].astype(int))
        self.cls_zero_count = len(self.data[:, 0]) - self.cls_ones_count

        temp = []
        for fvs in self.feature_values:
            temp.append(len(fvs))
        p_table = [[[0 for x in range(max(temp))] for y in range(len(self.data[0, :]) - 1)] for z in range(2)]
        for cls in range(0, 2):
            for f in range(1, len(self.data[0, :])):
                for fv in self.feature_values[f - 1]:
                    x = 0
                    for row_idx in range(0, len(self.data[:, 0])):
                        if self.data[row_idx, 0] == cls:
                            if self.data[row_idx, f] == fv:
                                x += 1
                    if cls == 0:
                        p_table[0][f - 1][self.feature_values[f - 1].index(fv)] = (x + 1) / (self.cls_zero_count + 1)
                    else:
                        p_table[1][f - 1][self.feature_values[f - 1].index(fv)] = (x + 1) / (self.cls_ones_count + 1)
        return p_table

    #Method for estimating class of an sample. sample is an list of feature values. values must have
    #the same order that they had in training. Only class feature is omitted.
    #This method returns a tuple: (cls_zero_probability, cls_one_probability)
    def estimate_class(self, sample):
        cls_zero_likelihood = 1
        cls_one_likelihood = 1
        for f in range(len(sample)):
            cls_zero_likelihood *= self.p_table[0][f][self.feature_values[f].index(sample[f])]
            cls_one_likelihood *= self.p_table[1][f][self.feature_values[f].index(sample[f])]
        cls_zero_probability = cls_zero_likelihood * self.cls_zero_count
        cls_one_probability = cls_one_likelihood * self.cls_ones_count

        return cls_zero_probability, cls_one_probability
