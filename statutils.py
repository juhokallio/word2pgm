#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import math
import gensim
import numpy as np
import nsphere
from scipy.stats import truncnorm

def get_average_variance(dimensions, samples=20):
    test_vectors_1 = [gensim.matutils.unitvec(v) for v in np.random.rand(samples, dimensions)]
    test_vectors_2 = [gensim.matutils.unitvec(v) for v in np.random.rand(samples, dimensions)]
    total_distance = 0
    for v1 in test_vectors_1:
        for v2 in test_vectors_2:
            s = np.dot(v1, v2)
            total_distance += distance(dimensions, s)
    return total_distance / (samples ** 2)

def distance(dimensions, cosine_similarity):
    return nsphere.cap(1, dimensions, math.acos(cosine_similarity))

def cosine_similarity(v1, v2):
    s = np.dot(gensim.matutils.unitvec(v1), gensim.matutils.unitvec(v2))
    if s > 1:
        return 1
    elif s < -1:
        return -1
    else:
        return s

def get_evidence_variance(predictions, means, iterations=40):
    dimensions = len(predictions[0])
    predictions = np.concatenate((
        predictions,
        [[-1 * v for v in pred] for pred in predictions]
        ))

    def calculate_prob(variance):
        dist = truncnorm(0, np.inf, 0, variance)
        p = 0
        for pred in predictions:
            pred_p = 0
            for mean in means:
                s = cosine_similarity(pred, mean)
                mean_p = dist.logpdf(distance(dimensions, s))
                if mean_p > pred_p:
                    pred_p = mean_p
            p += pred_p
        return p

    min_variance = 0
    max_variance = 1#get_average_variance(dimensions)
    for i in range(iterations):
        variance_1 = min_variance + (max_variance - min_variance) * 0.33
        variance_2 = min_variance + (max_variance - min_variance) * 0.66
        p_1 = calculate_prob(variance_1)
        p_2 = calculate_prob(variance_2)
        if p_1 > p_2:
            max_variance = variance_2
            print("p {}, new max: {} - {}".format(p_1, min_variance, max_variance))
        else:
            min_variance = variance_1
            print("p {}, new min: {} - {}".format(p_2, min_variance, max_variance))
    return min_variance + (max_variance - min_variance) * 0.5

class StatUtilTest(unittest.TestCase):

    def test_get_average_variance(self):
        self.assertTrue(get_average_variance(10) > 0,
                msg="Average variance wasn't above zero")

    def test_get_evidence_variance(self):
        predictions = np.random.rand(20, 20)
        variance = get_evidence_variance(predictions, predictions)
        self.assertAlmostEqual(variance, 0, delta=0.0001,
                msg="Variance wasn't close to 0 when predictions were all exactly means")
        random_variance = get_evidence_variance(predictions, np.random.rand(20, 20))
        self.assertTrue(random_variance > variance,
                msg="Variance wasn't larger with random means than means that were exactly predictions")
