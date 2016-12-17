#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
import pdb

class AnnModel:

    def __init__(self, vector_size, look_back=10, lstm_size=50, lstm_count=2):
        self.look_back = look_back
        self.word_vector_size = vector_size
        self.model = Sequential()
        self.model.add(LSTM(lstm_size, input_dim=self.word_vector_size, input_length=look_back, return_sequences=True))
        for i in range(0, lstm_count - 2):
            self.model.add(LSTM(lstm_size, return_sequences=True))
        self.model.add(LSTM(lstm_size))
        self.model.add(Dense(self.word_vector_size))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def create_training_data(self, vectors):
        dataX, dataY = [], []
        for i, vector in enumerate(vectors):
            sample = np.zeros((self.look_back, self.word_vector_size))
            for j in range(self.look_back):
                vector_index = i - self.look_back + j
                if vector_index >= 0:
                    sample[-j] = vectors[vector_index]
            dataX.append(sample)
            dataY.append(vector)
        return np.array(dataX), np.array(dataY)

    def create_input_data(self, history):
        X = np.zeros((1, self.look_back, self.word_vector_size))
        for i, vector in enumerate(history):
            age = len(history) - i 
            if age <= self.look_back:
                X[0][self.look_back - age] = vector
        return X

    def train(self, vectors, epochs=20, batch_size=100):
        trainX, trainY = self.create_training_data(vectors)
        print("Training data created")
        self.model.fit(trainX, trainY, nb_epoch=epochs, batch_size=batch_size, verbose=2)

    def predict(self, history):
        prediction_data = self.create_input_data(history)
        return self.model.predict(prediction_data)[0]
        #return self.text_model.likeliest(predicted_vector)

    def predict_sequence(self, words_to_predict, history=[]):
        if words_to_predict > 0:
            vector = self.predict(history)
            history.append(vector)
            self.predict_text(words_to_predict-1, history)
        else:
            return history

class testAnnModel(unittest.TestCase):

    def test_create_training_data(self):
        model = AnnModel(1, look_back=2)
        data = [[0.1], [0.2], [0.3], [0.4], [0.5]]
        X, Y = model.create_training_data(data)
        self.assertEqual((5, 2, 1), X.shape)
        self.assertEqual((5, 1), Y.shape)
        np.testing.assert_array_equal(X, [[[0.0], [0.0]], [[0.0], [0.1]], [[0.1], [0.2]], [[0.2], [0.3]], [[0.3], [0.4]]])
        np.testing.assert_array_equal(Y, [[0.1], [0.2], [0.3], [0.4], [0.5]])

    def test_simple_model(self):
        model = AnnModel(1, look_back=2, lstm_size=200, lstm_count=4)
        data = [[0.1], [0.2], [0.3], [0.4], [0.5]]
        model.train(data, epochs=400, batch_size=5)
        self.assertAlmostEqual(0.1, model.predict([[0.0], [0.0]])[0], delta=0.02)
        self.assertAlmostEqual(0.2, model.predict([[0.0], [0.1]])[0], delta=0.02)
        self.assertAlmostEqual(0.3, model.predict([[0.1], [0.2]])[0], delta=0.02)
        self.assertAlmostEqual(0.4, model.predict([[0.2], [0.3]])[0], delta=0.02)
        self.assertAlmostEqual(0.5, model.predict([[0.3], [0.4]])[0], delta=0.02)

    def test_create_input_data(self):
        model = AnnModel(2, look_back=4)
        data = [[2, 3], [1, 2], [1, 2], [1, 1], [0, 0]]
        input_data = model.create_input_data(data)
        self.assertEqual((1, 4, 2), input_data.shape)
        np.testing.assert_array_equal(input_data, [[[1, 2], [1, 2], [1, 1], [0, 0]]])


    def test_create_input_data_from_empty(self):
        model = AnnModel(10, look_back=4)
        data_from_empty = model.create_input_data([])
        self.assertEqual((1, 4, 10), data_from_empty.shape)
