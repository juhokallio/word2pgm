#!/usr/bin/env python
# -*- coding: utf-8 -*-

import unittest
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from vectorization import TextModel
from parsing import FinnishParser
import pdb

class AnnModel:

    def __init__(self, text_model, look_back):
        self.text_model = text_model
        self.look_back = look_back
        self.word_vector_size = text_model.size
        self.model = Sequential()
        self.model.add(LSTM(100, input_dim=look_back, input_length=self.word_vector_size, return_sequences=True))
        self.model.add(LSTM(100))
        self.model.add(Dense(self.word_vector_size))
        self.model.compile(loss='mean_squared_error', optimizer='adam')

    def create_training_data(self, parsed_text):
        dataX, dataY = [], []
        for i in range(len(parsed_text) - self.look_back - 1):
            a = [self.text_model.word_to_vector(w) for w in parsed_text[i:(i+self.look_back)]]
            dataX.append(np.transpose(a))
            dataY.append(self.text_model.word_to_vector(parsed_text[i + self.look_back]))
        return np.array(dataX), np.array(dataY)

    def create_input_data(self, history):
        X = np.zeros((self.look_back, self.word_vector_size))
        for i, w in enumerate(history):
            age = len(history) - i
            if age <= self.look_back:
                X[self.look_back - age] = self.text_model.word_to_vector(w)
        return np.array([np.transpose(X)])

    def train(self, parsed_text):
        trainX, trainY = self.create_training_data(parsed_text)
        print("Training data created")
        self.model.fit(trainX, trainY, nb_epoch=50, batch_size=100, verbose=2)

    def predict_word(self, history):
        prediction_data = self.create_input_data(history)
        predicted_vector = self.model.predict(prediction_data)[0]
        return self.text_model.likeliest(predicted_vector)

    def predict_text(self, words_to_predict, history=[]):
        if words_to_predict > 0:
            w = self.predict_word(history)
            print(w)
            history.append(w)
            self.predict_text(words_to_predict-1, history)

class testAnnModel(unittest.TestCase):

    def setUp(self):
        self.parser = FinnishParser()

    def test_create_input_data(self):
        parsed_words = self.parser.parse("Kissa kissalle kissan kissat.")
        text_model = TextModel(parsed_words)
        model = AnnModel(text_model, 4)
        data_from_empty = model.create_input_data(parsed_words)
        self.assertEqual((4, 10), data_from_empty.shape)

    def test_create_input_data_from_empty(self):
        parsed_words = self.parser.parse("Kissa kissalle kissan kissat.")
        text_model = TextModel(parsed_words)
        model = AnnModel(text_model, 4)
        data_from_empty = model.create_input_data([])
        self.assertEqual((4, 10), data_from_empty.shape)
