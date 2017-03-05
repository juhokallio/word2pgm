#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
import unittest
import numpy as np
from scipy.stats import truncnorm
from keras.models import Sequential, Model
from keras.layers import Dense, Input, LSTM, merge, Flatten, Reshape, Lambda
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot
import pandas as pd
from keras import backend as K
import tensorflow as tf
import pdb


def unitvectorize(x):
    norm = K.sqrt(K.sum(K.square(x), 1, keepdims=True))
    return x / norm


class AnnModel:

    def __init__(self, vector_size, look_back=10, lstm_size=50, lstm_count=2):
        self.look_back = look_back
        self.word_vector_size = vector_size

        h = Input(shape=(look_back, vector_size), dtype="float32", name="history")
        x = LSTM(lstm_size, input_dim=self.word_vector_size, input_length=look_back, return_sequences=True)(h)
        for i in range(0, lstm_count - 2):
            x = LSTM(lstm_size, return_sequences=True)(x)
        x = LSTM(lstm_size)(x)
        vector_predictions = Dense(self.word_vector_size)(x)
        self.generator = Model(input=h, output=vector_predictions, name="generator")
        self.generator.compile(loss="cosine_proximity", optimizer="adam")

        input_vector = Input(shape=(look_back+1, vector_size), dtype="float32", name="discriminator_input")
        x = Lambda(unitvectorize)(input_vector)
        x = LSTM(lstm_size, input_dim=self.word_vector_size, input_length=look_back+1, return_sequences=True)(input_vector)
        for i in range(0, lstm_count - 2):
            x = LSTM(lstm_size, return_sequences=True)(x)
        x = LSTM(lstm_size)(x)
        authenticity_classification = Dense(1, activation="sigmoid")(x)
        self.discriminator = Model(input=input_vector, output=authenticity_classification, name="discriminator")
        self.discriminator.compile(loss="binary_crossentropy", optimizer="adam")

        x = merge([h, Reshape((1, vector_size))(vector_predictions)], mode="concat", concat_axis=1)
        x = self.discriminator(x)
        self.discriminator_on_generator = Model(input=h, output=x)
        self.discriminator_on_generator.compile(loss="binary_crossentropy", optimizer="adam")
        plot(self.discriminator_on_generator, to_file="model.png", show_shapes=True)

    def create_training_data(self, vectors):
        dataX, dataY = [], []
        for i in range(len(vectors) - self.look_back):
            dataX.append(vectors[i:(i + self.look_back)])
            dataY.append(vectors[i + self.look_back])
        return np.array(dataX), np.array(dataY)

    def create_padded_training_data(self, vectors, sentence_start_indexes):
        def create_dataX(sentence_start):
            sampleX = np.zeros((self.look_back, self.look_back, self.word_vector_size))
            for i in range(self.look_back):
                for j in range(self.look_back):
                    vector_index = sentence_start - self.look_back + i + j
                    if vector_index >= sentence_start:
                        sampleX[i][j] = vectors[vector_index]
            return sampleX

        sentence_start_indexes.insert(0, 0)
        dataX = np.concatenate([create_dataX(start) for start in sentence_start_indexes])
        dataY = np.concatenate([vectors[start:start + self.look_back] for start in sentence_start_indexes])
        return dataX, dataY

    def create_input_data(self, history):
        X = np.zeros((1, self.look_back, self.word_vector_size))
        for i, vector in enumerate(history):
            age = len(history) - i 
            if age <= self.look_back:
                X[0][self.look_back - age] = vector
        return X

    def train(self, vectors, sentence_start_indexes=[], epochs=20, batch_size=100):
        print("max {}, min {} vector value".format(np.amax(vectors), np.amin(vectors)))
        unpaddedX, unpaddedY = self.create_training_data(vectors)
        paddedX, paddedY = self.create_padded_training_data(vectors, sentence_start_indexes)
        X = np.concatenate((unpaddedX, paddedX))
        Y = np.concatenate((unpaddedY, paddedY))
        print("Training data created")
        return self.generator.fit(X, Y, nb_epoch=epochs, batch_size=batch_size, verbose=2)

    def train_with_discriminator(self, vectors, sentence_start_indexes=[], epochs=20, batch_size=100, updates_per_batch=50):
        unpaddedX, unpaddedY = self.create_training_data(vectors)
        paddedX, paddedY = self.create_padded_training_data(vectors, sentence_start_indexes)
        X = np.concatenate((unpaddedX, paddedX))
        Y = np.concatenate((unpaddedY, paddedY))
        d_loss = None
        g_loss = None
        for epoch in range(epochs):
            for index in range(int(X.shape[0]/batch_size)):
                X_batch = X[index*batch_size:(index+1)*batch_size]
                Y_batch = Y[index*batch_size:(index+1)*batch_size]
                predicted = self.generator.predict(X_batch)
                X_discriminator = np.concatenate((
                    np.concatenate((X_batch, X_batch)),
                    np.expand_dims(np.concatenate((Y_batch, predicted)), 1)
                    ), axis=1)
                y_discriminator = [1] * batch_size + [0] * batch_size
                for i in range(updates_per_batch):
                    d_loss = self.discriminator.train_on_batch(X_discriminator, y_discriminator)
                self.discriminator.trainable = False
                for i in range(updates_per_batch):
                    g_loss = self.discriminator_on_generator.train_on_batch(X_batch, [1] * batch_size)
                self.discriminator.trainable = True
            print("Epoch {} finished".format(epoch))
            print("discriminator loss {}".format(d_loss))
            print("generator loss {}".format(g_loss))

    @staticmethod
    def cosine_similarity(v1, v2):
        return np.dot(gensim.matutils.unitvec(v1), gensim.matutils.unitvec(v2))

    def get_logpdf(self, vectors, normalizer=lambda x: x, sentence_start_indexes=[]):
        unpaddedX, unpaddedY = self.create_training_data(vectors)
        paddedX, paddedY = self.create_padded_training_data(vectors, sentence_start_indexes)
        X = np.concatenate((unpaddedX, paddedX))
        Y = np.concatenate((unpaddedY, paddedY))
        predictions = self.generator.predict(X)
        similarities = [normalizer(self.cosine_similarity(y, y_p)) for y, y_p in zip(Y, predictions)]
        similarities.extend([-1 * s for s in similarities])
        sd = np.std(similarities)
        return lambda s: truncnorm.logpdf(s, 0, np.inf, 0, sd)

    def predict(self, history):
        prediction_data = self.create_input_data(history)
        vector = self.generator.predict(prediction_data)[0]
        return gensim.matutils.unitvec(vector)

    def predict_batch(self, data):
        X = np.concatenate([self.create_input_data(h) for h in data])
        predictions = self.generator.predict(X)
        return  [gensim.matutils.unitvec(v) for v in predictions]


class TestAnnModel(unittest.TestCase):

    def test_create_training_data(self):
        model = AnnModel(1, look_back=2)
        data = [[0.1], [0.2], [0.3], [0.4], [0.5]]
        X, Y = model.create_training_data(data)
        self.assertEqual((3, 2, 1), X.shape)
        self.assertEqual((3, 1), Y.shape)
        np.testing.assert_array_equal(X, [[[0.1], [0.2]], [[0.2], [0.3]], [[0.3], [0.4]]])
        np.testing.assert_array_equal(Y, [[0.3], [0.4], [0.5]])

    def test_create_padded_training_data(self):
        model = AnnModel(1, look_back=2)
        data = [[0.1], [0.2], [0.3], [0.4], [0.5]]
        sentence_start_indexes = [2, 3]
        X, Y = model.create_padded_training_data(data, sentence_start_indexes)
        np.testing.assert_array_equal(X, [[[0.0], [0.0]], [[0.0], [0.1]], [[0.0], [0.0]], [[0.0], [0.3]], [[0.0], [0.0]], [[0.0], [0.4]]])
        np.testing.assert_array_equal(Y, [[0.1], [0.2], [0.3], [0.4], [0.4], [0.5]])

    def test_create_multidimensional_padded_training_data(self):
        model = AnnModel(3, look_back=3)
        data = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        X, Y = model.create_padded_training_data(data, [])
        expected_X = [
                [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                [[0, 0, 0], [0, 0, 0], [1, 2, 3]],
                [[0, 0, 0], [1, 2, 3], [4, 5, 6]]
                ]
        np.testing.assert_array_equal(X, expected_X)
        np.testing.assert_array_equal(Y, [[1, 2, 3], [4, 5, 6], [7, 8, 9]])

    def test_simple_model(self):
        model = AnnModel(2, look_back=2, lstm_size=40, lstm_count=2)
        training_data = [[0.1, 0.1], [0.1, 0.2], [0.1, 0.3], [0.1, 0.4], [0.1, 0.5]]
        model.train_with_discriminator(training_data, epochs=55, batch_size=5, updates_per_batch=200)

        test_cases = [
                (np.array([[0.0, 0.0], [0.0, 0.0]]), np.array([0.1, 0.1])),
                (np.array([[0.0, 0.0], [0.1, 0.1]]), np.array([0.1, 0.2])),
                (np.array([[0.1, 0.1], [0.1, 0.2]]), np.array([0.1, 0.3])),
                (np.array([[0.1, 0.2], [0.1, 0.3]]), np.array([0.1, 0.4])),
                (np.array([[0.1, 0.3], [0.1, 0.4]]), np.array([0.1, 0.5]))
                ]
        for x, expected_y in test_cases:
            discriminator_x = np.array([np.append(x, [expected_y], axis=0)])
            self.assertAlmostEqual(1.0, model.discriminator.predict(discriminator_x)[0][0], delta=0.1,
                    msg="Discriminator didn't recognize training data")
            y = model.predict(x)
            np.testing.assert_array_equal(y, gensim.matutils.unitvec(y),
                    err_msg="Predicted vector was not unit vector")
            s = model.cosine_similarity(expected_y, y)
            self.assertAlmostEqual(1.0, s, delta=0.2,
                    msg="Predicted vector didn't match training data")

    def test_create_input_data(self):
        model = AnnModel(2, look_back=4)
        data = [[2, 3], [1, 2], [1, 2], [1, 1], [0, 0]]
        input_data = model.create_input_data(data)
        self.assertEqual((1, 4, 2), input_data.shape)
        np.testing.assert_array_equal(input_data, [[[1, 2], [1, 2], [1, 1], [0, 0]]])

    def test_create_padded_input_data(self):
        model = AnnModel(1, look_back=4)
        history = [[1], [2]]
        input_data = model.create_input_data(history)
        np.testing.assert_array_equal(input_data, [[[0], [0], [1], [2]]])

    def test_create_input_data_from_empty(self):
        model = AnnModel(10, look_back=4)
        data_from_empty = model.create_input_data([])
        self.assertEqual((1, 4, 10), data_from_empty.shape)

    def test_unitvectorize(self):
        sess = tf.InteractiveSession()
        data = np.array([[0.0, 3.0], [1.0, 2.0]])
        unitvector_data = np.array([
                [0.0, 1.0],
                gensim.matutils.unitvec(data[1])
                ])
        np.testing.assert_array_almost_equal(unitvectorize(data).eval(), unitvector_data, decimal=5,
                err_msg="Incorrect result from unitvectorize tf function")
