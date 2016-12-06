#!/usr/bin/env python
# -*- coding: utf-8 -*-

from vectorization import TextModel
from parsing import FinnishParser
import unittest
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error


def read_file(file_name):
    with open(file_name, "r") as myfile:
        text = myfile.read().replace('\n', '')
    return text

def read_files(file_names):
    return " ".join([read_file(n) for n in file_names])

def test_grammar():
    text = read_file("data/finnish/pg45271.txt")
    print("Read {} words of training data".format(len(text)))
    sentences = create_sentences(text, word_to_grammar)
    print("Parsed into {} sentences".format(len(sentences)))
    model = create_word2vec_model(sentences)
    print("Model trained")

def test_unique_counts(file_names):
    words = split_to_words(read_files(file_names))
    print("Found {} words, counting distinct forms".format(len(words)))
    word_count = len({word_to_base(w) for w in words})
    grammar_form_count = len({word_to_grammar(w) for w in words})
    print("Forms counted:")
    print("{} distinct base words".format(word_count))
    print("{} distinct grammar forms".format(grammar_form_count))

def print_unique_words(file_names):
    words = split_to_words(read_files(file_names))
    unique_words = {word_to_base(w) for w in words}
    for w in unique_words:
        print(w)

def create_training_data(parsed_text, text_model, look_back=20):
    dataX, dataY = [], []
    for i in range(len(parsed_text) - look_back - 1):
        a = [text_model.word_to_vector(w) for w in parsed_text[i:(i+look_back)]]
        dataX.append(a)
        dataY.append(text_model.word_to_vector(parsed_text[i + look_back]))
    return numpy.array(dataX), numpy.array(dataY)

def create_rnn(look_back, word_vector_size):
    model = Sequential()
    model.add(LSTM(150, input_dim=word_vector_size, input_length=look_back))
    model.add(Dense(word_vector_size))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_rnn(model, parsed_text, text_model, look_back):
    trainX, trainY = create_training_data(parsed_text, text_model, look_back)
    model.fit(trainX, trainY, nb_epoch=100, batch_size=1, verbose=2)

def test_rnn_training():
    look_back = 4
    text = read_file("data/finnish/pg45271.txt")
    print("Read {} words of training data".format(len(text)))
    parser = FinnishParser()
    parsed_words = parser.parse(text)
    print("words parsed")
    text_model = TextModel(parsed_words)
    print("Text model trained")
    lstm_model = create_rnn(look_back, text_model.size)
    print("ann created, starting training")
    train_rnn(lstm_model, parsed_words, text_model, look_back)
    print("ann trained")

def main():
    #print_unique_words(["data/finnish/pg45271.txt"])
    #test_unique_counts(["data/finnish/pg45271.txt"])
    #test_base_form_word2vec()
    test_rnn_training()
    
if __name__ == "__main__":
    main()
