#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from vectorization import TextModel
from parsing import FinnishParser
from lstm import AnnModel
import matplotlib.pyplot as plt
import numpy as np
import unittest
import math
import nsphere
import pdb


class Word2pgm:

    def __init__(self, base_vector_size, grammar_vector_size, look_back, lstm_layer_size, lstm_layers):
        self.base_vector_size = base_vector_size
        self.grammar_vector_size = grammar_vector_size
        self.vector_size = base_vector_size + grammar_vector_size
        self.parser = FinnishParser()
        self.lstm_model = AnnModel(self.vector_size, look_back, lstm_layer_size, lstm_layers)

    def train(self, text, lstm_epochs, lstm_batch_size, word2vec_iterations=1000, test_data_portion=0.2):
        parsed_words, originals, sentence_start_indexes = self.parser.parse(text)
        self.original_words = {w: o for w, o in zip(parsed_words, originals)}
        print("words parsed")
        self.text_model = TextModel(parsed_words, sentence_start_indexes, base_size=self.base_vector_size, grammar_size=self.grammar_vector_size, word2vec_iterations=word2vec_iterations)
        self.vocabulary = self.text_model.get_vocabulary(self.parser.is_valid_word, 4)
        print("Vocabulary size {}".format(len(self.vocabulary)))
        vector_data = [self.text_model.word_to_vector(w) for w in parsed_words]
        split_index = int(len(vector_data) * test_data_portion)
        training_data = vector_data[split_index:]
        test_data = vector_data[:split_index] if split_index > 0 else training_data
        print("training data length: {}".format(len(training_data)))
        print("test data length: {}".format(len(test_data)))
        self.lstm_model.train(training_data, [], epochs=lstm_epochs, batch_size=lstm_batch_size)
        self.error_logpdf = self.lstm_model.get_logpdf(
                test_data,
                normalizer=self.distance
                )

    def distance(self, s):
        return nsphere.cap(1, self.vector_size, math.acos(s))

    def evidence_log_probability(self, s):
        return math.log(nsphere.surface(math.sin(math.acos(s)), self.vector_size - 1))

    def log_likelihood(self, s):
        return self.error_logpdf(self.distance(s))

    def predict_text(self, words_to_predict, history=[]):
        if words_to_predict > 0:
            vector = self.lstm_model.predict(history)
            word = self.get_likeliest_word(vector)
            history.append(self.text_model.word_to_vector(word))
            return [self.original_words[word]] + self.predict_text(words_to_predict-1, history)
        else:
            return []

    def get_likeliest_word(self, unit_vector):
        closest_w = None
        best_p = -1
        for w, v, log_prior in self.vocabulary:
            s = np.dot(unit_vector, v) 
            p = log_prior + self.log_likelihood(s) - self.evidence_log_probability(s)
            if (closest_w is None) or (best_p < p):
                best_p = p
                closest_w = w
        return closest_w

    def text_to_vectors(self, text):
        parsed_words, _, _ = self.parser.parse(text)
        return [self.text_model.word_to_vector(w) for w in parsed_words]


def read_file(file_name):
    with open(file_name, "r") as myfile:
        text = myfile.read()
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
    parser = FinnishParser()
    words = parser.tokenize(read_files(file_names))
    unique_words = set(words)
    for w in words:
        print(w)

def plot_similarities(text_model, vector):
    model_vectors = text_model.get_cosine_similarities(vector)
    df = pd.Series(model_vectors)
    plt.figure()
    df.hist(bins=100)
    plt.show()

def test_cosine_similarity_distribution():
    text = read_file("data/finnish/pg45271.txt")
    parser = FinnishParser()
    parsed_words, sentence_start_indexes = parser.parse(text)
    text_model = TextModel(parsed_words, sentence_start_indexes, base_size=10, grammar_size=10, word2vec_iterations=100)
    plot_similarities(text_model, np.random.rand(20))

def main():
    #print_unique_words(["data/finnish/pg45271.txt"])
    #test_unique_counts(["data/finnish/pg45271.txt"])
    #test_base_form_word2vec()
    text = read_file("data/finnish/pg45271.txt")[:1009]
    parser = FinnishParser()
    parsed_words, sentence_start_indexes = parser.parse(text)
    text_model = TextModel(parsed_words, sentence_start_indexes, base_size=2, grammar_size=2, word2vec_iterations=100)
    pdf = text_model.cosine_similarity_pdf
    #plot_similarities(text_model, np.random.rand(4))
    x = np.linspace(-1, 1, 100)
    plt.plot(x, pdf(x))
    print(pdf(0))
    plt.show()


if __name__ == "__main__":
    main()


class Word2pgmTest(unittest.TestCase):

    default_settings = {
            "base_vector_size": 15,
            "grammar_vector_size": 15,
            "look_back": 5,
            "lstm_layer_size": 100,
            "lstm_layers": 2
            }

    def test_predicting_with_tiny_input(self):
        word2pgm = Word2pgm(**self.default_settings)
        text = "Koiraa alkaa ärsyttämään ohjelmointi. Se ei enää jaksa."
        word2pgm.train(text, lstm_epochs=50, lstm_batch_size=300, word2vec_iterations=1, test_data_portion=0.0)

        history = word2pgm.text_to_vectors("Koiraa alkaa ärsyttämään")
        predicted = word2pgm.predict_text(1, history=history)
        self.assertEqual(predicted[0].base, "ohjelmointi")

        predicted_text = word2pgm.predict_text(30, history=[])
        self.assertEqual(predicted_text[0].base, "koira")
        self.assertEqual(predicted_text[1].base, "alkaa")
