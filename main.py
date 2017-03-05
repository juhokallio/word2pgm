#!/usr/bin/env python
# -*- coding: utf-8 -*-

import random
from vectorization import TextModel
from parsing import FinnishParser, AnalysedWord
from lstm import AnnModel
import matplotlib.pyplot as plt
import numpy as np
import unittest
import math
from scipy.stats import truncnorm
import nsphere
import pdb
import statutils
from operator import itemgetter


class Theorem:

    def __init__(self, name, vector_size, applier):
        self.name = name
        self.vector_size = vector_size
        self.apply = applier

class Word2pgm:

    def __init__(self, theorems, look_back, lstm_layer_size, lstm_layers):
        self.theorems = theorems
        self.vector_size = sum([t.vector_size for t in theorems])
        self.parser = FinnishParser()
        self.lstm_model = AnnModel(self.theorems, look_back, lstm_layer_size, lstm_layers)

    def train(self, text, lstm_epochs, lstm_batch_size, word2vec_iterations=1000, test_data_portion=0.2):
        parsed_words, originals, sentence_start_indexes = self.parser.parse(text)
        self.original_words = {w: o for w, o in zip(parsed_words, originals)}
        print("words parsed")
        self.text_model = TextModel(parsed_words, sentence_start_indexes, self.theorems, word2vec_iterations=word2vec_iterations)
        self.vocabulary = self.text_model.get_vocabulary(self.parser.is_valid_word, 1)
        print("Vocabulary size {}".format(len(self.vocabulary)))
        vector_data = [self.text_model.word_to_concat_vector(w) for w in parsed_words]
        split_index = int(len(vector_data) * test_data_portion)
        training_data = vector_data[split_index:]
        test_data = vector_data[:split_index] if split_index > 0 else training_data
        print("training data length: {}".format(len(training_data)))
        print("test data length: {}".format(len(test_data)))
        self.lstm_model.train_with_discriminator(training_data, [], epochs=lstm_epochs, batch_size=lstm_batch_size)
        self.error_dists = self.get_error_logpdfs(test_data)

    def get_error_logpdfs(self, test_data):
        test_predictions = self.lstm_model.predict_batch(test_data)
        dists = {}
        for theorem, theorem_predictions in zip(self.theorems, test_predictions):
            means = [v for _, v in self.text_model.get_theorem_vocabulary(theorem)]
            std = statutils.get_evidence_variance(theorem_predictions, means)
            dists[theorem.name] = truncnorm(0, np.inf, 0, std)
        return dists

    def evidence_log_probability(self, s):
        return math.log(nsphere.surface(math.sin(math.acos(s)), self.vector_size - 1))

    def log_likelihood(self, s, theorem):
        error = statutils.distance(theorem.vector_size, s)
        return self.error_dists[theorem.name].logpdf(error)

    def predict_text(self, words_to_predict, history=[]):
        if words_to_predict > 0:
            vectors = self.lstm_model.predict(history)
            word = self.get_likeliest_word(vectors)
            if word in self.original_words:
                original = self.original_words[word]
            else:
                original = "[{}] UNKNOWN".format(word.base)
            print(original)
            history.append(self.text_model.word_to_concat_vector(word))
            return [original] + self.predict_text(words_to_predict-1, history)
        else:
            return []

    def get_prior(self, word):
        counts = self.text_model.get_encounter_count(word)
        return math.log((counts + 1) / (self.text_model.counted_data_size * 2))

    def get_likeliest_theorem_solutions(self, vector, theorem, solution_count=10):
        solutions = [(w, np.dot(v, vector)) for w, v in self.text_model.get_theorem_vocabulary(theorem)]
        return sorted(solutions, key=itemgetter(1))[:solution_count]

    def get_likeliest_solutions(self, vectors, theorem_solution_count=10):
        solutions = []
        for i, theorem in enumerate(self.theorems):
            solutions.append(self.get_likeliest_theorem_solutions(vectors[i][0], theorem, theorem_solution_count))
        return solutions

    def get_likeliest_word(self, vectors):
        closest_w = None
        best_p = -1

        solutions = self.get_likeliest_solutions(vectors)
        for w_b, s_b in solutions[0]:
            for w_g, s_g in solutions[1]:
                log_likelihood = self.log_likelihood(s_b, self.theorems[0]) + self.log_likelihood(s_g, self.theorems[1])
                word = AnalysedWord(w_b, w_g)
                evidence = self.evidence_log_probability(s_b) + self.evidence_log_probability(s_g)
                p = self.get_prior(word) + log_likelihood - evidence
            if (closest_w is None) or (best_p < p):
                best_p = p
                closest_w = word
        return closest_w

    def text_to_vectors(self, text):
        parsed_words, _, _ = self.parser.parse(text)
        return [self.text_model.word_to_concat_vector(w) for w in parsed_words]


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
            "look_back": 5,
            "lstm_layer_size": 100,
            "lstm_layers": 2,
            "theorems": [
                Theorem("base", 15, lambda w: w.base),
                Theorem("grammar", 15, lambda w: w.grammar)
                ]
            }

    def test_predicting_with_tiny_input(self):
        word2pgm = Word2pgm(**self.default_settings)
        text = "Koiraa alkaa hermostuttamaan ohjelmointi. Se ei enää jaksa."
        word2pgm.train(text, lstm_epochs=50, lstm_batch_size=2, word2vec_iterations=1, test_data_portion=0.0)

        history = word2pgm.text_to_vectors("Koiraa alkaa hermostuttamaan")
        predicted = word2pgm.predict_text(2, history=history)
        self.assertEqual(predicted[0], "ohjelmointi")
        self.assertEqual(predicted[1], ".")

        predicted_text = word2pgm.predict_text(30, history=[])
        self.assertEqual(predicted_text[0], "Koiraa")
        self.assertEqual(predicted_text[1], "alkaa")
        self.assertEqual(predicted_text[2], "hermostuttamaan")
