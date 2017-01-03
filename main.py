#!/usr/bin/env python
# -*- coding: utf-8 -*-

from vectorization import TextModel
from parsing import FinnishParser
from lstm import AnnModel
import unittest
import pdb


class Word2pgm:

    def __init__(self, base_vector_size, grammar_vector_size, look_back, lstm_layer_size, lstm_layers):
        self.base_vector_size = base_vector_size
        self.grammar_vector_size = grammar_vector_size
        self.parser = FinnishParser()
        self.lstm_model = AnnModel(base_vector_size + grammar_vector_size, look_back, lstm_layer_size, lstm_layers)

    def train(self, text, lstm_epochs, lstm_batch_size, word2vec_iterations=1000, test_data_portion=0.2):
        parsed_words, sentence_start_indexes = self.parser.parse(text)
        print("words parsed")
        self.text_model = TextModel(parsed_words, sentence_start_indexes, base_size=self.base_vector_size, grammar_size=self.grammar_vector_size, word2vec_iterations=word2vec_iterations)
        self.cosine_similarity_pdf = self.text_model.get_cosine_similarity_pdf()
        vector_data = [self.text_model.word_to_vector(w) for w in parsed_words]
        split_index = int(len(vector_data) * test_data_portion)
        training_data = vector_data[split_index:]
        test_data = vector_data[:split_index]
        print("training data length: {}".format(len(training_data)))
        print("test data length: {}".format(len(test_data)))
        self.lstm_model.train(training_data, [], epochs=lstm_epochs, batch_size=lstm_batch_size)

    def predict_text(self, words_to_predict, history=[]):
        if words_to_predict > 0:
            vector = self.lstm_model.predict(history)
            word = self.text_model.likeliest(vector)
            history.append(self.text_model.word_to_vector(word))
            return [word] + self.predict_text(words_to_predict-1, history)
        else:
            return []

    def print_cosine_similarity_densities(self):
        print("cosine similarity densities (-1, 0, 1):")
        print(self.cosine_similarity_pdf(-1))
        print(self.cosine_similarity_pdf(0))
        print(self.cosine_similarity_pdf(1))

    def text_to_vectors(self, text):
        parsed_words, _ = self.parser.parse(text)
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

def test_rnn_training():
    look_back = 10
    base_vector_size = 10
    grammar_vector_size = 10
    lstm_neurons = 200
    lstm_count = 3

    text = read_file("data/finnish/pg45271.txt")
    print("Read {} words of training data".format(len(text)))
    parser = FinnishParser()
    parsed_words, sentence_start_indexes = parser.parse(text)
    print("words parsed")
    text_model = TextModel(parsed_words, sentence_start_indexes, base_size=base_vector_size, grammar_size=grammar_vector_size, word2vec_iterations=1000)
    print("Text model trained")
    lstm_model = AnnModel(base_vector_size + grammar_vector_size, look_back, lstm_neurons, lstm_count)
    print("ann created, starting training")
    training_vectors = [text_model.word_to_vector(w) for w in parsed_words]
    lstm_model.train(training_vectors, [], epochs=50, batch_size=500)
    print("ann trained")
    predicted_vectors = lstm_model.predict_sequence(30)
    for v in predicted_vectors:
        print(text_model.likeliest(v))

def predict_text(text_model, ann_model, words_to_predict, history=[], text=[]):
    if words_to_predict > 0:
        vector = ann_model.predict(history)
        word = text_model.likeliest(vector)
        history.append(text_model.word_to_vector(word))
        text.append(word)
        return predict_text(text_model, ann_model, words_to_predict-1, history, text)
    else:
        return text


def main():
    #print_unique_words(["data/finnish/pg45271.txt"])
    #test_unique_counts(["data/finnish/pg45271.txt"])
    #test_base_form_word2vec()
    test_rnn_training()
    
if __name__ == "__main__":
    main()


class Word2pgmTest(unittest.TestCase):

    default_settings = {
            "base_vector_size": 7,
            "grammar_vector_size": 5,
            "look_back": 2,
            "lstm_layer_size": 100,
            "lstm_layers": 2
            }

    def test_probability_distributions(self):
        word2pgm = Word2pgm(**self.default_settings)
        text = read_file("data/finnish/pg45271.txt")[:10000]
        word2pgm.train(text, lstm_epochs=1, lstm_batch_size=100, word2vec_iterations=1000, test_data_portion=0.5)
        self.assertTrue(word2pgm.cosine_similarity_pdf(-1) < word2pgm.cosine_similarity_pdf(0),
                msg="cosine similarity density in -1 was higher than in 0")
        self.assertTrue(word2pgm.cosine_similarity_pdf(1) < word2pgm.cosine_similarity_pdf(0),
                msg="cosine similarity density in 1 was higher than in 0")
        self.assertAlmostEqual(word2pgm.cosine_similarity_pdf(-1), word2pgm.cosine_similarity_pdf(1),
                delta=0.01,
                msg="cosine similarity density in -1 was not the same as in 1")

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
