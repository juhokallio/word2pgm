#!/usr/bin/env python
# -*- coding: utf-8 -*-

from vectorization import TextModel
from parsing import FinnishParser
from lstm import AnnModel
import unittest
import pdb


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

    def test_predicting_with_tiny_input(self):
        look_back = 4
        base_vector_size = 10
        grammar_vector_size = 10
        lstm_neurons = 100
        lstm_count = 2

        text = "Koiraa alkaa ärsyttämään ohjelmointi. Se ei enää jaksa."
        parser = FinnishParser()
        parsed_words, sentence_start_indexes = parser.parse(text)
        text_model = TextModel(parsed_words, sentence_start_indexes, base_size=base_vector_size, grammar_size=grammar_vector_size, word2vec_iterations=1)
        lstm_model = AnnModel(base_vector_size + grammar_vector_size, look_back, lstm_neurons, lstm_count)
        training_vectors = [text_model.word_to_vector(w) for w in parsed_words]
        lstm_model.train(training_vectors, [], epochs=500, batch_size=3000)

        history = [
                text_model.word_to_vector(parser.analyse("Koiraa")),
                text_model.word_to_vector(parser.analyse("alkaa")),
                text_model.word_to_vector(parser.analyse("ärsyttämään"))
                ]
        predicted = lstm_model.predict(history)
        self.assertEqual(text_model.likeliest(predicted).base, "ohjelmointi")

        predicted_text = predict_text(text_model, lstm_model, 30)
        self.assertEqual(predicted_text[0].base, "koira")
        self.assertEqual(predicted_text[1].base, "alkaa")
