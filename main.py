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
    look_back = 5
    text = read_file("data/finnish/pg45271.txt")
    print("Read {} words of training data".format(len(text)))
    parser = FinnishParser()
    parsed_words = parser.parse(text)
    print("words parsed")
    text_model = TextModel(parsed_words)
    print("Text model trained")
    lstm_model = AnnModel(text_model, look_back)
    print("ann created, starting training")
    lstm_model.train(parsed_words)
    print("ann trained")
    lstm_model.predict_text(30)

def main():
    print_unique_words(["data/finnish/pg45271.txt"])
    #test_unique_counts(["data/finnish/pg45271.txt"])
    #test_base_form_word2vec()
    #test_rnn_training()
    
if __name__ == "__main__":
    main()
