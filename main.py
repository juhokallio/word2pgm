#!/usr/bin/env python
# -*- coding: utf-8 -*-

from omorfi.omorfi import Omorfi
import re
import gensim
import unittest
import numpy
from nltk.tokenize import RegexpTokenizer
from collections import namedtuple
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error

omorfi = Omorfi()
omorfi.load_from_dir()

AnalysedWord = namedtuple("AnalysedWord", "base grammar")
TextModel = namedtuple("TextModel", "base grammar")


class TextModel:

    def __init__(self, parsed_words):
        self.base = self.create_word2vec_model(self.split_to_sentences(parsed_words, "base"))
        self.grammar = self.create_word2vec_model(self.split_to_sentences(parsed_words, "grammar"))
        self.size = self.base.vector_size + self.grammar.vector_size

    @staticmethod
    def create_word2vec_model(sentences):
        return gensim.models.Word2Vec(sentences, size=5, min_count=1, iter=10, workers=2)
 
    @staticmethod
    def split_to_sentences(words, dimension):
        sentences = []
        sentence = []
        for w in words:
            sentence.append(getattr(w, dimension))
            if w.base == ".":
                sentences.append(sentence)
                sentence = []
        return sentences

    def word_to_vector(self, word):
        return self.base[word.base] + self.grammar[word.grammar]


def analyse(word):
    omorfi_form = omorfi.analyse(word)
    first_form = omorfi_form[0][0]
    return AnalysedWord(omorfi_to_base(first_form), omorfi_to_grammar(first_form))

def omorfi_to_base(omorfi_form):
    return re.search(r"\[WORD_ID=(.*?)\]", omorfi_form).group(1)

def omorfi_to_grammar(omorfi_form):
    return re.sub(r"\[WORD_ID=.*?\]", "", omorfi_form)

def parse(text):
    tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')
    return [analyse(w) for w in tokenizer.tokenize(text)]


def read_file(file_name):
    with open(file_name, "r") as myfile:
        text = myfile.read().replace('\n', '')
    return text

def read_files(file_names):
    return " ".join([read_file(n) for n in file_names])

def test_similarity(model, w1, w2):
    print("{}={} {}".format(w1, w2, model.similarity(w1, w2)))

def test_base_form_word2vec():
    analysed = analyse("koiralle")
    print(analysed)
    parsed = parse(analysed[0][0])
    print(parsed)

    print(word_to_base("kissan"))
    text = read_file("data/finnish/input.txt")
    print("Read {} words of training data".format(len(text)))
    sentences = create_sentences(text, word_to_base)
    print("Parsed into {} sentences".format(len(sentences)))
    model = create_word2vec_model(sentences)
    print("Model trained")
    test_similarity(model, "koira", "kissa")
    test_similarity(model, "juosta", "mies")
    test_similarity(model, "nainen", "mies")
    test_similarity(model, "hän", "se")
    test_similarity(model, "mutta", "koska")

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
    parsed_words = parse(text)
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


class TestParsing(unittest.TestCase):

    def test_omorfi_to_base(self):
        self.assertEqual("koira", omorfi_to_base("[WORD_ID=koiralle][UPOS=NOUN][NUM=SG][CASE=ALL]"))

    def test_omorfi_to_base(self):
        self.assertEqual("[UPOS=NOUN][NUM=SG][CASE=ALL]", omorfi_to_grammar("[WORD_ID=koiralle][UPOS=NOUN][NUM=SG][CASE=ALL]"))

    def test_create_sentences(self):
        self.assertEqual([["kissa", "istua", "pöytä", "."]], create_sentences(parse("Kissa istui pöydällä."), "base"))
        self.assertEqual([["kissa", ",", "joka", "maata", "."]], create_sentences(parse("Kissa, joka makaa."), "base"))
        self.assertEqual([["kissa", "istua", "pöytä", "."], ["sataa", "."]], create_sentences(parse("Kissa istui pöydällä. Satoi."), "base"))

    def test_analyse(self):
        kissalle = analyse("kissalle")
        self.assertEqual("kissa", kissalle.base)
        self.assertEqual("[UPOS=NOUN][NUM=SG][CASE=ALL]", kissalle.grammar)

        pilkku = analyse(",")
        self.assertEqual(",", pilkku.base)
        self.assertEqual("[UPOS=PUNCT][BOUNDARY=CLAUSE][SUBCAT=COMMA]", pilkku.grammar)
