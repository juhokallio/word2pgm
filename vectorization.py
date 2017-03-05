#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
import nsphere
from scipy.stats import truncnorm, beta
import numpy as np
import itertools
import unittest
import math
import pdb
from parsing import FinnishParser, AnalysedWord


class TextModel:

    def __init__(self, parsed_words, sentence_start_indexes, theorems, word2vec_iterations=1500):
        self.theorems = theorems
        base_sentences, grammar_sentences = self.split_to_sentences(parsed_words, sentence_start_indexes)
        self.models = {
                theorems[0].name: self.create_word2vec_model(base_sentences, theorems[0].vector_size, word2vec_iterations),
                theorems[1].name: self.create_word2vec_model(grammar_sentences, theorems[1].vector_size, word2vec_iterations)
                }

        self.counts = self.get_counts(parsed_words)
        self.counted_data_size = len(parsed_words)

    @staticmethod
    def create_word2vec_model(sentences, size, iterations):
        model = gensim.models.Word2Vec(sentences, size=size, min_count=1, iter=iterations, workers=4)
        model.init_sims(replace=True)
        positions = {w: model[w] for w in model.vocab}
        return positions

    @staticmethod
    def get_counts(parsed_words):
        counts = {}
        for w in parsed_words:
            if w.base not in counts:
                counts[w.base] = {}
            old_count = counts[w.base][w.grammar] if w.grammar in counts[w.base] else 0
            counts[w.base][w.grammar] = old_count + 1
        return counts

    @staticmethod
    def split_to_sentences(words, sentence_start_indexes):
        def split_to_grammar_windows(sentence, window_size):
            if len(sentence) < window_size:
                return [[w[1] for w in sentence]]
            windows = []
            for i in range(window_size, len(sentence) + 1):
                window = sentence[i-window_size:i]
                windows.append([w[1] for w in window])
            return windows

        sentences = np.split(words, sentence_start_indexes)
        base_sentences = [[w[0] for w in s] for s in sentences]
        sentences_as_windows = [split_to_grammar_windows(s, 3) for s in sentences]
        grammar_sentences = [w for window in sentences_as_windows for w in window]
        return base_sentences, grammar_sentences

    def word_to_vector(self, word, theorem):
        vector = self.models[theorem.name][theorem.apply(word)]
        return gensim.matutils.unitvec(vector)

    def word_to_concat_vector(self, word):
        return np.concatenate([self.word_to_vector(word, t) for t in self.theorems])

    def get_encounter_count(self, word):
        word_encountered = word.base in self.counts and word.grammar in self.counts[word.base]
        return self.counts[word.base][word.grammar] if word_encountered else 0

    def get_theorem_vocabulary(self, theorem):
        return self.models[theorem.name].items()

    def get_vocabulary(self, word_filter, minimum_counts):
        vocabulary = []
        for b, v_b in self.get_theorem_vocabulary(self.theorems[0]):
            for g, v_g in self.get_theorem_vocabulary(self.theorems[1]):
                word = AnalysedWord(b, g)
                counts = self.get_encounter_count(word)
                if counts >= minimum_counts and word_filter(word):
                    vocabulary.append((
                        word,
                        gensim.matutils.unitvec(np.concatenate((v_b, v_g))),
                        math.log((counts + 1) / (self.counted_data_size + self.counted_data_size))
                        ))
        return vocabulary

    def get_cosine_similarities(self, vector):
        model_vectors = []
        unit_vector = gensim.matutils.unitvec(vector)
        for b, v_b in self.base.items():
            for g, v_g in self.grammar.items():
                target_unit_vector = gensim.matutils.unitvec(np.concatenate((v_b, v_g)))
                s = np.dot(unit_vector, target_unit_vector)
                model_vectors.append(s)
        return np.array(model_vectors)


class TestTextModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = FinnishParser()

    def test_init(self):
        parsed_words, sentence_start_indexes = self.parser.parse("Ohjelmointi on mukavaa. Se ei ikinä ärsytä.")
        self.assertEqual("ohjelmointi", parsed_words[0].base)

        text_model = TextModel(parsed_words, sentence_start_indexes, base_size=6, grammar_size=4)
        self.assertEqual(10, text_model.vector_size)

        word_vector = text_model.word_to_vector(parsed_words[0])
        self.assertEqual(10, len(word_vector))

    def test_split_to_sentences(self):
        parsed_words, _ = self.parser.parse("Kissa istui pöydällä.")
        sentences, _ = TextModel.split_to_sentences(parsed_words, [])
        self.assertEqual([["kissa", "istua", "pöytä", "."]], sentences)

        parsed_words, _ = self.parser.parse("Kissa, joka makaa.")
        sentences, _ = TextModel.split_to_sentences(parsed_words, [])
        self.assertEqual([["kissa", ",", "joka", "maata", "."]], sentences)

        parsed_words, _ = self.parser.parse("Kissa istui pöydällä. Satoi.")
        sentences, _ = TextModel.split_to_sentences(parsed_words, [4])
        self.assertEqual([["kissa", "istua", "pöytä", "."], ["sataa", "."]], sentences)

    def test_get_vocabulary(self):
        parsed_words, sentence_start_indexes = self.parser.parse("Kissa kissalle kissan kissan")
        text_model = TextModel(parsed_words, sentence_start_indexes, base_size=3, grammar_size=2)
        vocabulary = text_model.get_vocabulary(lambda w: True)
        self.assertEqual(3, len(vocabulary))
        base_forms = [w.base for w, v, prior in vocabulary]
        self.assertEqual(["kissa", "kissa", "kissa"], base_forms)
        first_vector = vocabulary[0][1]
        self.assertEqual(5, len(first_vector))

    def test_get_vocabulary_with_filter(self):
        parsed_words, sentence_start_indexes = self.parser.parse("Kissa kissa lintu apina koira")
        text_model = TextModel(parsed_words, sentence_start_indexes, base_size=3, grammar_size=2)
        vocabulary = text_model.get_vocabulary(lambda w: w.base == "koira")
        self.assertEqual(1, len(vocabulary))
        w, v, prior = vocabulary[0]
        self.assertEqual("koira", w.base)
