import gensim
import nsphere
from scipy.stats import truncnorm, beta
import numpy as np
import itertools
import unittest
import pdb
from parsing import FinnishParser, AnalysedWord


class TextModel:

    def __init__(self, parsed_words, sentence_start_indexes, base_size=30, grammar_size=30, word2vec_iterations=1500):
        base_sentences, grammar_sentences = self.split_to_sentences(parsed_words, sentence_start_indexes)
        self.base = self.create_word2vec_model(base_sentences, base_size, word2vec_iterations)
        self.grammar = self.create_word2vec_model(grammar_sentences, grammar_size, word2vec_iterations)
        self.base_length = base_size
        self.grammar_length = grammar_size
        self.vector_size = self.base_length + self.grammar_length
        self.counts = self.get_counts(parsed_words)
        self.counted_data_size = len(parsed_words)

    @staticmethod
    def create_word2vec_model(sentences, size, iterations):
        model = gensim.models.Word2Vec(sentences, size=size, min_count=1, iter=iterations, workers=2)
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
        sentences = np.split(words, sentence_start_indexes)
        base_sentences = [[w[0] for w in s] for s in sentences]
        grammar_sentences = [[w[1] for w in s] for s in sentences]
        return base_sentences, grammar_sentences

    def word_to_vector(self, word):
        vector = np.concatenate((self.base[word.base], self.grammar[word.grammar]))
        return gensim.matutils.unitvec(vector)

    def get_encounter_count(self, word):
        word_encountered = word.base in self.counts and word.grammar in self.counts[word.base]
        return self.counts[word.base][word.grammar] + 1 if word_encountered else 1

    def get_vocabulary(self):
        vocabulary = []
        for b, v_b in self.base.items():
            for g, v_g in self.grammar.items():
                word = AnalysedWord(b, g)
                vocabulary.append((
                    word,
                    gensim.matutils.unitvec(np.concatenate((v_b, v_g))),
                    self.get_encounter_count(word)
                    ))
        return vocabulary

    def likeliest(self, vector):
        closest_w = None
        similarity = -1
        unit_vector = gensim.matutils.unitvec(vector)
        for b, v_b in self.base.items():
            for g, v_g in self.grammar.items():
                target_unit_vector = gensim.matutils.unitvec(np.concatenate((v_b, v_g)))
                word = AnalysedWord(b, g)
                s = np.dot(unit_vector, target_unit_vector)# * self.get_encounter_count(word)
                if (closest_w is None) or (similarity < s):
                    similarity = s
                    closest_w = word
        return closest_w

    def get_cosine_similarities(self, vector):
        model_vectors = []
        unit_vector = gensim.matutils.unitvec(vector)
        for b, v_b in self.base.items():
            for g, v_g in self.grammar.items():
                target_unit_vector = gensim.matutils.unitvec(np.concatenate((v_b, v_g)))
                s = np.dot(unit_vector, target_unit_vector)
                model_vectors.append(s)
        return np.array(model_vectors)

    def cosine_similarity_pdf(self, similarity):
        r = np.sin(np.arccos(similarity))
        return nsphere.surface(r, self.vector_size - 1)


class TestTextModel(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = FinnishParser()

    def test_likeliest_base_form(self):
        parsed_words, sentence_start_indexes = self.parser.parse("Kissa kissalle kissan kissat")
        text_model = TextModel(parsed_words, sentence_start_indexes, base_size=3, grammar_size=2)
        self.assertEqual("kissa", text_model.likeliest(np.zeros(5)).base)

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
        vocabulary = text_model.get_vocabulary()
        self.assertEqual(3, len(vocabulary))
        priors = [prior for w, v, prior in vocabulary]
        self.assertEqual([2, 2, 3], sorted(priors))
        base_forms = [w.base for w, v, prior in vocabulary]
        self.assertEqual(["kissa", "kissa", "kissa"], base_forms)
        first_vector = vocabulary[0][1]
        self.assertEqual(5, len(first_vector))
