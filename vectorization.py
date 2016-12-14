import gensim
import numpy as np
import itertools
import unittest
import pdb
from parsing import FinnishParser, AnalysedWord


class TextModel:

    def __init__(self, parsed_words):
        self.base = self.create_word2vec_model(self.split_to_sentences(parsed_words, "base"), 35)
        self.grammar = self.create_word2vec_model(self.split_to_sentences(parsed_words, "grammar"), 35)
        self.base_length = len(list(self.base.values())[0]) if bool(self.base) else 0
        self.grammar_length = len(list(self.grammar.values())[0]) if bool(self.grammar) else 0
        self.size = self.base_length + self.grammar_length
        self.counts = self.get_counts(parsed_words)
        self.counted_data_size = len(parsed_words)

    @staticmethod
    def create_word2vec_model(sentences, size):
        model = gensim.models.Word2Vec(sentences, size=size, min_count=1, iter=1500, workers=2)
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
    def split_to_sentences(words, dimension):
        sentences = []
        sentence = []
        for w in words:
            sentence.append(getattr(w, dimension))
            if w.base == ".":
                sentences.append(sentence)
                sentence = []
        return sentences

    def get_encounter_count(self, word):
        word_encountered = word.base in self.counts and word.grammar in self.counts[word.base]
        return self.counts[word.base][word.grammar] if word_encountered else 0

    def word_to_vector(self, word):
        return list(itertools.chain(self.base[word.base], self.grammar[word.grammar]))

    def likeliest(self, vector):
        cleaned_target_v = gensim.matutils.unitvec(vector)
        closest_w = None
        similarity = 0
        target_v_b = gensim.matutils.unitvec(vector[:self.base_length])
        target_v_g = gensim.matutils.unitvec(vector[-self.grammar_length:])
        for b, v_b in self.base.items():
            cleaned_v_b = gensim.matutils.unitvec(v_b)
            base_p = np.dot(cleaned_v_b, target_v_b)**2
            for g, v_g in self.grammar.items():
                cleaned_v_g = gensim.matutils.unitvec(v_g)
                grammar_p = np.dot(cleaned_v_g, target_v_g)**2
                word = AnalysedWord(b, g)
                s = base_p * grammar_p * self.get_encounter_count(word)
                if (closest_w is None) or (similarity < s):
                    similarity = s
                    closest_w = word
        return closest_w


class TestTextModel(unittest.TestCase):

    def setUp(self):
        self.parser = FinnishParser()

    def test_likeliest_base_form(self):
        parsed_words = self.parser.parse("Kissa kissalle kissan kissat.")
        text_model = TextModel(parsed_words)
        self.assertEqual("kissa", text_model.likeliest(text_model.base, np.zeros(5)))

    def test_init(self):
        parsed_words = self.parser.parse("Ohjelmointi on mukavaa. Se ei ikinä ärsytä.")
        self.assertEqual("ohjelmointi", parsed_words[0].base)

        text_model = TextModel(parsed_words)
        self.assertEqual(10, text_model.size)

        word_vector = text_model.word_to_vector(parsed_words[0])
        self.assertEqual(10, len(word_vector))

    def test_split_to_sentences(self):
        parsed_words = self.parser.parse("Kissa istui pöydällä.")
        self.assertEqual([["kissa", "istua", "pöytä", "."]], TextModel.split_to_sentences(parsed_words, "base"))

        parsed_words = self.parser.parse("Kissa, joka makaa.")
        self.assertEqual([["kissa", ",", "joka", "maata", "."]], TextModel.split_to_sentences(parsed_words, "base"))

        parsed_words = self.parser.parse("Kissa istui pöydällä. Satoi.")
        self.assertEqual([["kissa", "istua", "pöytä", "."], ["sataa", "."]], TextModel.split_to_sentences(parsed_words, "base"))
