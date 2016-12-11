import gensim
import numpy as np
import itertools
import unittest
import pdb
from parsing import FinnishParser, AnalysedWord


class TextModel:

    def __init__(self, parsed_words):
        self.base = self.create_word2vec_model(self.split_to_sentences(parsed_words, "base"))
        self.grammar = self.create_word2vec_model(self.split_to_sentences(parsed_words, "grammar"))
        self.base_length = len(list(self.base.values())[0]) if bool(self.base) else 0
        self.grammar_length = len(list(self.grammar.values())[0]) if bool(self.grammar) else 0
        self.size = self.base_length + self.grammar_length

    @staticmethod
    def create_word2vec_model(sentences):
        model = gensim.models.Word2Vec(sentences, size=15, min_count=1, iter=10, workers=2)
        model.init_sims(replace=True)
        positions = {w: model[w] for w in model.vocab}
        return positions

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
        return list(itertools.chain(self.base[word.base], self.grammar[word.grammar]))

    def likeliest(self, model, vector):
        cleaned_target_v = gensim.matutils.unitvec(vector)
        closest_w = None
        similarity = 0
        for w, v in model.items():
            cleaned_v = gensim.matutils.unitvec(v)
            s = np.dot(cleaned_v, cleaned_target_v)
            if (closest_w is None) or (similarity < s):
                similarity = s
                closest_w = w
        return closest_w

    def likeliest_parsed_word(self, vector):
        print("Finding likeliest for {}".format(vector))
        base = self.likeliest(self.base, vector[:self.base_length])
        grammar = self.likeliest(self.grammar, vector[-self.grammar_length:])
        return AnalysedWord(base, grammar)


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
