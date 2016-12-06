import gensim
import itertools
import unittest
from parsing import FinnishParser


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
        return list(itertools.chain(self.base[word.base], self.grammar[word.grammar]))


class TestTextModel(unittest.TestCase):

    def setUp(self):
        self.parser = FinnishParser()

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
