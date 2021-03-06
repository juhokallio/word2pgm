from omorfi.omorfi import Omorfi
import re
import unittest
from collections import namedtuple
from nltk.tokenize import RegexpTokenizer

AnalysedWord = namedtuple("AnalysedWord", "base grammar")

class FinnishParser:

    def __init__(self):
        self.omorfi = Omorfi()
        self.omorfi.load_from_dir()
        self.tokenizer = RegexpTokenizer('\w+\-\w+|\w+|\$[\d\.]+|\.\.\.|[",!\.\(\)]|\S+')

    @staticmethod
    def omorfi_to_base(omorfi_form):
        return re.search(r"\[WORD_ID=(.*?)\]", omorfi_form).group(1)

    @staticmethod
    def omorfi_to_grammar(omorfi_form):
        return re.sub(r"\[WORD_ID=.*?\]", "", omorfi_form)

    def tokenize(self, text):
        text = re.sub("\[\d+\]|\ufeff", "", text)
        return self.tokenizer.tokenize(text)

    def get_sentence_start_indexes(self, tokens):
        start_indexes = []
        sentence_ended = False
        sentence_end_regex = r"\.\.\.|[\.!\?:;]"
        for i, token in enumerate(tokens):
            if re.match(sentence_end_regex, token):
                sentence_ended = True
            else:
                if sentence_ended:
                    start_indexes.append(i)
                sentence_ended = False
        return start_indexes

    def parse(self, text):
        tokens = self.tokenize(text)
        parsed_words = [self.analyse(t) for t in tokens]
        sentence_start_indexes = self.get_sentence_start_indexes(tokens)
        return parsed_words, tokens, sentence_start_indexes

    def analyse(self, word):
        omorfi_form = self.omorfi.analyse(word)
        first_form = omorfi_form[0][0]
        return AnalysedWord(self.omorfi_to_base(first_form), self.omorfi_to_grammar(first_form))

    def is_valid_word(self, word):
        return word.grammar != "[GUESS=UNKNOWN][WEIGHT=inf]"


class TestParsing(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.parser = FinnishParser()

    def test_get_sentence_start_indexes(self):
        tokens = ["Sataa", ".", "Koira", "jäi", "ulos", "!", "Mitähän", "nyt", "?", "Surettaa", "...", "Sataa", "."]
        self.assertEqual(self.parser.get_sentence_start_indexes(tokens), [2, 6, 9, 11])

    def test_tokenize_simple(self):
        self.assertEqual(self.parser.tokenize("Koiran ruoka on valmiina."), ["Koiran", "ruoka", "on", "valmiina", "."])

    def test_tokenize_bad_stuff_removal(self):
        self.assertEqual(self.parser.tokenize("Koira [2] haukkuu"), ["Koira", "haukkuu"])
        self.assertEqual(self.parser.tokenize("\ufeff \ufeffKoira"), ["Koira"])

    def test_tokenize_compounds(self):
        self.assertEqual(self.parser.tokenize("sota-aikana"), ["sota-aikana"])

    def test_tokenize_with_line_break(self):
        self.assertEqual(self.parser.tokenize("koira\nkissa"), ["koira", "kissa"])
        self.assertEqual(self.parser.tokenize("koira\rkissa"), ["koira", "kissa"])

    def test_tokenize_ellipsis(self):
        self.assertEqual(self.parser.tokenize("nukkuu..."), ["nukkuu", "..."])

    def test_tokenize_punctuation_with_missing_space(self):
        self.assertEqual(self.parser.tokenize("\"hän,ruhtinas"), ["\"", "hän", ",", "ruhtinas"])
        self.assertEqual(self.parser.tokenize("hän!ruhtinas"), ["hän", "!", "ruhtinas"])
        self.assertEqual(self.parser.tokenize("hän.ruhtinas"), ["hän", ".", "ruhtinas"])

    def test_tokenize_brackets(self):
        self.assertEqual(self.parser.tokenize("(mies)"), ["(", "mies", ")"])

    def test_omorfi_to_base(self):
        self.assertEqual("koira", FinnishParser.omorfi_to_base("[WORD_ID=koiralle][UPOS=NOUN][NUM=SG][CASE=ALL]"))

    def test_omorfi_to_base(self):
        self.assertEqual("[UPOS=NOUN][NUM=SG][CASE=ALL]", FinnishParser.omorfi_to_grammar("[WORD_ID=koiralle][UPOS=NOUN][NUM=SG][CASE=ALL]"))

    def test_analyse(self):
        kissalle = self.parser.analyse("kissalle")
        self.assertEqual("kissa", kissalle.base)
        self.assertEqual("[UPOS=NOUN][NUM=SG][CASE=ALL]", kissalle.grammar)

        pilkku = self.parser.analyse(",")
        self.assertEqual(",", pilkku.base)
        self.assertEqual("[UPOS=PUNCT][BOUNDARY=CLAUSE][SUBCAT=COMMA]", pilkku.grammar)

    def test_analyse_punctuation(self):
        ellipsis = self.parser.analyse("...")
        self.assertEqual(ellipsis.base, "...")
        self.assertEqual(ellipsis.grammar, "[UPOS=PUNCT][BOUNDARY=SENTENCE]")

    def test_is_valid_word(self):
        self.assertTrue(self.parser.is_valid_word(self.parser.analyse("koiralle")))
        self.assertTrue(self.parser.is_valid_word(self.parser.analyse("alkaa")))

        self.assertFalse(self.parser.is_valid_word(self.parser.analyse("koirelle")))
