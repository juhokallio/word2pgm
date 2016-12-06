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
        self.tokenizer = RegexpTokenizer('\w+|\$[\d\.]+|\S+')

    @staticmethod
    def omorfi_to_base(omorfi_form):
        return re.search(r"\[WORD_ID=(.*?)\]", omorfi_form).group(1)

    @staticmethod
    def omorfi_to_grammar(omorfi_form):
        return re.sub(r"\[WORD_ID=.*?\]", "", omorfi_form)

    def parse(self, text):
        return [self.analyse(w) for w in self.tokenizer.tokenize(text)]

    def analyse(self, word):
        omorfi_form = self.omorfi.analyse(word)
        first_form = omorfi_form[0][0]
        return AnalysedWord(self.omorfi_to_base(first_form), self.omorfi_to_grammar(first_form))


class TestParsing(unittest.TestCase):

    def test_omorfi_to_base(self):
        self.assertEqual("koira", FinnishParser.omorfi_to_base("[WORD_ID=koiralle][UPOS=NOUN][NUM=SG][CASE=ALL]"))

    def test_omorfi_to_base(self):
        self.assertEqual("[UPOS=NOUN][NUM=SG][CASE=ALL]", FinnishParser.omorfi_to_grammar("[WORD_ID=koiralle][UPOS=NOUN][NUM=SG][CASE=ALL]"))

    def test_analyse(self):
        parser = FinnishParser()

        kissalle = parser.analyse("kissalle")
        self.assertEqual("kissa", kissalle.base)
        self.assertEqual("[UPOS=NOUN][NUM=SG][CASE=ALL]", kissalle.grammar)

        pilkku = parser.analyse(",")
        self.assertEqual(",", pilkku.base)
        self.assertEqual("[UPOS=PUNCT][BOUNDARY=CLAUSE][SUBCAT=COMMA]", pilkku.grammar)
