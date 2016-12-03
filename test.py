from omorfi.omorfi import Omorfi
import re
import gensim

omorfi = Omorfi()
omorfi.load_from_dir()

def analyse(word):
    return omorfi.analyse(word)

def parse(omorfi_form):
    infos = [i.split("=") for i in re.findall(r"\[(.+?=.+?)\]", omorfi_form)]
    return {i[0]:i[1] for i in infos}

def word_to_base(word):
    return parse(analyse(word)[0][0])["WORD_ID"]

def create_word2vec_model(sentences):
    return gensim.models.Word2Vec(sentences, min_count=1)

def create_sentences(text):
    sentences = [s.split() for s in text.split(".")]
    return [[word_to_base(w) for w in s] for s in sentences]

def read_file(file_name):
    with open(file_name, "r") as myfile:
        text = myfile.read().replace('\n', '')
    return text

def test_similarity(model, w1, w2):
    print("{}={} {}".format(w1, w2, model.similarity(w1, w2)))

analysed = analyse("koiralle")
print(analysed)
parsed = parse(analysed[0][0])
print(parsed)

print(word_to_base("kissan"))

text = read_file("data/finnish/pg45448.txt")
print("Read {} words of training data".format(len(text)))
sentences = create_sentences(text)
print("Parsed into {} sentences".format(len(sentences)))
model = create_word2vec_model(sentences)
print("Model trained")
test_similarity(model, "koira", "kissa")
test_similarity(model, "juosta", "mies")
test_similarity(model, "nainen", "mies")
test_similarity(model, "hän", "se")
test_similarity(model, "mutta", "koska")

