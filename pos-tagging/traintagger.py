from random import shuffle

from spacy.vocab import Vocab
from spacy.tagger import Tagger
from spacy.tokens import Doc
from spacy.gold import GoldParse

from spacy.en.tag_map import TAG_MAP

def read_postagged(fn):
    alltags = set()
    with open(fn) as f:
        returnset = []
        for l in f.read().splitlines():
            ws, ts = [], []
            for wt in l.split():
                elems = wt.split('/')
                w = "/".join(elems[:-1])
                t = elems[-1]
                ws.append(w)
                ts.append(t)
                alltags.add(t)
            assert(len(ws) == len(ts))
            if ws != []:
                returnset.append((ws, ts))
    return returnset, alltags

trainset, alltags = read_postagged("/tmp/0-18.tagged")
testset , _       = read_postagged("/tmp/22.tagged")

def eval(t):
    right = 0
    wrong = 0
    for (ws, gold) in testset:
        doc = Doc(vocab, ws)
        t(doc)
        pred = [word.tag_ for word in doc]
        for (g, p) in zip(gold, pred):
            if g == p:
                right += 1
            else:
                wrong += 1
    acc = 100 * right / (right + wrong)
    print(f"Accuracy: {acc:.2f}")

tag_map = {t: {'pos': 'X'} for t in alltags}
#tag_map.update(TAG_MAP)
vocab = Vocab(tag_map=tag_map)

# Add all train words to vocab!
for (ws, _) in trainset + testset:
    for w in ws:
        _ = vocab[w]

tagger = Tagger(vocab)

for i in range(50):
    print(f"Epoch {i}:")
    for (ws, ts) in trainset:
        doc = Doc(vocab, words=ws)
        gold = GoldParse(doc, tags=ts)
        tagger.update(doc, gold)
    eval(tagger)
    tagger.model.end_training()
    eval(tagger)
    tagger.model.resume_training()
    shuffle(trainset)
