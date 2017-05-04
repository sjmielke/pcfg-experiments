from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

n = 100
train_path = "/tmp/2-21.tagged"
test_path = "/tmp/22.tagged"

def features(sentence, index):
    """ sentence: [w1, w2, ...], index: the index of the word """
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '<s>' if index == 0 else sentence[index - 1],
        'next_word': '</s>' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }

def gen_corpus(fn):
    alltags = set()
    with open(fn) as f:
        returnset = []
        for l in f.read().splitlines()[:n]:
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
    return returnset

def read_prepared_corpus(tt):
    pref = "/tmp/ptb.2-21" if tt == "train" else "/tmp/ptb.22"
    with open(pref + ".lex") as lex,\
         open(pref + ".tag") as tag:
        l = zip(lex.read().splitlines(), tag.read().splitlines())
        l = [(lexline.split(), tagline.split()) for (lexline, tagline) in l]
    if tt == "train":
        return l[0:n]
    else:
        return l

def transform_to_dataset(tagged_sentences):
    X, y = [], []

    for words, tags in tagged_sentences:
        for index, word  in enumerate(words):
            X.append(features(words, index))
            y.append(tags[index])
    return X, y
 
        
def pos_tag(sentence):
    tags = clf.predict([features(sentence, index) for index in range(len(sentence))])
    return list(zip(sentence, tags))

#training_sentences = list(gen_corpus(train_path))
#test_sentences     = list(gen_corpus(test_path))

training_sentences = list(read_prepared_corpus("train"))
test_sentences     = list(read_prepared_corpus("test"))

X, y = transform_to_dataset(training_sentences)

clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=True)),
    ('classifier',  LogisticRegression(n_jobs=4, max_iter=200, verbose=True))
])
clf.fit(X, y)

X_test, y_test = transform_to_dataset(test_sentences)
print( "Accuracy:", clf.score(X_test, y_test)) # Accuracy: 0.951851851852

y_pred, y_true = [], []
for words, tags in test_sentences:
    for ((word, pos), gold) in zip(pos_tag(words), tags):
        y_pred.append(pos)
        y_true.append(gold)
print(classification_report(y_true, y_pred))

with open(f'/tmp/sklearn.2-21.tagged.{n}.word', 'w') as fw,\
     open(f'/tmp/sklearn.2-21.tagged.{n}.pred', 'w') as ft,\
     open(f'/tmp/sklearn.2-21.tagged.{n}.gold', 'w') as fg:
    right = 0
    wrong = 0
    for (ws, gold) in training_sentences:
        pred = pos_tag(ws)
        print(" ".join([w for w, t in pred]), file = fw)
        print(" ".join([t for w, t in pred]), file = ft)
        print(" ".join(gold), file = fg)
        for (g, (_, p)) in zip(gold, pred):
            if g == p:
                right += 1
            else:
                wrong += 1
    acc = 100 * right / (right + wrong)
    print(f"SJM Accuracy: {acc:.2f}")

with open(f'/tmp/sklearn.22.tagged.{n}.word', 'w') as fw,\
     open(f'/tmp/sklearn.22.tagged.{n}.pred', 'w') as ft,\
     open(f'/tmp/sklearn.22.tagged.{n}.gold', 'w') as fg:
    right = 0
    wrong = 0
    for (ws, gold) in test_sentences:
        pred = list(pos_tag(ws))
        print(" ".join([w for w, t in pred]), file = fw)
        print(" ".join([t for w, t in pred]), file = ft)
        print(" ".join(gold), file = fg)
        for (g, (_, p)) in zip(gold, pred):
            if g == p:
                right += 1
            else:
                wrong += 1
    acc = 100 * right / (right + wrong)
    print(f"SJM Accuracy: {acc:.2f}")
