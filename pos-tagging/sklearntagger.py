from sklearn.feature_extraction import DictVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

pref = "/home/sjm/documents/Uni/FuzzySP/pcfg-experiments/pos-tagging/data/ptb."

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

def read_prepared_corpus(tt):
    sec = "2-21" if tt == "train" else "22"
    with open(pref + sec + ".lex") as lex,\
         open(pref + sec + ".tag") as tag:
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

def pos_tags_log_proba(sentence):
    tags = clf.predict_log_proba([features(sentence, index) for index in range(len(sentence))])
    bests = []
    outline = []
    for (w,l) in [(w, list(zip(clf.named_steps['classifier'].classes_, t))) for w, t in zip(sentence, tags)]:
        #print(w)
        s = []
        maxc = None
        maxp = -99999999999999999999999
        for (c, p) in l:
            if p > maxp:
                maxc = c
                maxp = p
            #print(f"\t{c}\t({p})")
            assert(';' not in c)
            assert('/' not in c)
            s.append(f"{c}/{p}")
        outline.append(";".join(s))
        bests.append(maxc)
    return bests, outline

for n in [10, 50, 100, 500, 1000, 5000, 10000, 20000, 39000]:
#for n in [500]:
    print(f"n = {n}")

    # Train classifier

    training_sentences = list(read_prepared_corpus("train"))
    clf = Pipeline([
        ('vectorizer', DictVectorizer(sparse=True)),
        ('classifier',  LogisticRegression(n_jobs=4, max_iter=200))
    ])
    X, y = transform_to_dataset(training_sentences)
    clf.fit(X, y)

    # Test model

    test_sentences = list(read_prepared_corpus("test"))
    X_test, y_test = transform_to_dataset(test_sentences)
    print( "Accuracy:", clf.score(X_test, y_test)) # Accuracy: 0.951851851852

    with open(pref + f'22.sklearn_tagged.{n}.word', 'w') as fw,\
         open(pref + f'22.sklearn_tagged.{n}.pred', 'w') as ft,\
         open(pref + f'22.sklearn_tagged.{n}.gold', 'w') as fg:
        right = 0
        wrong = 0
        for (ws, gold) in test_sentences:
            bests, pred = pos_tags_log_proba(ws)
            print(" ".join(ws), file = fw)
            print(" ".join(pred), file = ft)
            print(" ".join([f"{g}/0.0" for g in gold]), file = fg)
            #print(gold, bests)
            for (g, p) in zip(gold, bests):
                if g == p:
                    right += 1
                else:
                    wrong += 1
        acc = 100 * right / (right + wrong)
        print(f"SJM Accuracy: {acc:.2f}")
