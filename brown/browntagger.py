#! /bin/python3

import sys
import kenlm
import itertools

[_, brownpaths, lm] = sys.argv

if lm != "":
    model = kenlm.Model(lm)
    print('Loaded {0}-gram model.'.format(model.order), file = sys.stderr)

brown = {}
clusters = set()
with open(brownpaths) as b:
    for l in b.readlines():
        [cluster, word, freq] = l.split('\t')
        assert(word not in brown)
        brown[word] = cluster
        clusters.add(cluster)
clusters = list(clusters)

for line in sys.stdin:
    line = line.strip()
    words = list(line.split())
    
    oovs = [w for w in words if w not in brown]
    finalsent = []
    if oovs != []:
        # print(f"OOVs: {oovs}, ", end = '', flush = True, file = sys.stderr)
        for i, w in enumerate(words):
            if w in brown:
                finalsent.append(brown[w])
            else:
                bestscore = -999999999999999999999999999
                bestcluster = None
                for c in clusters:
                    variant = words[0:i] + [c] + words[i+1:]
                    variant = " ".join(variant)
                    s = model.score(variant)
                    # print(f"  {variant}", file = sys.stderr)
                    # print(f"  ~~> {s}", file = sys.stderr)
                    if s > bestscore:
                        bestscore = s
                        bestcluster = c
                finalsent.append(bestcluster)
    else:
        # print("Unique sentence.", file = sys.stderr)
        finalsent = [brown[w] for w in words]
    print(" ".join([f"{c}/0.0" for c in finalsent]))
