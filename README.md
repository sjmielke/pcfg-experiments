# Soft matching for PCFG parsing

This repository contains code that was used in:
"Soft matching of terminals for syntactic parsing", Sebastian J. Mielke (2017)

Please make sure to refresh git submodules when cloning and adapt hardcoded paths in `experiments.bash` and `plots.py`.

## Most important files and directories

### experiments.bash

Contains functions to perform baseline parsing (using the Rust implementation), evaluate different hyperparameter combinations (using the Rust implementation), train and call Morfessor, train and call BPE segmentation, obtain Brown clusters, train and call the Berkeley Parser.

### plots.py

Generates all plots of the thesis from experimental logs (not distributed in this repository).

### pcfg-rust

The pcfg-rust folder contains the main implementation, excutable by calling `cargo run --release --` followed by any options.
For information about the options, call `cargo run --release -- --help`.

## Other files and directories

### bpe

Contains Byte Pair Encoding (BPE) data after calling the corresponding `experiments.bash` functions.

### brown

Contains Brown cluster data after calling the corresponding `experiments.bash` functions.

### brown-cluster

(Percy Liangs implementation)[https://github.com/percyliang/brown-cluster] of Brown clusters. Should be populated when refreshing git submodules.

### evalb_spmrl2013.final

The evalb implementation that we use. Adapted from Michael Collins' version and distributed by Seddah et al., 2013.

### morfessor

Contains Morfessor data after calling the corresponding `experiments.bash` functions.

### pcfg-python

PCFG extraction and simple bottom-up CKY parsing, about 5 times slower than the reimplementation in Rust, hence abandoned.

### pos-tagging

Contains `sklearntagger.py`, which we used to as the "external POS tagger".
Note that `traintagger.py` is a remnant of a failed attempt to do so using the spacy framework.

Also contains POS tag data in subdirectory `data/` after calling the corresponding `experiments.bash` functions.

### subword-nmt

(Rico Sennrichs implementation)[https://github.com/rsennrich/subword-nmt] of BPE segmentation. Should be populated when refreshing git submodules.
