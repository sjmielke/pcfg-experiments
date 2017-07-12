#! /bin/bash

cd "/home/student/mielke/pcfg-experiments/pcfg-rust"

PCFGR="target/release/pcfg-rust --wsjpath=/home/student/mielke/ptb3/parsed/mrg/wsj --spmrlpath=/home/student/mielke/SPMRL_SHARED_2014"
LMPLZ="/home/sjm/programming/mosesdecoder/bin/lmplz"

#ETAVALS="0.0 0.001 0.003 0.006 0.01 0.03 0.06 0.1 0.3 0.6 0.95 1.0"
ETAVALS="0.0 0.001 0.006 0.01 0.06 0.1 0.6 1.0"
BETAVALS="1.0 2.0 5.0 10.0 20.0"
#TRAINSIZES="100 500 1000 5000 10000 40472"
TRAINSIZES="100 500 1000 10000"

run-baselines() {
	language="$1"
	trainsize="$2"

	$PCFGR --language=$language --trainsize=$trainsize --oovhandling=zero&
	$PCFGR --language=$language --trainsize=$trainsize --oovhandling=uniform &
	$PCFGR --language=$language --trainsize=$trainsize --oovhandling=marginal-all &
	$PCFGR --language=$language --trainsize=$trainsize &
	wait
}

########################## HYPERPARAM ROUTINES ############################

feat-goldtags() {
	for beta in $BETAVALS; do
		for eta in $ETAVALS; do
			$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=postagsonly &
		done
		wait
	done
}

feat-varitags-1best() {
	languc=$(echo $1 | tr [a-z] [A-Z])
	for suffix in "$2.gold" "$2.pred" "40472.pred"; do
		for eta in $ETAVALS; do
			$PCFGR --language=$1 --trainsize=$2 --eta=$eta --featurestructures=postagsonly --testtagsfile=../pos-tagging/data/spmrl.$languc.dev.sklearn_tagged.$suffix &
		done
		wait
	done
}

feat-varitags-nbest() {
	languc=$(echo $1 | tr [a-z] [A-Z])
	for suffix in "$2.pred --nbesttags=nbesttags" "40472.pred --nbesttags=nbesttags"; do
		for beta in 0.1 0.5 1.0 1.5 2.0 5.0; do
			for eta in $ETAVALS; do
				$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=postagsonly --testtagsfile=../pos-tagging/data/spmrl.$languc.dev.sklearn_tagged.$suffix &
			done
			wait
		done
	done
}

feat-varitags-faux-nbest() {
	languc=$(echo $1 | tr [a-z] [A-Z])
	for suffix in "$2.gold --nbesttags=faux-nbesttags" "$2.pred --nbesttags=faux-nbesttags" "40472.pred --nbesttags=faux-nbesttags"; do
		for beta in 0.1 0.5 1.0 1.5 2.0 5.0; do
			for eta in $ETAVALS; do
				$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=postagsonly --testtagsfile=../pos-tagging/data/spmrl.$languc.dev.sklearn_tagged.$suffix &
			done
			wait
		done
	done
}

feat-brown-1best() {
	NCLUSTERS="$1"
	LANG="$2"
	
	languc=$(echo "$LANG" | tr [a-z] [A-Z])
	for eta in $ETAVALS; do
		$PCFGR --language=$2 --trainsize=$3 --eta=$eta --featurestructures=postagsonly --testtagsfile=../brown/SPMRL.$languc.dev.c${NCLUSTERS}.browntagged --word2tagdictfile=../brown/SPMRL.$languc.train.word2tag.c${NCLUSTERS} &
	done
	wait
}

feat-brown-faux-nbest() {
	NCLUSTERS="$1"
	LANG="$2"
	
	languc=$(echo "$LANG" | tr [a-z] [A-Z])
	for beta in 0.5 1 1.5 2 5; do
		for eta in $ETAVALS; do
			$PCFGR --language=$2 --trainsize=$3 --eta=$eta --beta=$beta --featurestructures=postagsonly --testtagsfile=../brown/SPMRL.$languc.dev.c${NCLUSTERS}.browntagged --word2tagdictfile=../brown/SPMRL.$languc.train.word2tag.c${NCLUSTERS} --nbesttags=faux-nbesttags &
		done
		wait
	done
}

feat-freq-cont() {
	for beta in 1 10 100 1000 10000 100000 1000000; do
		for eta in $ETAVALS; do
			$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=freq-cont &
		done
		wait
	done
}

feat-lcsratio() {
	for beta in $BETAVALS; do
		for eta in $ETAVALS; do
			$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=lcsratio &
		done
		wait
	done
}

lcsratio-alphatune() {
	for alpha in 0.0 0.1 0.2 0.3 0.4 0.5; do
		$PCFGR --language=$1 --trainsize=$2 --trainsize=$trainsize --featurestructures=lcsratio --alpha=$alpha &
	done
	wait
	for alpha in 0.6 0.7 0.8 0.9 1.0; do
		$PCFGR --language=$1 --trainsize=$2 --trainsize=$trainsize --featurestructures=lcsratio --alpha=$alpha &
	done
	wait
}

feat-prefixsuffix-eta-beta-tau() {
	# not tuning omega and alpha :(
	for beta in $BETAVALS; do
		for tau in 0.01 0.1 0.25 0.5 0.75 1.0; do
			for eta in $ETAVALS; do
				$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=prefixsuffix --tau=$tau &
			done
			wait
		done
	done
}

feat-prefixsuffix-omega-alpha() {
	# not tuning eta, beta and tau :(
	for omega in 0.0 0.125 0.25 0.375 0.5 0.625 0.75 0.875 1.0; do
		for alpha in 0.0 0.2 0.4 0.6 0.8 1.0; do
			$PCFGR --language=$1 --trainsize=$2 --eta=0.6 --beta=10 --featurestructures=prefixsuffix --tau=0.5 --alpha=$alpha --omega=$omega &
		done
		wait
	done
}

feat-levenshtein() {
	for alpha in 0.0 0.2 0.4 0.6 0.8 1.0; do
		for beta in $BETAVALS; do
			for eta in $ETAVALS; do
				$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=levenshtein --alpha=$alpha &
			done
			wait
		done
	done
}

feat-ngrams() {
	for padmode in '' '--dualmono-pad'; do
		for kappa in 1 2 3 4 5 10; do
			for beta in $BETAVALS; do
				for eta in $ETAVALS; do
					$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=ngrams --kappa=$kappa $padmode &
				done
				wait
			done
		done
	done
}

feat-affixdice() {
	for segmenter in morfessor bpe; do
		for chi in 0.0001 0.1 0.2 0.333 0.5 0.9999; do
			for beta in 0.1 0.5 $BETAVALS; do
				for eta in $ETAVALS; do
					$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=affixdice --morftagfileprefix=../${segmenter}/SPMRL --chi=$chi &
				done
				wait
			done
		done
	done
}

ngrams-kappatune-constant() {
	for kappa in 1 2 3 4 5 10 20 50 ; do
		$PCFGR --language=$1 --trainsize=$2 --featurestructures=ngrams --kappa=$kappa &
	  $PCFGR --language=$1 --trainsize=$2 --featurestructures=ngrams --kappa=$kappa --dualmono-pad &
	  wait
	done
}

ngrams-kappatune-optimal() {
	for padding in '' '--dualmono-pad'; do
		kappas=( 1 2 3 4 5 10 )
		if   [[ $2 == 100 ]];   then etas=( 0.1  0.1 0.6 1   1   1   ); betas=( 10 5  5  5  5  5 )
		elif [[ $2 == 500 ]];   then etas=( 0.1  1   1   0.6 0.6 1   ); betas=( 20 10 10 5  5  5 )
		elif [[ $2 == 1000 ]];  then etas=( 0.1  1   1   1   1   1   ); betas=( 20 10 10 10 5  5 )
		elif [[ $2 == 10000 ]]; then etas=( 0.01 1   1   1   1   1   ); betas=( 20 20 10 10 10 10 )
		else echo "#\n#\n#\n#\n#\n# illegal train size\n#\n#\n#\n#\n#\n#\n"
		fi
		for i in 0 1 2 3 4 5; do
			echo $PCFGR --language=$1 --trainsize=$2 --featurestructures=ngrams --kappa=${kappas[$i]} --eta=${etas[$i]} --beta=${betas[$i]} $padding &
		done
		wait
	done
}

tune() {
	echo $'language\ttrainsize\tunbin_nts\tbin_nts\toov_handling\tuniform_oov_prob\tfeature_structures\ttesttagsfile\tword2tagdictfile\tmorftagfileprefix\tnbesttags\tdualmono_pad\tlogcompvalues\tkeepafterdash\teta\talpha\tbeta\tkappa\tomega\ttau\tmu\tchi\tall_terms_fallback\tonly_oovs_soft\texhaustive\tgram_ext_bin\tcky_prep\tcky_terms\tcky_higher\toov_words\toov_sents\tparsefails\tfmeasure\tfmeasure (fail ok)\ttagaccuracy'
	
	for trainsize in $TRAINSIZES; do
		# run-baselines                  German "$trainsize"
		# feat-goldtags                  German "$trainsize"
		# feat-varitags-1best            German "$trainsize"
		# feat-varitags-nbest            German "$trainsize"
		# feat-varitags-faux-nbest       German "$trainsize"
		# feat-brown-1best 100           German "$trainsize"
		# feat-brown-1best 1000          German "$trainsize"
		# feat-brown-faux-nbest 100      German "$trainsize"
		# feat-brown-faux-nbest 1000     German "$trainsize"
		feat-freq-cont                 German "$trainsize"
		# feat-lcsratio                  German "$trainsize"
		# feat-prefixsuffix-eta-beta-tau German "$trainsize"
		# feat-prefixsuffix-omega-alpha  German "$trainsize"
		# feat-prefixsuffix              German "$trainsize"
		# lcsratio-alphatune             German "$trainsize"
		# feat-levenshtein               German "$trainsize"
		# feat-ngrams                    German "$trainsize"
		# ngrams-kappatune-constant      German "$trainsize"
		# ngrams-kappatune-optimal       German "$trainsize"
		# feat-affixdice                 German "$trainsize"
	done
}

evalall() {
	echo $'language\ttrainsize\tunbin_nts\tbin_nts\toov_handling\tuniform_oov_prob\tfeature_structures\ttesttagsfile\tword2tagdictfile\tmorftagfileprefix\tnbesttags\tdualmono_pad\tlogcompvalues\tkeepafterdash\teta\talpha\tbeta\tkappa\tomega\ttau\tmu\tchi\tall_terms_fallback\tonly_oovs_soft\texhaustive\tgram_ext_bin\tcky_prep\tcky_terms\tcky_higher\toov_words\toov_sents\tparsefails\tfmeasure\tfmeasure (fail ok)\ttagaccuracy'
	
	languc=$(echo "$LANG" | tr [a-z] [A-Z])
	
	for trainsize in $TRAINSIZES; do
		$PCFGR --language=$1 --trainsize=$trainsize --eta=1.0   --beta=1.0 --featurestructures=postagsonly --nbesttags=faux-nbesttags &
		$PCFGR --language=$1 --trainsize=$trainsize --eta=0.006 --beta=1.5 --featurestructures=postagsonly --testtagsfile=../pos-tagging/data/spmrl.$languc.dev.sklearn_tagged.40472.pred --nbesttags=nbesttags &
		$PCFGR --language=$1 --trainsize=$trainsize --eta=0.01  --beta=1.0 --featurestructures=postagsonly --testtagsfile=../brown/SPMRL.${languc}.dev.c100.browntagged --word2tagdictfile=../brown/SPMRL.${languc}.train.word2tag.c100 --nbesttags=faux-nbesttags &
		$PCFGR --language=$1 --trainsize=$trainsize --eta=0.01  --beta=1.0 --featurestructures=postagsonly --testtagsfile=../brown/SPMRL.${languc}.dev.c1000.browntagged --word2tagdictfile=../brown/SPMRL.${languc}.train.word2tag.c1000 --nbesttags=faux-nbesttags &
		$PCFGR --language=$1 --trainsize=$trainsize --eta=0.1   --beta=10  --featurestructures=lcsratio &
		wait
		$PCFGR --language=$1 --trainsize=$trainsize --eta=0.6   --beta=10  --featurestructures=prefixsuffix --tau=0.5 &
		$PCFGR --language=$1 --trainsize=$trainsize --eta=0.1   --beta=10  --featurestructures=levenshtein &
		$PCFGR --language=$1 --trainsize=$trainsize --eta=1.0   --beta=10  --featurestructures=ngrams --kappa=4 &
		$PCFGR --language=$1 --trainsize=$trainsize --eta=0.1   --beta=5   --featurestructures=affixdice --morftagfileprefix=../morfessor/SPMRL --chi=0.5 &
		$PCFGR --language=$1 --trainsize=$trainsize --eta=0.1   --beta=5   --featurestructures=affixdice --morftagfileprefix=../bpe/SPMRL &
		wait
	done
}

########################## MORFESSOR ############################

morfessor-my-train() {
	TRAINING_TEXT="$1"
	MODEL_PREFIX="$2"
	
	morfessor-train "$TRAINING_TEXT" -S "${MODEL_PREFIX}.baseline.gz"
	
	flatcat-train "${MODEL_PREFIX}.baseline.gz" \
		--category-separator '#' \
		-p 100 \
		-s "${MODEL_PREFIX}.fc_analysis.tar.gz" \
		--save-parameters "${MODEL_PREFIX}.fc_parameters.txt"
}

morfessor-my-segment() {
	MODEL_PREFIX="$1"
	INCOMING="$2"
	OUTGOING="$3"
	
	flatcat-segment \
		--load-parameters "${MODEL_PREFIX}.fc_parameters.txt" \
		--output-format '{analysis} ' \
		--output-categories \
		--output-newlines \
		--output-category-separator '#' \
		--category-separator '#' \
		--remove-nonmorpheme \
		"${MODEL_PREFIX}.fc_analysis.tar.gz" \
		"$INCOMING" \
	> "$OUTGOING"

	paste -d '\n' \
		<(sed 's/^/Source             : /;s/$/\nDELMEDELMEDELME/' "$INCOMING") \
		<(sed 's/^/morf-Source-flat   : /;s/$/\n/;s/#/|/g'        "$OUTGOING") \
		| sed '/DELMEDELMEDELME/d' \
		| head -n 100 \
		> "$OUTGOING.100.viz"
}

morfize() {
	LANG="$1"
	TEXTFILE_TRAIN=$(ls /home/sjm/documents/Uni/FuzzySP/spmrl-2014/data/${LANG}_SPMRL/gold/ptb/train/train.*.gold.ptb.tobeparsed.raw)
	TEXTFILE_DEVEL=$(ls /home/sjm/documents/Uni/FuzzySP/spmrl-2014/data/${LANG}_SPMRL/gold/ptb/dev/dev.*.gold.ptb.tobeparsed.raw)
	MODEL_PREFIX="/home/sjm/documents/Uni/FuzzySP/pcfg-experiments/morfessor/SPMRL.${LANG}"
	morfessor-my-train "$TEXTFILE_TRAIN" "$MODEL_PREFIX"
	
	# morfessor-my-segment "$MODEL_PREFIX" "$TEXTFILE_TRAIN" "$MODEL_PREFIX.train.flatcatized.txt"
	# morfessor-my-segment "$MODEL_PREFIX" "$TEXTFILE_DEVEL" "$MODEL_PREFIX.dev.flatcatized.txt"
	cat "$TEXTFILE_TRAIN" "$TEXTFILE_DEVEL" | tr ' ' $'\n' | sort -u > "$MODEL_PREFIX.vocab.txt"
	morfessor-my-segment "$MODEL_PREFIX" "$MODEL_PREFIX.vocab.txt" "$MODEL_PREFIX.vocab.flatcatized.txt"
}

########################## BPE ############################

bpe-train() {
	TRAINING_TEXT="$1"
	MODEL_PREFIX="$2"
	
	python "../subword-nmt/learn_bpe.py" \
		< "$TRAINING_TEXT" \
		> "$MODEL_PREFIX.bpe.codes"
}

bpe-segment() {
	MODEL_PREFIX="$1"
	INCOMING="$2"
	OUTGOING="$3"
	
	python "../subword-nmt/apply_bpe.py" \
		-c "$MODEL_PREFIX.bpe.codes" \
		-s '#STM' \
		< "$INCOMING" \
		| sed 's/$/#STM /' \
		> "$OUTGOING"

	paste -d '\n' \
		<(sed 's/^/Source             : /;s/$/\nDELMEDELMEDELME/' "$INCOMING") \
		<(sed 's/^/morf-Source-flat   : /;s/$/\n/;s/#/|/g'        "$OUTGOING") \
		| sed '/DELMEDELMEDELME/d' \
		| head -n 100 \
		> "$OUTGOING.100.viz"
}

bpeize() {
	LANG="$1"
	TEXTFILE_TRAIN=$(ls /home/sjm/documents/Uni/FuzzySP/spmrl-2014/data/${LANG}_SPMRL/gold/ptb/train/train.*.gold.ptb.tobeparsed.raw)
	TEXTFILE_DEVEL=$(ls /home/sjm/documents/Uni/FuzzySP/spmrl-2014/data/${LANG}_SPMRL/gold/ptb/dev/dev.*.gold.ptb.tobeparsed.raw)
	MODEL_PREFIX="/home/sjm/documents/Uni/FuzzySP/pcfg-experiments/bpe/SPMRL.${LANG}"
	
	bpe-train "$TEXTFILE_TRAIN" "$MODEL_PREFIX"
	
	cat "$TEXTFILE_TRAIN" "$TEXTFILE_DEVEL" | tr ' ' $'\n' | sort -u > "$MODEL_PREFIX.vocab.txt"
	bpe-segment "$MODEL_PREFIX" "$MODEL_PREFIX.vocab.txt" "$MODEL_PREFIX.vocab.flatcatized.txt"
}

########################## BROWN CLUSTERING ############################

brownize() {
	LANG="$1"
	NCLUSTERS="$2"
	
	TEXTFILE_TRAIN=$(ls /home/sjm/documents/Uni/FuzzySP/spmrl-2014/data/${LANG}_SPMRL/gold/ptb/train/train.*.gold.ptb.tobeparsed.raw)
	TEXTFILE_DEVEL=$(ls /home/sjm/documents/Uni/FuzzySP/spmrl-2014/data/${LANG}_SPMRL/gold/ptb/dev/dev.*.gold.ptb.tobeparsed.raw)
	
	cp -n "$TEXTFILE_TRAIN" "../brown/SPMRL.${LANG}.train"
	cp -n "$TEXTFILE_DEVEL" "../brown/SPMRL.${LANG}.dev"
	
	pushd ../brown-cluster/
		make
	popd
	
	pushd ../brown/
		if [ ! -s "SPMRL.${LANG}.train-c${NCLUSTERS}-p1.out/paths" ]; then
			cp -n "SPMRL.${LANG}.train" "SPMRL.${LANG}.train.txt"
			../brown-cluster/wcluster --text "SPMRL.${LANG}.train.txt" --c $NCLUSTERS
			rm "SPMRL.${LANG}.train.txt"
		fi
		
		if [ ! -s "SPMRL.${LANG}.brownclass.lm.arpa" ]; then
			python3 browntagger.py "SPMRL.${LANG}.train-c${NCLUSTERS}-p1.out/paths" "" \
				< "SPMRL.${LANG}.train" \
				> "SPMRL.${LANG}.train.c${NCLUSTERS}.browntagged"
			
			"$LMPLZ" \
				-o 5 \
				--discount_fallback \
				--text "SPMRL.${LANG}.train.c${NCLUSTERS}.browntagged" \
				--arpa "SPMRL.${LANG}.train.c${NCLUSTERS}.brownclass.lm.arpa"
			
			# Dict for parser
			cat "SPMRL.${LANG}.train" \
				| tr ' ' '\n' \
				| sed '/^$/d' \
				| sort -u \
				> "SPMRL.${LANG}.train.words"
			python3 browntagger.py "SPMRL.${LANG}.train-c${NCLUSTERS}-p1.out/paths" "" \
				< "SPMRL.${LANG}.train.words" \
				| sed 's#/0.0$##' \
				> "SPMRL.${LANG}.train.tags.c${NCLUSTERS}"
			paste "SPMRL.${LANG}.train.words" "SPMRL.${LANG}.train.tags.c${NCLUSTERS}" \
				> "SPMRL.${LANG}.train.word2tag.c${NCLUSTERS}"
			rm "SPMRL.${LANG}.train.words" "SPMRL.${LANG}.train.tags.c${NCLUSTERS}"
		fi
		
		python3 browntagger.py "SPMRL.${LANG}.train-c${NCLUSTERS}-p1.out/paths" "SPMRL.${LANG}.train.c${NCLUSTERS}.brownclass.lm.arpa" \
			< "../brown/SPMRL.${LANG}.dev" \
			> "../brown/SPMRL.${LANG}.dev.c${NCLUSTERS}.browntagged"
	popd
}

########################## BERKELEY PARSER ############################

berkeley-baseline() {
	lang="$1"
	smcycles="$2"
	
	javac edu/berkeley/nlp/PCFGLA/*.java
	
	# java \
	# 	edu.berkeley.nlp.PCFGLA.GrammarTrainer \
	# 	-path <(head -n 10000 /home/sjm/documents/Uni/FuzzySP/pure-treebanks/${lang}.train.tb) \
	# 	-out /mnt/hylia/bp/${lang}.bpgrammar.nosmooth \
	# 	-treebank SINGLEFILE \
	# 	-SMcycles ${smcycles} -sm1 0 -sm2 0
	
	java \
		edu.berkeley.nlp.PCFGLA.GrammarTrainer \
		-path <(head -n 10000 /home/sjm/documents/Uni/FuzzySP/pure-treebanks/${lang}.train.tb) \
		-out /mnt/hylia/bp/${lang}.bpgrammar.smooth \
		-treebank SINGLEFILE \
		-SMcycles ${smcycles}
	
	# java edu.berkeley.nlp.PCFGLA.WriteGrammarToTextFile /tmp/bp/grammar /tmp/bp/grammar
	
	# java edu.berkeley.nlp.PCFGLA.BerkeleyParser -gr /tmp/bp/grammar -inputFile <(echo "Ich sah ein Objekt in der Ferne .")
	
	#for smoothing in nosmooth smooth; do
	for smoothing in smooth; do
		java \
			edu.berkeley.nlp.PCFGLA.BerkeleyParser \
			-gr /mnt/hylia/bp/${lang}.bpgrammar.${smoothing} \
			< /home/sjm/documents/Uni/FuzzySP/pure-treebanks/${lang}.test.yield \
			> /mnt/hylia/bp/${lang}.test.yield.bpparsed.${smoothing}
		
		/home/sjm/documents/Uni/FuzzySP/pcfg-experiments/evalb_spmrl2013.final/evalb_spmrl \
			-L -X \
			/home/sjm/documents/Uni/FuzzySP/pure-treebanks/${lang}.test.tb \
			/mnt/hylia/bp/${lang}.test.yield.bpparsed.${smoothing} \
			> /mnt/hylia/bp/${lang}.test.yield.bpparsed.${smoothing}.evalb
	done

}

########################## MAIN ############################

# for lang in ENGLISH; do
# 	morfize $lang
# 	bpeize $lang
# # 	brownize $lang 1000
# # 	brownize $lang 100
# done

#brownize GERMAN 1000
#brownize GERMAN 100
#tune

for smcycles in 0; do
	for lang in English German Korean French Arabic; do
		berkeley-baseline $lang $smcycles
	done
done

# for lang in English German Korean French Arabic; do
# 	evalall $lang
# done