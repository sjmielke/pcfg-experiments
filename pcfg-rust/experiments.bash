#! /bin/bash

#echo $''
echo $'language\ttrainsize\tunbin_nts\tbin_nts\toov_handling\tuniform_oov_prob\tfeature_structures\ttesttagsfile\tnbesttags\tdualmono_pad\tlogcompvalues\tkeepafterdash\teta\talpha\tbeta\tkappa\tall_terms_fallback\texhaustive\tgram_ext_bin\tcky_prep\tcky_terms\tcky_higher\toov_words\toov_sents\tparsefails\tfmeasure\tfmeasure (fail ok)\ttagaccuracy'

PCFGR="/home/student/mielke/pcfg-experiments/pcfg-rust/target/release/pcfg-rust --wsjpath=/home/student/mielke/ptb3/parsed/mrg/wsj --spmrlpath=/home/student/mielke/SPMRL_SHARED_2014"

#ETAVALS="0.0 0.001 0.003 0.006 0.01 0.03 0.06 0.1 0.3 0.6 0.95 1.0"
 ETAVALS="0.0 0.006 0.06 0.6 1.0"
BETAVALS="1.0 2.0 5.0 10.0 20.0"

run-baselines() {
	language="$1"
	trainsize="$2"

	$PCFGR --language=$language --trainsize=$trainsize --oovhandling=zero&
	$PCFGR --language=$language --trainsize=$trainsize --oovhandling=uniform &
	$PCFGR --language=$language --trainsize=$trainsize --oovhandling=marginal-all &
	$PCFGR --language=$language --trainsize=$trainsize &
	wait
}

feat-goldtags() {
	for beta in $BETAVALS; do
		for eta in $ETAVALS; do
			$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=postagsonly &
		done
		wait
	done
}

feat-varitags() {
	languc=$(echo $1 | tr [a-z] [A-Z])
	for suffix in $2.gold "$2.pred" "40472.pred" "$2.pred --nbesttags" "40472.pred --nbesttags"; do
		for beta in $BETAVALS; do
			for eta in $ETAVALS; do
				$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=postagsonly --testtagsfile=../pos-tagging/data/spmrl.$languc.dev.sklearn_tagged.$suffix &
			done
			wait
		done
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

feat-ngrams() {
	for padmode in '' '--dualmono-pad'; do
		for kappa in 1 2 3 5 10; do
			for beta in $BETAVALS; do
				for eta in $ETAVALS; do
					$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=ngrams --kappa=$kappa $padmode &
				done
				wait
			done
		done
	done
}

feat-levenshtein() {
	for beta in $BETAVALS; do
		for eta in $ETAVALS; do
			$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=levenshtein --beta=$beta &
		done
		wait
	done
}

for trainsize in 100 500 1000 5000 10000 40472; do
	# run-baselines    German "$trainsize"
	# feat-goldtags    German "$trainsize"
	# feat-varitags    German "$trainsize"
	# feat-lcsratio    German "$trainsize"
	feat-ngrams      German "$trainsize"
	# feat-levenshtein German "$trainsize"
done

# # German defaults alpha-tuning
# for trainsize in 40472 10000 5000 1000 500 100; do
# 	for alpha in 0.0 0.1 0.2 0.3 0.4 0.5; do
# 		$PCFGR --trainsize=$trainsize --featurestructures=lcsratio --alpha=$alpha &
# 	done
# 	wait
# 	for alpha in 0.6 0.7 0.8 0.9 1.0; do
# 		$PCFGR --trainsize=$trainsize --featurestructures=lcsratio --alpha=$alpha &
# 	done
# 	wait
# done
