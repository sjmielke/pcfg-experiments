#! /bin/bash

#echo $''
echo $'language\ttrainsize\tunbin_nts\tbin_nts\toov_handling\tuniform_oov_prob\tfeature_structures\ttesttagsfile\tnbesttags\tdualmono_pad\tlogcompvalues\tkeepafterdash\teta\talpha\tbeta\tkappa\tomega\ttau\tall_terms_fallback\tonly_oovs_soft\texhaustive\tgram_ext_bin\tcky_prep\tcky_terms\tcky_higher\toov_words\toov_sents\tparsefails\tfmeasure\tfmeasure (fail ok)\ttagaccuracy'

PCFGR="/home/student/mielke/pcfg-experiments/pcfg-rust/target/release/pcfg-rust --wsjpath=/home/student/mielke/ptb3/parsed/mrg/wsj --spmrlpath=/home/student/mielke/SPMRL_SHARED_2014"

#   ETAVALS="0.0 0.001 0.003 0.006 0.01 0.03 0.06 0.1 0.3 0.6 0.95 1.0"
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

feat-prefixsuffix() {
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

feat-levenshtein() {
	for beta in $BETAVALS; do
		for eta in $ETAVALS; do
			$PCFGR --language=$1 --trainsize=$2 --eta=$eta --beta=$beta --featurestructures=levenshtein --beta=$beta &
		done
		wait
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

for trainsize in $TRAINSIZES; do
# 	run-baselines             German "$trainsize"
# 	feat-goldtags             German "$trainsize"
# 	feat-varitags             German "$trainsize"
# 	feat-lcsratio             German "$trainsize"
	feat-prefixsuffix         German "$trainsize"
# 	lcsratio-alphatune        German "$trainsize"
# 	feat-levenshtein          German "$trainsize"
# 	feat-ngrams               German "$trainsize"
# 	ngrams-kappatune-constant German "$trainsize"
# 	ngrams-kappatune-optimal  German "$trainsize"
done

