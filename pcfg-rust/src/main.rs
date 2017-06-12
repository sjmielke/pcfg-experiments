#![feature(conservative_impl_trait)]

use std::collections::HashMap;
use std::f64; // log and exp
use std::process::Command;

extern crate tempdir;
use std::fs::File;
use std::io::Write;
use tempdir::TempDir;

extern crate argparse;
use argparse::{ArgumentParser, Store, StoreTrue};

extern crate ptb_reader;
use ptb_reader::PTBTree;

mod defs;
use defs::*;
mod extract;
mod featurestructures;
mod parse;

fn parsetree2ptbtree(ntdict: &HashMap<NT, String>, t: &ParseTree) -> PTBTree {
    match *t {
        ParseTree::TerminalNode { label } => {
            PTBTree::TerminalNode { label: label.to_string() }
        }
        ParseTree::InnerNode { label, ref children } => {
            let cs = children.iter().map(|c| parsetree2ptbtree(ntdict, c)).collect::<Vec<_>>();
            PTBTree::InnerNode { label: ntdict[&label].clone(), children: cs }
        }
    }
}

fn debinarize_parsetree<'a>(ntdict: &HashMap<NT, String>, t: &ParseTree<'a>) -> ParseTree<'a> {
    match *t {
        ParseTree::TerminalNode { label } => ParseTree::TerminalNode { label },
        ParseTree::InnerNode { label, ref children } => {
            // S(JJ _NNP_VBD_NN(NNP _VBD_NN(VBD NN))) => S(JJ NNP VBD NN)
            // Does this node contain children that are _-started?
            // Replace each of them with its children (recursively)!
            
            let mut newchildren = Vec::new();
            for c in children {
                match *c {
                    ParseTree::InnerNode { label, .. } => {
                        if ntdict[&label].chars().next().unwrap() == '_' {
                            // So this would be _NNP_VB_NN(NNP _VB_NN(VB(VBD..) NN))).
                            // It has be debinarized first!
                            let newchild = debinarize_parsetree(ntdict, c);
                            // Now we have _NNP_VB_NN(NNP VB(VBD..) NN), so just take its children!
                            if let ParseTree::InnerNode { children, .. } = newchild {
                                newchildren.extend(children)
                            } else {
                                unreachable!()
                            }
                        } else {
                            newchildren.push(debinarize_parsetree(ntdict, c))
                        }
                    }
                    _ => newchildren.push(debinarize_parsetree(ntdict, c))
                }
            }
            ParseTree::InnerNode { label, children: newchildren }
        }
    }
}

#[allow(dead_code)]
fn print_example(bin_ntdict: &HashMap<NT, String>, sent: &str, tree: &PTBTree, sorted_candidates: &[(NT, (f64, ParseTree))]) {
    if !sorted_candidates.is_empty() {
        println!("{}", sent);
        let mut got_s = false;
        // True parse
        println!("  {:28} -> {}", "", tree);
        // Algo
        for &(ref n, (p, ref ptree)) in sorted_candidates.iter().take(10) {
            println!("  {:10} ({:4.10}) -> {}", bin_ntdict[n], p, parsetree2ptbtree(bin_ntdict, &debinarize_parsetree(bin_ntdict, ptree)));
            if bin_ntdict[n] == "S" {
                got_s = true
            }
        }
        if sorted_candidates.len() > 10 {
            println!("\t...");
            if !got_s {
                for &(ref n, (p, ref ptree)) in sorted_candidates {
                    if bin_ntdict[n] == "S" {
                        println!("  {:10} ({:4.10}) -> {}", bin_ntdict[n], p, parsetree2ptbtree(bin_ntdict, &debinarize_parsetree(bin_ntdict, ptree)))
                    }
                }
            }
        }
    }
    else {
        println!("{}\n", sent);
    }
}

fn debinarize_and_sort_candidates<'a>(cell: &HashMap<NT, (f64, ParseTree<'a>)>, bin_ntdict: &HashMap<NT, String>) -> Vec<(NT, (f64, ParseTree<'a>))> {
    // Remove binarization traces
    let mut candidates: Vec<(NT, (f64, ParseTree))> = Vec::new();
    for (nt, &(p, ref ptree)) in cell {
        // Only keep candidates ending in proper NTs
        if bin_ntdict[nt].chars().next().unwrap() != '_' {
            // Remove inner binarization nodes
            candidates.push((*nt, (p, debinarize_parsetree(&bin_ntdict, ptree))))
        }
    }
    // Sort
    fn compare_results(r1: &(NT, (f64, ParseTree)), r2: &(NT, (f64, ParseTree))) -> std::cmp::Ordering {
        let &(nt1, (p1, ref pt1)) = r1;
        let &(nt2, (p2, ref pt2)) = r2;
        let c = p2.partial_cmp(&p1).unwrap_or(std::cmp::Ordering::Equal);
        match c {
            std::cmp::Ordering::Equal => (nt1,pt1).cmp(&(nt2,pt2)),
            _ => c
        }
    }
    candidates.sort_by(compare_results);
    candidates
}

fn eval_parses(testtrees: &Vec<PTBTree>, parses: Vec<Vec<(NT, (f64, PTBTree))>>, stats: &mut PCFGParsingStatistics) {
    // Save output for EVALB call
    let tmp_dir = TempDir::new("pcfg-rust").unwrap();
    let gold_path = tmp_dir.path().join("gold.txt");
    let best_or_fail_path = tmp_dir.path().join("best_or_fail.txt");
    let mut gold_file = File::create(&gold_path).unwrap();
    let mut best_or_fail_file = File::create(&best_or_fail_path).unwrap();
    
    for (gold_tree, candidates) in testtrees.iter().zip(parses.iter()) {
        //print_example(&bin_ntdict, &sent, &tree, &candidates);
        
        let gold_parse: String = format!("{}", gold_tree);
        let best_or_fail_parse: String = match candidates.get(0) {
            None => "".to_string(),
            Some(&(_, (_, ref parsetree))) => format!("{}", parsetree)
        };
        
        writeln!(gold_file, "{}", gold_parse).unwrap();
        writeln!(best_or_fail_file, "{}", best_or_fail_parse).unwrap();
    }
    
    // use std::io::prelude::*;
    // let _ = std::io::stdin().read(&mut [0u8]).unwrap();
    
    let out_strict = Command::new("../evalb_spmrl2013.final/evalb_spmrl").arg("-L").arg("-X").arg(&gold_path).arg(&best_or_fail_path).output().unwrap().stdout;
    let out_lenient = Command::new("../evalb_spmrl2013.final/evalb_spmrl").arg("-L").arg(&gold_path).arg(&best_or_fail_path).output().unwrap().stdout;
    fn run_call(out: Vec<u8>) -> (f64, f64) {
        let ress = String::from_utf8(out).unwrap().lines().last().unwrap().split('\t').map(|s| s.to_string()).collect::<Vec<String>>();
        assert_eq!(ress.len(), 8);
        let fmea = ress[0].split_whitespace().collect::<Vec<_>>();
        assert_eq!(fmea[0], "F1:");
        assert_eq!(fmea[2], "%");
        let taga = ress[3].split_whitespace().collect::<Vec<_>>();
        assert_eq!(taga[0], "POS:");
        assert_eq!(taga[2], "%");
        
        (fmea[1].parse().expect(&format!("F1:  {} % <- no number :(", fmea[1])),
         taga[1].parse().expect(&format!("POS:  {} % <- no number :(", taga[1])))
    }
    let strict = run_call(out_strict);
    stats.fmeasure = strict.0;
    stats.tagaccuracy = strict.1;
    stats.or_fail_fmeasure = run_call(out_lenient).0;
    
    stats.print(false);
    
    // safely close to get error messages just in case
    drop(gold_file);
    tmp_dir.close().unwrap();
}

fn extract_and_parse(wsj_path: &str, spmrl_path: &str, mut stats: &mut PCFGParsingStatistics) -> (Vec<PTBTree>, Vec<Vec<(NT, (f64, PTBTree))>>) {
    //println!("Now loading and processing all data!");
    let t = get_usertime();
    let ((bin_rules, bin_ntdict), initial_nts, pret_distr_all, pret_distr_le3, (testsents, testposs, testtrees)) = extract::get_data(wsj_path, spmrl_path, stats);
    stats.gram_ext_bin = get_usertime() - t;
    
    //println!("Now parsing!");
    let raw_parses = parse::agenda_cky_parse(&bin_rules, &bin_ntdict, &initial_nts, &testsents, &testposs, pret_distr_all, pret_distr_le3, &mut stats);
    let mut parses: Vec<Vec<(NT, (f64, PTBTree))>> = Vec::new();
    for cell in raw_parses {
        let candidates = debinarize_and_sort_candidates(&cell, &bin_ntdict);
        parses.push(candidates.into_iter().map(|(nt, res)| (nt, (res.0, parsetree2ptbtree(&bin_ntdict, &res.1)))).collect())
    }
    
    (testtrees, parses)
}

fn main() {
    let mut stats: PCFGParsingStatistics = PCFGParsingStatistics{
        // Placeholders
        gram_ext_bin:f64::NAN, cky_prep:f64::NAN, cky_terms:f64::NAN, cky_higher:f64::NAN, oov_words:0, oov_sents:0, parsefails:0, fmeasure:f64::NAN, or_fail_fmeasure:f64::NAN, tagaccuracy:f64::NAN, unbin_nts:std::usize::MAX, bin_nts:std::usize::MAX,
        // Values
        language: "German".to_string(),
        trainsize: 1000,
        oov_handling: OOVHandling::MarginalLe3,
        feature_structures: "exactmatch".to_string(),
        testtagsfile: "".to_string(),
        nbesttags: false,
        dualmono_pad: false,
        logcompvalues: false,
        keepafterdash: false,
        eta: 0.06,
        alpha: 0.2,
        beta: 10.0,
        kappa: 3,
        all_terms_fallback: false,
        only_oovs_soft: false,
        exhaustive: false,
        uniform_oov_prob: -10.0
    };
    
    let mut wsj_path: String = "/home/sjm/documents/Uni/FuzzySP/treebank-3_LDC99T42/treebank_3/parsed/mrg/wsj".to_string();
    let mut spmrl_path: String = "/home/sjm/documents/Uni/FuzzySP/spmrl-2014/data".to_string();
    
    { // this block limits scope of borrows by ap.refer() method
        let mut ap = ArgumentParser::new();
        ap.set_description("PCFG parsing");
        ap.refer(&mut stats.trainsize)
            .add_option(&["--trainsize"], Store,
            "Number of training sentences from sections 2-21");
        ap.refer(&mut stats.oov_handling)
            .add_option(&["--oovhandling"], Store,
            "OOV->POS handling: Zero (default), Uniform or Marginal");
        ap.refer(&mut stats.uniform_oov_prob)
            .add_option(&["--oovuniformlogprob"], Store,
            "Value for uniform OOV preterminal assignment");
        ap.refer(&mut stats.feature_structures)
            .add_option(&["--featurestructures"], Store,
            "Feature structures: exactmatch (default), postagsonly");
        ap.refer(&mut stats.testtagsfile)
            .add_option(&["--testtagsfile"], Store,
            "POS tags of the test tag file, if parsing with --featurestructures=postagsonly");
        ap.refer(&mut stats.nbesttags)
            .add_option(&["--nbesttags"], StoreTrue,
            "Parse with all possible tags and their weights instead of just the argmax tag (n/a for gold, duh)");
        ap.refer(&mut stats.keepafterdash)
            .add_option(&["--keepafterdash"], StoreTrue,
            "Keep everything after a - from SPMRL NTs (e.g., NP-SBJ instead of NP)");
        ap.refer(&mut stats.eta)
            .add_option(&["--eta"], Store,
            "Softness factor (default: 1.0)");
        ap.refer(&mut stats.alpha)
            .add_option(&["--alpha"], Store,
            "Hyperparameter alpha (default: 0.5)");
        ap.refer(&mut stats.beta)
            .add_option(&["--beta"], Store,
            "Hyperparameter beta (default: 1.0)");
        ap.refer(&mut stats.kappa)
            .add_option(&["--kappa"], Store,
            "Hyperparameter kappa (default: 1)");
        ap.refer(&mut stats.dualmono_pad)
            .add_option(&["--dualmono-pad"], StoreTrue,
            "grams = window(###word) ∪ window(word###) - instead of the standard grams = window(###word###)");
        ap.refer(&mut stats.logcompvalues)
            .add_option(&["--logcompvalues"], StoreTrue,
            "write binned comp value statistics at the end of parsing (caution: very slow!)");
        ap.refer(&mut stats.all_terms_fallback)
            .add_option(&["--all-terms-fallback"], StoreTrue,
            "Allows OOV-like treatment to all terms as fallback");
        ap.refer(&mut stats.only_oovs_soft)
            .add_option(&["--only-oovs-soft"], StoreTrue,
            "(set eta = 0.0)");
        ap.refer(&mut stats.exhaustive)
            .add_option(&["--exhaustive"], StoreTrue,
            "Forces exhaustive search for all parses");
        ap.refer(&mut wsj_path)
            .add_option(&["--wsjpath"], Store,
            "Path of WSL merged data (.../treebank_3/parsed/mrg/wsj)");
        ap.refer(&mut spmrl_path)
            .add_option(&["--spmrlpath"], Store,
            "Path of SPMRL data (has to contain GERMAN_SPMRL etc. folders)");
        ap.refer(&mut stats.language)
            .add_option(&["--language"], Store,
            "Language (case-insensitive) ∈ {english, german, ...}");
        ap.parse_args_or_exit();
    }
    
    let (testtrees, parses) = extract_and_parse(&wsj_path, &spmrl_path, &mut stats);
    eval_parses(&testtrees, parses, &mut stats);
}
