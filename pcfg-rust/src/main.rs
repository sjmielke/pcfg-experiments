use std::collections::HashMap;
use std::f64; // log and exp
use std::process::Command;

extern crate tempdir;
use std::fs::File;
use std::io::Write;
use tempdir::TempDir;

// Caching for feature structures
#[macro_use]
extern crate lazy_static;

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

fn prepare_candidates<'a>(cell: &HashMap<NT, (f64, ParseTree<'a>)>, bin_ntdict: &HashMap<NT, String>) -> Vec<(NT, (f64, ParseTree<'a>))> {
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

fn eval_parses(testsents: &Vec<String>, testtrees: &Vec<PTBTree>, parses: Vec<HashMap<NT, (f64, ParseTree)>>, bin_ntdict: &HashMap<NT, String>, stats: &mut PCFGParsingStatistics) {
    // Save output for EVALB call
    let tmp_dir = TempDir::new("pcfg-rust").unwrap();
    let gold_path = tmp_dir.path().join("gold.txt");
    let best_path = tmp_dir.path().join("best.txt");
    let best_or_fail_path = tmp_dir.path().join("best_or_fail.txt");
    let mut gold_file = File::create(&gold_path).unwrap();
    let mut best_file = File::create(&best_path).unwrap();
    let mut best_or_fail_file = File::create(&best_or_fail_path).unwrap();
    
    for ((sent, tree), cell) in testsents.iter().zip(testtrees).zip(parses.iter()) {
        let candidates = prepare_candidates(cell, bin_ntdict);
        
        //print_example(&bin_ntdict, &sent, &tree, &candidates);
        
        let gold_parse: String = format!("{}", tree);
        let best_parse: String = match candidates.get(0) {
            None => "(( ".to_string() + &sent.replace(" ", ") ( ") + "))",
            Some(&(_, (_, ref parsetree))) => format!("{}", parsetree2ptbtree(&bin_ntdict, &parsetree))
        };
        let best_or_fail_parse: String = match candidates.get(0) {
            None => "".to_string(),
            Some(&(_, (_, ref parsetree))) => format!("{}", parsetree2ptbtree(&bin_ntdict, &parsetree))
        };
        
        writeln!(gold_file, "{}", gold_parse).unwrap();
        writeln!(best_file, "{}", best_parse).unwrap();
        writeln!(best_or_fail_file, "{}", best_or_fail_parse).unwrap();
    }
    
    for line in String::from_utf8(Command::new("../EVALB/evalb").arg(&gold_path).arg(best_path).output().unwrap().stdout).unwrap().lines() {
        if line.starts_with("Bracketing FMeasure") {
            stats.fmeasure = line.split('=').nth(1).unwrap().trim().parse().unwrap();
            break
        }
    }
    for line in String::from_utf8(Command::new("../EVALB/evalb").arg(&gold_path).arg(best_or_fail_path).output().unwrap().stdout).unwrap().lines() {
        if line.starts_with("Bracketing FMeasure") {
            stats.or_fail_fmeasure = line.split('=').nth(1).unwrap().trim().parse().unwrap();
            break
        }
    }
    
    stats.print(false);
    
    // safely close to get error messages just in case
    drop(gold_file);
    drop(best_file);
    tmp_dir.close().unwrap();
}

fn main() {
    let mut stats: PCFGParsingStatistics = PCFGParsingStatistics{
        // Placeholders
        gram_ext_bin:f64::NAN, cky_prep:f64::NAN, cky_terms:f64::NAN, cky_higher:f64::NAN, oov_words:0, oov_sents:0, parsefails:0, fmeasure:f64::NAN, or_fail_fmeasure:f64::NAN, unbin_nts:std::usize::MAX, bin_nts:std::usize::MAX,
        // Values
        trainsize: 7500,
        testsize: 500,
        testmaxlen: 40,
        oov_handling: OOVHandling::Zero,
        feature_structures: "exactmatch".to_string(),
        testtagsfile: "".to_string(),
        nbesttags: false,
        eta: 1.0,
        alpha: 0.5,
        beta: 1.0,
        kappa: 1,
        all_terms_fallback: false,
        exhaustive: false,
        uniform_oov_prob: -10.0
    };
    
    let mut wsj_path: String = "/home/sjm/documents/Uni/FuzzySP/treebank-3_LDC99T42/treebank_3/parsed/mrg/wsj".to_string();
//    let mut spmrl_path: String = "/home/sjm/documents/Uni/FuzzySP/spmrl-2014/data/GERMAN_SPMRL/gold/ptb".to_string();
    let mut spmrl_path: String = "".to_string();
    
    { // this block limits scope of borrows by ap.refer() method
        let mut ap = ArgumentParser::new();
        ap.set_description("PCFG parsing");
        ap.refer(&mut stats.trainsize)
            .add_option(&["--trainsize"], Store,
            "Number of training sentences from sections 2-21");
        ap.refer(&mut stats.testsize)
            .add_option(&["--testsize"], Store,
            "Number of test sentences from section 22");
        ap.refer(&mut stats.testmaxlen)
            .add_option(&["--testmaxlen"], Store,
            "Maximum length of each test sentence (words)");
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
        ap.refer(&mut stats.all_terms_fallback)
            .add_option(&["--all-terms-fallback"], StoreTrue,
            "Allows OOV-like treatment to all terms as fallback");
        ap.refer(&mut stats.exhaustive)
            .add_option(&["--exhaustive"], StoreTrue,
            "Forces exhaustive search for all parses");
        ap.refer(&mut wsj_path)
            .add_option(&["--wsjpath"], Store,
            "Path of WSL merged data (.../treebank_3/parsed/mrg/wsj)");
        ap.refer(&mut spmrl_path)
            .add_option(&["--spmrlpath"], Store,
            "Path of SPMRL data (.../GERMAN_SPMRL/gold/ptb)");
        ap.parse_args_or_exit();
    }
    
    //println!("Now loading and processing all data!");
    let t = get_usertime();
    let ((bin_rules, bin_ntdict), (testsents, testposs, testtrees)) = extract::get_data(&wsj_path, &spmrl_path, &mut stats);
    stats.gram_ext_bin = get_usertime() - t;
    
    //println!("Now parsing!");
    let parses = parse::agenda_cky_parse(&bin_rules, &bin_ntdict, &testsents, &testposs, &mut stats);
    eval_parses(&testsents, &testtrees, parses, &bin_ntdict, &mut stats);
}
