//#![feature(alloc_system)]
//extern crate alloc_system;

use std::collections::HashMap;
use std::f64; // log and exp
use std::process::Command;

extern crate tempdir;
use std::fs::File;
use std::io::Write;
use tempdir::TempDir;

extern crate argparse;
use argparse::{ArgumentParser, Store};

extern crate ptb_reader;
use ptb_reader::PTBTree;

struct PCFGParsingStatistics {
    // Raw numbers
    trainsize: usize,
    testsize: usize,
    testmaxlen: usize,
    // Times in seconds
    gram_ext_cnf: f64,
    cky_prep: f64,
    cky_terms: f64,
    cky_higher: f64,
    // Raw numbers
    oov_words: usize,
    oov_sents: usize,
    parsefails: usize,
    // Percent
    fmeasure: f64,
    or_fail_fmeasure: f64
}

// taken from https://github.com/dikaiosune/rust-runtime-benchmarks/blob/master/bench-suite-linux/src/bencher.rs
extern crate libc;
use libc::{c_long, rusage, suseconds_t, timeval, time_t, getrusage, RUSAGE_SELF};
/// Returns time elapsed in userspace in seconds.
pub fn get_usertime() -> f64 {
    let mut usage = rusage {
        ru_utime: timeval{ tv_sec: 0 as time_t, tv_usec: 0 as suseconds_t, },
        ru_stime: timeval{ tv_sec: 0 as time_t, tv_usec: 0 as suseconds_t, },
        ru_maxrss: 0 as c_long,
        ru_ixrss: 0 as c_long,
        ru_idrss: 0 as c_long,
        ru_isrss: 0 as c_long,
        ru_minflt: 0 as c_long,
        ru_majflt: 0 as c_long,
        ru_nswap: 0 as c_long,
        ru_inblock: 0 as c_long,
        ru_oublock: 0 as c_long,
        ru_msgsnd: 0 as c_long,
        ru_msgrcv: 0 as c_long,
        ru_nsignals: 0 as c_long,
        ru_nvcsw: 0 as c_long,
        ru_nivcsw: 0 as c_long,
    };
    unsafe { getrusage(RUSAGE_SELF, (&mut usage) as *mut rusage); }
    let secs = usage.ru_utime.tv_sec as usize;
    let usecs = usage.ru_utime.tv_usec as usize;
    (secs * 1_000_000 + usecs) as f64 / 1000000.0
}

fn reverse_bijection<V: Clone + Eq + std::hash::Hash, K: Clone + Eq + std::hash::Hash>(indict: &HashMap<K, V>) -> HashMap<V, K> {
    let mut outdict: HashMap<V, K> = HashMap::new();
    for item in indict {
        let (k, v) = item;
        assert!(outdict.insert(v.clone(), k.clone()).is_none())
    }
    outdict
}

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
enum RHS {
    Terminal(String),
    NTs(Vec<usize>),
    CNFBin(usize, usize),
    CNFChain(usize)
}

#[derive(Debug)]
struct Rule {
    lhs: usize,
    rhs: RHS
}

#[derive(Debug, Clone)]
enum ParseTree<'a> {
    InnerNode {
        label: usize,
        children: Vec<ParseTree<'a>>
    },
    TerminalNode {
        label: &'a str
    }
}

fn parsetree2ptbtree(ntdict: &HashMap<usize, String>, t: &ParseTree) -> PTBTree {
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

fn debinarize_parsetree<'a>(ntdict: &HashMap<usize, String>, t: &ParseTree<'a>) -> ParseTree<'a> {
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

fn treat_nt(rev_ntdict: &mut HashMap<String, usize>, nt: &str) -> usize {
    match rev_ntdict.get(nt) {
        Some(&i) => i,
        None => {
            let next_value = rev_ntdict.values().max().unwrap_or(&0) + 1;
            //println!("{} ~ {}", next_value, nt);
            rev_ntdict.insert(nt.to_string(), next_value);
            next_value
        }
    }
}

fn normalize_grammar<'a, I>(rulelist: I) -> HashMap<usize, HashMap<RHS, f64>> where I: Iterator<Item=&'a Rule> {
    let mut lhs_to_rhs_count: HashMap<usize, HashMap<RHS, usize>> = HashMap::new();
    
    for r in rulelist {
        let innermap = lhs_to_rhs_count.entry(r.lhs).or_insert_with(HashMap::new);
        
        let newcount = match innermap.get(&r.rhs) {
            Some(&count) => count + 1,
            None => 1
        };
        innermap.insert(r.rhs.clone(), newcount);
    }
    
    let mut lhs_to_rhs_prob: HashMap<usize, HashMap<RHS, f64>> = HashMap::new();
    
    for (lhs, rhsdict) in lhs_to_rhs_count {
        let mut innermap = HashMap::new();
        let z: usize = rhsdict.values().sum();
        let z: f64 = z as f64;
        for (rhs, count) in rhsdict {
            innermap.insert(rhs, (count as f64) / z);
        }
        
        lhs_to_rhs_prob.insert(lhs, innermap);
    }
    
    lhs_to_rhs_prob
}

fn cnfize_grammar(in_rules: &HashMap<usize, HashMap<RHS, f64>>, ntdict: &HashMap<usize, String>) -> (HashMap<usize, HashMap<RHS, f64>>, HashMap<usize, String>) {
    let mut rev_ntdict: HashMap<String, usize> = reverse_bijection(ntdict);
    
    for nt in rev_ntdict.keys() {
        assert!(!nt.contains("_"));
    }
    
    fn binarize(ntdict: &HashMap<usize, String>, rev_ntdict: &mut HashMap<String, usize>, lhs: usize, rhs: &[usize]) -> Vec<Rule> {
        if rhs.len() > 2 {
            // right-branching
            let rest = &rhs[1..rhs.len()];
            let rest_names: Vec<String> = rest.iter().map(|i| ntdict[i].to_string()).collect();
            let newnt_str = "_".to_string() + &rest_names.join("_");
            let newnt_int = treat_nt(rev_ntdict, &newnt_str);
            //println!("{} ~> {}", newnt_str, newnt_int);
            
            let mut result: Vec<Rule> = binarize(ntdict, rev_ntdict, newnt_int, rest);
            //println!("{:?}", result);
            
            // root rule has to go last to be popped!
            result.push(Rule { lhs: lhs, rhs: RHS::CNFBin(rhs[0], newnt_int) });
            result
        } else if rhs.len() == 2 {
            // identity
            vec![Rule { lhs: lhs, rhs: RHS::CNFBin(rhs[0], rhs[1]) }]
        } else if rhs.len() == 1 {
            vec![Rule { lhs: lhs, rhs: RHS::CNFChain(rhs[0]) }]
        } else {
            panic!("Nullary rule detected!")
        }
    }
    
    let mut cnf_rules: HashMap<usize, HashMap<RHS, f64>> = HashMap::new();
    let mut new_rules_tmp: Vec<Rule> = Vec::new(); // all have prob 1.0
    
    for (lhs, rhsdict) in in_rules {
        let mut innermap: HashMap<RHS, f64> = HashMap::new();
        for (rhs, &prob) in rhsdict {
            match *rhs {
                RHS::Terminal(_) => assert!(innermap.insert(rhs.clone(), prob).is_none()),
                RHS::CNFBin(_, _) => panic!("Already partially CNFized!"),
                RHS::CNFChain(_) => panic!("Already partially CNFized!"),
                RHS::NTs(ref nts) => {
                    let mut newrules = binarize(ntdict, &mut rev_ntdict, *lhs, nts);
                    let Rule { lhs: lhs_, rhs: rhs_ } = newrules.pop().unwrap();
                    assert_eq!(*lhs, lhs_);
                    assert!(innermap.insert(rhs_, prob).is_none());
                    new_rules_tmp.extend(newrules)
                }
            }
        }
        assert!(cnf_rules.insert(lhs.clone(), innermap).is_none())
    }
    
    for Rule { lhs, rhs } in new_rules_tmp {
        cnf_rules.entry(lhs).or_insert_with(HashMap::new).insert(rhs, 1.0);
    }
    
    (cnf_rules, reverse_bijection(&rev_ntdict))
}

fn cky_parse<'a>(cnf_rules: &'a HashMap<usize, HashMap<RHS, f64>>, sents: &[String], stats: &mut PCFGParsingStatistics) -> Vec<HashMap<usize, (f64, ParseTree<'a>)>> {
    // Build helper dicts for quick access. All are bottom-up in the parse.
    let t = get_usertime();
    let mut word_to_preterminal: HashMap<String, Vec<(usize, (f64, ParseTree))>> = HashMap::new();
    let mut nt_chains: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    let mut rhss_to_lhs: HashMap<(usize, usize), Vec<(usize, f64)>> = HashMap::new();
    
    for (lhs, rhsdict) in cnf_rules {
        for (rhs, prob) in rhsdict {
            let logprob = prob.ln();
            match *rhs {
                RHS::Terminal(ref s) => {
                    if !word_to_preterminal.contains_key(s) {
                        word_to_preterminal.insert(s.clone(), Vec::new());
                    }
                    let tree = ParseTree::TerminalNode { label: s };
                    let tree = ParseTree::InnerNode { label: *lhs, children: vec![tree] };
                    word_to_preterminal.get_mut(s).unwrap().push((*lhs, (logprob, tree)))
                }
                RHS::CNFChain(ref r) => {
                    if !nt_chains.contains_key(r) {
                        nt_chains.insert(*r, Vec::new());
                    }
                    nt_chains.get_mut(r).unwrap().push((*lhs, logprob))
                }
                RHS::CNFBin(ref r1, ref r2) => {
                    rhss_to_lhs.entry((*r1, *r2)).or_insert_with(Vec::new).push((*lhs, logprob))
                }
                _ => panic!("Trying to use un-CNFized grammar!")
            }
        }
    }
    stats.cky_prep = get_usertime() - t;
    
    stats.cky_terms = 0.0;
    stats.cky_higher = 0.0;
    
    let mut results: Vec<HashMap<usize, (f64, ParseTree<'a>)>> = Vec::new();
    
    for raw_sent in sents {
        //println!("parsing: {}", raw_sent);
        
        let mut oov_in_this_sent = false;
        
        // Tokenize
        let sent: Vec<&str> = raw_sent.split(' ').collect();
        
        // Now populate a chart (0-based indexing)!
        let mut ckychart: HashMap<(usize, usize), HashMap<usize, (f64, ParseTree)>> = HashMap::new();
        
        // Populate leafs
        let t = get_usertime();
        for (i,w) in sent.iter().enumerate() {
            // TODO actually could just break if we don't recognize terminals :D
            let terminals: HashMap<usize, (f64, ParseTree)> = match word_to_preterminal.get(*w) {
                Some(prets) => prets.iter().cloned().collect(),
                None => {
                    stats.oov_words += 1;
                    oov_in_this_sent = true;
                    HashMap::new()
                }
            };
            ckychart.insert((i, i), terminals);
            //println!("{}, {:?}", i, ckychart[&(i,i)]);
        }
        stats.cky_terms += get_usertime() - t;
        
        // Populate inner cells
        let t = get_usertime();
        for width in 2 .. sent.len() + 1 {
            for start in 0 .. sent.len() + 1 - width {
                let end = start + width;
                let mut cell: HashMap<usize, (f64, ParseTree)> = HashMap::new();
                
                // get new ones from bin rules
                for sp in start + 1 .. end {
                    //println!("{} | {}", sent[start..sp].join(" "), sent[sp..end].join(" "));
                    for (r1, &(pr1, ref pt1)) in &ckychart[&(start, sp - 1)] {
                        for (r2, &(pr2, ref pt2)) in &ckychart[&(sp, end - 1)] {
                            if let Some(m) = rhss_to_lhs.get(&(*r1,*r2)) {
                                for &(lhs, prob) in m {
                                    let new_prob = pr1 + pr2 + prob;
                                    let tree = ParseTree::InnerNode {label: lhs, children: vec![pt1.clone(), pt2.clone()] };
                                    match cell.get(&lhs) {
                                        None => {
                                            cell.insert(lhs, (new_prob, tree));
                                        }
                                        Some(&(old_prob, _)) => if old_prob < new_prob {
                                            cell.insert(lhs, (new_prob, tree));
                                        }
                                    };
                                }
                            }
                        }
                    }
                }
                
                // get new ones from chain rules
                let mut got_new = true;
                while got_new {
                    got_new = false;
                    let clonecell = cell.clone();
                    let celllist: Vec<_> = clonecell.iter().collect();
                    for (reached_nt, &(lower_prob, ref lower_ptree)) in celllist {
                        if let Some(v) = nt_chains.get(reached_nt) {
                            for &(lhs, cr_prob) in v {
                                let new_prob = cr_prob + lower_prob;
                                let tree = ParseTree::InnerNode {label: lhs, children: vec![lower_ptree.clone()]};
                                match cell.get(&lhs) {
                                    None => {
                                        cell.insert(lhs, (new_prob, tree));
                                    }
                                    Some(&(old_prob, _)) => if old_prob < new_prob {
                                        cell.insert(lhs, (new_prob, tree));
                                    }
                                };
                            }
                        }
                    }
                }
                
                assert!(ckychart.insert((start, end - 1), cell).is_none())
            }
        }
        stats.cky_higher += get_usertime() - t;
        
        if oov_in_this_sent {
            stats.oov_sents += 1
        }
        
        if ckychart[&(0, sent.len() - 1)].is_empty() {
            stats.parsefails += 1
        }
        
        results.push(ckychart[&(0, sent.len() - 1)].clone())
    }
    
    results
}

fn ptbset(trainsize: usize, testsize: usize, testmaxlen: usize) -> ((Vec<Rule>, HashMap<usize, String>), (Vec<String>, Vec<PTBTree>)) {
    //println!("Reading in the PTB train set...");
    let mut train_trees = ptb_reader::parse_ptb_sections("/home/sjm/documents/Uni/FuzzySP/treebank-3_LDC99T42/treebank_3/parsed/mrg/wsj", (2..22).collect()); // sections 2-21
    //println!("Read in a total of {} trees, but limiting them to trainsize = {} trees.", train_trees.len(), trainsize);
    
    assert!(train_trees.len() >= trainsize);
    
    //println!("Removing unwanted annotations from train...");
    for ref mut t in &mut train_trees {
        t.strip_predicate_annotations()
    }
    
    //println!("Reading off rules from train...");
    let mut rulelist: Vec<Rule> = Vec::new();
    let mut rev_ntdict: HashMap<String, usize> = HashMap::new();
    
    fn getrules(t: &PTBTree, rulelist: &mut Vec<Rule>, rev_ntdict: &mut HashMap<String, usize>) {
        match *t {
            PTBTree::InnerNode { ref label, ref children } => {
                let lhs = treat_nt(rev_ntdict, label);
                // Terminal case
                if children.len() == 1 {
                    if let PTBTree::TerminalNode { ref label } = children[0] {
                        rulelist.push(Rule { lhs: lhs, rhs: RHS::Terminal(label.to_string()) });
                        return
                    }
                }
                // Other (NT) case
                let mut child_ids: Vec<usize> = Vec::new();
                for c in children {
                    let s = match *c {
                        PTBTree::InnerNode { ref label, .. } | PTBTree::TerminalNode { ref label } => label
                    };
                    child_ids.push(treat_nt(rev_ntdict, s));
                    getrules(c, rulelist, rev_ntdict);
                }
                let r = Rule { lhs: lhs, rhs: RHS::NTs(child_ids) };
                rulelist.push(r);
            }
            _ => {
                panic!("Unusable tree!")
            }
        }
    }
    
    for t in &train_trees[0..trainsize] {
        //println!("{}", t);
        getrules(t, &mut rulelist, &mut rev_ntdict);
    }
    
    //println!("From {} trees: NTs: {}, Rules: {}", &train_trees[0..2000].len(), rev_ntdict.len(), rulelist.len());
    
    let ntdict = reverse_bijection(&rev_ntdict);
    
    // load test sents
    
    //println!("Reading, stripping and yielding test sentences...");
    let read_devtrees = ptb_reader::parse_ptb_sections("/home/sjm/documents/Uni/FuzzySP/treebank-3_LDC99T42/treebank_3/parsed/mrg/wsj", vec![22]);
    let read_devtrees_len = read_devtrees.len();
    
    let mut devsents: Vec<String> = Vec::new();
    let mut devtrees: Vec<PTBTree> = Vec::new();
    for mut t in read_devtrees {
        if t.front_length() <= testmaxlen && devtrees.len() < testsize {
            t.strip_predicate_annotations();
            devsents.push(t.front());
            devtrees.push(t);
        }
    }
    assert_eq!(devtrees.len(), testsize);
    //println!("From {} candidates we took {} dev sentences (max length {})!", read_devtrees_len, devtrees.len(), testmaxlen);
    
    //println!("PTB set done!");
    ((rulelist, ntdict), (devsents, devtrees))
}

fn print_example(cnf_ntdict: &HashMap<usize, String>, sent: &str, tree: &PTBTree, sorted_candidates: &[(usize, (f64, ParseTree))]) {
    if !sorted_candidates.is_empty() {
        println!("{}", sent);
        let mut got_s = false;
        // True parse
        println!("  {:28} -> {}", "", tree);
        // Algo
        for &(ref n, (p, ref ptree)) in sorted_candidates.iter().take(10) {
            println!("  {:10} ({:4.10}) -> {}", cnf_ntdict[n], p, parsetree2ptbtree(cnf_ntdict, &debinarize_parsetree(cnf_ntdict, ptree)));
            if cnf_ntdict[n] == "S" {
                got_s = true
            }
        }
        if sorted_candidates.len() > 10 {
            println!("\t...");
            if !got_s {
                for &(ref n, (p, ref ptree)) in sorted_candidates {
                    if cnf_ntdict[n] == "S" {
                        println!("  {:10} ({:4.10}) -> {}", cnf_ntdict[n], p, parsetree2ptbtree(cnf_ntdict, &debinarize_parsetree(cnf_ntdict, ptree)))
                    }
                }
            }
        }
    }
    else {
        println!("{}\n", sent);
    }
}

fn main() {
    let mut stats: PCFGParsingStatistics = PCFGParsingStatistics{
        // Placeholders
        gram_ext_cnf:f64::NAN, cky_prep:f64::NAN, cky_terms:f64::NAN, cky_higher:f64::NAN, oov_words:0, oov_sents:0, parsefails:0, fmeasure:f64::NAN, or_fail_fmeasure:f64::NAN,
        // Values
        trainsize: 7500,
        testsize: 500,
        testmaxlen: 30
    };
    
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
        ap.parse_args_or_exit();
    }
    
    let t = get_usertime();
    let ((rulelist, ntdict), (testsents, testtrees)) = ptbset(stats.trainsize, stats.testsize, stats.testmaxlen);
    let rules = normalize_grammar(rulelist.iter());
    //println!("Now CNFing!");
    let (cnf_rules, cnf_ntdict) = cnfize_grammar(&rules, &ntdict);
    stats.gram_ext_cnf = get_usertime() - t;
    
    //println!("Now parsing!");
    let parses = cky_parse(&cnf_rules, &testsents, &mut stats);
    
    // Save output for EVALB call
    let tmp_dir = TempDir::new("pcfg-rust").unwrap();
    let gold_path = tmp_dir.path().join("gold.txt");
    let best_path = tmp_dir.path().join("best.txt");
    let best_or_fail_path = tmp_dir.path().join("best_or_fail.txt");
    let mut gold_file = File::create(&gold_path).unwrap();
    let mut best_file = File::create(&best_path).unwrap();
    let mut best_or_fail_file = File::create(&best_or_fail_path).unwrap();
    
    for ((sent, tree), cell) in testsents.iter().zip(testtrees).zip(parses.iter()) {
        // Remove binarization traces
        let mut candidates: Vec<(usize, (f64, ParseTree))> = Vec::new();
        for (nt, &(p, ref ptree)) in cell {
            // Only keep candidates ending in proper NTs
            if cnf_ntdict[nt].chars().next().unwrap() != '_' {
                // Remove inner binarization nodes
                candidates.push((*nt, (p, debinarize_parsetree(&cnf_ntdict, ptree))))
            }
        }
        // Sort
        candidates.sort_by(|&(_, (p1, _)), &(_, (p2, _))| p2.partial_cmp(&p1).unwrap_or(std::cmp::Ordering::Equal));
        
        //print_example(&cnf_ntdict, &sent, &tree, &candidates);
        
        let gold_parse: String = format!("{}", tree);
        let best_parse: String = match candidates.get(0) {
            None => "(( ".to_string() + &sent.replace(" ", ") ( ") + "))",
            Some(&(_, (_, ref parsetree))) => format!("{}", parsetree2ptbtree(&cnf_ntdict, &parsetree))
        };
        let best_or_fail_parse: String = match candidates.get(0) {
            None => "".to_string(),
            Some(&(_, (_, ref parsetree))) => format!("{}", parsetree2ptbtree(&cnf_ntdict, &parsetree))
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
    
    // Print statistics
    //println!("trainsize\ttestsize\ttestmaxlen\tgram_ext_cnf\tcky_prep\tcky_terms\tcky_higher\toov_words\toov_sents\tparsefails\tfmeasure\tfmeasure (fail ok)");
    print!("{}\t", stats.trainsize);
    print!("{}\t", stats.testsize);
    print!("{}\t", stats.testmaxlen);
    print!("{}\t", stats.gram_ext_cnf);
    print!("{}\t", stats.cky_prep);
    print!("{}\t", stats.cky_terms);
    print!("{}\t", stats.cky_higher);
    print!("{}\t", stats.oov_words);
    print!("{}\t", stats.oov_sents);
    print!("{}\t", stats.parsefails);
    print!("{}\t", stats.fmeasure);
    print!("{}\n", stats.or_fail_fmeasure);
    
    drop(gold_file);
    drop(best_file);
    tmp_dir.close().unwrap();
}
