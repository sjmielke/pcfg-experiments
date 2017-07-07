use std::collections::{HashMap,HashSet};
use std::fs::File;
use std::io::prelude::*;

extern crate ptb_reader;
use ptb_reader::PTBTree;

use defs::*;

fn reverse_bijection<
        V: Clone + Eq + ::std::hash::Hash,
        K: Clone + Eq + ::std::hash::Hash>
    (indict: &HashMap<K, V>) -> HashMap<V, K> {
    let mut outdict: HashMap<V, K> = HashMap::new();
    for item in indict {
        let (k, v) = item;
        assert!(outdict.insert(v.clone(), k.clone()).is_none())
    }
    outdict
}

fn incr_in_innermap<T: Eq + ::std::hash::Hash>(l: NT, r: T, outermap: &mut HashMap<NT, HashMap<T, usize>>) {
    let innermap = outermap.entry(l).or_insert_with(HashMap::new);
    let newcount = innermap.get(&r).unwrap_or(&0) + 1;
    innermap.insert(r, newcount);
}

fn treat_nt(rev_ntdict: &mut HashMap<String, NT>, nt: &str) -> NT {
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

pub fn binarize_grammar(in_rules: &HashMap<NT, HashMap<RHS, f64>>, ntdict: &HashMap<NT, String>) -> (HashMap<NT, HashMap<RHS, f64>>, HashMap<NT, String>) {
    let mut rev_ntdict: HashMap<String, NT> = reverse_bijection(ntdict);
    
    for nt in rev_ntdict.keys() {
        assert!(!nt.contains("_"));
    }
    
    fn binarize(ntdict: &HashMap<NT, String>, rev_ntdict: &mut HashMap<String, NT>, lhs: NT, rhs: &[NT]) -> Vec<Rule> {
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
            result.push(Rule { lhs: lhs, rhs: RHS::Binary(rhs[0], newnt_int) });
            result
        } else if rhs.len() == 2 {
            // identity
            vec![Rule { lhs: lhs, rhs: RHS::Binary(rhs[0], rhs[1]) }]
        } else if rhs.len() == 1 {
            vec![Rule { lhs: lhs, rhs: RHS::Unary(rhs[0]) }]
        } else {
            panic!("Nullary rule detected!")
        }
    }
    
    let mut bin_rules: HashMap<NT, HashMap<RHS, f64>> = HashMap::new();
    let mut new_rules_tmp: Vec<Rule> = Vec::new(); // all have prob 1.0
    
    for (lhs, rhsmap) in in_rules {
        let mut innermap: HashMap<RHS, f64> = HashMap::new();
        for (rhs, &prob) in rhsmap {
            match *rhs {
                RHS::Terminal(_) => assert!(innermap.insert(rhs.clone(), prob).is_none()),
                RHS::Binary(_, _) => panic!("Already partially binarized!"),
                RHS::Unary(_) => panic!("Already partially binarized!"),
                RHS::NTs(ref nts) => {
                    let mut newrules = binarize(ntdict, &mut rev_ntdict, *lhs, nts);
                    let Rule { lhs: lhs_, rhs: rhs_ } = newrules.pop().unwrap();
                    assert_eq!(*lhs, lhs_);
                    assert!(innermap.insert(rhs_, prob).is_none());
                    new_rules_tmp.extend(newrules)
                }
            }
        }
        assert!(bin_rules.insert(lhs.clone(), innermap).is_none())
    }
    
    for Rule { lhs, rhs } in new_rules_tmp {
        bin_rules.entry(lhs).or_insert_with(HashMap::new).insert(rhs, 1.0);
    }
    
    (bin_rules, reverse_bijection(&rev_ntdict))
}

pub fn crunch_train_trees(mut train_trees: Vec<PTBTree>, stats: &PCFGParsingStatistics) -> (HashMap<NT, HashMap<RHS, f64>>, HashMap<NT, String>, Vec<NT>, HashMap<NT, f64>, HashMap<NT, f64>, (HashMap<String, f64>, f64)) {
    
    assert!(train_trees.len() >= stats.trainsize);
    
    let mut lhs_to_rhs_count: HashMap<NT, HashMap<RHS, usize>> = HashMap::new();
    let mut rev_ntdict: HashMap<String, NT> = HashMap::new();
    let mut initial_nts: HashSet<NT> = HashSet::new();
    
    fn readoff_rules_into(t: &PTBTree, lhs_to_rhs_count: &mut HashMap<NT, HashMap<RHS, usize>>, rev_ntdict: &mut HashMap<String, NT>, o_initial_nts: Option<&mut HashSet<NT>>) {
        match *t {
            PTBTree::InnerNode { ref label, ref children } => {
                let lhs = treat_nt(rev_ntdict, label);
                
                if let Some(mut initial_nts) = o_initial_nts {
                    initial_nts.insert(lhs);
                }
                
                // Terminal case maybe?
                if children.len() == 1 {
                    if let PTBTree::TerminalNode { ref label } = children[0] {
                        incr_in_innermap(lhs, RHS::Terminal(label.to_string()), lhs_to_rhs_count);
                        return
                    } // else continue with NT case
                }
                // Other (NT) case
                let mut child_ids: Vec<NT> = Vec::new();
                for c in children {
                    let s = match *c {
                        PTBTree::InnerNode { ref label, .. } | PTBTree::TerminalNode { ref label } => label
                    };
                    child_ids.push(treat_nt(rev_ntdict, s));
                    readoff_rules_into(c, lhs_to_rhs_count, rev_ntdict, None);
                }
                incr_in_innermap(lhs, RHS::NTs(child_ids), lhs_to_rhs_count);
            }
            _ => {
                panic!("Unusable tree (maybe you don't have proper preterminals?)!")
            }
        }
    }
    
    for ref mut t in &mut train_trees[0..stats.trainsize] {
        t.strip_all_predicate_annotations();
        readoff_rules_into(t, &mut lhs_to_rhs_count, &mut rev_ntdict, Some(&mut initial_nts));
    }
    
    // Normalize grammar
    let mut lhs_to_rhs_prob: HashMap<NT, HashMap<RHS, f64>> = HashMap::new();
    
    // First: what are the rare words?
    let mut word_counter: HashMap<String, usize> = HashMap::new();
    for (_, rhsmap) in &lhs_to_rhs_count {
        for (rhs, count) in rhsmap {
            if let RHS::Terminal(ref w) = *rhs {
                let c = word_counter.entry(w.clone()).or_insert(0);
                *c = *c + count;
            }
        }
    }
    
    // Construct better baselines
    let mut pret_counter_all: HashMap<NT, usize> = HashMap::new();
    let mut pret_counter_le3: HashMap<NT, usize> = HashMap::new();
    let mut wc_all: usize = 0;
    let mut wc_le3: usize = 0;
    for (lhs, rhsmap) in &lhs_to_rhs_count {
        for (rhs, count) in rhsmap {
            if let RHS::Terminal(ref w) = *rhs {
                // Make sure there's an entry so we get it in the final distribution!
                // Also, note that we do +1 smoothing to allow everything :)
                pret_counter_all.entry(*lhs).or_insert(1);
                pret_counter_le3.entry(*lhs).or_insert(1);
                let c = pret_counter_all.get_mut(lhs).unwrap();
                *c = *c + *count;
                wc_all += *count;
                if *word_counter.get(w).unwrap() <= 3 {
                    let c = pret_counter_le3.get_mut(lhs).unwrap();
                    *c = *c + *count;
                    wc_le3 += *count;
                }
            }
        }
    }
    
    // Normalize these distributions
    let pret_distr_all: HashMap<NT, f64> = pret_counter_all.into_iter().map(|(nt, c)| (nt, (c as f64) / (wc_all as f64))).collect();
    let pret_distr_le3: HashMap<NT, f64> = pret_counter_le3.into_iter().map(|(nt, c)| (nt, (c as f64) / (wc_le3 as f64))).collect();
    
    // {
    //     let ntdict = reverse_bijection(&rev_ntdict);
    //     let mut v = pret_distr_all.iter().collect::<Vec<_>>();
    //     v.sort_by_key(|&(_,b)| (b * 100000000.0) as usize);
    //     println!("{:?}\n\n", v.into_iter().map(|(a,b)| (ntdict.get(a).unwrap(),b)).collect::<Vec<_>>());
    //     let mut v = pret_distr_le3.iter().collect::<Vec<_>>();
    //     v.sort_by_key(|&(_,b)| (b * 100000000.0) as usize);
    //     println!("{:?}", v.into_iter().map(|(a,b)| (ntdict.get(a).unwrap(),b)).collect::<Vec<_>>());
    // }
    
    // ... and finally normalize rule set!
    for (lhs, rhsmap) in lhs_to_rhs_count {
        let mut innermap = HashMap::new();
        let z: usize = rhsmap.values().sum();
        let z: f64 = z as f64;
        for (rhs, count) in rhsmap {
            innermap.insert(rhs, (count as f64) / z);
        }
        lhs_to_rhs_prob.insert(lhs, innermap);
    }
    
    // get to log freqs from `wordcounter` for the frequency based embedding space
    
    let mut word2logfreq: HashMap<String, f64> = HashMap::new();
    let mut wordsum: f64 = 0.0;
    for (w, c) in word_counter {
        assert_eq!(word2logfreq.insert(w, (c as f64).ln()), None);
        wordsum += c as f64
    }
    
    (lhs_to_rhs_prob, reverse_bijection(&rev_ntdict), initial_nts.into_iter().collect::<Vec<usize>>(), pret_distr_all, pret_distr_le3, (word2logfreq, wordsum.ln()))
}

pub fn crunch_test_trees(read_devtrees: Vec<PTBTree>, stats: &PCFGParsingStatistics) -> (Vec<String>, Vec<String>, Vec<PTBTree>) {
    let mut devsents: Vec<String> = Vec::new();
    let mut devposs: Vec<String> = Vec::new();
    let mut devtrees: Vec<PTBTree> = Vec::new();
    for mut t in read_devtrees {
        if stats.language.to_uppercase() == "ENGLISH" {
            t.strip_all_predicate_annotations()
        }
        if stats.language.to_uppercase() != "ENGLISH" || t.front_length() <= 40 {
            devsents.push(t.front());
            devposs.push(t.pos_front());
            devtrees.push(t);
        }
    }
    //println!("From {} candidates we took {} dev sentences (max length {})!", read_devtrees.len(), devtrees.len(), stats.testmaxlen);
    
    (devsents, devposs, devtrees)
}

/// Returns a vector of tag-LOGprob pairs for each word of each sentence.
/// Assumes space separated tag descriptors:
/// * 1-best: simply the tag.
/// * n-best: class1/logprob1;c2/l2;...
pub fn read_testtagsfile(filename: &str, golddata: Vec<String>, testmaxlen: Option<usize>) -> Vec<Vec<Vec<(POSTag, f64)>>> {
    fn decode_td(td: &str) -> Vec<(POSTag, f64)> {
        let mut ress: Vec<(POSTag, f64)> = Vec::new();
        for cd in td.split(';') {
            let v = cd.split('/').collect::<Vec<&str>>();
            if v.len() != 2 {
                panic!("'{}' of '{}'", cd, td)
            }
            let classname = v[0];
            let logprob: f64 = v[1].parse().unwrap();
            ress.push((classname.to_string(), logprob))
        }
        ress
    }
    
    if filename != "" {
        let mut contents = String::new();
        File::open(filename)
            .expect("no pos file :(")
            .read_to_string(&mut contents)
            .expect("unreadable pos file :(");
        let poss = contents.split("\n")
            .collect::<Vec<&str>>()
            .into_iter()
            .filter(|s| *s != "")
            .map(|s| s.split(' ')
                      .map(decode_td)
                      .collect::<Vec<Vec<(POSTag, f64)>>>())
            .collect::<Vec<Vec<Vec<(POSTag, f64)>>>>();
        
        match testmaxlen {
            Some(mln) => poss.into_iter().filter(|v| v.len() <= mln).collect(),
            None => poss
        }
    } else {
        golddata.iter()
                .map(|s| s.split(' ')
                          .map(|t| vec![(t.to_string(), 0.0)]) // keep in mind: these are log probs!
                          .collect::<Vec<Vec<(POSTag, f64)>>>())
        .collect::<Vec<Vec<Vec<(POSTag, f64)>>>>()
    }
}

pub fn get_data(wsj_path: &str, spmrl_path: &str, stats: &mut PCFGParsingStatistics)
        -> ((HashMap<NT, HashMap<RHS, f64>>, HashMap<NT, String>), Vec<NT>, HashMap<NT, f64>, HashMap<NT, f64>, (HashMap<String, f64>, f64), (Vec<String>, Vec<Vec<Vec<(POSTag, f64)>>>, Vec<PTBTree>))  {
    
    fn read_caseinsensitive(prefix: &String, camellang: &String, stats: &PCFGParsingStatistics, bracketing: bool) -> Result<Vec<PTBTree>, Box<::std::error::Error>> {
        let name1 = prefix.to_string() + &camellang + ".gold.ptb";
        let name2 = prefix.to_string() + &camellang.to_lowercase() + ".gold.ptb";
        match ptb_reader::parse_spmrl_ptb_file(&name1, bracketing, !stats.keepafterdash) {
            t@Ok(_) => t,
            Err(_) => ptb_reader::parse_spmrl_ptb_file(&name2, bracketing, !stats.keepafterdash)
        }
    }
    
    fn read_bracketinginsensitive(prefix: &String, camellang: &String, stats: &PCFGParsingStatistics) -> Result<Vec<PTBTree>, Box<::std::error::Error>> {
        match read_caseinsensitive(prefix, camellang, stats, false) {
            t@Ok(_) => t,
            Err(_) => read_caseinsensitive(prefix, camellang, stats, true)
        }
    }
    let lang = stats.language.to_uppercase();
    
    let (train_trees, test_trees) = if lang == "ENGLISH" {
        let train = ptb_reader::parse_ptb_sections(wsj_path, (2..22).collect()); // sections 2-21
        let  test = ptb_reader::parse_ptb_sections(wsj_path, vec![22]); // section 22
        (train, test)
    } else {
        let mut camellang = lang.chars().next().unwrap().to_string();
        camellang += &lang.to_lowercase()[1..];
        let alltrain_prefix = spmrl_path.to_string() + "/" + &lang + "_SPMRL/gold/ptb/train/train.";
        let train_5k_prefix = spmrl_path.to_string() + "/" + &lang + "_SPMRL/gold/ptb/train5k/train5k.";
        let testfile_prefix = spmrl_path.to_string() + "/" + &lang + "_SPMRL/gold/ptb/dev/dev.";
        
        let train = match read_bracketinginsensitive(&alltrain_prefix, &camellang, stats) {
            Ok(t) => t,
            Err(_) => read_bracketinginsensitive(&train_5k_prefix, &camellang, stats).expect(&format!("Din't find {} train!", lang))
        };
        (train, read_bracketinginsensitive(&testfile_prefix, &camellang, stats).expect(&format!("Didn't find {} dev!", lang)))
    };
    
    let (unb_rules, unb_ntdict, initial_nts, pret_distr_all, pret_distr_le3, wordcounts) = crunch_train_trees(train_trees, &stats);
    let (mut testsents, gold_testposs, mut testtrees) = crunch_test_trees(test_trees, &stats);
    
    let maxlen = if lang == "ENGLISH" {Some(40)} else {None};
    let mut testposs = read_testtagsfile(&stats.testtagsfile, gold_testposs, maxlen);
    
    if lang == "ENGLISH" {
        testsents.truncate(500);
        testposs.truncate(500);
        testtrees.truncate(500)
    }
    
    let intended_ts_len = match lang.as_ref() {
        "ENGLISH" => 500,
        "GERMAN" => 5000,
        "KOREAN" => 2066,
        "ARABIC" => 1985,
        "FRENCH" => 1235,
        "HUNGARIAN" => 1051,
        "BASQUE" => 948,
        "POLISH" => 821,
        "HEBREW" => 500,
        "SWEDISH" => 494,
        _ => unreachable!()
    };
    assert_eq!(testsents.len(), intended_ts_len);
    assert_eq!(testposs.len(), intended_ts_len);
    assert_eq!(testtrees.len(), intended_ts_len);
    
    let (bin_rules, bin_ntdict) = binarize_grammar(&unb_rules, &unb_ntdict);
    
    stats.unbin_nts = unb_ntdict.len();
    stats.bin_nts   = bin_ntdict.len();
    
    ((bin_rules, bin_ntdict), initial_nts, pret_distr_all, pret_distr_le3, wordcounts, (testsents, testposs, testtrees))
}