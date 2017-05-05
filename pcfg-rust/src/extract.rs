use std::collections::HashMap;
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

pub fn crunch_train_trees(mut train_trees: Vec<PTBTree>, stats: &PCFGParsingStatistics) -> (HashMap<NT, HashMap<RHS, f64>>, HashMap<NT, String>) {
    
    assert!(train_trees.len() >= stats.trainsize);
    
    let mut lhs_to_rhs_count: HashMap<NT, HashMap<RHS, usize>> = HashMap::new();
    let mut rev_ntdict: HashMap<String, NT> = HashMap::new();
    
    fn readoff_rules_into(t: &PTBTree, lhs_to_rhs_count: &mut HashMap<NT, HashMap<RHS, usize>>, rev_ntdict: &mut HashMap<String, NT>) {
        match *t {
            PTBTree::InnerNode { ref label, ref children } => {
                let lhs = treat_nt(rev_ntdict, label);
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
                    readoff_rules_into(c, lhs_to_rhs_count, rev_ntdict);
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
        readoff_rules_into(t, &mut lhs_to_rhs_count, &mut rev_ntdict);
    }
    
    // Normalize grammar
    let mut lhs_to_rhs_prob: HashMap<NT, HashMap<RHS, f64>> = HashMap::new();
    
    for (lhs, rhsmap) in lhs_to_rhs_count {
        let mut innermap = HashMap::new();
        let z: usize = rhsmap.values().sum();
        let z: f64 = z as f64;
        for (rhs, count) in rhsmap {
            innermap.insert(rhs, (count as f64) / z);
        }
        lhs_to_rhs_prob.insert(lhs, innermap);
    }
    
    (lhs_to_rhs_prob, reverse_bijection(&rev_ntdict))
}

pub fn crunch_test_trees(read_devtrees: Vec<PTBTree>, stats: &PCFGParsingStatistics) -> (Vec<String>, Vec<String>, Vec<PTBTree>) {
    let mut devsents: Vec<String> = Vec::new();
    let mut devposs: Vec<String> = Vec::new();
    let mut devtrees: Vec<PTBTree> = Vec::new();
    for mut t in read_devtrees {
        t.strip_all_predicate_annotations();
        if t.front_length() <= stats.testmaxlen && devtrees.len() < stats.testsize {
            devsents.push(t.front());
            devposs.push(t.pos_front());
            devtrees.push(t);
        }
    }
    assert_eq!(devtrees.len(), stats.testsize);
    //println!("From {} candidates we took {} dev sentences (max length {})!", read_devtrees.len(), devtrees.len(), stats.testmaxlen);
    
    (devsents, devposs, devtrees)
}

/// Returns a vector of tag-LOGprob pairs for each word of each sentence.
/// Assumes space separated tag descriptors:
/// * 1-best: simply the tag.
/// * n-best: class1/logprob1;c2/l2;...
pub fn read_testtagsfile(filename: &str, golddata: Vec<String>, testmaxlen: usize) -> Vec<Vec<Vec<(POSTag, f64)>>> {
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
        contents.split("\n")
            .collect::<Vec<&str>>()
            .into_iter()
            .filter(|s| *s != "")
            .map(|s| s.split(' ')
                      .map(decode_td)
                      .collect::<Vec<Vec<(POSTag, f64)>>>())
            .filter(|v| v.len() <= testmaxlen)
            .collect::<Vec<Vec<Vec<(POSTag, f64)>>>>()
    } else {
        golddata.iter()
                .map(|s| s.split(' ')
                          .map(|t| vec![(t.to_string(), 1.0)])
                          .collect::<Vec<Vec<(POSTag, f64)>>>())
        .collect::<Vec<Vec<Vec<(POSTag, f64)>>>>()
    }
}

pub fn get_data(wsj_path: &str, spmrl_path: &str, stats: &mut PCFGParsingStatistics)
        -> ((HashMap<NT, HashMap<RHS, f64>>, HashMap<NT, String>), (Vec<String>, Vec<Vec<Vec<(POSTag, f64)>>>, Vec<PTBTree>))  {
    
    let train_trees = match (wsj_path, spmrl_path) {
        ("", _) => ptb_reader::parse_spmrl_ptb_file(&(spmrl_path.to_string() + "/train/train.German.gold.ptb")).unwrap(),
        (_, "") => ptb_reader::parse_ptb_sections(wsj_path, (2..22).collect()), // sections 2-21
        _ => unreachable!()
    };
    let test_trees = match (wsj_path, spmrl_path) {
        ("", _) => ptb_reader::parse_spmrl_ptb_file(&(spmrl_path.to_string() + "/dev/dev.German.gold.ptb")).unwrap(),
        (_, "") => ptb_reader::parse_ptb_sections(wsj_path, vec![22]), // section 22
        _ => unreachable!()
    };
    
    let (unb_rules, unb_ntdict) = crunch_train_trees(train_trees, &stats);
    let (testsents, testposs, testtrees) = crunch_test_trees(test_trees, &stats);
    
    let testposs = read_testtagsfile(&stats.testtagsfile, testposs, stats.testmaxlen);
    
    let (bin_rules, bin_ntdict) = binarize_grammar(&unb_rules, &unb_ntdict);
    
    stats.unbin_nts = unb_ntdict.len();
    stats.bin_nts   = bin_ntdict.len();
    
    ((bin_rules, bin_ntdict), (testsents, testposs, testtrees))
}