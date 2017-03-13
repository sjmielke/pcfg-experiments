use std::collections::HashMap;

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

pub fn binarize_grammar(in_rules: &HashMap<usize, HashMap<RHS, f64>>, ntdict: &HashMap<usize, String>) -> (HashMap<usize, HashMap<RHS, f64>>, HashMap<usize, String>) {
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
    
    let mut bin_rules: HashMap<usize, HashMap<RHS, f64>> = HashMap::new();
    let mut new_rules_tmp: Vec<Rule> = Vec::new(); // all have prob 1.0
    
    for (lhs, rhsdict) in in_rules {
        let mut innermap: HashMap<RHS, f64> = HashMap::new();
        for (rhs, &prob) in rhsdict {
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

fn ptbrulelist(wsj_path: &str, trainsize: usize) -> (Vec<Rule>, HashMap<usize, String>) {
    //println!("Reading in the PTB train set...");
    let mut train_trees = ptb_reader::parse_ptb_sections(wsj_path, (2..22).collect()); // sections 2-21
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
    
    
    //println!("PTB set done!");
    (rulelist, ntdict)
}

pub fn ptb_train(wsj_path: &str, stats: &mut PCFGParsingStatistics) -> (HashMap<usize, HashMap<RHS, f64>>, HashMap<usize, String>) {
    let (rulelist, ntdict) = ptbrulelist(wsj_path, stats.trainsize);
    let rules = normalize_grammar(rulelist.iter());
    (rules, ntdict)
}

pub fn ptb_test(wsj_path: &str, stats: &PCFGParsingStatistics) -> (Vec<String>, Vec<PTBTree>) {
    //println!("Reading, stripping and yielding test sentences...");
    let read_devtrees = ptb_reader::parse_ptb_sections(wsj_path, vec![22]);
    
    let mut devsents: Vec<String> = Vec::new();
    let mut devtrees: Vec<PTBTree> = Vec::new();
    for mut t in read_devtrees {
        t.strip_predicate_annotations();
        if t.front_length() <= stats.testmaxlen && devtrees.len() < stats.testsize {
            devsents.push(t.front());
            devtrees.push(t);
        }
    }
    assert_eq!(devtrees.len(), stats.testsize);
    //println!("From {} candidates we took {} dev sentences (max length {})!", read_devtrees.len(), devtrees.len(), stats.testmaxlen);
    
    (devsents, devtrees)
}
