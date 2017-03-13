use std::collections::{HashMap, HashSet};

use defs::*;

type ToNT<T> = HashMap<T, Vec<(NT, f64)>>;

fn preprocess_rules(bin_rules: &HashMap<NT, HashMap<RHS, f64>>) -> (ToNT<String>, Vec<NT>, ToNT<NT>, ToNT<(NT, NT)>) {
    let mut word_to_preterminal: ToNT<String> = HashMap::new();
    let mut preterminals_set: HashSet<NT> = HashSet::new();
    let mut nt_chains: ToNT<NT> = HashMap::new();
    let mut rhss_to_lhs: ToNT<(NT, NT)> = HashMap::new();
    
    for (lhs, rhsdict) in bin_rules {
        for (rhs, prob) in rhsdict {
            let logprob = prob.ln();
            match *rhs {
                RHS::Terminal(ref s) => {
                    word_to_preterminal.entry(s.to_string()).or_insert_with(Vec::new).push((*lhs, logprob));
                    preterminals_set.insert(*lhs);
                }
                RHS::Unary(r) => {
                    nt_chains.entry(r).or_insert_with(Vec::new).push((*lhs, logprob))
                }
                RHS::Binary(r1, r2) => {
                    rhss_to_lhs.entry((r1, r2)).or_insert_with(Vec::new).push((*lhs, logprob))
                }
                _ => panic!("Trying to use unbinarized grammar!")
            }
        }
    }
    let preterminals: Vec<NT> = preterminals_set.into_iter().collect();
    
    (word_to_preterminal, preterminals, nt_chains, rhss_to_lhs)
}

pub fn cky_parse<'a>(bin_rules: &'a HashMap<NT, HashMap<RHS, f64>>, sents: &'a [String], stats: &mut PCFGParsingStatistics) -> Vec<HashMap<NT, (f64, ParseTree<'a>)>> {
    // Build helper dicts for quick access. All are bottom-up in the parse.
    let t = get_usertime();
    let (word_to_preterminal, preterminals, nt_chains, rhss_to_lhs) = preprocess_rules(bin_rules);
    stats.cky_prep = get_usertime() - t;
    
    stats.cky_terms = 0.0;
    stats.cky_higher = 0.0;
    
    let mut results: Vec<HashMap<NT, (f64, ParseTree<'a>)>> = Vec::new();
    
    for raw_sent in sents {
        //println!("parsing: {}", raw_sent);
        
        let mut oov_in_this_sent = false;
        
        // Tokenize
        let sent: Vec<&str> = raw_sent.split(' ').collect();
        
        // Now populate a chart (0-based indexing)!
        let mut ckychart: HashMap<(NT, NT), HashMap<NT, (f64, ParseTree)>> = HashMap::new();
        
        // Populate leafs
        let t = get_usertime();
        for (i,w) in sent.iter().enumerate() {
            // TODO actually could just break if we don't recognize terminals :D
            let terminals: HashMap<NT, (f64, ParseTree)> = match word_to_preterminal.get(*w) {
                Some(prets) => prets.iter().map(|&(nt, logprob)| (nt, (logprob, ParseTree::InnerNode { label: nt, children: vec![ParseTree::TerminalNode { label: w }] }))).collect(),
                None => {
                    stats.oov_words += 1;
                    oov_in_this_sent = true;
                    match stats.oov_handling {
                        OOVHandling::Zero => HashMap::new(),
                        OOVHandling::Uniform => preterminals.clone().into_iter().map(|pt| (pt, (-300.0, ParseTree::InnerNode { label: pt, children: vec![ParseTree::TerminalNode { label: w }] }))).collect(),
                        OOVHandling::Marginal => panic!("Unimplemented")
                    }
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
                let mut cell: HashMap<NT, (f64, ParseTree)> = HashMap::new();
                
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

pub fn agenda_cky_parse<'a>(bin_rules: &'a HashMap<NT, HashMap<RHS, f64>>, sents: &'a [String], stats: &mut PCFGParsingStatistics) -> Vec<HashMap<NT, (f64, ParseTree<'a>)>> {
    stats.print(false);
    panic!("{:?} {:?}", bin_rules, sents)
}