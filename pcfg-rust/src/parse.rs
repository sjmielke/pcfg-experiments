use std::collections::{HashMap, HashSet};

use defs::*;

pub fn cky_parse<'a>(bin_rules: &'a HashMap<usize, HashMap<RHS, f64>>, sents: &'a [String], stats: &mut PCFGParsingStatistics) -> Vec<HashMap<usize, (f64, ParseTree<'a>)>> {
    // Build helper dicts for quick access. All are bottom-up in the parse.
    let t = get_usertime();
    let mut word_to_preterminal: HashMap<String, Vec<(usize, (f64, ParseTree))>> = HashMap::new();
    let mut nt_chains: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    let mut rhss_to_lhs: HashMap<(usize, usize), Vec<(usize, f64)>> = HashMap::new();
    let mut preterminals: HashSet<usize> = HashSet::new();
    
    for (lhs, rhsdict) in bin_rules {
        for (rhs, prob) in rhsdict {
            let logprob = prob.ln();
            match *rhs {
                RHS::Terminal(ref s) => {
                    let tree = ParseTree::TerminalNode { label: s };
                    let tree = ParseTree::InnerNode { label: *lhs, children: vec![tree] };
                    word_to_preterminal.entry(s.clone()).or_insert_with(Vec::new).push((*lhs, (logprob, tree.clone())));
                    preterminals.insert(*lhs);
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
    let preterminals: Vec<usize> = preterminals.into_iter().collect();
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
