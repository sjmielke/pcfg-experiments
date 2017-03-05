use std::collections::HashMap;
use std::f64; // log and exp

extern crate ptb_reader;

use ptb_reader::PTBTree;

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

fn pp_tree<'a>(ntdict: &HashMap<usize, String>, t: &ParseTree<'a>) -> String {
    match t {
        &ParseTree::TerminalNode { label } => label.to_string(),
        &ParseTree::InnerNode { label, ref children } => {
            let cs = children.iter().map(|c| pp_tree(ntdict, &c)).collect::<Vec<_>>().join(", ");
            ntdict[&label].clone() + "(" + &cs + ")"
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
        // First insert new inner map if necessary
        if !lhs_to_rhs_count.contains_key(&r.lhs) {
            lhs_to_rhs_count.insert(r.lhs.clone(), HashMap::new());
        }
        // Then unwrap.
        let ref mut innermap = lhs_to_rhs_count.get_mut(&r.lhs).unwrap();
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

fn print_grammar(rules: &HashMap<usize, HashMap<RHS, f64>>, ntdict: &HashMap<usize, String>) {
    let numrules: usize = rules.values().map(|d| d.len()).sum();
    println!("{} NTs, {} rules.", ntdict.len(), numrules);
    
    if true {
        println!("{:?}", ntdict);
        
        for (ref lhs, ref rhsdict) in rules {
            for (ref rhs, ref prob) in *rhsdict {
                match *rhs {
                    &RHS::Terminal(ref s) => println!("{:>15} -> {:<15} # {:1.10}", &ntdict[*lhs], s, prob),
                    &RHS::NTs(ref nts) => {
                        let ntlist: Vec<String> = nts.iter().map(|n| ntdict.get(n).unwrap().to_string()).collect();
                        println!("{:>15} -> {:<15} # {:1.10}", ntdict[*lhs], ntlist.join(" "), prob)
                    }
                    &RHS::CNFBin(ref r1, ref r2) => {
                        println!("{:>15} -> {:<15} # {:1.10}", ntdict[*lhs], format!("{} {}", &ntdict[r1], &ntdict[r2]), prob)
                    }
                    &RHS::CNFChain(ref r) => {
                        println!("{:>15} -> {:<15} # {:1.10}", ntdict[*lhs], &ntdict[r], prob)
                    }
                }
            }
        }
    }
}

fn cnfize_grammar(in_rules: &HashMap<usize, HashMap<RHS, f64>>, ntdict: &HashMap<usize, String>) -> (HashMap<usize, HashMap<RHS, f64>>, HashMap<usize, String>) {
    let mut rev_ntdict: HashMap<String, usize> = reverse_bijection(&ntdict);
    
    fn binarize(ntdict: &HashMap<usize, String>, rev_ntdict: &mut HashMap<String, usize>, lhs: usize, rhs: &[usize]) -> Vec<Rule> {
        if rhs.len() > 2 {
            // right-branching
            let ref rest = rhs[1..rhs.len()];
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
            match rhs {
                &RHS::Terminal(_) => assert!(innermap.insert(rhs.clone(), prob).is_none()),
                &RHS::CNFBin(_, _) => panic!("Already partially CNFized!"),
                &RHS::CNFChain(_) => panic!("Already partially CNFized!"),
                &RHS::NTs(ref nts) => {
                    let mut newrules = binarize(&ntdict, &mut rev_ntdict, lhs.clone(), nts);
                    let Rule { lhs: lhs_, rhs: rhs_ } = newrules.pop().unwrap();
                    assert!(*lhs == lhs_);
                    assert!(innermap.insert(rhs_, prob).is_none());
                    new_rules_tmp.extend(newrules)
                }
            }
        }
        assert!(cnf_rules.insert(lhs.clone(), innermap).is_none())
    }
    
    for Rule { lhs, rhs } in new_rules_tmp {
        // First insert new inner map if necessary
        if !cnf_rules.contains_key(&lhs) {
            cnf_rules.insert(lhs, HashMap::new());
        }
        // Then unwrap.
        cnf_rules.get_mut(&lhs).unwrap().insert(rhs, 1.0);
    }
    
    (cnf_rules, reverse_bijection(&rev_ntdict))
}

fn cky_parse<'a>(cnf_rules: &'a HashMap<usize, HashMap<RHS, f64>>, sents: &Vec<String>) -> Vec<HashMap<usize, (f64, ParseTree<'a>)>> {
    // Build helper dicts for quick access. All are bottom-up in the parse.
    let mut word_to_preterminal: HashMap<String, Vec<(usize, (f64, ParseTree))>> = HashMap::new();
    let mut nt_chains: HashMap<usize, Vec<(usize, f64)>> = HashMap::new();
    let mut rhss_to_lhs: HashMap<(usize, usize), Vec<(usize, f64)>> = HashMap::new();
    
    for (lhs, rhsdict) in cnf_rules {
        for (rhs, prob) in rhsdict {
            let logprob = prob.ln();
            match rhs {
                &RHS::Terminal(ref s) => {
                    if !word_to_preterminal.contains_key(s) {
                        word_to_preterminal.insert(s.clone(), Vec::new());
                    }
                    let tree = ParseTree::TerminalNode { label: s };
                    let tree = ParseTree::InnerNode { label: *lhs, children: vec![tree] };
                    word_to_preterminal.get_mut(s).unwrap().push((*lhs, (logprob, tree)))
                }
                &RHS::CNFChain(ref r) => {
                    if !nt_chains.contains_key(r) {
                        nt_chains.insert(r.clone(), Vec::new());
                    }
                    nt_chains.get_mut(&r).unwrap().push((*lhs, logprob))
                }
                &RHS::CNFBin(ref r1, ref r2) => {
                    if !rhss_to_lhs.contains_key(&(*r1, *r2)) {
                        rhss_to_lhs.insert((r1.clone(), r2.clone()), Vec::new());
                    }
                    rhss_to_lhs.get_mut(&(*r1, *r2)).unwrap().push((*lhs, logprob))
                }
                _ => panic!("Trying to use un-CNFized grammar!")
            }
        }
    }
    
    let mut results: Vec<HashMap<usize, (f64, ParseTree<'a>)>> = Vec::new();
    
    for raw_sent in sents {
        println!("parsing: {}", raw_sent);
        
        // Tokenize
        let sent: Vec<&str> = raw_sent.split(" ").collect();
        
        // Now populate a chart (0-based indexing)!
        let mut ckychart: HashMap<(usize, usize), HashMap<usize, (f64, ParseTree)>> = HashMap::new();
        
        // Populate leafs
        for (i,w) in sent.iter().enumerate() {
            // TODO actually could just break if we don't recognize terminals :D
            let terminals: HashMap<usize, (f64, ParseTree)> = match word_to_preterminal.get(*w) {
                Some(prets) => prets.iter().map(|x| x.clone()).collect(),
                None => HashMap::new()
            };
            ckychart.insert((i, i), terminals);
            //println!("{}, {:?}", i, ckychart[&(i,i)]);
        }
        
        // Populate inner cells
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
        
        results.push(ckychart[&(0, sent.len() - 1)].clone())
    }
    
    results
}

fn bananaset() -> ((Vec<Rule>, HashMap<usize, String>), Vec<String>) {
    let s = "
        S -> NP
        NP -> DET NN
        NP -> DET JJ NN
        DET -> 'the'
        DET -> 'the'
        DET -> 'a'
        DET -> 'an'
        JJ -> 'fresh'
        NN -> 'apple'
        NN -> 'orange'
        ";
    
    let mut rulelist: Vec<Rule> = Vec::new();
    let mut rev_ntdict: HashMap<String, usize> = HashMap::new();
    
    for l in s.lines() {
        let l: &str = l.trim();
        if !l.is_empty() {
            let ruleparts: Vec<&str> = l.split(" -> ").collect();
            assert!(ruleparts.len() == 2);
            let rule = Rule { lhs: treat_nt(&mut rev_ntdict, ruleparts[0]), rhs:
                match ruleparts[1].chars().next() {
                    Some('\'') => {
                        let mut trimmed_rhs = ruleparts[1][1..].to_string();
                        let newlen = trimmed_rhs.len() - 1;
                        trimmed_rhs.truncate(newlen);
                        RHS::Terminal(trimmed_rhs)
                    },
                    Some(_) => { 
                        let mut nts: Vec<usize> = Vec::new();
                        for p in ruleparts[1].split(" ") {
                            nts.push(treat_nt(&mut rev_ntdict, p))
                        }
                        RHS::NTs(nts)
                    },
                    None => panic!("Empty LHS!"),
                }
            };
            rulelist.push(rule);
        }
    }
    
    let ntdict = reverse_bijection(&rev_ntdict);
    
    ((rulelist, ntdict), vec!["the fresh apple".to_string(), "the fresh banana".to_string()])
}

fn ptbset() -> ((Vec<Rule>, HashMap<usize, String>), Vec<String>) {
    let all_trees = ptb_reader::parse_ptb_sample_dir("/home/sjm/documents/Uni/penn-treebank-sample/treebank/combined/");
    
    let mut rulelist: Vec<Rule> = Vec::new();
    let mut rev_ntdict: HashMap<String, usize> = HashMap::new();
    
    fn getrules(t: &PTBTree, rulelist: &mut Vec<Rule>, rev_ntdict: &mut HashMap<String, usize>) {
        match t {
            &PTBTree::InnerNode { ref label, ref children } => {
                let lhs = treat_nt(rev_ntdict, &label);
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
                    let s = match c {
                        &PTBTree::InnerNode { ref label, children: _ } => label,
                        &PTBTree::TerminalNode { ref label } => label
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
    
    for t in &all_trees[0..2000] {
        //println!("{}", t);
        getrules(&t, &mut rulelist, &mut rev_ntdict);
    }
    
    //println!("From {} trees: NTs: {}, Rules: {}", &all_trees[0..2000].len(), rev_ntdict.len(), rulelist.len());
    
    let ntdict = reverse_bijection(&rev_ntdict);
    
    // load test sents
    
    let mut testsents: Vec<String> = Vec::new();
    for t in &all_trees[2000..2100] {
        let s: String = From::from(t.clone()); // yield
        testsents.push(s)
    }
    
    ((rulelist, ntdict), testsents)
}

fn main() {
    let ((cnf_rules, cnf_ntdict), testsents) = {
        let ((rulelist, ntdict), testsents) = ptbset();
        let rules = normalize_grammar(rulelist.iter());
        //print_grammar(&rules, &ntdict);
        println!("Now CNFing!");
        (cnfize_grammar(&rules, &ntdict), testsents)
    };
    //println!("What came out?");
    //print_grammar(&cnf_rules, &cnf_ntdict);
    //println!("try printing original again");
    //print_grammar(&rules, &ntdict);
    //println!("try printing cnfized again");
    //print_grammar(&cnf_rules, &cnf_ntdict);
    
    println!("Now parsing!");
    let parses = cky_parse(&cnf_rules, &testsents);
    
    for (s, cell) in testsents.iter().zip(parses.iter()) {
        if !cell.is_empty() {
            println!("{}", s);
            let mut got_s = false;
            let mut usablecell: Vec<_> = cell.iter().filter(|&(n, _)| cnf_ntdict[n].chars().next().unwrap() != '_').collect();
            usablecell.sort_by(|&(_, &(p1, _)), &(_, &(p2, _))| p2.partial_cmp(&p1).unwrap_or(std::cmp::Ordering::Equal));
            for &(n, &(p, ref ptree)) in usablecell.iter().take(10) {
                println!("\t{:15} ({:1.20}) -> {}", cnf_ntdict[n], p.exp(), pp_tree(&cnf_ntdict, ptree));
                if cnf_ntdict[n] == "S" {
                    got_s = true
                }
            }
            if usablecell.len() > 10 {
                println!("\t...");
                if !got_s {
                    for (n, &(p, ref ptree)) in usablecell {
                        if cnf_ntdict[n] == "S" {
                            println!("\t{:15} ({:1.20}) -> {}", cnf_ntdict[n], p.exp(), pp_tree(&cnf_ntdict, ptree))
                        }
                    }
                }
            }
        }
        else {
            println!("{}\n", s);
        }
    }
}
