use std::collections::{HashMap, HashSet};
use std::collections::BinaryHeap;

use defs::*;

type ToVecOf<T,S> = HashMap<T, Vec<S>>;
type ToNTVec<T> = HashMap<T, Vec<(NT, f64)>>;

fn preprocess_rules(bin_rules: &HashMap<NT, HashMap<RHS, f64>>)
        ->  ( ToNTVec<String>            // word_to_preterminal
            , Vec<NT>                    // preterminals
            , ToNTVec<NT>                // nt_chains
            , ToNTVec<(NT, NT)>          // rhss_to_lhs
            , ToVecOf<NT, (NT, NT, f64)> // rhs_l_to_lhs
            , ToVecOf<NT, (NT, NT, f64)> // rhs_r_to_lhs
        ) {
    let mut word_to_preterminal: ToNTVec<String>            = HashMap::new();
    let mut preterminals_set   : HashSet<NT>                = HashSet::new();
    let mut nt_chains          : ToNTVec<NT>                = HashMap::new();
    let mut rhss_to_lhs        : ToNTVec<(NT, NT)>          = HashMap::new();
    let mut rhs_l_to_lhs       : ToVecOf<NT, (NT, NT, f64)> = HashMap::new();
    let mut rhs_r_to_lhs       : ToVecOf<NT, (NT, NT, f64)> = HashMap::new();
    
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
                    rhss_to_lhs.entry((r1, r2)).or_insert_with(Vec::new).push((*lhs, logprob));
                    rhs_l_to_lhs.entry(r1).or_insert_with(Vec::new).push((*lhs, r2, logprob));
                    rhs_r_to_lhs.entry(r2).or_insert_with(Vec::new).push((*lhs, r1, logprob))
                }
                _ => panic!("Trying to use unbinarized grammar!")
            }
        }
    }
    let preterminals: Vec<NT> = preterminals_set.into_iter().collect();
    
    (word_to_preterminal, preterminals, nt_chains, rhss_to_lhs, rhs_l_to_lhs, rhs_r_to_lhs)
}

#[allow(dead_code)]
pub fn cky_parse<'a>(bin_rules: &'a HashMap<NT, HashMap<RHS, f64>>, sents: &'a [String], stats: &mut PCFGParsingStatistics) -> Vec<HashMap<NT, (f64, ParseTree<'a>)>> {
    // Build helper dicts for quick access. All are bottom-up in the parse.
    let t = get_usertime();
    let (word_to_preterminal, all_preterminals, nt_chains, rhss_to_lhs, _, _) = preprocess_rules(bin_rules);
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
        
        #[allow(dead_code)]
        fn print_chart(ckychart: &HashMap<(NT, NT), HashMap<NT, (f64, ParseTree)>>, sent: &Vec<&str>) {
            for width in 1 .. sent.len() + 1 {
                println!("Width {:2}: ", width);
                for start in 0 .. sent.len() - (width-1) {
                    let end = start + width;
                    print!("    {:?}   ->  ", &sent[start..end]);
                    for (nt, _) in ckychart.get(&(start, end - 1)).clone().unwrap_or(&HashMap::new()) {
                        print!("{}, ", nt);
                    }
                    println!("");
                }
                println!("");
            }
        }
        
        // Populate leafs
        let t = get_usertime();
        for (i,w) in sent.iter().enumerate() {
            // TODO actually could just break if we don't recognize terminals :D
            let mut preterminals: HashMap<NT, (f64, ParseTree)> = match word_to_preterminal.get(*w) {
                Some(prets) => prets.iter().map(|&(nt, logprob)| (nt, (logprob, ParseTree::InnerNode { label: nt, children: vec![ParseTree::TerminalNode { label: w }] }))).collect(),
                None => {
                    stats.oov_words += 1;
                    oov_in_this_sent = true;
                    match stats.oov_handling {
                        OOVHandling::Zero => HashMap::new(),
                        OOVHandling::Uniform => all_preterminals.clone().into_iter().map(|pt| (pt, (-300.0, ParseTree::InnerNode { label: pt, children: vec![ParseTree::TerminalNode { label: w }] }))).collect(),
                        OOVHandling::Marginal => panic!("Unimplemented")
                    }
                }
            };
            // Apply unary rules!
            let mut got_new = true;
            while got_new {
                got_new = false;
                let celllist = preterminals.clone();
                let celllist: Vec<_> = celllist.iter().collect();
                for (reached_nt, &(lower_prob, ref lower_ptree)) in celllist {
                    if let Some(v) = nt_chains.get(reached_nt) {
                        for &(lhs, cr_prob) in v {
                            let new_prob = cr_prob + lower_prob;
                            let tree = ParseTree::InnerNode {label: lhs, children: vec![lower_ptree.clone()]};
                            match preterminals.get(&lhs) {
                                None => {
                                    preterminals.insert(lhs, (new_prob, tree));
                                    got_new = true;
                                }
                                Some(&(old_prob, _)) => {
                                    if old_prob < new_prob {
                                        preterminals.insert(lhs, (new_prob, tree));
                                        got_new = true;
                                    }
                                }
                            };
                        }
                    }
                }
            }
            
            ckychart.insert((i, i), preterminals);
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
                                        Some(&(old_prob, _)) => {
                                            if old_prob < new_prob {
                                                cell.insert(lhs, (new_prob, tree));
                                            }
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
                                        got_new = true;
                                    }
                                    Some(&(old_prob, _)) => {
                                        if old_prob < new_prob {
                                            cell.insert(lhs, (new_prob, tree));
                                            got_new = true;
                                        }
                                    }
                                };
                            }
                        }
                    }
                }
                
                assert!(ckychart.insert((start, end - 1), cell).is_none())
            }
            
            //print_chart(&ckychart, &sent);
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

/// Adressing a CKY chart
/// 
/// _0_ some _1_ test _2_ string _3_
/// ...meaning (0, 3) covers the whole sentence, any NTs here are goal items.
///
///    Map<(NT, NT), Map<NT, (f64, lpred, rpred)>>
/// => Vec< Map<NT, (f64, lpred, rpred)> >         (span (a, b) -> [a * sentlen + b])
/// => Vec< (f64, usize, usize) >                  (span (a, b), NT nt -> [(a * sentlen + b) * ntcount + (nt - 1)], because NTs start at 1)
/// ...yielding a Vec of up to 60 mio items (approx. 1.5 GB) for 120 words, although just 30 words less than a 10th.
/// 
/// Over half of the entries are unused, but... eh, who cares, it's small enough.
#[inline]
fn chart_adr(sentlen: usize, ntcount: usize, a: usize, b: usize, nt: NT) -> usize {
    // assert!(a < b);
    // assert!(b <= sentlen);
    // assert!(nt > 0);
    // assert!(nt <= ntcount);
    (a * (sentlen+1) + b) * ntcount + (nt - 1)
}
/// `chart_adr` is reversible, given sentlen and ntcount, but only very slowly (modulo).
/// Hence we should avoid doing it in the inner loop!
fn rev_chart_adr(sentlen: usize, ntcount: usize, addr: usize) -> (usize, usize, usize) {
    let nt = (addr % ntcount) + 1;
    let a_sentlen_plus_b = (addr - (nt - 1)) / ntcount;
    let b = a_sentlen_plus_b % (sentlen+1);
    let a = (a_sentlen_plus_b - b) / (sentlen+1);
    (a, b, nt)
}
#[test]
fn test_chart_adr() {
    let sentlen = 30;
    let ntcount = 20;
    assert!(::std::usize::MAX > 60000000);
    // cheap test that it works!
    let mut alladdrs: HashSet<usize> = HashSet::new();
    for a in 0..sentlen+1 {
        for b in 0..sentlen+1 {
            for nt in 1..ntcount+1 {
                let addr = chart_adr(sentlen, ntcount, a, b, nt);
                assert!(alladdrs.insert(addr));
                assert_eq!(rev_chart_adr(sentlen, ntcount, addr), (a, b, nt));
            }
        }
    }
}

#[derive(PartialEq, PartialOrd)]
/// Score, span l/r, finished NT
struct AgendaItem(f64, usize, usize, NT);
impl ::std::cmp::Eq for AgendaItem {}
impl ::std::cmp::Ord for AgendaItem {
    fn cmp(&self, other: &Self) -> ::std::cmp::Ordering {
        //if other.0.is_nan() {
        //    ::std::cmp::Ordering::Greater
        //} else {
            self.0.partial_cmp(&other.0).expect("Trying to compare a NaN score!")
        //}
    }
}

fn recover_parsetree<'a>(ckychart: &[(f64, usize, usize)], sentlen: usize, ntcount: usize, i: usize, j: usize, nt: NT, sent: &[&'a str]) -> ParseTree<'a> {
    let addr = chart_adr(sentlen, ntcount, i, j, nt);
    let (_, lc_addr, rc_addr) = ckychart[addr];
    let cs = if lc_addr == ::std::usize::MAX {
        // terminal rule
        assert!(i + 1 == j);
        assert!(rc_addr == ::std::usize::MAX);
        let lc_tree = ParseTree::TerminalNode { label: sent[i] };
        vec![lc_tree]
    } else if rc_addr == ::std::usize::MAX {
        // unary rule
        let (lc_span_i, lc_span_j, lc_nt) = rev_chart_adr(sentlen, ntcount, lc_addr);
        let lc_tree = recover_parsetree(ckychart, sentlen, ntcount, lc_span_i, lc_span_j, lc_nt, sent);
        vec![lc_tree]
    } else {
        // binary rule
        let (lc_span_i, lc_span_j, lc_nt) = rev_chart_adr(sentlen, ntcount, lc_addr);
        let (rc_span_i, rc_span_j, rc_nt) = rev_chart_adr(sentlen, ntcount, rc_addr);
        let lc_tree = recover_parsetree(ckychart, sentlen, ntcount, lc_span_i, lc_span_j, lc_nt, sent);
        let rc_tree = recover_parsetree(ckychart, sentlen, ntcount, rc_span_i, rc_span_j, rc_nt, sent);
        vec![lc_tree, rc_tree]
    };
    ParseTree::InnerNode { label: nt, children: cs }
}

fn hashmap_to_vec<V: Default>(hm: HashMap<usize, V>) -> Vec<V> {
    let mut mapvec: Vec<V> = Vec::new();
    for (k, v) in hm {
        while mapvec.len() <= k {
            mapvec.push(V::default())
        }
        mapvec[k] = v
    }
    mapvec
}

pub fn agenda_cky_parse<'a>(bin_rules: &'a HashMap<NT, HashMap<RHS, f64>>, sents: &'a [String], stats: &mut PCFGParsingStatistics) -> Vec<HashMap<NT, (f64, ParseTree<'a>)>> {
    // Build helper dicts for quick access. All are bottom-up in the parse.
    let t = get_usertime();
    let ntcount = bin_rules.len();
    // Get the fat HashMaps...
    let (word_to_preterminal, all_preterminals, nt_chains, _, rhs_l_to_lhs, rhs_r_to_lhs) = preprocess_rules(bin_rules);
    // ...and convert some to Vecs for faster addressing
    let nt_chains_vec: Vec<Vec<(NT, f64)>> = hashmap_to_vec(nt_chains);
    let rhs_l_to_lhs_vec: Vec<Vec<(NT, NT, f64)>> = hashmap_to_vec(rhs_l_to_lhs);
    let rhs_r_to_lhs_vec: Vec<Vec<(NT, NT, f64)>> = hashmap_to_vec(rhs_r_to_lhs);
    stats.cky_prep = get_usertime() - t;
    
    stats.cky_terms = 0.0;
    stats.cky_higher = 0.0;
    
    let mut results: Vec<HashMap<NT, (f64, ParseTree<'a>)>> = Vec::new();
    
    let mut useless_pops = 0;
    let mut used_pops = 0;
    
    let mut skips = 0;
    let mut noskips = 0;
    
    for raw_sent in sents {
        //println!("parsing: {}", raw_sent);
        
        let mut oov_in_this_sent = false;
        
        // Tokenize
        let sent: Vec<&str> = raw_sent.split(' ').collect();
        let sentlen = sent.len();
        
        // score and left and right predecessors in compact notation
        let chartlength = chart_adr(sentlen, ntcount, (sentlen-1), sentlen, ntcount) + 1;
        let mut ckychart: Vec<(f64, usize, usize)> = Vec::with_capacity(chartlength);
        //println!("Need {} cells.", chartlength);
        
        ckychart.resize(chartlength, (::std::f64::NEG_INFINITY, ::std::usize::MAX, ::std::usize::MAX));
        
        // The agenda into these addresses will contain newly constructed chart items that are
        // ready for recombination.
        //let mut agenda: BinaryHeap<AgendaItem> = BinaryHeap::new(); // TODO maybe: with_capacity(chartlength);
        let mut agenda: BinaryHeap<AgendaItem> = BinaryHeap::with_capacity(chartlength);
        
        // Kick it off with the terminals!
        let t = get_usertime();
        let uniform_oov_prob = stats.uniform_oov_prob;
        for (i, w) in sent.iter().enumerate() {
            match word_to_preterminal.get(*w) {
                Some(prets) => {
                    for &(nt, logprob) in prets {
                        let addr = chart_adr(sentlen, ntcount, i, i + 1, nt);
                        ckychart[addr].0 = logprob;
                        agenda.push(AgendaItem(logprob, i, i+1, nt))
                    }
                    if stats.all_terms_fallback {
                        for nt in &all_preterminals {
                            let addr = chart_adr(sentlen, ntcount, i, i + 1, *nt);
                            if ckychart[addr].0 == ::std::f64::NEG_INFINITY {
                                ckychart[addr].0 = uniform_oov_prob;
                                agenda.push(AgendaItem(uniform_oov_prob, i, i+1, *nt))
                            }
                        }
                    }
                }
                None => {
                    oov_in_this_sent = true;
                    stats.oov_words += 1;
                    match stats.oov_handling {
                        OOVHandling::Zero => (),
                        OOVHandling::Uniform => {
                            for nt in &all_preterminals {
                                let addr = chart_adr(sentlen, ntcount, i, i + 1, *nt);
                                ckychart[addr].0 = uniform_oov_prob;
                                agenda.push(AgendaItem(uniform_oov_prob, i, i+1, *nt))
                            }
                        },
                        OOVHandling::Marginal => panic!("Unimplemented")
                    }
                }
            }
        }
        stats.cky_terms += get_usertime() - t;
        
        // Now do the lengthy agenda-working
        let t = get_usertime();
        let mut done = false;
        while let Some(AgendaItem(base_score, i, j, base_nt)) = agenda.pop() {
            // Let's check if we have a goal item (and can stop) first:
            if i == 0 && j == sentlen && base_nt <= stats.unbin_nts {
                done = true;
                if !stats.exhaustive {
                    break
                }
            }
            
            let base_addr = chart_adr(sentlen, ntcount, i, j, base_nt);
            //assert_eq!(ckychart[base_addr].0, base_score); // <- this actually does not hold, since it could have been updated in the meantime!
            
            // Check if the score is still the same in the chart as in the agenda...
            if base_score < ckychart[base_addr].0 {
                // ...if not we found (and used!) a better one already and can skip this one!
                skips += 1;
                continue
            } else { noskips += 1 }
            
            // Counting
            if done {
                useless_pops += 1
            } else {
                used_pops += 1
            }
            
            //println!("Popping ({}, {}, {})", i, j, base_nt);
            
            // base is RHS of a unary rule
            if let Some(v) = nt_chains_vec.get(base_nt) {
                for &(lhs, logprob) in v {
                    let high_addr = chart_adr(sentlen, ntcount, i, j, lhs);
                    let newscore = base_score + logprob;
                    
                    if ckychart[high_addr].0 < newscore {
                        ckychart[high_addr] = (newscore, base_addr, ::std::usize::MAX);
                        agenda.push(AgendaItem(newscore, i, j, lhs))
                    }
                }
            }
            
            // base is left RHS of a binary rule
            if let Some(v) = rhs_l_to_lhs_vec.get(base_nt) {
                for &(lhs, rhs_r, logprob) in v {
                    // Try finding the rhs_r at span (j, k)
                    for k in j+1..sentlen+1 {
                        let rhs_r_addr = chart_adr(sentlen, ntcount, j, k, rhs_r);
                        let rhs_r_score = ckychart[rhs_r_addr].0;
                        if rhs_r_score != ::std::f64::NEG_INFINITY {
                            let high_addr = chart_adr(sentlen, ntcount, i, k, lhs);
                            let newscore = base_score + rhs_r_score + logprob;
                            
                            if ckychart[high_addr].0 < newscore {
                                ckychart[high_addr] = (newscore, base_addr, rhs_r_addr);
                                agenda.push(AgendaItem(newscore, i, k, lhs))
                            }
                        }
                    }
                }
            }
            
            // base is right RHS of a binary rule
            if i > 0 { // <- need this check to prevent i-1 underflow!!!
                if let Some(v) = rhs_r_to_lhs_vec.get(base_nt) {
                    for &(lhs, rhs_l, logprob) in v {
                        // Try finding the rhs_l at span (h, i)
                        for h in 0..i {
                            let rhs_l_addr = chart_adr(sentlen, ntcount, h, i, rhs_l);
                            let rhs_l_score = ckychart[rhs_l_addr].0;
                            if rhs_l_score != ::std::f64::NEG_INFINITY {
                                let high_addr = chart_adr(sentlen, ntcount, h, j, lhs);
                                let newscore = base_score + rhs_l_score + logprob;
                                
                                if ckychart[high_addr].0 < newscore {
                                    ckychart[high_addr] = (newscore, rhs_l_addr, base_addr);
                                    agenda.push(AgendaItem(newscore, h, j, lhs))
                                }
                            }
                        }
                    }
                }
            }
        }
        stats.cky_higher += get_usertime() - t;
        
        if oov_in_this_sent {
            stats.oov_sents += 1
        }
        
        let mut r: HashMap<NT, (f64, ParseTree)> = HashMap::new();
        for nt in 1..ntcount+1 {
            let item = ckychart[chart_adr(sentlen, ntcount, 0, sent.len(), nt)];
            if item.0 != ::std::f64::NEG_INFINITY {
                let tree = recover_parsetree(&ckychart, sentlen, ntcount, 0, sent.len(), nt, &sent);
                //println!("Tree:\n{}", tree.render());
                r.insert(nt, (item.0, tree));
            }
        }
        if r.is_empty() {
            stats.parsefails += 1
        }
        results.push(r)
    }
    
    println!("useless: {}, used: {}", useless_pops, used_pops);
    println!("skips: {}, noskips: {}", skips, noskips);
    
    results
}

