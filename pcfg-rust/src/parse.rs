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

// Returns a HashMap: F -> P(R), i.e. gives all rules deriving a certain feature structure.
// This way terminals are only compared with the number of unique feature structures
// and not with every rule itself.
//
// For now it gives you the POS tag (original name, i.e., a String!)
fn embed_rules_pos(word_to_preterminal: &ToNTVec<String>, bin_ntdict: &HashMap<NT, String>) -> HashMap<String, Vec<(String, NT, f64)>> {
    let mut result = HashMap::new();
    for (word, pret_vect) in word_to_preterminal {
        for &(nt, logprob) in pret_vect {
            result.entry(bin_ntdict.get(&nt).unwrap().clone()).or_insert_with(Vec::new).push((word.clone(), nt, logprob));
        }
    }
    return result;
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

pub fn agenda_cky_parse<'a>(bin_rules: &'a HashMap<NT, HashMap<RHS, f64>>, bin_ntdict: &HashMap<NT, String>, sents: &'a [String], poss: &'a [String], stats: &mut PCFGParsingStatistics) -> Vec<HashMap<NT, (f64, ParseTree<'a>)>> {
    // Build helper dicts for quick access. All are bottom-up in the parse.
    let t = get_usertime();
    let ntcount = bin_rules.len();
    // Get the fat HashMaps...
    let (word_to_preterminal, all_preterminals, nt_chains, _, rhs_l_to_lhs, rhs_r_to_lhs) = preprocess_rules(bin_rules);
    let feature_to_rules = embed_rules_pos(&word_to_preterminal, bin_ntdict);
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
    
    for (raw_sent, raw_pos) in sents.iter().zip(poss) {
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
        match stats.feature_structures {
            FeatureStructures::ExactMatch => {
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
            },
            FeatureStructures::POSTagsOnly => {
                for (i, (w1, pos)) in sent.iter().zip(raw_pos.split(' ')).enumerate() {
                    // go through HashMap<String, Vec<(String, NT, f64)>>
                    let mut n: usize = 999999999;
                    for (pos_r, rules) in &feature_to_rules {
                        if pos_r == pos {
                            for &(ref w2, nt, logprob) in rules {
                                if n == 999999999 {n = nt} else { assert!(n == nt) };
                                let addr = chart_adr(sentlen, ntcount, i, i + 1, nt);
                                // p ̃(r(σ'))
                                //   = p(r) ⋅ (comp + μ ⋅ δ(σ = σ'))   (since we know A = B, comp = 1)
                                //   = p(r) ⋅ (1.0  + μ ⋅ δ(σ = σ'))   (now into log space...)
                                //   = p(r) + ln(1 + μ ⋅ δ(σ = σ'))
                                let logprob = logprob + (1.0 + (if w1 == w2 {stats.mu} else {0.0})).ln();
                                if ckychart[addr].0 < logprob {
                                    ckychart[addr].0 = logprob;
                                    agenda.push(AgendaItem(logprob, i, i+1, nt))
                                }
                            }
                        }
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
            assert!(base_score <= ckychart[base_addr].0); // This does. We should never have better agenda entries than chart entries.
            
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

