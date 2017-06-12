use std::collections::{HashMap, HashSet};
use std::collections::BinaryHeap;

use defs::*;
use featurestructures::*;

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

pub fn agenda_cky_parse<'a>(
    bin_rules: &'a HashMap<NT, HashMap<RHS, f64>>,
    bin_ntdict: &HashMap<NT, String>,
    initial_nts: &Vec<NT>,
    sents: &'a [String],
    poss: &'a Vec<Vec<Vec<(POSTag, f64)>>>,
    pret_distr_all: HashMap<NT, f64>,
    pret_distr_le3: HashMap<NT, f64>,
    mut stats: &mut PCFGParsingStatistics)
    -> Vec<HashMap<NT, (f64, ParseTree<'a>)>>
{
    #[inline]
    fn bin(val: f64) -> usize {
        let noofbins = 20 as f64;
        (val * noofbins).ceil() as usize
    }
    fn unbin(bin: usize) -> f64 {
        let noofbins = 20 as f64;
        (bin as f64) / noofbins
    }
    
    let maxbin = bin(1.0);
    let mut bins: Vec<usize> = Vec::with_capacity(maxbin+1);
    bins.resize(maxbin+1, 0);
    
    let mut fullmatches = 0;
    
    
    // Build helper dicts for quick access. All are bottom-up in the parse.
    let t = get_usertime();
    let ntcount = bin_rules.len();
    // Get the fat HashMaps...
    let (word_to_preterminal, all_preterminals, nt_chains, _, rhs_l_to_lhs, rhs_r_to_lhs) = preprocess_rules(bin_rules);
    let mut terminal_matcher = embed_rules(&word_to_preterminal, bin_ntdict, stats);
    // ...and convert some to Vecs for faster addressing
    let nt_chains_vec: Vec<Vec<(NT, f64)>> = hashmap_to_vec(nt_chains);
    let rhs_l_to_lhs_vec: Vec<Vec<(NT, NT, f64)>> = hashmap_to_vec(rhs_l_to_lhs);
    let rhs_r_to_lhs_vec: Vec<Vec<(NT, NT, f64)>> = hashmap_to_vec(rhs_r_to_lhs);
    stats.cky_prep = get_usertime() - t;
    
    stats.cky_terms = 0.0;
    stats.cky_higher = 0.0;
    
    let mut results: Vec<HashMap<NT, (f64, ParseTree<'a>)>> = Vec::new();
    
    for (raw_sent, pos_descs) in sents.iter().zip(poss) {
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
        // First do the good ones...
        
        fn handle_terminals<E: IsEmbedding> (
            embdr: &mut E,
            ckychart: &mut Vec<(f64, usize, usize)>,
            agenda: &mut BinaryHeap<AgendaItem>,
            i: usize,
            sentlen: usize,
            ntcount: usize,
            wsent: &str,
            wsent_pos_desc: &Vec<(String, f64)>,
            bins: &mut Vec<usize>,
            fullmatches: &mut usize,
            stats: &mut PCFGParsingStatistics)
        {
            let esent: usize = embdr.embed_sent(wsent, wsent_pos_desc);
            
            for &(erule, ref rules) in embdr.get_e_id_to_rules() {
                let comp = embdr.comp(erule, esent);
                assert!(comp - ::std::f64::EPSILON <= 1.0);
                assert!(comp + ::std::f64::EPSILON >= 0.0);
                
                for &(ref wrule, nt, logprob) in rules {
                    if stats.logcompvalues {
                        bins[bin(comp)] += 1;
                        if wsent == wrule {*fullmatches += 1}
                    };
                    let lp_add = if wrule == wsent {
                        (stats.eta * comp.powf(stats.beta) + (1.0-stats.eta)).ln()
                    } else {
                        (stats.eta * comp.powf(stats.beta)                  ).ln()
                    };
                    if lp_add == ::std::f64::NEG_INFINITY {continue};
                    
                    let addr = chart_adr(sentlen, ntcount, i, i + 1, nt);
                    let logprob = logprob + lp_add;
                    if ckychart[addr].0 < logprob {
                        ckychart[addr].0 = logprob;
                        agenda.push(AgendaItem(logprob, i, i+1, nt))
                    }
                }
            }
        }
        
        assert_eq!(pos_descs.len(), sentlen);
        for (i, (wsent, wsent_pos_desc)) in sent.iter().zip(pos_descs).enumerate() {
            if word_to_preterminal.get(*wsent) == None {
                oov_in_this_sent = true;
                stats.oov_words += 1
            }
            
            let is_oov = word_to_preterminal.get(*wsent) == None;
            
            if !stats.only_oovs_soft || is_oov {
                match terminal_matcher {
                    TerminalMatcher::ExactMatcher(ref mut embdr) => {
                        handle_terminals(embdr, &mut ckychart, &mut agenda, i, sentlen, ntcount, wsent, wsent_pos_desc, &mut bins, &mut fullmatches, &mut stats)
                    }
                    TerminalMatcher::POSTagMatcher(ref mut embdr) => {
                        handle_terminals(embdr, &mut ckychart, &mut agenda, i, sentlen, ntcount, wsent, wsent_pos_desc, &mut bins, &mut fullmatches, &mut stats)
                    }
                    TerminalMatcher::LCSMatcher(ref mut embdr) => {
                        handle_terminals(embdr, &mut ckychart, &mut agenda, i, sentlen, ntcount, wsent, wsent_pos_desc, &mut bins, &mut fullmatches, &mut stats)
                    }
                    TerminalMatcher::NGramMatcher(ref mut embdr) => {
                        handle_terminals(embdr, &mut ckychart, &mut agenda, i, sentlen, ntcount, wsent, wsent_pos_desc, &mut bins, &mut fullmatches, &mut stats)
                    },
                    TerminalMatcher::LevenshteinMatcher(ref mut embdr) => {
                        handle_terminals(embdr, &mut ckychart, &mut agenda, i, sentlen, ntcount, wsent, wsent_pos_desc, &mut bins, &mut fullmatches, &mut stats)
                    }
                }
            } else {
                // Otherwise only do exact matching!
                if let Some(tagginglogprobs) = word_to_preterminal.get(*wsent) {
                    for &(nt, logprob) in tagginglogprobs {
                        let addr = chart_adr(sentlen, ntcount, i, i + 1, nt);
                        ckychart[addr].0 = logprob;
                        agenda.push(AgendaItem(logprob, i, i+1, nt))
                    }
                }
            }
        }
        // ...then do the fallback.
        for (i, _) in sent.iter().enumerate() {
            let mut nothing_was_found = true;
            for nt in &all_preterminals {
                let addr = chart_adr(sentlen, ntcount, i, i + 1, *nt);
                if ckychart[addr].0 != ::std::f64::NEG_INFINITY {
                    nothing_was_found = false;
                }
            }
            if nothing_was_found || stats.all_terms_fallback {
                match stats.oov_handling {
                    OOVHandling::Zero => (),
                    OOVHandling::Uniform => {
                        for nt in &all_preterminals {
                            let addr = chart_adr(sentlen, ntcount, i, i + 1, *nt);
                            if uniform_oov_prob > ckychart[addr].0 { // for all_terms_fallback that should be checked
                                ckychart[addr].0 = uniform_oov_prob;
                                agenda.push(AgendaItem(uniform_oov_prob, i, i+1, *nt))
                            }
                        }
                    },
                    OOVHandling::MarginalAll => {
                        for (nt, p) in &pret_distr_all {
                            let addr = chart_adr(sentlen, ntcount, i, i + 1, *nt);
                            let logprob = p.ln() + uniform_oov_prob;
                            if logprob > ckychart[addr].0 { // for all_terms_fallback that should be checked
                                ckychart[addr].0 = logprob;
                                agenda.push(AgendaItem(logprob, i, i+1, *nt))
                            }
                        }
                    }
                    OOVHandling::MarginalLe3 => {
                        for (nt, p) in &pret_distr_le3 {
                            let addr = chart_adr(sentlen, ntcount, i, i + 1, *nt);
                            let logprob = p.ln() + uniform_oov_prob;
                            if logprob > ckychart[addr].0 { // for all_terms_fallback that should be checked
                                ckychart[addr].0 = logprob;
                                agenda.push(AgendaItem(logprob, i, i+1, *nt))
                            }
                        }
                    }
                }
            }
        }
        stats.cky_terms += get_usertime() - t;
        
        
        // Now do the lengthy agenda-working
        let t = get_usertime();
        while let Some(AgendaItem(base_score, i, j, base_nt)) = agenda.pop() {
            // Let's check if we have a goal item (and can stop) first:
            if i == 0 && j == sentlen && base_nt <= stats.unbin_nts {
                if initial_nts.contains(&base_nt) && !stats.exhaustive {
                    break
                }
            }
            
            let base_addr = chart_adr(sentlen, ntcount, i, j, base_nt);
            //assert_eq!(ckychart[base_addr].0, base_score); // <- this actually does not hold, since it could have been updated in the meantime!
            assert!(base_score <= ckychart[base_addr].0); // This does. We should never have better agenda entries than chart entries.
            
            // Check if the score is still the same in the chart as in the agenda...
            if base_score < ckychart[base_addr].0 {
                // ...if not we found (and used!) a better one already and can skip this one!
                continue
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
        for &nt in initial_nts {
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
    
    if stats.logcompvalues {
        for i in 0..(maxbin+1) {
            println!(">= {}: {}", unbin(i), bins[i])
        }
        println!("full: {}", fullmatches);
    }
    
    results
}

