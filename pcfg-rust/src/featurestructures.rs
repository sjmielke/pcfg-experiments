use std::collections::{HashMap, HashSet};

use defs::*;

pub enum TerminalMatcher {
    POSTagMatcher (POSTagEmbedder),
    LCSRatioMatcher (LCSEmbedder),
    DiceMatcher (usize, bool, Vec<(HashSet<String>, Vec<(String, NT, f64)>)>), // kappa; dualmono_pad; assoc list from set of ngrams to list of rules
    LevenshteinMatcher(f64), // beta, TODO: trie for all for speed?
    ExactMatchOnly
}

pub trait IsEmbedding {
    type emb_rule: ::std::cmp::Eq + ::std::hash::Hash;
    type emb_sent;
    
    fn build_e_to_rules(&mut self, word_to_preterminal: &HashMap<String, Vec<(NT, f64)>>) {
        let mut e_to_rules: HashMap<_, Vec<(String, NT, f64)>> = HashMap::new();
        for (word, pret_vect) in word_to_preterminal {
            for &(nt, logprob) in pret_vect {
                let e = self.embed_rule(&(word, nt, logprob));
                e_to_rules.entry(e).or_insert_with(Vec::new).push((word.clone(), nt, logprob))
            }
        }
        *self.get_e_to_rules_mut() = e_to_rules.into_iter().collect()
    }
    fn get_e_to_rules(&self) -> &Vec<(Self::emb_rule, Vec<(String, NT, f64)>)>;
    fn get_e_to_rules_mut(&mut self) -> &mut Vec<(Self::emb_rule, Vec<(String, NT, f64)>)>;
    
    fn comp(&self, erule: &Self::emb_rule, esent: &Self::emb_sent) -> f64;
    fn embed_rule(&self, rule: &(&str, NT, f64)) -> Self::emb_rule;
    fn embed_sent(&self, word: &str, posdesc: &Vec<(POSTag, f64)>) -> Self::emb_sent;
}

pub struct POSTagEmbedder {
    e_to_rules: Vec<(POSTag, Vec<(String, NT, f64)>)>,
    bin_ntdict: HashMap<NT, String>,
    nbesttags: bool
}
impl IsEmbedding for POSTagEmbedder {
    type emb_rule = POSTag;
    type emb_sent = (POSTag, HashMap<POSTag, f64>); // argmax and scores
    
    fn get_e_to_rules(&self) -> &Vec<(Self::emb_rule, Vec<(String, NT, f64)>)> {&self.e_to_rules}
    fn get_e_to_rules_mut(&mut self) -> &mut Vec<(Self::emb_rule, Vec<(String, NT, f64)>)> {&mut self.e_to_rules}
    
    fn comp(&self, erule: &POSTag, esent: &(POSTag, HashMap<POSTag, f64>)) -> f64 {
        if self.nbesttags {
            esent.1.get(erule).unwrap_or(&::std::f64::NEG_INFINITY).exp()
        } else {
            if *erule == esent.0 {1.0} else {0.0}
        }
    }
    fn embed_rule(&self, rule: &(&str, NT, f64)) -> POSTag {
        let &(_, nt, _) = rule;
        self.bin_ntdict.get(&nt).unwrap().clone()
    }
    fn embed_sent(&self, _: &str, posdesc: &Vec<(POSTag, f64)>) -> (POSTag, HashMap<POSTag, f64>) {
        let mut max_lp = ::std::f64::NEG_INFINITY;
        let mut max_pos: &str = "";
        let mut hm: HashMap<String, f64> = HashMap::new();
        for &(ref p, lp) in posdesc {
            if lp >= max_lp {
                max_pos = p;
                max_lp = lp
            }
            hm.insert(p.to_string(), lp);
        }
        assert_eq!(max_lp, 0.0);
        (max_pos.to_string(), hm)
    }
}

pub struct LCSEmbedder {
    e_to_rules: Vec<(Vec<char>, Vec<(String, NT, f64)>)>,
    alpha: f64,
    beta: f64
}
impl IsEmbedding for LCSEmbedder {
    type emb_rule = Vec<char>;
    type emb_sent = Vec<char>;
    
    fn get_e_to_rules(&self) -> &Vec<(Self::emb_rule, Vec<(String, NT, f64)>)> {&self.e_to_rules}
    fn get_e_to_rules_mut(&mut self) -> &mut Vec<(Self::emb_rule, Vec<(String, NT, f64)>)> {&mut self.e_to_rules}
    
    fn comp(&self, erule: &Vec<char>, esent: &Vec<char>) -> f64 {
        (
            (lcs_dyn_prog(erule.as_slice(), esent.as_slice()) as f64)
            /
            (self.alpha * (erule.len() as f64) + (1.0-self.alpha) * (esent.len() as f64))
        ).powf(self.beta)
    }
    fn embed_rule(&self, rule: &(&str, NT, f64)) -> Vec<char> {
        let &(w, _, _) = rule;
        w.chars().collect()
    }
    fn embed_sent(&self, w: &str, _: &Vec<(POSTag, f64)>) -> Vec<char> {
        w.chars().collect()
    }
}

pub fn lcs_dyn_prog<T: Eq>(a: &[T], b: &[T]) -> usize {
    let mut table: Vec<Vec<usize>> = Vec::new();
    // Fill with 0s in first row/col
    // table_ai=0, table_bi ∈ {0..b.len()}
    let mut v = Vec::new();
    for _ in 0..b.len() + 1 {
        v.push(0)
    }
    table.push(v);
    // table_bi=0, table)ai ∈ {1..a.len()} (already did ai=0)
    for _ in 1..a.len() + 1 {
        let mut v = Vec::new();
        v.push(0);
        table.push(v)
    }
    // Complete table bottom-up
    for table_ai in 1..a.len() + 1 {
        for table_bi in 1..b.len() + 1 {
            let byte_ai = table_ai - 1;
            let byte_bi = table_bi - 1;
            // get longest suffix of a[0..ai] and b[0..bi]
            if a[byte_ai] != b[byte_bi] {
                table[table_ai].push(0)
            } else {
                let oldval = table[table_ai-1][table_bi-1];
                table[table_ai].push(1 + oldval)
            }
        }
    }
    // max entry is the length of the LCS
    let mut max = 0;
    for table_ai in 1..a.len() + 1 {
        for table_bi in 1..b.len() + 1 {
            if table[table_ai][table_bi] > max {
                max = table[table_ai][table_bi]
            }
        }
    }
    
    // print!("\n");
    // for ai in 0..a.len() + 1 {
    //     for bi in 0..b.len() + 1 {
    //         print!("{} ", table[ai][bi])
    //     }
    //     print!("\n")
    // }
    
    max
}


use std::sync::Mutex;
lazy_static! {
    // Assuming kappa stays constant during program execution!
    static ref NGRAMMAP: Mutex<HashMap<String,HashSet<String>>> = Mutex::new(HashMap::new());
}

pub fn get_ngrams(kappa: usize, dualmono_pad: bool, word: &str) -> HashSet<String> {
    if let Some(r) = NGRAMMAP.lock().unwrap().get(word) {
        return r.clone()
    }
    
    fn getgrams(s: String, kappa: usize) -> HashSet<String> {
        s
            .chars()
            .collect::<Vec<_>>()
            .windows(kappa)
            .map(|w| w.iter().cloned().collect())
            .collect()
    }
    
    let sharpstring = "#".repeat(kappa - 1);
    
    let r: HashSet<String> = if dualmono_pad {
        let r1: HashSet<String> = getgrams(sharpstring.clone() + word, kappa);
        let r2: HashSet<String> = getgrams(word.to_string() + &sharpstring, kappa);
        
        r1.union(&r2).cloned().collect()
    } else {
        getgrams(sharpstring.clone() + word + &sharpstring, kappa)
    };
    
    // println!("{}:", word);
    // for g in &r {
    //     println!("\t{:?}", g);
    // }
    
    NGRAMMAP.lock().unwrap().insert(word.to_string(), r.clone());
    
    r
}

// Returns a HashMap: F -> P(R), i.e. gives all rules deriving a certain feature structure.
// This way terminals are only compared with the number of unique feature structures
// and not with every rule itself.
pub fn embed_rules(
    word_to_preterminal: &HashMap<String, Vec<(NT, f64)>>,
    bin_ntdict: &HashMap<NT, String>,
    stats: &PCFGParsingStatistics)
    -> TerminalMatcher {
    
    if stats.feature_structures == "exactmatch" {
        return TerminalMatcher::ExactMatchOnly
    }
    
    // Init embedder
    let embdr_box = match &*stats.feature_structures {
        "postagsonly" => {
            let mut embdr = POSTagEmbedder { e_to_rules: vec![], bin_ntdict: bin_ntdict.clone(), nbesttags: stats.nbesttags };
            embdr.build_e_to_rules(word_to_preterminal);
            TerminalMatcher::POSTagMatcher(embdr)
        },
        "lcsratio" => {
            let mut embdr = LCSEmbedder { e_to_rules: vec![], alpha: stats.alpha, beta: stats.beta };
            embdr.build_e_to_rules(word_to_preterminal);
            TerminalMatcher::LCSRatioMatcher(embdr)
        },
        _ => unreachable!()
    };
    
    match &*stats.feature_structures {
        "postagsonly" => {
            embdr_box
        }
        "lcsratio" => {
            embdr_box
        }
        "dice" => {
            let mut result = Vec::new();
            for (word, pret_vect) in word_to_preterminal {
                let ngrams = get_ngrams(stats.kappa, stats.dualmono_pad, word);

                let mut rules = Vec::new();
                for &(nt, logprob) in pret_vect {
                    rules.push((word.clone(), nt, logprob));
                }
                result.push((ngrams, rules));
            }
            TerminalMatcher::DiceMatcher(stats.kappa, stats.dualmono_pad, result)
        }
        "levenshtein" => {
            TerminalMatcher::LevenshteinMatcher(stats.beta)
        }
        _ => {panic!("Incorrect feature structure / matching algorithm {} requested!", stats.feature_structures)}
    }
}
