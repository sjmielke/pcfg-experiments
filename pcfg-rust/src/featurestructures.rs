use std::collections::{HashMap, BTreeMap, BTreeSet};

use defs::*;
extern crate strsim;

pub enum TerminalMatcher {
    ExactMatcher (ExactEmbedder),
    POSTagMatcher (POSTagEmbedder),
    LCSMatcher (LCSEmbedder),
    NGramMatcher (NGramEmbedder),
    LevenshteinMatcher(LevenshteinEmbedder)
}

fn get_id<T: Clone + Eq + ::std::hash::Hash>(w: &T, id2e: &mut Vec<T>, e2id: &mut HashMap<T, usize>) -> usize {
    if let Some(i) = e2id.get(w) {
        return *i
    }
    
    let i = id2e.len();
    id2e.push(w.clone());
    e2id.insert(w.clone(), i);
    i
}

pub trait IsEmbedding {
    fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)>;
    fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)>;
    fn build_e_to_rules(&mut self, word_to_preterminal: &HashMap<String, Vec<(NT, f64)>>) {
        let mut e_id_to_rules: HashMap<_, Vec<(String, NT, f64)>> = HashMap::new();
        for (word, pret_vect) in word_to_preterminal {
            for &(nt, logprob) in pret_vect {
                let e_id = self.embed_rule(&(word, nt, logprob));
                e_id_to_rules.entry(e_id).or_insert_with(Vec::new).push((word.clone(), nt, logprob))
            }
        }
        *self.get_e_id_to_rules_mut() = e_id_to_rules.into_iter().collect()
    }
    
    fn comp(&self, erule: usize, esent: usize) -> f64;
    fn embed_rule(&mut self, rule: &(&str, NT, f64)) -> usize;
    fn embed_sent(&mut self, word: &str, posdesc: &Vec<(POSTag, f64)>) -> usize;
}

pub struct ExactEmbedder {
    e_id_to_rules: Vec<(usize, Vec<(String, NT, f64)>)>,
    id2e: Vec<String>,
    e2id: HashMap<String, usize>
}
impl IsEmbedding for ExactEmbedder {
    fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)> {&self.e_id_to_rules}
    fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)> {&mut self.e_id_to_rules}
    
    fn comp(&self, erule: usize, esent: usize) -> f64 {
        if self.id2e[erule] == self.id2e[esent] {1.0} else {0.0}
    }
    fn embed_rule(&mut self, rule: &(&str, NT, f64)) -> usize {
        let &(w, _, _) = rule;
        get_id(&w.to_string(), &mut self.id2e, &mut self.e2id)
    }
    fn embed_sent(&mut self, w: &str, _: &Vec<(POSTag, f64)>) -> usize {
        get_id(&w.to_string(), &mut self.id2e, &mut self.e2id)
    }
}

pub struct POSTagEmbedder {
    e_id_to_rules: Vec<(usize, Vec<(String, NT, f64)>)>,
    id2e: Vec<(POSTag, BTreeMap<POSTag, LogProb>)>,
    e2id: HashMap<(POSTag, BTreeMap<POSTag, LogProb>), usize>,
    
    bin_ntdict: HashMap<NT, String>,
    nbesttags: bool
}
impl IsEmbedding for POSTagEmbedder {
    fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)> {&self.e_id_to_rules}
    fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)> {&mut self.e_id_to_rules}
    
    fn comp(&self, erule: usize, esent: usize) -> f64 {
        if self.nbesttags {
            let &LogProb(lp) = self.id2e[esent].1.get(&self.id2e[erule].0).unwrap_or(&LogProb(::std::f64::NEG_INFINITY));
            lp.exp()
        } else {
            if self.id2e[erule].0 == self.id2e[esent].0 {1.0} else {0.0}
        }
    }
    fn embed_rule(&mut self, rule: &(&str, NT, f64)) -> usize {
        let &(_, nt, _) = rule;
        let tag = self.bin_ntdict.get(&nt).unwrap().clone();
        let mut cpd: BTreeMap<String, LogProb> = BTreeMap::new();
        cpd.insert(tag.clone(), LogProb(0.0));
        let e = (tag, cpd);
        get_id(&e, &mut self.id2e, &mut self.e2id)
    }
    fn embed_sent(&mut self, _: &str, posdesc: &Vec<(POSTag, f64)>) -> usize {
        let mut max_lp = ::std::f64::NEG_INFINITY;
        let mut max_pos: &str = "";
        let mut btm: BTreeMap<String, LogProb> = BTreeMap::new();
        for &(ref p, lp) in posdesc {
            if lp >= max_lp {
                max_pos = p;
                max_lp = lp
            }
            btm.insert(p.to_string(), LogProb(lp));
        }
        assert_eq!(max_lp, 0.0);
        let e = (max_pos.to_string(), btm);
        get_id(&e, &mut self.id2e, &mut self.e2id)
    }
}

pub struct LCSEmbedder {
    e_id_to_rules: Vec<(usize, Vec<(String, NT, f64)>)>,
    id2e: Vec<Vec<char>>,
    e2id: HashMap<Vec<char>, usize>,
    
    alpha: f64
}
impl LCSEmbedder {
    fn lcs_dyn_prog<T: Eq>(a: &[T], b: &[T]) -> usize {
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
        
        max
    }
}
impl IsEmbedding for LCSEmbedder {
    fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)> {&self.e_id_to_rules}
    fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)> {&mut self.e_id_to_rules}
    
    fn comp(&self, erule: usize, esent: usize) -> f64 {
        (LCSEmbedder::lcs_dyn_prog(self.id2e[erule].as_slice(), self.id2e[esent].as_slice()) as f64)
        /
        (self.alpha * (self.id2e[erule].len() as f64)
            + (1.0-self.alpha) * (self.id2e[esent].len() as f64))
    }
    fn embed_rule(&mut self, rule: &(&str, NT, f64)) -> usize {
        let &(w, _, _) = rule;
        let e = w.chars().collect();
        get_id(&e, &mut self.id2e, &mut self.e2id)
    }
    fn embed_sent(&mut self, w: &str, _: &Vec<(POSTag, f64)>) -> usize {
        let e = w.chars().collect();
        get_id(&e, &mut self.id2e, &mut self.e2id)
    }
}

pub struct NGramEmbedder {
    e_id_to_rules: Vec<(usize, Vec<(String, NT, f64)>)>,
    id2e: Vec<BTreeSet<String>>,
    e2id: HashMap<BTreeSet<String>, usize>,
    
    kappa: usize,
    dualmono_pad: bool,
    
    ngram_cache: HashMap<String, usize>
}
impl NGramEmbedder {
    fn get_ngrams_index(&mut self, word: &str) -> usize {
        if let Some(i) = self.ngram_cache.get(word) {
            return *i
        }
        
        fn getgrams(s: String, kappa: usize) -> BTreeSet<String> {
            s
                .chars()
                .collect::<Vec<_>>()
                .windows(kappa)
                .map(|w| w.iter().cloned().collect())
                .collect()
        }
        
        let sharpstring = "#".repeat(self.kappa - 1);
        
        let r: BTreeSet<String> = if self.dualmono_pad {
            let r1: BTreeSet<String> = getgrams(sharpstring.clone() + word, self.kappa);
            let r2: BTreeSet<String> = getgrams(word.to_string() + &sharpstring, self.kappa);
            
            r1.union(&r2).cloned().collect()
        } else {
            getgrams(sharpstring.clone() + word + &sharpstring, self.kappa)
        };
        
        let i = get_id(&r, &mut self.id2e, &mut self.e2id);
        
        self.ngram_cache.insert(word.to_string(), i);
        
        i
    }
}
impl IsEmbedding for NGramEmbedder {
    fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)> {&self.e_id_to_rules}
    fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)> {&mut self.e_id_to_rules}
    
    fn comp(&self, erule: usize, esent: usize) -> f64 {
        let inter = self.id2e[esent].intersection(&self.id2e[erule]).count() as f64;
        let sum = (self.id2e[esent].len() + self.id2e[erule].len()) as f64;
        2.0 * inter / sum // dice
    }
    fn embed_rule(&mut self, rule: &(&str, NT, f64)) -> usize {
        let &(w, _, _) = rule;
        self.get_ngrams_index(w)
    }
    fn embed_sent(&mut self, w: &str, _: &Vec<(POSTag, f64)>) -> usize {
        self.get_ngrams_index(w)
    }
}

pub struct LevenshteinEmbedder {
    e_id_to_rules: Vec<(usize, Vec<(String, NT, f64)>)>,
    id2e: Vec<(String, usize)>,
    e2id: HashMap<(String, usize), usize>
}
impl IsEmbedding for LevenshteinEmbedder {
    fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)> {&self.e_id_to_rules}
    fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)> {&mut self.e_id_to_rules}
    
    fn comp(&self, erule: usize, esent: usize) -> f64 {
        let (ref wrule, lrule) = self.id2e[erule];
        let (ref wsent, lsent) = self.id2e[esent];
        (1.0 -
            (strsim::levenshtein(wsent, wrule) as f64)
            /
            (::std::cmp::max(lrule, lsent) as f64)
        )
    }
    fn embed_rule(&mut self, rule: &(&str, NT, f64)) -> usize {
        let &(w, _, _) = rule;
        let e = (w.to_string(), w.chars().count());
        get_id(&e, &mut self.id2e, &mut self.e2id)
    }
    fn embed_sent(&mut self, w: &str, _: &Vec<(POSTag, f64)>) -> usize {
        let e = (w.to_string(), w.chars().count());
        get_id(&e, &mut self.id2e, &mut self.e2id)
    }
}

pub fn embed_rules(
    word_to_preterminal: &HashMap<String, Vec<(NT, f64)>>,
    bin_ntdict: &HashMap<NT, String>,
    stats: &PCFGParsingStatistics)
    -> TerminalMatcher {
    
    match &*stats.feature_structures {
        "exactmatch" => {
            let mut embdr = ExactEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new() };
            embdr.build_e_to_rules(word_to_preterminal);
            TerminalMatcher::ExactMatcher(embdr)
        },
        "postagsonly" => {
            let mut embdr = POSTagEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), bin_ntdict: bin_ntdict.clone(), nbesttags: stats.nbesttags };
            embdr.build_e_to_rules(word_to_preterminal);
            TerminalMatcher::POSTagMatcher(embdr)
        },
        "lcsratio" => {
            let mut embdr = LCSEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), alpha: stats.alpha };
            embdr.build_e_to_rules(word_to_preterminal);
            TerminalMatcher::LCSMatcher(embdr)
        },
        "ngrams" => {
            let mut embdr = NGramEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), kappa: stats.kappa, dualmono_pad: stats.dualmono_pad, ngram_cache: HashMap::new() };
            embdr.build_e_to_rules(word_to_preterminal);
            TerminalMatcher::NGramMatcher(embdr)
        },
        "levenshtein" => {
            let mut embdr = LevenshteinEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new() };
            embdr.build_e_to_rules(word_to_preterminal);
            TerminalMatcher::LevenshteinMatcher(embdr)
        },
        _ => panic!("Incorrect feature structure / matching algorithm {} requested!", stats.feature_structures)
    }
}
