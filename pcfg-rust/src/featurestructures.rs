use std::collections::{HashMap, BTreeMap, BTreeSet};

use defs::*;
extern crate strsim;

pub enum TerminalMatcher {
    ExactMatcher (ExactEmbedder),
    POSTagMatcher (POSTagEmbedder),
    LCSMatcher (LCSEmbedder),
    PrefixSuffixMatcher (PrefixSuffixEmbedder),
    LevenshteinMatcher(LevenshteinEmbedder),
    NGramMatcher (NGramEmbedder),
    ContinuousFrequencyMatcher (ContinuousFrequencyEmbedder),
    AffixDiceMatcher (AffixDiceEmbedder),
    JointMatcher(JointEmbedder)
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
    word2tag: Option<HashMap<String, String>>,
    faux_nbesttags: bool,
    nbesttags: bool,
    mu: f64
}
impl IsEmbedding for POSTagEmbedder {
    fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)> {&self.e_id_to_rules}
    fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)> {&mut self.e_id_to_rules}
    
    fn comp(&self, erule: usize, esent: usize) -> f64 {
        if self.nbesttags {
            let &LogProb(lp) = self.id2e[esent].1.get(&self.id2e[erule].0).unwrap_or(&LogProb(::std::f64::NEG_INFINITY));
            lp.exp()
        } else if self.faux_nbesttags {
            if self.id2e[erule].0 == self.id2e[esent].0 {self.mu} else {(1.0 - self.mu) / ((self.e_id_to_rules.len() - 1) as f64)}
        } else {
            if self.id2e[erule].0 == self.id2e[esent].0 {1.0} else {0.0}
        }
    }
    fn embed_rule(&mut self, rule: &(&str, NT, f64)) -> usize {
        let &(w, nt, _) = rule;
        let tag = match self.word2tag {
            None => self.bin_ntdict.get(&nt).unwrap().clone(),
            Some(ref w2t) => w2t[w].clone()
        };
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
        // only if gold tags // assert_eq!(max_lp, 0.0);
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
        let lrule: f64 = self.id2e[erule].len() as f64;
        let lsent: f64 = self.id2e[esent].len() as f64;
        let lcslen: f64 = LCSEmbedder::lcs_dyn_prog(self.id2e[erule].as_slice(), self.id2e[esent].as_slice()) as f64;
        let w_avg: f64 = self.alpha * lrule + (1.0-self.alpha) * lsent;
        // lcslen <= min{lrule, lsent}
        // min{lrule, lsent} <= w_avg{lrule, lsent}
        lcslen / w_avg
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

pub struct PrefixSuffixEmbedder {
    e_id_to_rules: Vec<(usize, Vec<(String, NT, f64)>)>,
    id2e: Vec<Vec<char>>,
    e2id: HashMap<Vec<char>, usize>,
    
    alpha: f64,
    omega: f64,
    tau: f64
}
impl PrefixSuffixEmbedder {
    fn cp_len<T: Eq + ::std::fmt::Display>(a: &[T], b: &[T]) -> usize {
        let al = a.len();
        let bl = b.len();
        let mut l = 0;
        while l < al && l < bl && a[l] == b[l] {
            l += 1
        }
        return l
    }
    fn cs_len<T: Eq + ::std::fmt::Display>(a: &[T], b: &[T]) -> usize {
        let al = a.len();
        let bl = b.len();
        let mut l = 0;
        while l < al && l < bl && a[al-l-1] == b[bl-l-1] {
            l += 1
        }
        return l
    }
    #[inline]
    fn geom_upto_i(&self, k: i32) -> f64 {
        (1.0 - self.tau.powi(k)) / (1.0 - self.tau)
    }
    #[inline]
    fn geom_upto_f(&self, k: f64) -> f64 {
        (1.0 - self.tau.powf(k)) / (1.0 - self.tau)
    }
}
impl IsEmbedding for PrefixSuffixEmbedder {
    fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)> {&self.e_id_to_rules}
    fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)> {&mut self.e_id_to_rules}
    
    fn comp(&self, erule: usize, esent: usize) -> f64 {
        let lrule: f64 = self.id2e[erule].len() as f64;
        let lsent: f64 = self.id2e[esent].len() as f64;
        let cplen: i32 = PrefixSuffixEmbedder::cp_len(self.id2e[erule].as_slice(), self.id2e[esent].as_slice()) as i32;
        let cslen: i32 = PrefixSuffixEmbedder::cs_len(self.id2e[erule].as_slice(), self.id2e[esent].as_slice()) as i32;
        let w_avg: f64 = self.alpha * lrule + (1.0-self.alpha) * lsent;
        
        let (cp, cs, dn) = if self.tau == 1.0 {
            (cplen as f64, cslen as f64, w_avg)
        } else {
            (self.geom_upto_i(cplen), self.geom_upto_i(cslen), self.geom_upto_f(w_avg))
        };
        
        ((1.0 - self.omega) * cp + self.omega * cs) / dn
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

pub struct LevenshteinEmbedder {
    e_id_to_rules: Vec<(usize, Vec<(String, NT, f64)>)>,
    id2e: Vec<(String, usize)>,
    e2id: HashMap<(String, usize), usize>,
    
    alpha: f64
}
impl IsEmbedding for LevenshteinEmbedder {
    fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)> {&self.e_id_to_rules}
    fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)> {&mut self.e_id_to_rules}
    
    fn comp(&self, erule: usize, esent: usize) -> f64 {
        let (ref wrule, lrule) = self.id2e[erule];
        let (ref wsent, lsent) = self.id2e[esent];
        (1.0 -
            (
                (strsim::levenshtein(wsent, wrule) as f64)
                /
                (self.alpha * (lrule as f64) + (1.0 - self.alpha) * (lsent as f64))
            ) *
            (
                (::std::cmp::min(lrule, lsent) as f64)
                /
                (::std::cmp::max(lrule, lsent) as f64)
            )
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

pub struct ContinuousFrequencyEmbedder {
    e_id_to_rules: Vec<(usize, Vec<(String, NT, f64)>)>,
    id2e: Vec<LogProb>,
    e2id: HashMap<LogProb, usize>,
    
    word2logfreq: HashMap<String, f64>,
    unklogfreq: f64,
    norm_factor: f64
}
impl IsEmbedding for ContinuousFrequencyEmbedder {
    fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)> {&self.e_id_to_rules}
    fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)> {&mut self.e_id_to_rules}
    
    fn comp(&self, erule: usize, esent: usize) -> f64 {
        let diff: f64 = self.id2e[esent].0 - self.id2e[erule].0;
        1.0 - (diff * diff) / self.norm_factor
    }
    fn embed_rule(&mut self, rule: &(&str, NT, f64)) -> usize {
        let &(s, _, _) = rule;
        let e: &f64 =self.word2logfreq.get(s).unwrap_or(&self.unklogfreq);
        get_id(&LogProb(*e), &mut self.id2e, &mut self.e2id)
    }
    fn embed_sent(&mut self, s: &str, _: &Vec<(POSTag, f64)>) -> usize {
        let e: &f64 = self.word2logfreq.get(s).unwrap_or(&self.unklogfreq);
        get_id(&LogProb(*e), &mut self.id2e, &mut self.e2id)
    }
}

pub struct AffixDiceEmbedder {
    e_id_to_rules: Vec<(usize, Vec<(String, NT, f64)>)>,
    id2e: Vec<(BTreeSet<(String, Morph)>, Probability)>,
    e2id: HashMap<(BTreeSet<(String, Morph)>, Probability), usize>,
    
    morfdict: HashMap<String, (BTreeSet<(String, Morph)>, Probability)>,
    chi: f64
}
impl IsEmbedding for AffixDiceEmbedder {
    fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)> {&self.e_id_to_rules}
    fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)> {&mut self.e_id_to_rules}
    
    fn comp(&self, erule: usize, esent: usize) -> f64 {
        let &(ref set1, Probability(sum1)) = &self.id2e[esent];
        let &(ref set2, Probability(sum2)) = &self.id2e[erule];
        
        let inter_vec: Vec<&(String, Morph)> = set1.intersection(&set2).collect();
        let mut intersum: f64 = (inter_vec.iter().filter(|&&&(_, ref c)| *c == Morph::STM).count() as f64) * self.chi;
        intersum             += (inter_vec.iter().filter(|&&&(_, ref c)| *c != Morph::STM).count() as f64) * (1.0-self.chi) / 2.0;
        
        2.0 * intersum / (sum1 + sum2) // affixdice
    }
    fn embed_rule(&mut self, rule: &(&str, NT, f64)) -> usize {
        let &(s, _, _) = rule;
        let e = &self.morfdict[s];
        get_id(e, &mut self.id2e, &mut self.e2id)
    }
    fn embed_sent(&mut self, s: &str, _: &Vec<(POSTag, f64)>) -> usize {
        let e = &self.morfdict[s];
        get_id(e, &mut self.id2e, &mut self.e2id)
    }
}

// pub struct DummyEmbedder {
//     e_id_to_rules: Vec<(usize, Vec<(String, NT, f64)>)>,
//     id2e: Vec<XXXXXXXXXXX>,
//     e2id: HashMap<XXXXXXXXXXX, usize>
// }
// impl IsEmbedding for DummyEmbedder {
//     fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)> {&self.e_id_to_rules}
//     fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)> {&mut self.e_id_to_rules}
//     
//     fn comp(&self, erule: usize, esent: usize) -> f64 {
//         self.id2e[esent] == self.id2e[erule]
//     }
//     fn embed_rule(&mut self, rule: &(&str, NT, f64)) -> usize {
//         let &(s, _, _) = rule;
//         let e = s;
//         get_id(&e, &mut self.id2e, &mut self.e2id)
//     }
//     fn embed_sent(&mut self, s: &str, posdesc: &Vec<(POSTag, f64)>) -> usize {
//         let e = s;
//         get_id(&e, &mut self.id2e, &mut self.e2id)
//     }
// }


pub struct JointEmbedder {
    e_id_to_rules: Vec<(usize, Vec<(String, NT, f64)>)>,
    id2e: Vec<Vec<usize>>,
    e2id: HashMap<Vec<usize>, usize>,
    
    embedders: Vec<Box<IsEmbedding>>,
    betas: Vec<f64>
}
impl IsEmbedding for JointEmbedder {
    fn get_e_id_to_rules(&self) -> &Vec<(usize, Vec<(String, NT, f64)>)> {&self.e_id_to_rules}
    fn get_e_id_to_rules_mut(&mut self) -> &mut Vec<(usize, Vec<(String, NT, f64)>)> {&mut self.e_id_to_rules}
    
    fn comp(&self, erule: usize, esent: usize) -> f64 {
        let mut comp: f64 = 1.0;
        let pairs = self.id2e[esent].iter().zip(&self.id2e[erule]);
        for ((emb, (isent, irule)), beta) in self.embedders.iter().zip(pairs).zip(&self.betas) {
            comp *= emb.comp(*isent, *irule).powf(*beta)
        }
        comp
    }
    fn embed_rule(&mut self, rule: &(&str, NT, f64)) -> usize {
        let mut ids: Vec<usize> = Vec::new();
        for ref mut emb in &mut self.embedders {
            ids.push(emb.embed_rule(rule))
        }
        get_id(&ids, &mut self.id2e, &mut self.e2id)
    }
    fn embed_sent(&mut self, s: &str, posdesc: &Vec<(POSTag, f64)>) -> usize {
        let mut ids: Vec<usize> = Vec::new();
        for ref mut emb in &mut self.embedders {
            ids.push(emb.embed_sent(s, posdesc))
        }
        get_id(&ids, &mut self.id2e, &mut self.e2id)
    }
}


pub fn embed_rules(
    word_to_preterminal: &HashMap<String, Vec<(NT, f64)>>,
    bin_ntdict: &HashMap<NT, String>,
    wordcounts: (HashMap<String, f64>, f64),
    word2tag: Option<HashMap<String, String>>,
    morfdict: HashMap<String, BTreeSet<(String, Morph)>>,
    stats: &PCFGParsingStatistics)
    -> TerminalMatcher {
    
    fn get_single_embedder(
        embedding_space: &str,
        word_to_preterminal: &HashMap<String, Vec<(NT, f64)>>,
        bin_ntdict: &HashMap<NT, String>,
        wordcounts: (HashMap<String, f64>, f64),
        word2tag: Option<HashMap<String, String>>,
        morfdict: HashMap<String, BTreeSet<(String, Morph)>>,
        stats: &PCFGParsingStatistics) -> TerminalMatcher {
        match embedding_space {
            "exactmatch" => {
                let mut embdr = ExactEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new() };
                embdr.build_e_to_rules(word_to_preterminal);
                TerminalMatcher::ExactMatcher(embdr)
            },
            "postagsonly" => {
                assert!(stats.nbesttags == "nbesttags" || stats.nbesttags == "faux-nbesttags" || stats.nbesttags == "1besttags");
                let mut embdr = POSTagEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), bin_ntdict: bin_ntdict.clone(), nbesttags: stats.nbesttags == "nbesttags", faux_nbesttags: stats.nbesttags == "faux-nbesttags", mu: stats.mu, word2tag: word2tag };
                embdr.build_e_to_rules(word_to_preterminal);
                TerminalMatcher::POSTagMatcher(embdr)
            },
            "lcsratio" => {
                let mut embdr = LCSEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), alpha: stats.alpha };
                embdr.build_e_to_rules(word_to_preterminal);
                TerminalMatcher::LCSMatcher(embdr)
            },
            "prefixsuffix" => {
                let mut embdr = PrefixSuffixEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), alpha: stats.alpha, omega: stats.omega, tau: stats.tau };
                embdr.build_e_to_rules(word_to_preterminal);
                TerminalMatcher::PrefixSuffixMatcher(embdr)
            },
            "levenshtein" => {
                let mut embdr = LevenshteinEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), alpha: stats.alpha };
                embdr.build_e_to_rules(word_to_preterminal);
                TerminalMatcher::LevenshteinMatcher(embdr)
            },
            "ngrams" => {
                let mut embdr = NGramEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), kappa: stats.kappa, dualmono_pad: stats.dualmono_pad, ngram_cache: HashMap::new() };
                embdr.build_e_to_rules(word_to_preterminal);
                TerminalMatcher::NGramMatcher(embdr)
            },
            "freq-cont" => {
                let (word2logfreq, logtotal) = wordcounts;
                let mut embdr = ContinuousFrequencyEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), word2logfreq: word2logfreq, unklogfreq: 0.0, norm_factor: logtotal * logtotal };
                embdr.build_e_to_rules(word_to_preterminal);
                TerminalMatcher::ContinuousFrequencyMatcher(embdr)
            },
            "affixdice" => {
                let mut morfdict_w_sum: HashMap<String, (BTreeSet<(String, Morph)>, Probability)> = HashMap::new();
                for (w, set) in morfdict.into_iter() {
                    let mut weightedsum: f64 = (set.iter().filter(|&&(_, ref c)| *c == Morph::STM).count() as f64) * stats.chi;
                    weightedsum             += (set.iter().filter(|&&(_, ref c)| *c != Morph::STM).count() as f64) * (1.0-stats.chi) / 2.0;
                    morfdict_w_sum.insert(w, (set, Probability(weightedsum)));
                }
                
                let mut embdr = AffixDiceEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), morfdict: morfdict_w_sum, chi: stats.chi };
                embdr.build_e_to_rules(word_to_preterminal);
                TerminalMatcher::AffixDiceMatcher(embdr)
            }
            _ => panic!("Incorrect feature structure / matching algorithm {} requested!", stats.feature_structures)
        }
    }
    
    if stats.feature_structures.contains('+') {
        let mut ngrams_embdr = NGramEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), kappa: stats.kappa, dualmono_pad: stats.dualmono_pad, ngram_cache: HashMap::new() };
        ngrams_embdr.build_e_to_rules(word_to_preterminal);
        
        let (word2logfreq, logtotal) = wordcounts;
        let mut freq_embdr = ContinuousFrequencyEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), word2logfreq: word2logfreq, unklogfreq: 0.0, norm_factor: logtotal * logtotal };
        freq_embdr.build_e_to_rules(word_to_preterminal);
        
        let mut embdr = JointEmbedder { e_id_to_rules: Vec::new(), e2id: HashMap::new(), id2e: Vec::new(), embedders: vec![Box::new(ngrams_embdr), Box::new(freq_embdr)], betas: vec![10.0, 80.0] };
        embdr.build_e_to_rules(word_to_preterminal);
        
        TerminalMatcher::JointMatcher(embdr)
    } else {
        get_single_embedder(&stats.feature_structures, word_to_preterminal, bin_ntdict, wordcounts, word2tag, morfdict, stats)
    }
}
