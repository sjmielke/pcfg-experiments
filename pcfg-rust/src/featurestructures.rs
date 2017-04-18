use std::collections::HashMap;

use defs::*;

type POSTag = String;

#[derive(Debug, Clone)]
pub enum TerminalMatcher {
    POSTagMatcher (HashMap<POSTag, Vec<(String, NT, f64)>>),
    LCSRatioMatcher (f64, f64), // alpha, beta
    ExactMatchOnly
}

pub fn lcs_dyn_prog<T: Eq>(a: &[T], b: &[T]) -> usize {
    let mut table: Vec<Vec<usize>> = Vec::new();
    // Fill with 0s in first row/col
    // table_ai=0, table_bi ∈ {0..b.len()}
    let mut v = Vec::new();
    for bi in 0..b.len() + 1 {
        v.push(0)
    }
    table.push(v);
    // table_bi=0, table)ai ∈ {1..a.len()} (already did ai=0)
    for ai in 1..a.len() + 1 {
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

// Returns a HashMap: F -> P(R), i.e. gives all rules deriving a certain feature structure.
// This way terminals are only compared with the number of unique feature structures
// and not with every rule itself.
pub fn embed_rules(
    word_to_preterminal: &HashMap<String, Vec<(NT, f64)>>,
    bin_ntdict: &HashMap<NT, String>,
    stats: &PCFGParsingStatistics)
    -> TerminalMatcher {
    
    match &*stats.feature_structures {
        "exactmatch" => {
            TerminalMatcher::ExactMatchOnly
        },
        "postagsonly" => {
            TerminalMatcher::POSTagMatcher(embed_rules_pos(word_to_preterminal, bin_ntdict))
        }
        "lcsratio" => {
            TerminalMatcher::LCSRatioMatcher(stats.alpha, stats.beta)
        }
        _ => {panic!("Incorrect feature structure / matching algorithm {} requested!", stats.feature_structures)}
    }
}

// Gives you the POS tag (original name, i.e., a String!)
fn embed_rules_pos(
    word_to_preterminal: &HashMap<String, Vec<(NT, f64)>>,
    bin_ntdict: &HashMap<NT, String>)
    -> HashMap<String, Vec<(String, NT, f64)>> {
    
    let mut result = HashMap::new();
    for (word, pret_vect) in word_to_preterminal {
        for &(nt, logprob) in pret_vect {
            result.entry(bin_ntdict.get(&nt).unwrap().clone()).or_insert_with(Vec::new).push((word.clone(), nt, logprob));
        }
    }
    return result;
}
