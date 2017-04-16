use std::collections::HashMap;

use defs::*;

type POSTag = String;

#[derive(Debug, Clone)]
pub enum TerminalMatcher {
    POSTagMatcher (HashMap<POSTag, Vec<(String, NT, f64)>>),
    ExactMatchOnly
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
