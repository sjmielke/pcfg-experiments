// ######################################### Float hashing ########################################
// (taken from https://github.com/reem/rust-ordered-float, MIT licensed)
// (updated w/ integer_decode_f64 from rust-num: https://github.com/rust-num/num/blob/338e4799e675118789bffa95860ccef8a3abba38/traits/src/float.rs)

use std::mem;
use std::hash::{Hash, Hasher};

// masks for the parts of the IEEE 754 float
const SIGN_MASK: u64 = 0x8000000000000000u64;
const EXP_MASK: u64 = 0x7ff0000000000000u64;
const MAN_MASK: u64 = 0x000fffffffffffffu64;

// canonical raw bit patterns (for hashing)
const CANONICAL_NAN_BITS: u64 = 0x7ff8000000000000u64;
const CANONICAL_ZERO_BITS: u64 = 0x0u64;

#[inline]
fn raw_double_bits(f: &f64) -> u64 {
    if f.is_nan() {
        return CANONICAL_NAN_BITS;
    }

    let bits: u64 = unsafe { mem::transmute(f) };
    let sign: i8 = if bits >> 63 == 0 {
        1
    } else {
        -1
    };
    let mut exponent: i16 = ((bits >> 52) & 0x7ff) as i16;
    let mantissa = if exponent == 0 {
        (bits & 0xfffffffffffff) << 1
    } else {
        (bits & 0xfffffffffffff) | 0x10000000000000
    };
    // Exponent bias + mantissa shift
    exponent -= 1023 + 52;
    let (man, exp, sign) = (mantissa, exponent, sign);
    if man == 0 {
        return CANONICAL_ZERO_BITS;
    }

    let exp_u64 = unsafe { mem::transmute::<i16, u16>(exp) } as u64;
    let sign_u64 = if sign > 0 { 1u64 } else { 0u64 };
    (man & MAN_MASK) | ((exp_u64 << 52) & EXP_MASK) | ((sign_u64 << 63) & SIGN_MASK)
}

// ######################################## Rules, trees ##########################################

#[derive(Clone)]
pub struct Probability(pub f64);
impl PartialEq for Probability {
    fn eq(&self, other: &Probability) -> bool {
        self.0 == other.0
    }
}
impl Eq for Probability {}
impl Hash for Probability {
    fn hash<H: Hasher>(&self, state: &mut H) {
        raw_double_bits(&self.0).hash(state);
    }
}

#[derive(Clone)]
pub struct LogProb(pub f64);
impl PartialEq for LogProb {
    fn eq(&self, other: &LogProb) -> bool {
        self.0 == other.0
    }
}
impl Eq for LogProb {}
impl Hash for LogProb {
    fn hash<H: Hasher>(&self, state: &mut H) {
        raw_double_bits(&self.0).hash(state);
    }
}

pub type NT = usize;
pub type POSTag = String;

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
pub enum RHS {
    Terminal(String),
    NTs(Vec<NT>),
    Binary(NT, NT),
    Unary(NT)
}

#[derive(Debug)]
pub struct Rule {
    pub lhs: NT,
    pub rhs: RHS
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
pub enum ParseTree<'a> {
    InnerNode {
        label: NT,
        children: Vec<ParseTree<'a>>
    },
    TerminalNode {
        label: &'a str
    }
}
impl<'a> ParseTree<'a> {
    #[allow(dead_code)]
    pub fn render(&self) -> String {
        match *self {
            ParseTree::TerminalNode {label} => label.to_string(),
            ParseTree::InnerNode {label, ref children} => {
                label.to_string() + "(" + &children.iter().map(|c| c.render()).collect::<Vec<_>>().join(",") + ")"
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, PartialOrd, Ord, Clone, Hash)]
pub enum Morph {
    PRE,
    STM,
    SUF
}
impl ::std::fmt::Display for Morph {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            Morph::PRE => write!(f, "PRE"),
            Morph::STM => write!(f, "STM"),
            Morph::SUF => write!(f, "SUF")
        }
    }
}
impl ::std::str::FromStr for Morph {
    type Err = String;
    
    fn from_str(s: &str) -> Result<Morph, String> {
        match s {
            "PRE" => Ok(Morph::PRE),
            "STM" => Ok(Morph::STM),
            "SUF" => Ok(Morph::SUF),
            _ => Err(format!("Invalid morph type »{}«", s))
        }
    }
}


// ###################################### OOV handling flag #######################################

#[derive(Clone)]
pub enum OOVHandling {
    /// Fail on every OOV (standard, fastest)
    Zero,
    /// Accept every POS tag with `1.0 * oovuniformprob` "probability"
    Uniform,
    /// Accept every POS tag with `(c_POS/sum_POS' c_POS') * oovuniformprob` "probability"
    MarginalAll,
    /// Accept every POS tag with `(c_rare_POS/sum_rare_POS' c_rare_POS') * oovuniformprob` "probability" (where rare is defined as belonging to a word that appears less or equal to 3 times)
    MarginalLe3
}
impl ::std::fmt::Display for OOVHandling {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            OOVHandling::Zero => write!(f, "zero"),
            OOVHandling::Uniform => write!(f, "uniform"),
            OOVHandling::MarginalAll => write!(f, "marginal-all"),
            OOVHandling::MarginalLe3 => write!(f, "marginal-le3")
        }
    }
}
impl ::std::str::FromStr for OOVHandling {
    type Err = String;
    
    fn from_str(s: &str) -> Result<OOVHandling, String> {
        match s {
            "zero" => Ok(OOVHandling::Zero),
            "uniform" => Ok(OOVHandling::Uniform),
            "marginal-all" => Ok(OOVHandling::MarginalAll),
            "marginal-le3" => Ok(OOVHandling::MarginalLe3),
            _ => Err(format!("Invalid OOV handling parameter »{}«", s))
        }
    }
}

// #################################### All flags / statistics ####################################

#[derive(Clone)]
pub struct PCFGParsingStatistics {
    pub language: String,
    // Raw numbers
    pub trainsize: usize,
    pub oov_handling: OOVHandling,
    pub all_terms_fallback: bool,
    pub only_oovs_soft: bool,
    pub exhaustive: bool,
    pub uniform_oov_prob: f64,
    pub feature_structures: String,
    pub testtagsfile: String,
    pub morftagfileprefix: String,
    pub nbesttags: String,
    pub dualmono_pad: bool,
    pub logcompvalues: bool,
    pub keepafterdash: bool,
    pub eta: f64,
    pub alpha: f64,
    pub beta: f64,
    pub kappa: usize,
    pub omega: f64,
    pub tau: f64,
    pub mu: f64,
    pub chi: f64,
    // Raw numbers
    pub unbin_nts: usize,
    pub bin_nts: usize,
    // Times in seconds
    pub gram_ext_bin: f64,
    pub cky_prep: f64,
    pub cky_terms: f64,
    pub cky_higher: f64,
    // Raw numbers
    pub oov_words: usize,
    pub oov_sents: usize,
    pub parsefails: usize,
    // Percent
    pub fmeasure: f64,
    pub or_fail_fmeasure: f64,
    pub tagaccuracy: f64
}
impl PCFGParsingStatistics {
    pub fn print(&self, head: bool) {
        if head {
            println!("language\ttrainsize\tunbin_nts\tbin_nts\toov_handling\tuniform_oov_prob\tfeature_structures\ttesttagsfile\tmorftagfileprefix\tnbesttags\tdualmono_pad\tlogcompvalues\tkeepafterdash\teta\talpha\tbeta\tkappa\tomega\ttau\tmu\tchi\tall_terms_fallback\tonly_oovs_soft\texhaustive\tgram_ext_bin\tcky_prep\tcky_terms\tcky_higher\toov_words\toov_sents\tparsefails\tfmeasure\tfmeasure (fail ok)\ttagaccuracy");
        }
        print!("{}\t", self.language);
        print!("{}\t", self.trainsize);
        print!("{}\t", self.unbin_nts);
        print!("{}\t", self.bin_nts);
        print!("{}\t", self.oov_handling);
        print!("{}\t", self.uniform_oov_prob);
        print!("{}\t", self.feature_structures);
        print!("{}\t", self.testtagsfile);
        print!("{}\t", self.morftagfileprefix);
        print!("{}\t", self.nbesttags);
        print!("{}\t", if self.dualmono_pad {"dualmonopad"} else {"fullpad"});
        print!("{}\t", if self.logcompvalues {"logcompvalues"} else {"nocompvallog"});
        print!("{}\t", if self.keepafterdash {"keepafterdash"} else {"noafterdash"});
        print!("{}\t", self.eta);
        print!("{}\t", self.alpha);
        print!("{}\t", self.beta);
        print!("{}\t", self.kappa);
        print!("{}\t", self.omega);
        print!("{}\t", self.tau);
        print!("{}\t", self.mu);
        print!("{}\t", self.chi);
        print!("{}\t", if self.all_terms_fallback {"all_terms_fallback"} else {"no_fallback"});
        print!("{}\t", if self.only_oovs_soft {"onlyoovssoft"} else {"allsoft"});
        print!("{}\t", if self.exhaustive {"exhaustive"} else {"stop_on_first_goal"});
        print!("{:.3}\t", self.gram_ext_bin);
        print!("{:.3}\t", self.cky_prep);
        print!("{:.3}\t", self.cky_terms);
        print!("{:.3}\t", self.cky_higher);
        print!("{}\t", self.oov_words);
        print!("{}\t", self.oov_sents);
        print!("{}\t", self.parsefails);
        print!("{}\t", self.fmeasure);
        print!("{}\t", self.or_fail_fmeasure);
        println!("{}", self.tagaccuracy);
    }
}

// ############################################ TIMING ############################################

// taken from https://github.com/dikaiosune/rust-runtime-benchmarks/blob/master/bench-suite-linux/src/bencher.rs
extern crate libc;
use self::libc::{c_long, rusage, suseconds_t, timeval, time_t, getrusage, RUSAGE_SELF};
/// Returns time elapsed in userspace in seconds.
pub fn get_usertime() -> f64 {
    let mut usage = rusage {
        ru_utime: timeval{ tv_sec: 0 as time_t, tv_usec: 0 as suseconds_t, },
        ru_stime: timeval{ tv_sec: 0 as time_t, tv_usec: 0 as suseconds_t, },
        ru_maxrss: 0 as c_long,
        ru_ixrss: 0 as c_long,
        ru_idrss: 0 as c_long,
        ru_isrss: 0 as c_long,
        ru_minflt: 0 as c_long,
        ru_majflt: 0 as c_long,
        ru_nswap: 0 as c_long,
        ru_inblock: 0 as c_long,
        ru_oublock: 0 as c_long,
        ru_msgsnd: 0 as c_long,
        ru_msgrcv: 0 as c_long,
        ru_nsignals: 0 as c_long,
        ru_nvcsw: 0 as c_long,
        ru_nivcsw: 0 as c_long,
    };
    unsafe { getrusage(RUSAGE_SELF, (&mut usage) as *mut rusage); }
    let secs = usage.ru_utime.tv_sec as usize;
    let usecs = usage.ru_utime.tv_usec as usize;
    (secs * 1_000_000 + usecs) as f64 / 1000000.0
}
