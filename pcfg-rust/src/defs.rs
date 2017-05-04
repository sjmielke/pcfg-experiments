// ######################################## Rules, trees ##########################################

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

// ###################################### OOV handling flag #######################################

#[derive(Clone)]
pub enum OOVHandling {
    /// Fail on every OOV (standard, fastest)
    Zero,
    /// Accept every POS tag with e^{-300} probability
    Uniform,
    /// Accept every POS tag with (c_POS/sum_POS' c_POS') * e^{-300} probability
    Marginal
}
impl ::std::fmt::Display for OOVHandling {
    fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match *self {
            OOVHandling::Zero => write!(f, "zero"),
            OOVHandling::Uniform => write!(f, "uniform"),
            OOVHandling::Marginal => write!(f, "marginal")
        }
    }
}
impl ::std::str::FromStr for OOVHandling {
    type Err = String;
    
    fn from_str(s: &str) -> Result<OOVHandling, String> {
        match s {
            "zero" => Ok(OOVHandling::Zero),
            "uniform" => Ok(OOVHandling::Uniform),
            "marginal" => Ok(OOVHandling::Marginal),
            _ => Err(format!("Invalid OOV handling parameter »{}«", s))
        }
    }
}

// #################################### All flags / statistics ####################################

#[derive(Clone)]
pub struct PCFGParsingStatistics {
    // Raw numbers
    pub trainsize: usize,
    pub testsize: usize,
    pub testmaxlen: usize,
    pub oov_handling: OOVHandling,
    pub all_terms_fallback: bool,
    pub exhaustive: bool,
    pub uniform_oov_prob: f64,
    pub feature_structures: String,
    pub testtagsfile: String,
    pub nbesttags: bool,
    pub eta: f64,
    pub alpha: f64,
    pub beta: f64,
    pub kappa: usize,
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
    pub or_fail_fmeasure: f64
}
impl PCFGParsingStatistics {
    pub fn print(&self, head: bool) {
        if head {
            println!("trainsize\ttestsize\ttestmaxlen\tunbin_nts\tbin_nts\toov_handling\tuniform_oov_prob\tfeature_structures\ttesttagsfile\tnbesttags\teta\talpha\tbeta\tkappa\tall_terms_fallback\texhaustive\tgram_ext_bin\tcky_prep\tcky_terms\tcky_higher\toov_words\toov_sents\tparsefails\tfmeasure\tfmeasure (fail ok)");
        }
        print!("{}\t", self.trainsize);
        print!("{}\t", self.testsize);
        print!("{}\t", self.testmaxlen);
        print!("{}\t", self.unbin_nts);
        print!("{}\t", self.bin_nts);
        print!("{}\t", self.oov_handling);
        print!("{}\t", self.uniform_oov_prob);
        print!("{}\t", self.feature_structures);
        print!("{}\t", self.testtagsfile);
        print!("{}\t", if self.nbesttags {"nbesttags"} else {"1besttags"});
        print!("{}\t", self.eta);
        print!("{}\t", self.alpha);
        print!("{}\t", self.beta);
        print!("{}\t", self.kappa);
        print!("{}\t", if self.all_terms_fallback {"all_terms_fallback"} else {"no_fallback"});
        print!("{}\t", if self.exhaustive {"exhaustive"} else {"stop_on_first_goal"});
        print!("{:.3}\t", self.gram_ext_bin);
        print!("{:.3}\t", self.cky_prep);
        print!("{:.3}\t", self.cky_terms);
        print!("{:.3}\t", self.cky_higher);
        print!("{}\t", self.oov_words);
        print!("{}\t", self.oov_sents);
        print!("{}\t", self.parsefails);
        print!("{}\t", self.fmeasure);
        println!("{}", self.or_fail_fmeasure);
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
