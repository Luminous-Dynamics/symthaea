// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Semantic Enrichment: Self-Learning, Compositional, Context-Aware Semantics
//!
//! This module implements six revolutionary improvements to the semantic layer:
//!
//! 1. **Compositional Prime Inference** - Derive primes from morphology
//! 2. **Contextual Prime Activation** - Disambiguate based on surrounding words
//! 3. **Prime Coherence as Φ Signal** - Use disagreement as integration metric
//! 4. **Self-Learning Vocabulary** - Learn from successful interactions
//! 5. **Emergent Frame Induction** - Induce frames from patterns
//! 6. **Grounded Symbol Learning** - Connect words to system states
//!
//! ## Architecture
//!
//! ```text
//! Input Word
//!     │
//!     ├──► Morphological Analyzer ──► Base + Affixes
//!     │                                    │
//!     │         ┌──────────────────────────┘
//!     │         ▼
//!     ├──► Compositional Inference ──► Candidate Primes
//!     │                                    │
//!     │         ┌──────────────────────────┘
//!     │         ▼
//!     ├──► Context Disambiguator ──► Filtered Primes
//!     │                                    │
//!     │         ┌──────────────────────────┘
//!     │         ▼
//!     └──► Coherence Calculator ──► Φ Signal
//! ```

use std::collections::{HashMap, HashSet, VecDeque};
use symthaea_core::hdc::universal_semantics::SemanticPrime;

// ============================================================================
// MORPHOLOGICAL ANALYSIS
// ============================================================================

/// Result of morphological analysis
#[derive(Debug, Clone)]
pub struct MorphologicalAnalysis {
    /// The original word
    pub word: String,
    /// Detected prefix (if any)
    pub prefix: Option<MorphemePrefix>,
    /// The root/stem
    pub root: String,
    /// Detected suffix (if any)
    pub suffix: Option<MorphemeSuffix>,
    /// Confidence in the analysis
    pub confidence: f64,
}

/// Common English prefixes with semantic meaning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MorphemePrefix {
    /// "un-" = negation (undo, uninstall)
    Un,
    /// "re-" = again (redo, reinstall, rebuild)
    Re,
    /// "pre-" = before (preload, preconfigure)
    Pre,
    /// "post-" = after (postinstall)
    Post,
    /// "dis-" = negation/reversal (disable, disconnect)
    Dis,
    /// "mis-" = wrong (misconfigure)
    Mis,
    /// "over-" = excessive (override, overwrite)
    Over,
    /// "under-" = insufficient (underpowered)
    Under,
    /// "out-" = external/surpass (output, outperform)
    Out,
    /// "in-/im-" = into/not (input, import, impossible)
    In,
    /// "de-" = reversal (decompress, decrypt)
    De,
    /// "auto-" = self (automate, autocomplete)
    Auto,
    /// "multi-" = many (multithread)
    Multi,
    /// "sub-" = under/part (subprocess, submodule)
    Sub,
    /// "super-" = above (superuser)
    Super,
    /// "co-" = together (coprocess)
    Co,
}

/// Common English suffixes with semantic meaning
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MorphemeSuffix {
    /// "-ing" = ongoing action (running, installing)
    Ing,
    /// "-ed" = completed action (installed, configured)
    Ed,
    /// "-er" = agent/tool (installer, compiler)
    Er,
    /// "-or" = agent (executor, processor)
    Or,
    /// "-tion/-sion" = result/process (installation, compression)
    Tion,
    /// "-able/-ible" = capability (installable, accessible)
    Able,
    /// "-ment" = result (deployment, environment)
    Ment,
    /// "-ness" = state (readiness)
    Ness,
    /// "-ly" = manner (quickly, safely)
    Ly,
    /// "-ful" = full of (powerful, successful)
    Ful,
    /// "-less" = without (wireless, serverless)
    Less,
    /// "-ize/-ise" = make into (optimize, customize)
    Ize,
    /// "-ify" = make (simplify, verify)
    Ify,
}

impl MorphemePrefix {
    /// Get the string representation
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Un => "un",
            Self::Re => "re",
            Self::Pre => "pre",
            Self::Post => "post",
            Self::Dis => "dis",
            Self::Mis => "mis",
            Self::Over => "over",
            Self::Under => "under",
            Self::Out => "out",
            Self::In => "in",
            Self::De => "de",
            Self::Auto => "auto",
            Self::Multi => "multi",
            Self::Sub => "sub",
            Self::Super => "super",
            Self::Co => "co",
        }
    }

    /// Get the semantic primes this prefix contributes
    pub fn primes(&self) -> Vec<SemanticPrime> {
        match self {
            Self::Un => vec![SemanticPrime::Not],
            Self::Re => vec![SemanticPrime::Before],
            Self::Pre => vec![SemanticPrime::Before],
            Self::Post => vec![SemanticPrime::After],
            Self::Dis => vec![SemanticPrime::Not],
            Self::Mis => vec![SemanticPrime::Bad],
            Self::Over => vec![SemanticPrime::Much, SemanticPrime::Above],
            Self::Under => vec![SemanticPrime::Little, SemanticPrime::Below],
            Self::Out => vec![SemanticPrime::Far],
            Self::In => vec![SemanticPrime::Inside],
            Self::De => vec![SemanticPrime::Not],    // reversal
            Self::Auto => vec![SemanticPrime::Same], // self = same as self
            Self::Multi => vec![SemanticPrime::Much],
            Self::Sub => vec![SemanticPrime::PartOf, SemanticPrime::Below],
            Self::Super => vec![SemanticPrime::Above, SemanticPrime::Big],
            Self::Co => vec![SemanticPrime::Same], // together
        }
    }
}

impl MorphemeSuffix {
    /// Get possible string representations (some have variants)
    pub fn patterns(&self) -> Vec<&'static str> {
        match self {
            Self::Ing => vec!["ing"],
            Self::Ed => vec!["ed", "d"],
            Self::Er => vec!["er"],
            Self::Or => vec!["or"],
            Self::Tion => vec!["tion", "sion", "ation"],
            Self::Able => vec!["able", "ible"],
            Self::Ment => vec!["ment"],
            Self::Ness => vec!["ness"],
            Self::Ly => vec!["ly"],
            Self::Ful => vec!["ful"],
            Self::Less => vec!["less"],
            Self::Ize => vec!["ize", "ise"],
            Self::Ify => vec!["ify"],
        }
    }

    /// Get the semantic primes this suffix contributes
    pub fn primes(&self) -> Vec<SemanticPrime> {
        match self {
            Self::Ing => vec![SemanticPrime::Now, SemanticPrime::Happen],
            Self::Ed => vec![SemanticPrime::Before, SemanticPrime::Happen],
            Self::Er | Self::Or => vec![SemanticPrime::Someone, SemanticPrime::Do],
            Self::Tion => vec![SemanticPrime::Something, SemanticPrime::Happen],
            Self::Able => vec![SemanticPrime::Can],
            Self::Ment => vec![SemanticPrime::Something],
            Self::Ness => vec![SemanticPrime::KindOf],
            Self::Ly => vec![], // manner - modifies but doesn't add primes
            Self::Ful => vec![SemanticPrime::Much],
            Self::Less => vec![SemanticPrime::Not, SemanticPrime::Have],
            Self::Ize | Self::Ify => vec![SemanticPrime::Do, SemanticPrime::Happen],
        }
    }
}

/// Morphological analyzer
#[derive(Debug, Clone)]
pub struct MorphologicalAnalyzer {
    /// Minimum root length (to avoid over-stripping)
    min_root_length: usize,
    /// Known roots that shouldn't be further decomposed
    protected_roots: HashSet<String>,
}

impl Default for MorphologicalAnalyzer {
    fn default() -> Self {
        let mut protected_roots = HashSet::new();
        // Words that look like they have prefixes but don't
        // These are common words where the apparent prefix is actually part of the root
        for word in &[
            // Words starting with "in-" that aren't "in" + something
            "install",
            "inside",
            "input",
            "into",
            "information",
            "include",
            "increase",
            "indeed",
            "index",
            "indicate",
            "industry",
            "initial",
            "instead",
            "interest",
            "internet",
            "introduce",
            "investigate",
            // Words starting with "un-" that aren't negations
            "under",
            "understand",
            "unit",
            "until",
            "upon",
            "use",
            "usual",
            // Words starting with "re-" that aren't "again"
            "real",
            "reason",
            "recent",
            "receive",
            "record",
            "reduce",
            "refer",
            "region",
            "relate",
            "release",
            "remain",
            "remember",
            "remove",
            "render",
            "report",
            "represent",
            "require",
            "research",
            "resolve",
            "resource",
            "respond",
            "result",
            "return",
            "reveal",
            "review",
            // Words starting with "de-" that aren't reversals
            "deal",
            "decide",
            "deep",
            "define",
            "degree",
            "deliver",
            "demand",
            "describe",
            "design",
            "desire",
            "detail",
            "determine",
            "develop",
            "device",
            // Words starting with "dis-" that aren't negations
            "discuss",
            "display",
            "distance",
            "distribute",
            // Words starting with "pre-" that aren't "before"
            "present",
            "president",
            "press",
            "pretty",
            "prevent",
            "price",
            "print",
            "private",
            "probably",
            "problem",
            "process",
            "produce",
            "product",
            "program",
            "project",
            "promise",
            "property",
            "protect",
            "provide",
            "public",
            // Words starting with "over-" that aren't "excessive"
            "over",
            "output",
            // Words starting with "out-" that aren't "external"
            "other",
            "otherwise",
            // Other common false-prefix words
            "auto",
            "super",
            "disco",
            "video",
            "audio",
        ] {
            protected_roots.insert(word.to_string());
        }
        Self {
            min_root_length: 3,
            protected_roots,
        }
    }
}

impl MorphologicalAnalyzer {
    /// Create a new analyzer
    pub fn new() -> Self {
        Self::default()
    }

    /// Analyze a word into morphemes
    pub fn analyze(&self, word: &str) -> MorphologicalAnalysis {
        let word_lower = word.to_lowercase();

        // Check if it's a protected root
        if self.protected_roots.contains(&word_lower) {
            return MorphologicalAnalysis {
                word: word.to_string(),
                prefix: None,
                root: word_lower,
                suffix: None,
                confidence: 1.0,
            };
        }

        let mut prefix = None;
        let mut working = word_lower.clone();
        let mut confidence = 1.0;

        // Try to strip prefix (longest match first)
        let prefixes_by_length = [
            (MorphemePrefix::Super, "super"),
            (MorphemePrefix::Under, "under"),
            (MorphemePrefix::Multi, "multi"),
            (MorphemePrefix::Over, "over"),
            (MorphemePrefix::Post, "post"),
            (MorphemePrefix::Auto, "auto"),
            (MorphemePrefix::Pre, "pre"),
            (MorphemePrefix::Dis, "dis"),
            (MorphemePrefix::Mis, "mis"),
            (MorphemePrefix::Out, "out"),
            (MorphemePrefix::Sub, "sub"),
            (MorphemePrefix::Un, "un"),
            (MorphemePrefix::Re, "re"),
            (MorphemePrefix::De, "de"),
            (MorphemePrefix::In, "in"),
            (MorphemePrefix::Co, "co"),
        ];

        for (pref, pref_str) in &prefixes_by_length {
            if working.starts_with(pref_str) {
                let remainder = &working[pref_str.len()..];
                if remainder.len() >= self.min_root_length {
                    prefix = Some(*pref);
                    working = remainder.to_string();
                    confidence *= 0.9; // Slight uncertainty for morphological analysis
                    break;
                }
            }
        }

        // Try to strip suffix
        let mut suffix = None;
        let suffixes_by_length = [
            (MorphemeSuffix::Tion, vec!["ation", "tion", "sion"]),
            (MorphemeSuffix::Able, vec!["able", "ible"]),
            (MorphemeSuffix::Less, vec!["less"]),
            (MorphemeSuffix::Ment, vec!["ment"]),
            (MorphemeSuffix::Ness, vec!["ness"]),
            (MorphemeSuffix::Ize, vec!["ize", "ise"]),
            (MorphemeSuffix::Ify, vec!["ify"]),
            (MorphemeSuffix::Ful, vec!["ful"]),
            (MorphemeSuffix::Ing, vec!["ing"]),
            (MorphemeSuffix::Er, vec!["er"]),
            (MorphemeSuffix::Or, vec!["or"]),
            (MorphemeSuffix::Ed, vec!["ed"]),
            (MorphemeSuffix::Ly, vec!["ly"]),
        ];

        for (suf, suf_strs) in &suffixes_by_length {
            for suf_str in suf_strs {
                if working.ends_with(suf_str) {
                    let remainder = &working[..working.len() - suf_str.len()];
                    if remainder.len() >= self.min_root_length {
                        suffix = Some(*suf);
                        working = remainder.to_string();
                        confidence *= 0.9;
                        break;
                    }
                }
            }
            if suffix.is_some() {
                break;
            }
        }

        MorphologicalAnalysis {
            word: word.to_string(),
            prefix,
            root: working,
            suffix,
            confidence,
        }
    }
}

// ============================================================================
// COMPOSITIONAL PRIME INFERENCE
// ============================================================================

/// Base vocabulary: common roots mapped to semantic primes
#[derive(Debug, Clone)]
pub struct BaseVocabulary {
    /// Root word -> semantic primes
    roots: HashMap<String, Vec<SemanticPrime>>,
}

impl Default for BaseVocabulary {
    fn default() -> Self {
        let mut roots = HashMap::new();

        // === ACTION VERBS ===
        roots.insert(
            "install".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Move,
                SemanticPrime::Something,
            ],
        );
        roots.insert(
            "search".into(),
            vec![SemanticPrime::Want, SemanticPrime::Know, SemanticPrime::See],
        );
        roots.insert("find".into(), vec![SemanticPrime::See, SemanticPrime::Know]);
        roots.insert(
            "config".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Something,
                SemanticPrime::PartOf,
            ],
        );
        roots.insert(
            "configur".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Something,
                SemanticPrime::PartOf,
            ],
        );
        roots.insert(
            "build".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Something,
                SemanticPrime::Happen,
            ],
        );
        roots.insert("run".into(), vec![SemanticPrime::Do, SemanticPrime::Happen]);
        roots.insert(
            "start".into(),
            vec![SemanticPrime::Do, SemanticPrime::Happen, SemanticPrime::Now],
        );
        roots.insert(
            "stop".into(),
            vec![SemanticPrime::Do, SemanticPrime::Not, SemanticPrime::Happen],
        );
        roots.insert(
            "update".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Happen,
                SemanticPrime::Good,
            ],
        );
        roots.insert(
            "upgrad".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Happen,
                SemanticPrime::Good,
                SemanticPrime::Big,
            ],
        );
        roots.insert(
            "remov".into(),
            vec![SemanticPrime::Do, SemanticPrime::Move, SemanticPrime::Not],
        );
        roots.insert(
            "delet".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Not,
                SemanticPrime::ThereIs,
            ],
        );
        roots.insert(
            "creat".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Happen,
                SemanticPrime::ThereIs,
            ],
        );
        roots.insert(
            "add".into(),
            vec![SemanticPrime::Do, SemanticPrime::Move, SemanticPrime::Have],
        );
        roots.insert("enabl".into(), vec![SemanticPrime::Do, SemanticPrime::Can]);
        roots.insert(
            "disabl".into(),
            vec![SemanticPrime::Do, SemanticPrime::Not, SemanticPrime::Can],
        );
        roots.insert(
            "check".into(),
            vec![SemanticPrime::See, SemanticPrime::Know],
        );
        roots.insert(
            "test".into(),
            vec![SemanticPrime::Do, SemanticPrime::See, SemanticPrime::Know],
        );
        roots.insert(
            "verif".into(),
            vec![SemanticPrime::See, SemanticPrime::Know, SemanticPrime::True],
        );
        roots.insert(
            "help".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Good,
                SemanticPrime::Someone,
            ],
        );
        roots.insert("show".into(), vec![SemanticPrime::Do, SemanticPrime::See]);
        roots.insert("list".into(), vec![SemanticPrime::See, SemanticPrime::All]);
        roots.insert("get".into(), vec![SemanticPrime::Do, SemanticPrime::Have]);
        roots.insert("set".into(), vec![SemanticPrime::Do, SemanticPrime::Happen]);
        roots.insert(
            "load".into(),
            vec![SemanticPrime::Move, SemanticPrime::Inside],
        );
        roots.insert(
            "save".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Have,
                SemanticPrime::LongTime,
            ],
        );
        roots.insert(
            "open".into(),
            vec![SemanticPrime::Do, SemanticPrime::Can, SemanticPrime::Inside],
        );
        roots.insert(
            "clos".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Not,
                SemanticPrime::Can,
                SemanticPrime::Inside,
            ],
        );
        roots.insert(
            "read".into(),
            vec![
                SemanticPrime::See,
                SemanticPrime::Know,
                SemanticPrime::Words,
            ],
        );
        roots.insert("writ".into(), vec![SemanticPrime::Do, SemanticPrime::Words]);
        roots.insert(
            "edit".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Happen,
                SemanticPrime::Other,
            ],
        );
        roots.insert(
            "chang".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Happen,
                SemanticPrime::Other,
            ],
        );
        roots.insert("mov".into(), vec![SemanticPrime::Move]);
        roots.insert(
            "copy".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Same,
                SemanticPrime::Something,
            ],
        );
        roots.insert("sync".into(), vec![SemanticPrime::Do, SemanticPrime::Same]);
        roots.insert(
            "connect".into(),
            vec![SemanticPrime::Do, SemanticPrime::Touch],
        );
        roots.insert(
            "send".into(),
            vec![SemanticPrime::Do, SemanticPrime::Move, SemanticPrime::Far],
        );
        roots.insert(
            "receiv".into(),
            vec![SemanticPrime::Have, SemanticPrime::Move],
        );
        roots.insert(
            "download".into(),
            vec![
                SemanticPrime::Move,
                SemanticPrime::Have,
                SemanticPrime::Below,
            ],
        );
        roots.insert(
            "upload".into(),
            vec![SemanticPrime::Move, SemanticPrime::Above],
        );
        roots.insert(
            "compil".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Happen,
                SemanticPrime::Something,
            ],
        );
        roots.insert(
            "link".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Touch,
                SemanticPrime::PartOf,
            ],
        );
        roots.insert(
            "execut".into(),
            vec![SemanticPrime::Do, SemanticPrime::Happen],
        );
        roots.insert(
            "process".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Happen,
                SemanticPrime::Something,
            ],
        );
        roots.insert(
            "generat".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Happen,
                SemanticPrime::ThereIs,
            ],
        );
        roots.insert(
            "deploy".into(),
            vec![SemanticPrime::Move, SemanticPrime::Do, SemanticPrime::Where],
        );
        roots.insert(
            "collect".into(),
            vec![SemanticPrime::Do, SemanticPrime::Have, SemanticPrime::All],
        );
        roots.insert(
            "clean".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Good,
                SemanticPrime::Not,
                SemanticPrime::Bad,
            ],
        );
        roots.insert(
            "garbage".into(),
            vec![SemanticPrime::Bad, SemanticPrime::Something],
        );
        roots.insert(
            "switch".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Happen,
                SemanticPrime::Other,
            ],
        );
        roots.insert(
            "rollback".into(),
            vec![SemanticPrime::Move, SemanticPrime::Before],
        );
        roots.insert(
            "boot".into(),
            vec![SemanticPrime::Do, SemanticPrime::Happen, SemanticPrime::Now],
        );
        roots.insert(
            "rebuild".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Something,
                SemanticPrime::Before,
            ],
        );
        roots.insert(
            "flake".into(),
            vec![SemanticPrime::Something, SemanticPrime::PartOf],
        ); // NixOS specific
        roots.insert("nix".into(), vec![SemanticPrime::Something]); // NixOS namespace

        // === NOUNS / OBJECTS ===
        roots.insert(
            "file".into(),
            vec![SemanticPrime::Something, SemanticPrime::Words],
        );
        roots.insert(
            "direct".into(),
            vec![SemanticPrime::Where, SemanticPrime::PartOf],
        );
        roots.insert(
            "fold".into(),
            vec![SemanticPrime::Where, SemanticPrime::Inside],
        );
        roots.insert(
            "packag".into(),
            vec![SemanticPrime::Something, SemanticPrime::PartOf],
        );
        roots.insert(
            "servic".into(),
            vec![SemanticPrime::Something, SemanticPrime::Do],
        );
        roots.insert(
            "system".into(),
            vec![
                SemanticPrime::Something,
                SemanticPrime::All,
                SemanticPrime::PartOf,
            ],
        );
        roots.insert(
            "program".into(),
            vec![SemanticPrime::Something, SemanticPrime::Do],
        );
        roots.insert(
            "option".into(),
            vec![
                SemanticPrime::Something,
                SemanticPrime::Can,
                SemanticPrime::Other,
            ],
        );
        roots.insert("valu".into(), vec![SemanticPrime::Something]);
        roots.insert(
            "nam".into(),
            vec![SemanticPrime::Words, SemanticPrime::This],
        );
        roots.insert(
            "path".into(),
            vec![SemanticPrime::Where, SemanticPrime::PartOf],
        );
        roots.insert(
            "user".into(),
            vec![SemanticPrime::Someone, SemanticPrime::Do],
        );
        roots.insert(
            "group".into(),
            vec![SemanticPrime::People, SemanticPrime::Same],
        );
        roots.insert(
            "permiss".into(),
            vec![SemanticPrime::Can, SemanticPrime::Do],
        );
        roots.insert(
            "error".into(),
            vec![SemanticPrime::Bad, SemanticPrime::Happen],
        );
        roots.insert(
            "warn".into(),
            vec![SemanticPrime::Say, SemanticPrime::Bad, SemanticPrime::Maybe],
        );
        roots.insert(
            "log".into(),
            vec![
                SemanticPrime::Words,
                SemanticPrime::Before,
                SemanticPrime::Happen,
            ],
        );
        roots.insert(
            "output".into(),
            vec![SemanticPrime::Something, SemanticPrime::Far],
        );
        roots.insert(
            "input".into(),
            vec![SemanticPrime::Something, SemanticPrime::Inside],
        );
        roots.insert(
            "data".into(),
            vec![SemanticPrime::Something, SemanticPrime::Know],
        );
        roots.insert(
            "info".into(),
            vec![SemanticPrime::Know, SemanticPrime::Something],
        );
        roots.insert(
            "inform".into(),
            vec![SemanticPrime::Know, SemanticPrime::Something],
        );
        roots.insert(
            "messag".into(),
            vec![SemanticPrime::Words, SemanticPrime::Say],
        );
        roots.insert(
            "command".into(),
            vec![SemanticPrime::Words, SemanticPrime::Do, SemanticPrime::Want],
        );
        roots.insert(
            "request".into(),
            vec![SemanticPrime::Want, SemanticPrime::Say],
        );
        roots.insert(
            "respons".into(),
            vec![SemanticPrime::Say, SemanticPrime::After],
        );
        roots.insert(
            "result".into(),
            vec![
                SemanticPrime::Something,
                SemanticPrime::After,
                SemanticPrime::Happen,
            ],
        );
        roots.insert(
            "status".into(),
            vec![SemanticPrime::Know, SemanticPrime::Now],
        );
        roots.insert("state".into(), vec![SemanticPrime::Be, SemanticPrime::Now]);
        roots.insert(
            "version".into(),
            vec![SemanticPrime::This, SemanticPrime::KindOf],
        );
        roots.insert(
            "depend".into(),
            vec![SemanticPrime::Want, SemanticPrime::Have],
        );
        roots.insert(
            "modul".into(),
            vec![SemanticPrime::Something, SemanticPrime::PartOf],
        );
        roots.insert(
            "librari".into(),
            vec![
                SemanticPrime::Something,
                SemanticPrime::Have,
                SemanticPrime::Much,
            ],
        );
        roots.insert(
            "lib".into(),
            vec![SemanticPrime::Something, SemanticPrime::Have],
        );
        roots.insert(
            "bin".into(),
            vec![SemanticPrime::Something, SemanticPrime::Do],
        );
        roots.insert(
            "src".into(),
            vec![SemanticPrime::Words, SemanticPrime::Something],
        );
        roots.insert(
            "sourc".into(),
            vec![SemanticPrime::Words, SemanticPrime::Something],
        );
        roots.insert(
            "doc".into(),
            vec![SemanticPrime::Words, SemanticPrime::Know],
        );
        roots.insert(
            "document".into(),
            vec![SemanticPrime::Words, SemanticPrime::Know],
        );
        roots.insert(
            "video".into(),
            vec![SemanticPrime::See, SemanticPrime::Move],
        );
        roots.insert("audio".into(), vec![SemanticPrime::Hear]);
        roots.insert(
            "imag".into(),
            vec![SemanticPrime::See, SemanticPrime::Something],
        );
        roots.insert("text".into(), vec![SemanticPrime::Words]);
        roots.insert(
            "editor".into(),
            vec![
                SemanticPrime::Do,
                SemanticPrime::Words,
                SemanticPrime::Happen,
            ],
        );

        // === ADJECTIVES / DESCRIPTORS ===
        roots.insert(
            "new".into(),
            vec![
                SemanticPrime::Now,
                SemanticPrime::Not,
                SemanticPrime::Before,
            ],
        );
        roots.insert(
            "old".into(),
            vec![SemanticPrime::Before, SemanticPrime::LongTime],
        );
        roots.insert(
            "fast".into(),
            vec![SemanticPrime::ShortTime, SemanticPrime::Do],
        );
        roots.insert(
            "slow".into(),
            vec![SemanticPrime::LongTime, SemanticPrime::Do],
        );
        roots.insert("good".into(), vec![SemanticPrime::Good]);
        roots.insert("bad".into(), vec![SemanticPrime::Bad]);
        roots.insert("big".into(), vec![SemanticPrime::Big]);
        roots.insert("small".into(), vec![SemanticPrime::Small]);
        roots.insert("long".into(), vec![SemanticPrime::Big]);
        roots.insert("short".into(), vec![SemanticPrime::Small]);
        roots.insert("all".into(), vec![SemanticPrime::All]);
        roots.insert("some".into(), vec![SemanticPrime::Some]);
        roots.insert("many".into(), vec![SemanticPrime::Much]);
        roots.insert("few".into(), vec![SemanticPrime::Little]);
        roots.insert(
            "first".into(),
            vec![SemanticPrime::One, SemanticPrime::Before],
        );
        roots.insert(
            "last".into(),
            vec![SemanticPrime::One, SemanticPrime::After],
        );
        roots.insert(
            "next".into(),
            vec![SemanticPrime::After, SemanticPrime::One],
        );
        roots.insert(
            "prev".into(),
            vec![SemanticPrime::Before, SemanticPrime::One],
        );
        roots.insert(
            "current".into(),
            vec![SemanticPrime::Now, SemanticPrime::This],
        );
        roots.insert(
            "default".into(),
            vec![
                SemanticPrime::Same,
                SemanticPrime::Not,
                SemanticPrime::Other,
            ],
        );
        roots.insert(
            "custom".into(),
            vec![SemanticPrime::Other, SemanticPrime::Want],
        );
        roots.insert(
            "local".into(),
            vec![SemanticPrime::Near, SemanticPrime::This],
        );
        roots.insert("remot".into(), vec![SemanticPrime::Far]);
        roots.insert(
            "global".into(),
            vec![SemanticPrime::All, SemanticPrime::Where],
        );
        roots.insert(
            "public".into(),
            vec![SemanticPrime::All, SemanticPrime::Can, SemanticPrime::See],
        );
        roots.insert(
            "privat".into(),
            vec![
                SemanticPrime::Not,
                SemanticPrime::All,
                SemanticPrime::Can,
                SemanticPrime::See,
            ],
        );
        roots.insert(
            "secret".into(),
            vec![SemanticPrime::Not, SemanticPrime::All, SemanticPrime::Know],
        );
        roots.insert(
            "safe".into(),
            vec![SemanticPrime::Good, SemanticPrime::Not, SemanticPrime::Bad],
        );
        roots.insert(
            "danger".into(),
            vec![
                SemanticPrime::Bad,
                SemanticPrime::Maybe,
                SemanticPrime::Happen,
            ],
        );
        roots.insert(
            "valid".into(),
            vec![SemanticPrime::Good, SemanticPrime::True],
        );
        roots.insert(
            "invalid".into(),
            vec![SemanticPrime::Bad, SemanticPrime::Not, SemanticPrime::True],
        );

        // === COMMON WORDS ===
        roots.insert("the".into(), vec![SemanticPrime::This]);
        roots.insert("a".into(), vec![SemanticPrime::One]);
        roots.insert("an".into(), vec![SemanticPrime::One]);
        roots.insert("this".into(), vec![SemanticPrime::This]);
        roots.insert("that".into(), vec![SemanticPrime::This]);
        roots.insert("for".into(), vec![SemanticPrime::Because]);
        roots.insert("with".into(), vec![SemanticPrime::Have]);
        roots.insert(
            "from".into(),
            vec![SemanticPrime::Where, SemanticPrime::Move],
        );
        roots.insert("to".into(), vec![SemanticPrime::Where]);
        roots.insert("in".into(), vec![SemanticPrime::Inside]);
        roots.insert(
            "on".into(),
            vec![SemanticPrime::Above, SemanticPrime::Touch],
        );
        roots.insert("at".into(), vec![SemanticPrime::Where]);
        roots.insert(
            "by".into(),
            vec![SemanticPrime::Near, SemanticPrime::Because],
        );
        roots.insert("of".into(), vec![SemanticPrime::PartOf]);
        roots.insert("and".into(), vec![]); // Connector, no primes
        roots.insert("or".into(), vec![SemanticPrime::Maybe]);
        roots.insert("not".into(), vec![SemanticPrime::Not]);
        roots.insert("if".into(), vec![SemanticPrime::If]);
        roots.insert(
            "then".into(),
            vec![SemanticPrime::After, SemanticPrime::Because],
        );
        roots.insert("else".into(), vec![SemanticPrime::Other, SemanticPrime::If]);
        roots.insert("when".into(), vec![SemanticPrime::When]);
        roots.insert("where".into(), vec![SemanticPrime::Where]);
        roots.insert("what".into(), vec![SemanticPrime::Something]);
        roots.insert(
            "which".into(),
            vec![SemanticPrime::This, SemanticPrime::Something],
        );
        roots.insert("who".into(), vec![SemanticPrime::Someone]);
        roots.insert("how".into(), vec![SemanticPrime::KindOf]);
        roots.insert("why".into(), vec![SemanticPrime::Because]);
        roots.insert("thing".into(), vec![SemanticPrime::Something]);
        roots.insert(
            "stuff".into(),
            vec![SemanticPrime::Something, SemanticPrime::Some],
        );
        roots.insert("do".into(), vec![SemanticPrime::Do]);
        roots.insert("have".into(), vec![SemanticPrime::Have]);
        roots.insert("be".into(), vec![SemanticPrime::Be]);
        roots.insert("want".into(), vec![SemanticPrime::Want]);
        roots.insert(
            "need".into(),
            vec![SemanticPrime::Want, SemanticPrime::Want],
        );
        roots.insert("can".into(), vec![SemanticPrime::Can]);
        roots.insert(
            "could".into(),
            vec![SemanticPrime::Can, SemanticPrime::Maybe],
        );
        roots.insert("would".into(), vec![SemanticPrime::If, SemanticPrime::Do]);
        roots.insert(
            "should".into(),
            vec![SemanticPrime::Good, SemanticPrime::Do],
        );
        roots.insert("must".into(), vec![SemanticPrime::Want]);
        roots.insert("maybe".into(), vec![SemanticPrime::Maybe]);
        roots.insert(
            "pleas".into(),
            vec![SemanticPrime::Want, SemanticPrime::Good],
        );

        Self { roots }
    }
}

impl BaseVocabulary {
    /// Create a new base vocabulary
    pub fn new() -> Self {
        Self::default()
    }

    /// Look up primes for a root
    pub fn get(&self, root: &str) -> Option<&Vec<SemanticPrime>> {
        self.roots.get(root)
    }

    /// Add a new root mapping
    pub fn insert(&mut self, root: String, primes: Vec<SemanticPrime>) {
        self.roots.insert(root, primes);
    }

    /// Get the number of known roots
    pub fn len(&self) -> usize {
        self.roots.len()
    }

    /// Check if vocabulary is empty
    pub fn is_empty(&self) -> bool {
        self.roots.is_empty()
    }
}

/// Compositional prime inference engine
#[derive(Debug, Clone)]
pub struct CompositionalInference {
    /// Morphological analyzer
    analyzer: MorphologicalAnalyzer,
    /// Base vocabulary
    vocabulary: BaseVocabulary,
}

impl Default for CompositionalInference {
    fn default() -> Self {
        Self {
            analyzer: MorphologicalAnalyzer::new(),
            vocabulary: BaseVocabulary::new(),
        }
    }
}

impl CompositionalInference {
    /// Create a new compositional inference engine
    pub fn new() -> Self {
        Self::default()
    }

    /// Infer semantic primes for a word compositionally
    pub fn infer(&self, word: &str) -> CompositionResult {
        let analysis = self.analyzer.analyze(word);
        let mut primes = Vec::new();
        let mut confidence = analysis.confidence;
        let mut sources = Vec::new();

        // Add primes from prefix
        if let Some(prefix) = &analysis.prefix {
            let prefix_primes = prefix.primes();
            sources.push(format!(
                "prefix '{}' -> {:?}",
                prefix.as_str(),
                prefix_primes
            ));
            primes.extend(prefix_primes);
        }

        // Add primes from root
        if let Some(root_primes) = self.vocabulary.get(&analysis.root) {
            sources.push(format!("root '{}' -> {:?}", analysis.root, root_primes));
            primes.extend(root_primes.clone());
        } else {
            // Unknown root - reduce confidence
            confidence *= 0.5;
            sources.push(format!("root '{}' -> UNKNOWN", analysis.root));
        }

        // Add primes from suffix
        if let Some(suffix) = &analysis.suffix {
            let suffix_primes = suffix.primes();
            sources.push(format!("suffix -> {:?}", suffix_primes));
            primes.extend(suffix_primes);
        }

        // Deduplicate primes while preserving order
        let mut seen = HashSet::new();
        primes.retain(|p| seen.insert(*p));

        CompositionResult {
            word: word.to_string(),
            analysis,
            primes,
            confidence,
            sources,
        }
    }

    /// Get the base vocabulary (for learning)
    pub fn vocabulary_mut(&mut self) -> &mut BaseVocabulary {
        &mut self.vocabulary
    }
}

/// Result of compositional inference
#[derive(Debug, Clone)]
pub struct CompositionResult {
    /// Original word
    pub word: String,
    /// Morphological analysis
    pub analysis: MorphologicalAnalysis,
    /// Inferred primes
    pub primes: Vec<SemanticPrime>,
    /// Confidence in the inference
    pub confidence: f64,
    /// Sources of each prime (for explainability)
    pub sources: Vec<String>,
}

// ============================================================================
// CONTEXTUAL PRIME ACTIVATION
// ============================================================================

/// Context window for disambiguation
#[derive(Debug, Clone)]
pub struct ContextWindow {
    /// Words before the target
    pub before: Vec<String>,
    /// The target word
    pub target: String,
    /// Words after the target
    pub after: Vec<String>,
}

/// Contextual disambiguator
#[derive(Debug, Clone)]
pub struct ContextualDisambiguator {
    /// Context patterns that influence prime selection
    patterns: Vec<ContextPattern>,
}

/// A pattern that influences prime selection based on context
#[derive(Debug, Clone)]
pub struct ContextPattern {
    /// Pattern name
    pub name: String,
    /// Words that trigger this pattern (in context)
    pub triggers: Vec<String>,
    /// Primes to boost when triggered
    pub boost_primes: Vec<SemanticPrime>,
    /// Primes to suppress when triggered
    pub suppress_primes: Vec<SemanticPrime>,
}

impl Default for ContextualDisambiguator {
    fn default() -> Self {
        let patterns = vec![
            // "run" in computing context
            ContextPattern {
                name: "computing_run".into(),
                triggers: vec![
                    "program".into(),
                    "command".into(),
                    "script".into(),
                    "code".into(),
                    "process".into(),
                ],
                boost_primes: vec![SemanticPrime::Do, SemanticPrime::Happen],
                suppress_primes: vec![SemanticPrime::Move],
            },
            // "run" in physical context
            ContextPattern {
                name: "physical_run".into(),
                triggers: vec![
                    "fast".into(),
                    "walk".into(),
                    "jog".into(),
                    "sprint".into(),
                    "exercise".into(),
                ],
                boost_primes: vec![SemanticPrime::Move],
                suppress_primes: vec![],
            },
            // File system context
            ContextPattern {
                name: "filesystem".into(),
                triggers: vec![
                    "file".into(),
                    "directory".into(),
                    "folder".into(),
                    "path".into(),
                ],
                boost_primes: vec![SemanticPrime::Where, SemanticPrime::PartOf],
                suppress_primes: vec![],
            },
            // Network context
            ContextPattern {
                name: "network".into(),
                triggers: vec![
                    "server".into(),
                    "client".into(),
                    "network".into(),
                    "remote".into(),
                    "connect".into(),
                ],
                boost_primes: vec![SemanticPrime::Far, SemanticPrime::Touch],
                suppress_primes: vec![SemanticPrime::Near],
            },
            // Time context
            ContextPattern {
                name: "temporal".into(),
                triggers: vec![
                    "schedule".into(),
                    "cron".into(),
                    "timer".into(),
                    "wait".into(),
                    "delay".into(),
                ],
                boost_primes: vec![SemanticPrime::When, SemanticPrime::ForSomeTime],
                suppress_primes: vec![],
            },
            // Package management context
            ContextPattern {
                name: "package_management".into(),
                triggers: vec![
                    "package".into(),
                    "install".into(),
                    "nixpkgs".into(),
                    "derivation".into(),
                    "flake".into(),
                ],
                boost_primes: vec![
                    SemanticPrime::Something,
                    SemanticPrime::PartOf,
                    SemanticPrime::Move,
                ],
                suppress_primes: vec![],
            },
        ];

        Self { patterns }
    }
}

impl ContextualDisambiguator {
    /// Create a new disambiguator
    pub fn new() -> Self {
        Self::default()
    }

    /// Disambiguate primes based on context
    pub fn disambiguate(
        &self,
        context: &ContextWindow,
        candidate_primes: &[SemanticPrime],
    ) -> DisambiguationResult {
        let mut scores: HashMap<SemanticPrime, f64> = HashMap::new();

        // Initialize scores
        for prime in candidate_primes {
            scores.insert(*prime, 1.0);
        }

        // Collect all context words
        let context_words: Vec<&str> = context
            .before
            .iter()
            .chain(context.after.iter())
            .map(|s| s.as_str())
            .collect();

        let mut activated_patterns = Vec::new();

        // Apply patterns
        for pattern in &self.patterns {
            let trigger_count: usize = pattern
                .triggers
                .iter()
                .filter(|t| context_words.iter().any(|w| w.contains(t.as_str())))
                .count();

            if trigger_count > 0 {
                activated_patterns.push(pattern.name.clone());
                let boost = 1.0 + (trigger_count as f64 * 0.3);

                for prime in &pattern.boost_primes {
                    if let Some(score) = scores.get_mut(prime) {
                        *score *= boost;
                    } else {
                        // Add the boosted prime even if not in candidates
                        scores.insert(*prime, boost * 0.5);
                    }
                }

                for prime in &pattern.suppress_primes {
                    if let Some(score) = scores.get_mut(prime) {
                        *score *= 0.5;
                    }
                }
            }
        }

        // Sort by score and filter
        let mut ranked: Vec<_> = scores.into_iter().collect();
        ranked.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Keep primes with score >= 0.5
        let filtered_primes: Vec<SemanticPrime> = ranked
            .iter()
            .filter(|(_, score)| *score >= 0.5)
            .map(|(prime, _)| *prime)
            .collect();

        let confidence = if activated_patterns.is_empty() {
            0.7
        } else {
            0.9
        };

        DisambiguationResult {
            original_primes: candidate_primes.to_vec(),
            filtered_primes,
            activated_patterns,
            confidence,
        }
    }
}

/// Result of contextual disambiguation
#[derive(Debug, Clone)]
pub struct DisambiguationResult {
    /// Original candidate primes
    pub original_primes: Vec<SemanticPrime>,
    /// Filtered/reranked primes
    pub filtered_primes: Vec<SemanticPrime>,
    /// Patterns that were activated
    pub activated_patterns: Vec<String>,
    /// Confidence in disambiguation
    pub confidence: f64,
}

// ============================================================================
// PRIME COHERENCE AS Φ SIGNAL
// ============================================================================

/// Prime coherence calculator
#[derive(Debug, Clone, Default)]
pub struct PrimeCoherenceCalculator;

impl PrimeCoherenceCalculator {
    /// Create a new calculator
    pub fn new() -> Self {
        Self
    }

    /// Calculate coherence between multiple prime decompositions
    /// Returns a value between 0.0 (total disagreement) and 1.0 (total agreement)
    pub fn calculate_coherence(&self, decompositions: &[Vec<SemanticPrime>]) -> CoherenceResult {
        if decompositions.is_empty() {
            return CoherenceResult {
                coherence: 0.0,
                agreement_pairs: 0,
                disagreement_pairs: 0,
                shared_primes: vec![],
                unique_primes: vec![],
            };
        }

        if decompositions.len() == 1 {
            return CoherenceResult {
                coherence: 1.0,
                agreement_pairs: 0,
                disagreement_pairs: 0,
                shared_primes: decompositions[0].clone(),
                unique_primes: vec![],
            };
        }

        // Count prime occurrences across decompositions
        let mut prime_counts: HashMap<SemanticPrime, usize> = HashMap::new();
        for decomp in decompositions {
            let unique_in_decomp: HashSet<_> = decomp.iter().cloned().collect();
            for prime in unique_in_decomp {
                *prime_counts.entry(prime).or_insert(0) += 1;
            }
        }

        let n = decompositions.len();

        // Shared primes appear in all decompositions
        let shared_primes: Vec<_> = prime_counts
            .iter()
            .filter(|(_, count)| **count == n)
            .map(|(prime, _)| *prime)
            .collect();

        // Unique primes appear in only one
        let unique_primes: Vec<_> = prime_counts
            .iter()
            .filter(|(_, count)| **count == 1)
            .map(|(prime, _)| *prime)
            .collect();

        // Calculate pairwise agreement
        let mut agreement_pairs = 0;
        let mut disagreement_pairs = 0;

        for i in 0..decompositions.len() {
            for j in (i + 1)..decompositions.len() {
                let set_i: HashSet<_> = decompositions[i].iter().cloned().collect();
                let set_j: HashSet<_> = decompositions[j].iter().cloned().collect();

                let intersection = set_i.intersection(&set_j).count();
                let union = set_i.union(&set_j).count();

                if union > 0 {
                    let jaccard = intersection as f64 / union as f64;
                    if jaccard >= 0.5 {
                        agreement_pairs += 1;
                    } else {
                        disagreement_pairs += 1;
                    }
                }
            }
        }

        let total_pairs = agreement_pairs + disagreement_pairs;
        let coherence = if total_pairs > 0 {
            agreement_pairs as f64 / total_pairs as f64
        } else {
            1.0
        };

        CoherenceResult {
            coherence,
            agreement_pairs,
            disagreement_pairs,
            shared_primes,
            unique_primes,
        }
    }
}

/// Result of coherence calculation
#[derive(Debug, Clone)]
pub struct CoherenceResult {
    /// Overall coherence score (0.0 - 1.0)
    pub coherence: f64,
    /// Number of agreeing pairs
    pub agreement_pairs: usize,
    /// Number of disagreeing pairs
    pub disagreement_pairs: usize,
    /// Primes shared across all decompositions
    pub shared_primes: Vec<SemanticPrime>,
    /// Primes unique to single decompositions
    pub unique_primes: Vec<SemanticPrime>,
}

// ============================================================================
// SELF-LEARNING VOCABULARY
// ============================================================================

/// A successful interaction to learn from
#[derive(Debug, Clone)]
pub struct LearnableInteraction {
    /// Input text
    pub input: String,
    /// Words in the input
    pub words: Vec<String>,
    /// Was the interaction successful?
    pub success: bool,
    /// Inferred primes per word
    pub word_primes: HashMap<String, Vec<SemanticPrime>>,
    /// Feedback score (0.0 - 1.0)
    pub feedback_score: f64,
}

/// Self-learning vocabulary system
#[derive(Debug, Clone)]
pub struct SelfLearningVocabulary {
    /// Compositional inference engine
    inference: CompositionalInference,
    /// Learned associations: word -> (primes, confidence, observation_count)
    learned: HashMap<String, (Vec<SemanticPrime>, f64, usize)>,
    /// Interaction history (recent)
    history: VecDeque<LearnableInteraction>,
    /// Max history size
    max_history: usize,
    /// Learning rate
    learning_rate: f64,
}

impl Default for SelfLearningVocabulary {
    fn default() -> Self {
        Self {
            inference: CompositionalInference::new(),
            learned: HashMap::new(),
            history: VecDeque::new(),
            max_history: 1000,
            learning_rate: 0.1,
        }
    }
}

impl SelfLearningVocabulary {
    /// Create a new self-learning vocabulary
    pub fn new() -> Self {
        Self::default()
    }

    /// Get primes for a word (compositional + learned)
    pub fn get_primes(&self, word: &str) -> (Vec<SemanticPrime>, f64) {
        let word_lower = word.to_lowercase();

        // Check learned associations first
        if let Some((primes, confidence, _)) = self.learned.get(&word_lower) {
            return (primes.clone(), *confidence);
        }

        // Fall back to compositional inference
        let result = self.inference.infer(&word_lower);
        (result.primes, result.confidence)
    }

    /// Record a successful interaction for learning
    pub fn record_success(
        &mut self,
        input: &str,
        word_primes: HashMap<String, Vec<SemanticPrime>>,
    ) {
        let interaction = LearnableInteraction {
            input: input.to_string(),
            words: input.split_whitespace().map(|s| s.to_lowercase()).collect(),
            success: true,
            word_primes: word_primes.clone(),
            feedback_score: 1.0,
        };

        // Learn from the interaction
        for (word, primes) in word_primes {
            self.learn_association(&word, &primes, 1.0);
        }

        // Add to history
        self.history.push_back(interaction);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Record a failed interaction (negative learning)
    pub fn record_failure(
        &mut self,
        input: &str,
        word_primes: HashMap<String, Vec<SemanticPrime>>,
    ) {
        let interaction = LearnableInteraction {
            input: input.to_string(),
            words: input.split_whitespace().map(|s| s.to_lowercase()).collect(),
            success: false,
            word_primes,
            feedback_score: 0.0,
        };

        // Add to history but don't reinforce
        self.history.push_back(interaction);
        if self.history.len() > self.max_history {
            self.history.pop_front();
        }
    }

    /// Learn an association between a word and primes
    fn learn_association(&mut self, word: &str, primes: &[SemanticPrime], feedback: f64) {
        let word_lower = word.to_lowercase();

        if let Some((existing_primes, confidence, count)) = self.learned.get_mut(&word_lower) {
            // Update existing association
            let new_confidence = *confidence + self.learning_rate * (feedback - *confidence);
            *confidence = new_confidence;
            *count += 1;

            // Merge primes if they're new and feedback is positive
            if feedback > 0.5 {
                for prime in primes {
                    if !existing_primes.contains(prime) {
                        existing_primes.push(*prime);
                    }
                }
            }
        } else {
            // New association
            self.learned.insert(
                word_lower,
                (primes.to_vec(), feedback * 0.5, 1), // Start with lower confidence
            );
        }
    }

    /// Get the compositional inference engine
    pub fn inference(&self) -> &CompositionalInference {
        &self.inference
    }

    /// Get learning statistics
    pub fn stats(&self) -> LearningStats {
        let successful = self.history.iter().filter(|i| i.success).count();
        LearningStats {
            learned_words: self.learned.len(),
            base_vocabulary_size: self.inference.vocabulary.len(),
            total_interactions: self.history.len(),
            successful_interactions: successful,
            success_rate: if self.history.is_empty() {
                0.0
            } else {
                successful as f64 / self.history.len() as f64
            },
        }
    }
}

/// Learning statistics
#[derive(Debug, Clone)]
pub struct LearningStats {
    /// Number of learned word associations
    pub learned_words: usize,
    /// Size of base vocabulary
    pub base_vocabulary_size: usize,
    /// Total recorded interactions
    pub total_interactions: usize,
    /// Successful interactions
    pub successful_interactions: usize,
    /// Success rate
    pub success_rate: f64,
}

// ============================================================================
// EMERGENT FRAME INDUCTION
// ============================================================================

/// An observed verb-argument pattern
#[derive(Debug, Clone)]
pub struct ObservedPattern {
    /// The verb/action
    pub verb: String,
    /// Argument roles detected
    pub arguments: Vec<(String, String)>, // (role, filler)
    /// Frequency of observation
    pub frequency: usize,
    /// Semantic primes associated
    pub primes: Vec<SemanticPrime>,
}

/// Induced frame from patterns
#[derive(Debug, Clone)]
pub struct InducedFrame {
    /// Frame name (auto-generated)
    pub name: String,
    /// Core primes defining this frame
    pub core_primes: Vec<SemanticPrime>,
    /// Expected roles
    pub roles: Vec<String>,
    /// Example patterns
    pub examples: Vec<String>,
    /// Confidence in the frame
    pub confidence: f64,
    /// How many patterns support this frame
    pub support_count: usize,
}

/// Frame induction engine
#[derive(Debug, Clone)]
pub struct FrameInduction {
    /// Observed patterns
    patterns: Vec<ObservedPattern>,
    /// Induced frames
    frames: Vec<InducedFrame>,
    /// Minimum support for frame induction
    min_support: usize,
}

impl Default for FrameInduction {
    fn default() -> Self {
        Self {
            patterns: Vec::new(),
            frames: Vec::new(),
            min_support: 3,
        }
    }
}

impl FrameInduction {
    /// Create a new frame induction engine
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an observed pattern
    pub fn observe(
        &mut self,
        verb: &str,
        arguments: Vec<(String, String)>,
        primes: Vec<SemanticPrime>,
    ) {
        // Check if we've seen this pattern before
        let verb_lower = verb.to_lowercase();
        if let Some(existing) = self.patterns.iter_mut().find(|p| p.verb == verb_lower) {
            existing.frequency += 1;
            // Merge primes
            for prime in primes {
                if !existing.primes.contains(&prime) {
                    existing.primes.push(prime);
                }
            }
        } else {
            self.patterns.push(ObservedPattern {
                verb: verb_lower,
                arguments,
                frequency: 1,
                primes,
            });
        }

        // Try to induce new frames
        self.induce_frames();
    }

    /// Induce frames from observed patterns
    fn induce_frames(&mut self) {
        // Group patterns by core primes
        let mut prime_groups: HashMap<Vec<SemanticPrime>, Vec<&ObservedPattern>> = HashMap::new();

        for pattern in &self.patterns {
            if pattern.frequency >= self.min_support {
                let mut key = pattern.primes.clone();
                key.sort();
                prime_groups.entry(key).or_default().push(pattern);
            }
        }

        // Create frames from groups with multiple patterns
        for (core_primes, patterns) in prime_groups {
            if patterns.len() >= 2 && !core_primes.is_empty() {
                // Generate frame name from primes
                let name = core_primes
                    .iter()
                    .take(3)
                    .map(|p| format!("{:?}", p))
                    .collect::<Vec<_>>()
                    .join("_");

                // Collect roles from patterns
                let mut roles: HashSet<String> = HashSet::new();
                for pattern in &patterns {
                    for (role, _) in &pattern.arguments {
                        roles.insert(role.clone());
                    }
                }

                // Create or update frame
                let examples: Vec<_> = patterns.iter().map(|p| &p.verb).cloned().collect();
                let support = patterns.iter().map(|p| p.frequency).sum();
                let confidence = (support as f64 / 10.0).min(1.0);

                if let Some(existing) = self
                    .frames
                    .iter_mut()
                    .find(|f| f.core_primes == core_primes)
                {
                    existing.support_count = support;
                    existing.confidence = confidence;
                    for ex in &examples {
                        if !existing.examples.contains(ex) {
                            existing.examples.push(ex.clone());
                        }
                    }
                } else {
                    self.frames.push(InducedFrame {
                        name,
                        core_primes,
                        roles: roles.into_iter().collect(),
                        examples,
                        confidence,
                        support_count: support,
                    });
                }
            }
        }
    }

    /// Get induced frames
    pub fn frames(&self) -> &[InducedFrame] {
        &self.frames
    }

    /// Find frames matching given primes
    pub fn find_matching_frames(&self, primes: &[SemanticPrime]) -> Vec<&InducedFrame> {
        let prime_set: HashSet<_> = primes.iter().cloned().collect();

        self.frames
            .iter()
            .filter(|f| {
                let frame_primes: HashSet<_> = f.core_primes.iter().cloned().collect();
                let intersection = prime_set.intersection(&frame_primes).count();
                intersection as f64 / frame_primes.len().max(1) as f64 >= 0.5
            })
            .collect()
    }

    /// Get statistics
    pub fn stats(&self) -> FrameInductionStats {
        FrameInductionStats {
            observed_patterns: self.patterns.len(),
            total_observations: self.patterns.iter().map(|p| p.frequency).sum(),
            induced_frames: self.frames.len(),
            average_support: if self.frames.is_empty() {
                0.0
            } else {
                self.frames.iter().map(|f| f.support_count).sum::<usize>() as f64
                    / self.frames.len() as f64
            },
        }
    }
}

/// Frame induction statistics
#[derive(Debug, Clone)]
pub struct FrameInductionStats {
    /// Number of unique patterns observed
    pub observed_patterns: usize,
    /// Total observations
    pub total_observations: usize,
    /// Number of induced frames
    pub induced_frames: usize,
    /// Average support per frame
    pub average_support: f64,
}

// ============================================================================
// GROUNDED SYMBOL LEARNING
// ============================================================================

/// A grounded symbol with real-world connections
#[derive(Debug, Clone)]
pub struct GroundedSymbol {
    /// The symbol/word
    pub symbol: String,
    /// Semantic primes
    pub primes: Vec<SemanticPrime>,
    /// Grounding observations
    pub groundings: Vec<SymbolGrounding>,
    /// Confidence in grounding
    pub confidence: f64,
}

/// A grounding observation
#[derive(Debug, Clone)]
pub struct SymbolGrounding {
    /// Type of grounding
    pub grounding_type: GroundingType,
    /// Description
    pub description: String,
    /// When observed
    pub timestamp: std::time::SystemTime,
}

/// Types of symbol grounding
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GroundingType {
    /// Package exists in nixpkgs
    PackageExists,
    /// Service can be enabled
    ServiceExists,
    /// File/path exists
    PathExists,
    /// Command executed successfully
    CommandSucceeded,
    /// Command failed
    CommandFailed,
    /// User provided feedback
    UserFeedback,
}

/// Grounded symbol learning system
#[derive(Debug, Clone)]
pub struct GroundedSymbolLearning {
    /// Known grounded symbols
    symbols: HashMap<String, GroundedSymbol>,
}

impl Default for GroundedSymbolLearning {
    fn default() -> Self {
        Self {
            symbols: HashMap::new(),
        }
    }
}

impl GroundedSymbolLearning {
    /// Create a new grounded symbol learning system
    pub fn new() -> Self {
        Self::default()
    }

    /// Record a grounding observation
    pub fn ground(&mut self, symbol: &str, grounding_type: GroundingType, description: &str) {
        let symbol_lower = symbol.to_lowercase();

        let grounding = SymbolGrounding {
            grounding_type: grounding_type.clone(),
            description: description.to_string(),
            timestamp: std::time::SystemTime::now(),
        };

        if let Some(existing) = self.symbols.get_mut(&symbol_lower) {
            existing.groundings.push(grounding);
            // Increase confidence for positive groundings
            if grounding_type != GroundingType::CommandFailed {
                existing.confidence = (existing.confidence + 0.1).min(1.0);
            } else {
                existing.confidence = (existing.confidence - 0.1).max(0.0);
            }
        } else {
            let confidence = match grounding_type {
                GroundingType::CommandFailed => 0.3,
                _ => 0.6,
            };
            self.symbols.insert(
                symbol_lower.clone(),
                GroundedSymbol {
                    symbol: symbol_lower,
                    primes: vec![], // To be filled by integration
                    groundings: vec![grounding],
                    confidence,
                },
            );
        }
    }

    /// Check if a symbol is grounded
    pub fn is_grounded(&self, symbol: &str) -> bool {
        self.symbols.contains_key(&symbol.to_lowercase())
    }

    /// Get grounding for a symbol
    pub fn get_grounding(&self, symbol: &str) -> Option<&GroundedSymbol> {
        self.symbols.get(&symbol.to_lowercase())
    }

    /// Get all grounded symbols
    pub fn symbols(&self) -> &HashMap<String, GroundedSymbol> {
        &self.symbols
    }

    /// Get statistics
    pub fn stats(&self) -> GroundingStats {
        let total_groundings: usize = self.symbols.values().map(|s| s.groundings.len()).sum();
        let well_grounded = self
            .symbols
            .values()
            .filter(|s| s.confidence >= 0.7)
            .count();

        GroundingStats {
            total_symbols: self.symbols.len(),
            total_groundings,
            well_grounded_symbols: well_grounded,
            average_confidence: if self.symbols.is_empty() {
                0.0
            } else {
                self.symbols.values().map(|s| s.confidence).sum::<f64>() / self.symbols.len() as f64
            },
        }
    }
}

/// Grounding statistics
#[derive(Debug, Clone)]
pub struct GroundingStats {
    /// Total grounded symbols
    pub total_symbols: usize,
    /// Total grounding observations
    pub total_groundings: usize,
    /// Symbols with high confidence grounding
    pub well_grounded_symbols: usize,
    /// Average confidence
    pub average_confidence: f64,
}

// ============================================================================
// UNIFIED SEMANTIC ENRICHMENT ENGINE
// ============================================================================

/// Unified semantic enrichment engine combining all capabilities
#[derive(Debug, Clone)]
pub struct SemanticEnrichmentEngine {
    /// Self-learning vocabulary (includes compositional inference)
    pub vocabulary: SelfLearningVocabulary,
    /// Contextual disambiguator
    pub disambiguator: ContextualDisambiguator,
    /// Prime coherence calculator
    pub coherence: PrimeCoherenceCalculator,
    /// Frame induction
    pub frame_induction: FrameInduction,
    /// Grounded symbol learning
    pub grounding: GroundedSymbolLearning,
}

impl Default for SemanticEnrichmentEngine {
    fn default() -> Self {
        Self {
            vocabulary: SelfLearningVocabulary::new(),
            disambiguator: ContextualDisambiguator::new(),
            coherence: PrimeCoherenceCalculator::new(),
            frame_induction: FrameInduction::new(),
            grounding: GroundedSymbolLearning::new(),
        }
    }
}

impl SemanticEnrichmentEngine {
    /// Create a new semantic enrichment engine
    pub fn new() -> Self {
        Self::default()
    }

    /// Process a sentence and get enriched semantic understanding
    pub fn process(&mut self, input: &str) -> EnrichmentResult {
        let words: Vec<String> = input
            .split_whitespace()
            .map(|w| {
                w.to_lowercase()
                    .trim_matches(|c: char| !c.is_alphanumeric())
                    .to_string()
            })
            .filter(|w| !w.is_empty())
            .collect();

        // Get compositional primes for each word
        let mut word_primes: HashMap<String, Vec<SemanticPrime>> = HashMap::new();
        let mut word_confidences: HashMap<String, f64> = HashMap::new();

        for (i, word) in words.iter().enumerate() {
            let (primes, confidence) = self.vocabulary.get_primes(word);

            // Apply contextual disambiguation
            let context = ContextWindow {
                before: words[..i].to_vec(),
                target: word.clone(),
                after: words[i + 1..].to_vec(),
            };
            let disambiguated = self.disambiguator.disambiguate(&context, &primes);

            word_primes.insert(word.clone(), disambiguated.filtered_primes);
            word_confidences.insert(word.clone(), confidence * disambiguated.confidence);
        }

        // Calculate coherence across word decompositions
        let decompositions: Vec<Vec<SemanticPrime>> = word_primes.values().cloned().collect();
        let coherence_result = self.coherence.calculate_coherence(&decompositions);

        // Collect all primes
        let mut all_primes: Vec<SemanticPrime> = word_primes.values().flatten().cloned().collect();
        let mut seen = HashSet::new();
        all_primes.retain(|p| seen.insert(*p));

        // Find matching induced frames
        let matching_frames: Vec<String> = self
            .frame_induction
            .find_matching_frames(&all_primes)
            .iter()
            .map(|f| f.name.clone())
            .collect();

        // Calculate overall confidence
        let avg_word_confidence = if word_confidences.is_empty() {
            0.0
        } else {
            word_confidences.values().sum::<f64>() / word_confidences.len() as f64
        };
        let overall_confidence = avg_word_confidence * coherence_result.coherence;

        EnrichmentResult {
            input: input.to_string(),
            word_primes,
            word_confidences,
            all_primes,
            coherence: coherence_result,
            matching_frames,
            overall_confidence,
        }
    }

    /// Record feedback for learning
    pub fn record_feedback(
        &mut self,
        input: &str,
        success: bool,
        word_primes: HashMap<String, Vec<SemanticPrime>>,
    ) {
        if success {
            self.vocabulary.record_success(input, word_primes.clone());

            // Also record patterns for frame induction
            let words: Vec<_> = input.split_whitespace().collect();
            if let Some(first_word) = words.first() {
                let all_primes: Vec<SemanticPrime> =
                    word_primes.values().flatten().cloned().collect();
                self.frame_induction.observe(first_word, vec![], all_primes);
            }
        } else {
            self.vocabulary.record_failure(input, word_primes);
        }
    }

    /// Get comprehensive statistics
    pub fn stats(&self) -> EnrichmentStats {
        EnrichmentStats {
            learning: self.vocabulary.stats(),
            frame_induction: self.frame_induction.stats(),
            grounding: self.grounding.stats(),
        }
    }
}

/// Result of semantic enrichment
#[derive(Debug, Clone)]
pub struct EnrichmentResult {
    /// Original input
    pub input: String,
    /// Primes per word
    pub word_primes: HashMap<String, Vec<SemanticPrime>>,
    /// Confidence per word
    pub word_confidences: HashMap<String, f64>,
    /// All unique primes
    pub all_primes: Vec<SemanticPrime>,
    /// Coherence result
    pub coherence: CoherenceResult,
    /// Matching induced frames
    pub matching_frames: Vec<String>,
    /// Overall confidence
    pub overall_confidence: f64,
}

/// Comprehensive enrichment statistics
#[derive(Debug, Clone)]
pub struct EnrichmentStats {
    /// Learning statistics
    pub learning: LearningStats,
    /// Frame induction statistics
    pub frame_induction: FrameInductionStats,
    /// Grounding statistics
    pub grounding: GroundingStats,
}

// ============================================================================
// TESTS
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_morphological_analysis() {
        let analyzer = MorphologicalAnalyzer::new();

        let result = analyzer.analyze("reinstall");
        assert_eq!(result.prefix, Some(MorphemePrefix::Re));
        assert_eq!(result.root, "install");

        let result = analyzer.analyze("uninstallable");
        assert_eq!(result.prefix, Some(MorphemePrefix::Un));
        assert!(result.root.contains("install"));
        assert_eq!(result.suffix, Some(MorphemeSuffix::Able));
    }

    #[test]
    fn test_compositional_inference() {
        let inference = CompositionalInference::new();

        let result = inference.infer("install");
        assert!(!result.primes.is_empty());
        assert!(result.primes.contains(&SemanticPrime::Do));

        let result = inference.infer("reinstall");
        assert!(result.primes.contains(&SemanticPrime::Before));
        assert!(result.primes.contains(&SemanticPrime::Do));
    }

    #[test]
    fn test_contextual_disambiguation() {
        let disambiguator = ContextualDisambiguator::new();

        let context = ContextWindow {
            before: vec!["run".into(), "the".into()],
            target: "program".into(),
            after: vec![],
        };

        let candidate_primes = vec![
            SemanticPrime::Do,
            SemanticPrime::Something,
            SemanticPrime::Move,
        ];
        let result = disambiguator.disambiguate(&context, &candidate_primes);

        // Should boost DO and suppress MOVE in computing context
        assert!(result.filtered_primes.contains(&SemanticPrime::Do));
    }

    #[test]
    fn test_prime_coherence() {
        let calculator = PrimeCoherenceCalculator::new();

        // High coherence: similar decompositions
        let decompositions = vec![
            vec![SemanticPrime::Do, SemanticPrime::Something],
            vec![
                SemanticPrime::Do,
                SemanticPrime::Something,
                SemanticPrime::Move,
            ],
        ];
        let result = calculator.calculate_coherence(&decompositions);
        assert!(result.coherence > 0.5);

        // Low coherence: different decompositions
        let decompositions = vec![
            vec![SemanticPrime::Do],
            vec![SemanticPrime::See, SemanticPrime::Know],
        ];
        let result = calculator.calculate_coherence(&decompositions);
        assert!(result.coherence < 0.5);
    }

    #[test]
    fn test_self_learning() {
        let mut vocab = SelfLearningVocabulary::new();

        // Initially unknown word
        let (primes, conf) = vocab.get_primes("frobulate");
        assert!(conf < 1.0); // Low confidence for unknown

        // Record success
        let mut word_primes = HashMap::new();
        word_primes.insert(
            "frobulate".into(),
            vec![SemanticPrime::Do, SemanticPrime::Something],
        );
        vocab.record_success("frobulate the thing", word_primes);

        // Should have learned
        let (primes, conf) = vocab.get_primes("frobulate");
        assert!(!primes.is_empty());
    }

    #[test]
    fn test_unified_engine() {
        let mut engine = SemanticEnrichmentEngine::new();

        let result = engine.process("install firefox");
        assert!(!result.all_primes.is_empty());
        assert!(result.overall_confidence > 0.0);

        // Check word-level decomposition
        assert!(result.word_primes.contains_key("install"));
        assert!(
            result
                .word_primes
                .get("install")
                .unwrap()
                .contains(&SemanticPrime::Do)
        );
    }
}