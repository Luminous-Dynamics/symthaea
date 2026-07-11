// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Grounded creativity benchmarks — RAT + DAT over *real* word embeddings.
//!
//! # The honest-grounding contract
//!
//! The sibling creativity benchmarks in this domain are HDC-algebra sanity
//! checks: their stimulus adapters *plant* the cue↔solution association they
//! then recover (see the scope-honesty note in `creativity/mod.rs`). This
//! module is the first honestly-grounded counterpart. Its contract:
//!
//! - **No planted links.** The only knowledge source is an externally-trained
//!   embedding space supplied by the caller through [`WordEmbedder`]. Nothing
//!   in this module encodes which solution belongs to which cues.
//! - **What it measures**: whether an embedding space places the normed RAT
//!   solution word near the centroid of its three cue words
//!   ([`RatGroundedBenchmark`]), and the semantic spread of an arbitrary word
//!   list under the published DAT scoring rule ([`score_dat`]). Both are
//!   properties of the *embedding space* (plus, for DAT, of whoever generated
//!   the word list).
//! - **What it does NOT measure (yet)**: Symthaea's own generative creativity.
//!   Until Symthaea's generation side is wired in (producing DAT word lists /
//!   RAT guesses itself), these numbers say nothing about her creative
//!   capability — they characterize the grounding substrate.
//! - **Mock numbers are meaningless.** [`DeterministicHashEmbedder`] carries
//!   zero semantics; any accuracy or DAT score computed with it is noise and
//!   must never be reported as a result. It exists solely so the harness
//!   logic is unit-testable without a 100+ MB embedding file.
//!
//! # Why a GloVe file loader instead of `symthaea-embeddings`
//!
//! `symthaea-embeddings::Qwen3Embedder` was considered and rejected for now:
//! its default config is `use_simulated: true` (hash-quality vectors, no
//! semantics), and when a real model fails to load it *silently falls back*
//! to simulated mode (`qwen3/mod.rs`). For a module whose whole point is
//! honest grounding, a silent fake-semantics fallback is disqualifying.
//! Getting real vectors requires the `burn` + `burn-hub` features (the Burn
//! ML framework, tokenizers, and an HF-hub download) — a heavy dependency
//! surface for a benchmark crate. [`GloVeFileEmbedder`] instead parses the
//! standard GloVe text format with zero new dependencies, is deterministic
//! and offline once the file is downloaded, and *fails loudly* (returns
//! `Err` / `None`) rather than degrading to fake semantics. A
//! `symthaea-embeddings` adapter can be added later behind a feature once
//! the silent-fallback hazard is addressed.
//!
//! # Why not `PsychBenchmark`
//!
//! The crate's [`PsychBenchmark`](crate::harness::PsychBenchmark) trait is
//! `fn run(&self, config: &BenchmarkConfig) -> BenchmarkResult` — config-only,
//! with no channel for injecting an external grounding source. Implementing
//! it would force this benchmark to construct an embedder internally, and the
//! only embedder that is always constructible is the semantics-free mock —
//! which would produce an official-looking `BenchmarkResult` from meaningless
//! vectors, exactly the failure mode this module exists to eliminate. The
//! grounded API is therefore standalone:
//! [`RatGroundedBenchmark::run`]`(&mut impl WordEmbedder, &RatItemSet) -> RatReport`.
//!
//! # Obtaining real data
//!
//! - **Embeddings**: download GloVe from <https://nlp.stanford.edu/projects/glove/>
//!   (e.g. `glove.6B.zip`; `glove.6B.300d.txt` is a good default) and load it
//!   with [`GloVeFileEmbedder::from_path`].
//! - **RAT norms**: the published 144-item normed compound-remote-associate
//!   set is Bowden & Jung-Beeman (2003), *Behavior Research Methods,
//!   Instruments, & Computers* 35, 634–639 (doi:10.3758/BF03195543). Convert
//!   the published table to TSV (`cue1<TAB>cue2<TAB>cue3<TAB>solution`, one
//!   item per line) and load with [`RatItemSet::from_tsv_file`]. The built-in
//!   [`RatItemSet::demo`] is a small canonical subset for smoke-testing only —
//!   do not report demo-set accuracy as "RAT performance".
//! - **DAT**: Olson et al. (2021), *PNAS* 118(25) e2022340118 — "Naming
//!   unrelated words predicts creativity". Scoring implemented per the paper:
//!   mean pairwise cosine *distance* of the first 7 valid words, × 100.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io;
use std::path::Path;

// ---------------------------------------------------------------------------
// Embedder abstraction
// ---------------------------------------------------------------------------

/// A source of dense word vectors.
///
/// `embed` returns `None` for out-of-vocabulary words; implementations must
/// **never** substitute fabricated vectors for unknown words (that would
/// silently un-ground the benchmark). `&mut self` allows caching /
/// lazy-loading implementations.
pub trait WordEmbedder {
    /// Return the embedding for `word`, or `None` if the word is not in the
    /// embedder's vocabulary.
    fn embed(&mut self, word: &str) -> Option<Vec<f32>>;

    /// Provided helper: cosine similarity between two words, `None` if either
    /// is out-of-vocabulary or a vector is degenerate (zero norm).
    fn word_similarity(&mut self, a: &str, b: &str) -> Option<f32> {
        let va = self.embed(a)?;
        let vb = self.embed(b)?;
        cosine_similarity(&va, &vb)
    }
}

/// Cosine similarity between two vectors. `None` if lengths differ or either
/// vector has (near-)zero norm.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> Option<f32> {
    if a.len() != b.len() || a.is_empty() {
        return None;
    }
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += f64::from(*x) * f64::from(*y);
        na += f64::from(*x) * f64::from(*x);
        nb += f64::from(*y) * f64::from(*y);
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom < 1e-12 {
        return None;
    }
    Some((dot / denom) as f32)
}

/// Element-wise mean of a set of equal-length vectors. `None` if empty or
/// lengths mismatch.
fn centroid(vectors: &[Vec<f32>]) -> Option<Vec<f32>> {
    let first = vectors.first()?;
    let dim = first.len();
    if vectors.iter().any(|v| v.len() != dim) {
        return None;
    }
    let n = vectors.len() as f32;
    let mut out = vec![0.0f32; dim];
    for v in vectors {
        for (o, x) in out.iter_mut().zip(v.iter()) {
            *o += *x / n;
        }
    }
    Some(out)
}

// ---------------------------------------------------------------------------
// Mock embedder (test/harness plumbing only)
// ---------------------------------------------------------------------------

/// Deterministic hash-based embedder — **test/mock only**.
///
/// Maps each word to a pseudo-random unit-ish vector derived from a hash of
/// its bytes. It carries **no semantics whatsoever**: "cat" is no closer to
/// "dog" than to "carburetor". It exists so the RAT/DAT harness logic
/// (parsing, ranking, edge cases) is testable without shipping or downloading
/// an embedding file. **Any accuracy or DAT number produced with this
/// embedder is meaningless and must never be reported as a result.**
#[derive(Debug, Clone)]
pub struct DeterministicHashEmbedder {
    dim: usize,
    seed: u64,
}

impl Default for DeterministicHashEmbedder {
    fn default() -> Self {
        Self {
            dim: 64,
            seed: 0x5EED,
        }
    }
}

impl DeterministicHashEmbedder {
    /// Create a mock embedder with the given dimensionality and seed.
    pub fn new(dim: usize, seed: u64) -> Self {
        Self { dim, seed }
    }
}

impl WordEmbedder for DeterministicHashEmbedder {
    fn embed(&mut self, word: &str) -> Option<Vec<f32>> {
        // FNV-1a over the bytes, then splitmix64 per component.
        let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ self.seed;
        for b in word.as_bytes() {
            h ^= u64::from(*b);
            h = h.wrapping_mul(0x0000_0100_0000_01B3);
        }
        let mut out = Vec::with_capacity(self.dim);
        let mut state = h;
        for _ in 0..self.dim {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^= z >> 31;
            // Map to [-1, 1)
            out.push((z >> 40) as f32 / (1u64 << 23) as f32 - 1.0);
        }
        Some(out)
    }
}

// ---------------------------------------------------------------------------
// GloVe file embedder (the real-grounding path)
// ---------------------------------------------------------------------------

/// Real-embedding adapter: loads a standard GloVe `.txt` file
/// (`word v1 v2 ... vD`, space-separated, one word per line).
///
/// Download from <https://nlp.stanford.edu/projects/glove/> — e.g.
/// `glove.6B.zip` and use `glove.6B.300d.txt` (or `glove.6B.50d.txt` for a
/// smaller memory footprint). Lookups are lowercased, matching GloVe's
/// lowercase vocabulary. Construction is fallible (I/O + parse errors) and
/// requires the file to have been downloaded beforehand — there is no
/// network access and, deliberately, **no fallback to synthetic vectors**.
pub struct GloVeFileEmbedder {
    vectors: BTreeMap<String, Vec<f32>>,
    dim: usize,
}

impl fmt::Debug for GloVeFileEmbedder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("GloVeFileEmbedder")
            .field("words", &self.vectors.len())
            .field("dim", &self.dim)
            .finish()
    }
}

impl GloVeFileEmbedder {
    /// Load every vector in the file.
    pub fn from_path(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::from_path_with_limit(path, usize::MAX)
    }

    /// Load at most `max_words` vectors (GloVe files are frequency-ordered,
    /// so a prefix is the most-frequent subset — useful to bound memory).
    pub fn from_path_with_limit(path: impl AsRef<Path>, max_words: usize) -> io::Result<Self> {
        let text = fs::read_to_string(path.as_ref())?;
        Self::from_glove_text(&text, max_words)
    }

    /// Parse GloVe-format text (`word v1 v2 ... vD` per line).
    pub fn from_glove_text(text: &str, max_words: usize) -> io::Result<Self> {
        let mut vectors = BTreeMap::new();
        let mut dim = 0usize;
        for (line_no, line) in text.lines().enumerate() {
            if vectors.len() >= max_words {
                break;
            }
            let line = line.trim_end();
            if line.is_empty() {
                continue;
            }
            let mut parts = line.split(' ');
            let word = parts.next().unwrap_or_default();
            let vec: Result<Vec<f32>, _> = parts.map(str::parse::<f32>).collect();
            let vec = vec.map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("GloVe parse error on line {}: {e}", line_no + 1),
                )
            })?;
            if vec.is_empty() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("GloVe line {} has no vector components", line_no + 1),
                ));
            }
            if dim == 0 {
                dim = vec.len();
            } else if vec.len() != dim {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "GloVe line {}: dimension {} != expected {dim}",
                        line_no + 1,
                        vec.len()
                    ),
                ));
            }
            vectors.insert(word.to_lowercase(), vec);
        }
        if vectors.is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "GloVe file contained no vectors",
            ));
        }
        Ok(Self { vectors, dim })
    }

    /// Number of words loaded.
    pub fn len(&self) -> usize {
        self.vectors.len()
    }

    /// Whether the vocabulary is empty (cannot happen post-construction).
    pub fn is_empty(&self) -> bool {
        self.vectors.is_empty()
    }

    /// Embedding dimensionality.
    pub fn dim(&self) -> usize {
        self.dim
    }
}

impl WordEmbedder for GloVeFileEmbedder {
    fn embed(&mut self, word: &str) -> Option<Vec<f32>> {
        self.vectors.get(&word.to_lowercase()).cloned()
    }
}

// ---------------------------------------------------------------------------
// RAT (Remote Associates Test), grounded
// ---------------------------------------------------------------------------

/// One compound-remote-associate item: three cue words and the normed
/// solution word.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RatItem {
    /// The three cue words.
    pub cues: [String; 3],
    /// The normed solution word.
    pub solution: String,
}

/// A set of RAT items plus loaders.
#[derive(Debug, Clone, Default)]
pub struct RatItemSet {
    /// The items.
    pub items: Vec<RatItem>,
}

impl RatItemSet {
    /// Load items from a TSV file: `cue1<TAB>cue2<TAB>cue3<TAB>solution`,
    /// one item per line. Blank lines and lines starting with `#` are
    /// skipped. Use this to load the published Bowden & Jung-Beeman (2003)
    /// 144-item normed set (see module docs for the citation and format).
    pub fn from_tsv_file(path: impl AsRef<Path>) -> io::Result<Self> {
        let text = fs::read_to_string(path.as_ref())?;
        Self::from_tsv_str(&text)
    }

    /// Parse TSV content (same format as [`Self::from_tsv_file`]).
    pub fn from_tsv_str(text: &str) -> io::Result<Self> {
        let mut items = Vec::new();
        for (line_no, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let fields: Vec<&str> = line.split('\t').map(str::trim).collect();
            if fields.len() != 4 || fields.iter().any(|f| f.is_empty()) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "RAT TSV line {}: expected 4 non-empty tab-separated fields \
                         (cue1 cue2 cue3 solution), got {}",
                        line_no + 1,
                        fields.len()
                    ),
                ));
            }
            items.push(RatItem {
                cues: [
                    fields[0].to_lowercase(),
                    fields[1].to_lowercase(),
                    fields[2].to_lowercase(),
                ],
                solution: fields[3].to_lowercase(),
            });
        }
        Ok(Self { items })
    }

    /// Serialize back to TSV (round-trips with [`Self::from_tsv_str`]).
    pub fn to_tsv_string(&self) -> String {
        let mut out = String::new();
        for item in &self.items {
            out.push_str(&format!(
                "{}\t{}\t{}\t{}\n",
                item.cues[0], item.cues[1], item.cues[2], item.solution
            ));
        }
        out
    }

    /// **Demo subset only** — a handful of canonical compound-remote-associate
    /// items that are widely reproduced in the literature (Mednick, 1962;
    /// Bowden & Jung-Beeman, 2003). This is NOT the published normed set;
    /// accuracy on it is a smoke test, not a reportable RAT result. For real
    /// evaluation load the 144-item Bowden & Jung-Beeman (2003) norms via
    /// [`Self::from_tsv_file`].
    pub fn demo() -> Self {
        // Each item below is a canonical CRA example reproduced across many
        // published papers and reviews of the Bowden & Jung-Beeman (2003)
        // normed set. Deliberately small: recalling the full 144-item table
        // from memory would risk fabricating psychometric data.
        let raw: [([&str; 3], &str); 12] = [
            // The two textbook examples used in virtually every CRA paper:
            (["cottage", "swiss", "cake"], "cheese"),
            (["cream", "skate", "water"], "ice"),
            // Canonical B&JB (2003) items widely reproduced in the insight
            // literature:
            (["show", "life", "row"], "boat"),
            (["night", "wrist", "stop"], "watch"),
            (["duck", "fold", "dollar"], "bill"),
            (["rocking", "wheel", "high"], "chair"),
            (["fountain", "baking", "pop"], "soda"),
            (["aid", "rubber", "wagon"], "band"),
            (["widow", "bite", "monkey"], "spider"),
            (["pine", "crab", "sauce"], "apple"),
            (["fish", "mine", "rush"], "gold"),
            (["dew", "comb", "bee"], "honey"),
        ];
        Self {
            items: raw
                .iter()
                .map(|(cues, sol)| RatItem {
                    cues: [
                        cues[0].to_string(),
                        cues[1].to_string(),
                        cues[2].to_string(),
                    ],
                    solution: (*sol).to_string(),
                })
                .collect(),
        }
    }

    /// The distractor vocabulary used for ranking: the union of all solution
    /// words in the set (deduplicated, sorted).
    pub fn solution_vocabulary(&self) -> Vec<String> {
        let set: BTreeSet<&str> = self.items.iter().map(|i| i.solution.as_str()).collect();
        set.into_iter().map(str::to_string).collect()
    }
}

/// Why a RAT item could not be scored.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RatSkipReason {
    /// One or more cue words were out of the embedder's vocabulary.
    CueOutOfVocabulary(String),
    /// The solution word was out of the embedder's vocabulary.
    SolutionOutOfVocabulary,
}

/// Per-item outcome of a grounded RAT run.
#[derive(Debug, Clone)]
pub struct RatItemResult {
    /// Index of the item in the input set.
    pub item_index: usize,
    /// The solution word.
    pub solution: String,
    /// 1-based rank of the true solution among the vocabulary (1 = best),
    /// or `None` if the item was skipped.
    pub rank: Option<usize>,
    /// Populated when the item was skipped.
    pub skip_reason: Option<RatSkipReason>,
}

/// Aggregate report for a grounded RAT run.
#[derive(Debug, Clone)]
pub struct RatReport {
    /// Total items in the set.
    pub total_items: usize,
    /// Items that could be scored (all cues + solution embeddable).
    pub scored_items: usize,
    /// Scored items where the true solution ranked 1st.
    pub top1_correct: usize,
    /// Scored items where the true solution ranked in the top 5.
    pub top5_correct: usize,
    /// Size of the ranking vocabulary (embeddable solution words).
    pub vocabulary_size: usize,
    /// Per-item detail.
    pub per_item: Vec<RatItemResult>,
}

impl RatReport {
    /// Top-1 accuracy over scored items. `None` if nothing was scored.
    pub fn top1_accuracy(&self) -> Option<f64> {
        (self.scored_items > 0).then(|| self.top1_correct as f64 / self.scored_items as f64)
    }

    /// Top-5 accuracy over scored items. `None` if nothing was scored.
    pub fn top5_accuracy(&self) -> Option<f64> {
        (self.scored_items > 0).then(|| self.top5_correct as f64 / self.scored_items as f64)
    }
}

/// Grounded Remote Associates Test.
///
/// For each item: embed the three cues, take their centroid, and rank every
/// word in the set's solution vocabulary by cosine similarity to that
/// centroid. The item is top-1 correct if the true solution outranks all
/// other vocabulary words. There are **no planted links** — the embedding
/// space is the only knowledge source, so accuracy is a property of the
/// embedding space's associative structure.
///
/// Standalone API rather than [`PsychBenchmark`](crate::harness::PsychBenchmark)
/// — see the module docs for why the config-only trait shape does not fit an
/// externally-grounded evaluation.
pub struct RatGroundedBenchmark;

impl RatGroundedBenchmark {
    /// Run the grounded RAT over `items` using `embedder` as the sole
    /// knowledge source.
    pub fn run(embedder: &mut impl WordEmbedder, items: &RatItemSet) -> RatReport {
        // Embed the ranking vocabulary once.
        let vocab_words = items.solution_vocabulary();
        let vocab: Vec<(String, Vec<f32>)> = vocab_words
            .into_iter()
            .filter_map(|w| embedder.embed(&w).map(|v| (w, v)))
            .collect();

        let mut report = RatReport {
            total_items: items.items.len(),
            scored_items: 0,
            top1_correct: 0,
            top5_correct: 0,
            vocabulary_size: vocab.len(),
            per_item: Vec::with_capacity(items.items.len()),
        };

        for (idx, item) in items.items.iter().enumerate() {
            // Skip if the solution itself is not embeddable (it can then
            // never appear in the ranking).
            if !vocab.iter().any(|(w, _)| *w == item.solution) {
                report.per_item.push(RatItemResult {
                    item_index: idx,
                    solution: item.solution.clone(),
                    rank: None,
                    skip_reason: Some(RatSkipReason::SolutionOutOfVocabulary),
                });
                continue;
            }
            // Embed cues; skip on any OOV cue.
            let mut cue_vecs = Vec::with_capacity(3);
            let mut oov_cue = None;
            for cue in &item.cues {
                match embedder.embed(cue) {
                    Some(v) => cue_vecs.push(v),
                    None => {
                        oov_cue = Some(cue.clone());
                        break;
                    }
                }
            }
            if let Some(cue) = oov_cue {
                report.per_item.push(RatItemResult {
                    item_index: idx,
                    solution: item.solution.clone(),
                    rank: None,
                    skip_reason: Some(RatSkipReason::CueOutOfVocabulary(cue)),
                });
                continue;
            }
            let Some(center) = centroid(&cue_vecs) else {
                report.per_item.push(RatItemResult {
                    item_index: idx,
                    solution: item.solution.clone(),
                    rank: None,
                    skip_reason: Some(RatSkipReason::CueOutOfVocabulary(
                        "cue embedding dimension mismatch".to_string(),
                    )),
                });
                continue;
            };

            // Rank vocabulary by cosine similarity to the cue centroid.
            let mut sims: Vec<(&str, f32)> = vocab
                .iter()
                .map(|(w, v)| {
                    (
                        w.as_str(),
                        cosine_similarity(&center, v).unwrap_or(f32::NEG_INFINITY),
                    )
                })
                .collect();
            sims.sort_by(|(_, a), (_, b)| b.total_cmp(a));
            let rank = sims
                .iter()
                .position(|(w, _)| *w == item.solution)
                .map(|p| p + 1)
                .expect("solution verified present in vocabulary above");

            report.scored_items += 1;
            if rank == 1 {
                report.top1_correct += 1;
            }
            if rank <= 5 {
                report.top5_correct += 1;
            }
            report.per_item.push(RatItemResult {
                item_index: idx,
                solution: item.solution.clone(),
                rank: Some(rank),
                skip_reason: None,
            });
        }
        report
    }
}

// ---------------------------------------------------------------------------
// DAT (Divergent Association Task)
// ---------------------------------------------------------------------------

/// Number of words the published DAT scoring rule uses (first 7 valid of the
/// 10 the participant provides).
pub const DAT_WORDS_REQUIRED: usize = 7;

/// Result of scoring a word list with the DAT rule.
#[derive(Debug, Clone)]
pub struct DatScore {
    /// The DAT score: mean pairwise cosine *distance* of the first
    /// [`DAT_WORDS_REQUIRED`] embeddable words, × 100 (Olson et al., 2021).
    /// Human norms (with GloVe-840B embeddings): mean ≈ 78, roughly 65–90
    /// range. Scores from a different embedding space are internally
    /// comparable but not directly comparable to the published norms.
    pub score: f64,
    /// The words actually used (first 7 embeddable).
    pub words_used: Vec<String>,
    /// Words skipped because the embedder had no vector for them.
    pub skipped_words: Vec<String>,
}

/// Score a word list with the Divergent Association Task rule
/// (Olson et al., 2021, PNAS 118(25) e2022340118): take the first
/// [`DAT_WORDS_REQUIRED`] words the embedder knows, compute cosine distance
/// (`1 − cosine similarity`) for all 21 pairs, and return the mean × 100.
/// Returns `None` if fewer than 7 words are embeddable.
///
/// This scores **any** word list — human-typed, sampled, or (later)
/// Symthaea-generated. Note: the published pipeline also validates words
/// (single English nouns, spell-checked); that validation is the caller's
/// responsibility here — this function only enforces embeddability.
///
/// # Example
///
/// ```
/// use symthaea_psych_bench::benchmarks::creativity::grounded::{
///     score_dat, DeterministicHashEmbedder,
/// };
///
/// // With a REAL embedder (e.g. GloVeFileEmbedder), a semantically spread
/// // list like the one below scores far higher than a clustered list:
/// let divergent = ["arm", "eyes", "feather", "steel", "tumor", "unicorn", "vodka"];
/// let clustered = ["cat", "dog", "hamster", "rabbit", "ferret", "parrot", "goldfish"];
///
/// // The mock embedder proves only the machinery, not semantics — its
/// // numbers are meaningless (see its docs):
/// let mut mock = DeterministicHashEmbedder::default();
/// assert!(score_dat(&divergent, &mut mock).is_some());
/// assert!(score_dat(&clustered, &mut mock).is_some());
/// ```
pub fn score_dat(words: &[&str], embedder: &mut impl WordEmbedder) -> Option<DatScore> {
    let mut used: Vec<(String, Vec<f32>)> = Vec::with_capacity(DAT_WORDS_REQUIRED);
    let mut skipped = Vec::new();
    for w in words {
        if used.len() >= DAT_WORDS_REQUIRED {
            break;
        }
        let key = w.trim().to_lowercase();
        if key.is_empty() {
            continue;
        }
        match embedder.embed(&key) {
            Some(v) => used.push((key, v)),
            None => skipped.push(key),
        }
    }
    if used.len() < DAT_WORDS_REQUIRED {
        return None;
    }

    let mut total = 0.0f64;
    let mut pairs = 0usize;
    for i in 0..used.len() {
        for j in (i + 1)..used.len() {
            let sim = cosine_similarity(&used[i].1, &used[j].1)?;
            total += f64::from(1.0 - sim);
            pairs += 1;
        }
    }
    debug_assert_eq!(pairs, DAT_WORDS_REQUIRED * (DAT_WORDS_REQUIRED - 1) / 2);
    Some(DatScore {
        score: total / pairs as f64 * 100.0,
        words_used: used.into_iter().map(|(w, _)| w).collect(),
        skipped_words: skipped,
    })
}

// ---------------------------------------------------------------------------
// Tests (mock embedder — machinery only, no semantic claims)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Test-only embedder with hand-assigned vectors, so tests can control
    /// geometry exactly.
    struct FixtureEmbedder(HashMap<String, Vec<f32>>);

    impl FixtureEmbedder {
        fn new(entries: &[(&str, Vec<f32>)]) -> Self {
            Self(
                entries
                    .iter()
                    .map(|(w, v)| ((*w).to_string(), v.clone()))
                    .collect(),
            )
        }
    }

    impl WordEmbedder for FixtureEmbedder {
        fn embed(&mut self, word: &str) -> Option<Vec<f32>> {
            self.0.get(word).cloned()
        }
    }

    #[test]
    fn tsv_parse_round_trip() {
        let set = RatItemSet::demo();
        let tsv = set.to_tsv_string();
        let reparsed = RatItemSet::from_tsv_str(&tsv).expect("round-trip parse");
        assert_eq!(set.items, reparsed.items);
    }

    #[test]
    fn tsv_skips_comments_and_blanks_rejects_malformed() {
        let ok = "# comment\n\ncottage\tswiss\tcake\tcheese\n";
        let set = RatItemSet::from_tsv_str(ok).unwrap();
        assert_eq!(set.items.len(), 1);
        assert_eq!(set.items[0].solution, "cheese");

        let bad = "cottage\tswiss\tcake\n"; // only 3 fields
        assert!(RatItemSet::from_tsv_str(bad).is_err());
    }

    #[test]
    fn demo_set_loads_and_is_sane() {
        let set = RatItemSet::demo();
        assert!(
            (8..=12).contains(&set.items.len()),
            "demo subset should stay small (8-12 items), got {}",
            set.items.len()
        );
        for item in &set.items {
            assert!(item.cues.iter().all(|c| !c.is_empty()));
            assert!(!item.solution.is_empty());
        }
    }

    #[test]
    fn demo_set_has_no_duplicate_solutions() {
        // The ranking vocabulary is the union of solutions; duplicates would
        // silently shrink it and inflate accuracy.
        let set = RatItemSet::demo();
        let vocab = set.solution_vocabulary();
        assert_eq!(
            vocab.len(),
            set.items.len(),
            "duplicate solution words in demo set"
        );
    }

    #[test]
    fn rat_ranks_centroid_aligned_solution_top1() {
        // Geometry fixture: solution vector == cue centroid, all other
        // vocabulary words orthogonal to it. The true solution must rank 1.
        let tsv = "red\tgreen\tblue\ttarget\nfoo\tbar\tbaz\tother\nqux\tquux\tcorge\tthird\n";
        let set = RatItemSet::from_tsv_str(tsv).unwrap();
        let mut embedder = FixtureEmbedder::new(&[
            ("red", vec![1.0, 0.0, 0.0, 0.0]),
            ("green", vec![0.0, 1.0, 0.0, 0.0]),
            ("blue", vec![0.0, 0.0, 1.0, 0.0]),
            // centroid of the cues, exactly:
            ("target", vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]),
            // orthogonal to the centroid:
            ("other", vec![0.0, 0.0, 0.0, 1.0]),
            ("third", vec![1.0, -1.0, 0.0, 0.0]),
            // second item's cues resolve; third item's cues are OOV
            ("foo", vec![0.0, 0.0, 0.0, 1.0]),
            ("bar", vec![0.0, 0.0, 0.0, 1.0]),
            ("baz", vec![0.0, 0.0, 0.0, 1.0]),
        ]);
        let report = RatGroundedBenchmark::run(&mut embedder, &set);
        assert_eq!(report.total_items, 3);
        assert_eq!(report.vocabulary_size, 3);
        // Item 0: solution at centroid → rank 1.
        assert_eq!(report.per_item[0].rank, Some(1));
        // Item 1: cues all point at "other"'s axis → "other" ranks 1 too.
        assert_eq!(report.per_item[1].rank, Some(1));
        // Item 2: cues OOV → skipped.
        assert!(matches!(
            report.per_item[2].skip_reason,
            Some(RatSkipReason::CueOutOfVocabulary(_))
        ));
        assert_eq!(report.scored_items, 2);
        assert_eq!(report.top1_correct, 2);
        assert_eq!(report.top1_accuracy(), Some(1.0));
    }

    #[test]
    fn rat_skips_oov_solution() {
        let tsv = "red\tgreen\tblue\tmissing\n";
        let set = RatItemSet::from_tsv_str(tsv).unwrap();
        let mut embedder = FixtureEmbedder::new(&[
            ("red", vec![1.0, 0.0]),
            ("green", vec![0.0, 1.0]),
            ("blue", vec![1.0, 1.0]),
        ]);
        let report = RatGroundedBenchmark::run(&mut embedder, &set);
        assert_eq!(report.scored_items, 0);
        assert_eq!(
            report.per_item[0].skip_reason,
            Some(RatSkipReason::SolutionOutOfVocabulary)
        );
        assert_eq!(report.top1_accuracy(), None);
    }

    #[test]
    fn rat_runs_on_demo_set_with_mock_embedder() {
        // Machinery smoke test only — the mock has no semantics, so the
        // accuracy value here is meaningless by construction and asserted
        // only to be a valid probability.
        let mut mock = DeterministicHashEmbedder::default();
        let report = RatGroundedBenchmark::run(&mut mock, &RatItemSet::demo());
        assert_eq!(report.total_items, report.scored_items);
        let acc = report.top1_accuracy().unwrap();
        assert!((0.0..=1.0).contains(&acc));
    }

    #[test]
    fn dat_identical_words_score_zero() {
        // Seven copies of the same word: every pairwise distance is 0.
        let words = ["same"; 7];
        let mut embedder = FixtureEmbedder::new(&[("same", vec![0.3, -0.7, 0.2])]);
        let score = score_dat(&words, &mut embedder).expect("7 embeddable words");
        assert!(score.score.abs() < 1e-4, "got {}", score.score);
        assert_eq!(score.words_used.len(), 7);
    }

    #[test]
    fn dat_orthogonal_words_score_high() {
        // Seven mutually orthogonal vectors: every pairwise cosine is 0,
        // distance 1 → score 100.
        let entries: Vec<(String, Vec<f32>)> = (0..7)
            .map(|i| {
                let mut v = vec![0.0f32; 7];
                v[i] = 1.0;
                (format!("w{i}"), v)
            })
            .collect();
        let refs: Vec<(&str, Vec<f32>)> = entries
            .iter()
            .map(|(w, v)| (w.as_str(), v.clone()))
            .collect();
        let mut embedder = FixtureEmbedder::new(&refs);
        let words: Vec<&str> = entries.iter().map(|(w, _)| w.as_str()).collect();
        let score = score_dat(&words, &mut embedder).expect("7 embeddable words");
        assert!((score.score - 100.0).abs() < 1e-3, "got {}", score.score);
    }

    #[test]
    fn dat_fewer_than_seven_embeddable_is_none() {
        let mut embedder = FixtureEmbedder::new(&[("a", vec![1.0, 0.0]), ("b", vec![0.0, 1.0])]);
        // 8 words offered but only 2 embeddable.
        let words = ["a", "b", "x", "y", "z", "p", "q", "r"];
        assert!(score_dat(&words, &mut embedder).is_none());
        // Exactly 7 embeddable works even with extra OOV words in front.
        let mut full = DeterministicHashEmbedder::default();
        let ten = [
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
        ];
        let s = score_dat(&ten, &mut full).unwrap();
        assert_eq!(s.words_used.len(), DAT_WORDS_REQUIRED);
        // Only the first 7 are used (published rule).
        assert_eq!(
            s.words_used,
            ten[..7].iter().map(|w| w.to_string()).collect::<Vec<_>>()
        );
    }

    #[test]
    fn dat_skips_oov_and_uses_next_words() {
        let mut embedder = DeterministicHashEmbedder::default();
        // Hash embedder embeds everything; wrap to blocklist one word.
        struct Blocklist<'a>(&'a mut DeterministicHashEmbedder);
        impl WordEmbedder for Blocklist<'_> {
            fn embed(&mut self, word: &str) -> Option<Vec<f32>> {
                (word != "blocked").then(|| self.0.embed(word)).flatten()
            }
        }
        let words = ["blocked", "a", "b", "c", "d", "e", "f", "g"];
        let s = score_dat(&words, &mut Blocklist(&mut embedder)).unwrap();
        assert_eq!(s.skipped_words, vec!["blocked".to_string()]);
        assert_eq!(s.words_used.len(), 7);
        assert!(!s.words_used.contains(&"blocked".to_string()));
    }

    #[test]
    fn hash_embedder_is_deterministic_and_nonsemantic_by_contract() {
        let mut e1 = DeterministicHashEmbedder::default();
        let mut e2 = DeterministicHashEmbedder::default();
        assert_eq!(e1.embed("cheese"), e2.embed("cheese"));
        // Different words get different vectors.
        assert_ne!(e1.embed("cheese"), e1.embed("ice"));
        // A word is maximally similar to itself.
        let sim = e1.word_similarity("cheese", "cheese").unwrap();
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn glove_text_parser_works_and_validates() {
        let text = "cat 0.1 0.2 0.3\ndog 0.2 0.1 0.4\n";
        let mut g = GloVeFileEmbedder::from_glove_text(text, usize::MAX).unwrap();
        assert_eq!(g.len(), 2);
        assert_eq!(g.dim(), 3);
        assert_eq!(g.embed("CAT"), Some(vec![0.1, 0.2, 0.3])); // lowercased lookup
        assert_eq!(g.embed("bird"), None); // no fabricated fallback

        // Dimension mismatch rejected.
        assert!(GloVeFileEmbedder::from_glove_text("a 0.1 0.2\nb 0.1\n", usize::MAX).is_err());
        // Non-numeric rejected.
        assert!(GloVeFileEmbedder::from_glove_text("a x y\n", usize::MAX).is_err());
        // Empty rejected.
        assert!(GloVeFileEmbedder::from_glove_text("", usize::MAX).is_err());
        // Word limit respected (frequency-ordered prefix).
        let limited = GloVeFileEmbedder::from_glove_text(text, 1).unwrap();
        assert_eq!(limited.len(), 1);
    }

    #[test]
    fn cosine_similarity_edge_cases() {
        assert_eq!(cosine_similarity(&[1.0], &[1.0, 2.0]), None); // length mismatch
        assert_eq!(cosine_similarity(&[], &[]), None); // empty
        assert_eq!(cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]), None); // zero norm
        let s = cosine_similarity(&[1.0, 0.0], &[0.0, 1.0]).unwrap();
        assert!(s.abs() < 1e-6);
    }
}
