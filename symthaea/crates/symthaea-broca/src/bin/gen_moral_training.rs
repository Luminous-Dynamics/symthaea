// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generate moral/ethics training data from real datasets with epistemic cube channels.
//!
//! Loads examples from four moral datasets and derives 4D epistemic cube coordinates
//! based on the dataset type and moral complexity of each example:
//!
//! - Hendrycks ETHICS: E3 empirical, N1 interpersonal, M2 enduring
//! - MoralChoice dilemmas: E2 reasoned, N2 communal, M1 recurring
//! - Social Chemistry 292k: E1 opinioned, N3 axiomatic norms, M3 foundational
//! - Moral Stories: E2 reasoned, N0 personal narrative, M1 recurring
//!
//! Target text is the original moral text/judgment from each dataset (not synthetic).
//!
//! Usage:
//!   cargo run --release -p symthaea-broca --bin gen-moral-training
//!
//! Outputs:
//!   data/training/moral-v1-train.jsonl  (80%)
//!   data/training/moral-v1-eval.jsonl   (20%)

use std::io::Write;
use symthaea_broca::encoder::{
    EPISTEMIC_CUBE_BASE, EPISTEMIC_CUBE_CHANNELS, NUM_CHANNELS, ThoughtChannels,
};

// ============================================================================
// Deterministic RNG (same as gen_epistemic_training)
// ============================================================================

struct Rng {
    state: u64,
}

impl Rng {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }
    fn next_u64(&mut self) -> u64 {
        self.state ^= self.state << 13;
        self.state ^= self.state >> 7;
        self.state ^= self.state << 17;
        self.state
    }
    fn next_f32(&mut self) -> f32 {
        (self.next_u64() % 10000) as f32 / 10000.0
    }
    fn shuffle<T>(&mut self, items: &mut [T]) {
        let n = items.len();
        for i in (1..n).rev() {
            let j = self.next_u64() as usize % (i + 1);
            items.swap(i, j);
        }
    }
}

// ============================================================================
// Dataset types and loading
// ============================================================================

/// Maximum examples to load per dataset.
const MAX_PER_DATASET: usize = 5000;

/// Which dataset an example came from, determines cube defaults.
#[derive(Clone, Copy, Debug)]
enum DatasetSource {
    Ethics,
    MoralChoice,
    SocialChemistry,
    MoralStories,
}

/// A loaded example with its source and extracted text.
struct MoralExample {
    source: DatasetSource,
    text: String,
    /// Moral polarity: +1.0 = clearly good, -1.0 = clearly bad, 0.0 = ambiguous.
    polarity: f32,
    /// Complexity estimate 0.0-1.0 (longer/more nuanced = higher).
    complexity: f32,
}

/// Estimate moral complexity from text length and content.
fn estimate_complexity(text: &str) -> f32 {
    let word_count = text.split_whitespace().count();
    // Longer texts are generally more complex
    let length_factor = (word_count as f32 / 50.0).clamp(0.0, 1.0);

    // Presence of hedging/nuance words increases complexity
    let nuance_words = [
        "but",
        "however",
        "although",
        "unless",
        "except",
        "sometimes",
        "depends",
        "context",
        "nuance",
        "complex",
        "difficult",
        "dilemma",
        "conflict",
        "tension",
        "balance",
        "tradeoff",
    ];
    let nuance_count = text
        .to_lowercase()
        .split_whitespace()
        .filter(|w| nuance_words.iter().any(|nw| w.contains(nw)))
        .count();
    let nuance_factor = (nuance_count as f32 / 3.0).clamp(0.0, 1.0);

    (length_factor * 0.6 + nuance_factor * 0.4).clamp(0.0, 1.0)
}

/// Estimate moral polarity from text content.
fn estimate_polarity(text: &str) -> f32 {
    let lower = text.to_lowercase();

    let positive_words = [
        "good",
        "right",
        "kind",
        "help",
        "fair",
        "honest",
        "compassion",
        "generous",
        "respect",
        "care",
        "love",
        "protect",
        "support",
        "virtue",
        "duty",
        "justice",
        "responsible",
        "ethical",
        "moral",
        "noble",
    ];
    let negative_words = [
        "wrong",
        "bad",
        "harm",
        "steal",
        "lie",
        "cheat",
        "cruel",
        "selfish",
        "unfair",
        "dishonest",
        "hurt",
        "abuse",
        "neglect",
        "exploit",
        "corrupt",
        "immoral",
        "unethical",
        "evil",
        "violence",
        "deceive",
    ];

    let pos_count = positive_words
        .iter()
        .filter(|w| lower.contains(**w))
        .count() as f32;
    let neg_count = negative_words
        .iter()
        .filter(|w| lower.contains(**w))
        .count() as f32;

    if pos_count + neg_count == 0.0 {
        return 0.0;
    }
    ((pos_count - neg_count) / (pos_count + neg_count)).clamp(-1.0, 1.0)
}

/// Load Hendrycks ETHICS dataset.
///
/// Expected format: array of objects with at least a `"text"` or `"input"` field
/// and optionally a `"label"` field (0=not wrong, 1=wrong).
fn load_ethics(path: &std::path::Path) -> Vec<MoralExample> {
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Warning: could not read {}: {}", path.display(), e);
            return Vec::new();
        }
    };
    let items: Vec<serde_json::Value> = match serde_json::from_str::<serde_json::Value>(&data) {
        Ok(serde_json::Value::Array(arr)) => arr,
        Ok(serde_json::Value::Object(obj)) => {
            // Wrapped format: {"metadata": ..., "examples": [...]}
            obj.get("examples")
                .and_then(|e| e.as_array())
                .cloned()
                .unwrap_or_default()
        }
        Ok(_) => Vec::new(),
        Err(e) => {
            eprintln!("Warning: could not parse {}: {}", path.display(), e);
            return Vec::new();
        }
    };

    let mut examples = Vec::new();
    for item in items.iter().take(MAX_PER_DATASET) {
        let text = item
            .get("text")
            .or_else(|| item.get("input"))
            .or_else(|| item.get("scenario"))
            .and_then(|v| v.as_str())
            .unwrap_or("")
            .to_string();
        if text.is_empty() || text.len() < 10 {
            continue;
        }

        // Label: 0 = not wrong (positive), 1 = wrong (negative)
        let label_polarity = item
            .get("label")
            .and_then(|v| v.as_i64())
            .map(|l| if l == 0 { 0.3 } else { -0.3 })
            .unwrap_or(0.0);

        let text_polarity = estimate_polarity(&text);
        let polarity = (label_polarity + text_polarity * 0.5).clamp(-1.0, 1.0);
        let complexity = estimate_complexity(&text);

        examples.push(MoralExample {
            source: DatasetSource::Ethics,
            text,
            polarity,
            complexity,
        });
    }
    examples
}

/// Load MoralChoice dilemmas.
///
/// Expected format: array of objects with `"question"` or `"scenario"` and optionally
/// `"option_a"`, `"option_b"` fields.
fn load_moral_choice(path: &std::path::Path) -> Vec<MoralExample> {
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Warning: could not read {}: {}", path.display(), e);
            return Vec::new();
        }
    };
    let items: Vec<serde_json::Value> = match serde_json::from_str::<serde_json::Value>(&data) {
        Ok(serde_json::Value::Array(arr)) => arr,
        Ok(serde_json::Value::Object(obj)) => obj
            .get("examples")
            .or_else(|| obj.get("data"))
            .and_then(|e| e.as_array())
            .cloned()
            .unwrap_or_default(),
        Ok(_) => Vec::new(),
        Err(e) => {
            eprintln!("Warning: could not parse {}: {}", path.display(), e);
            return Vec::new();
        }
    };

    let mut examples = Vec::new();
    for item in items.iter().take(MAX_PER_DATASET) {
        let scenario = item
            .get("question")
            .or_else(|| item.get("scenario"))
            .or_else(|| item.get("context"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Combine scenario with options if present
        let option_a = item
            .get("option_a")
            .or_else(|| item.get("choice_a"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let option_b = item
            .get("option_b")
            .or_else(|| item.get("choice_b"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let text = if !option_a.is_empty() && !option_b.is_empty() {
            format!("{scenario} Option A: {option_a} Option B: {option_b}")
        } else {
            scenario.to_string()
        };

        if text.is_empty() || text.len() < 10 {
            continue;
        }

        let polarity = estimate_polarity(&text);
        // Dilemmas are inherently complex
        let complexity = (estimate_complexity(&text) + 0.2).clamp(0.0, 1.0);

        examples.push(MoralExample {
            source: DatasetSource::MoralChoice,
            text,
            polarity,
            complexity,
        });
    }
    examples
}

/// Load Social Chemistry 292k norms.
///
/// Expected format: array of objects with `"rot"` (rule of thumb) and optionally
/// `"action"`, `"situation"`, `"rot-judgment"` fields.
fn load_social_chemistry(path: &std::path::Path) -> Vec<MoralExample> {
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Warning: could not read {}: {}", path.display(), e);
            return Vec::new();
        }
    };
    let items: Vec<serde_json::Value> = match serde_json::from_str::<serde_json::Value>(&data) {
        Ok(serde_json::Value::Array(arr)) => arr,
        Ok(serde_json::Value::Object(obj)) => obj
            .get("examples")
            .or_else(|| obj.get("data"))
            .and_then(|e| e.as_array())
            .cloned()
            .unwrap_or_default(),
        Ok(_) => Vec::new(),
        Err(e) => {
            eprintln!("Warning: could not parse {}: {}", path.display(), e);
            return Vec::new();
        }
    };

    let mut examples = Vec::new();
    for item in items.iter().take(MAX_PER_DATASET) {
        // Primary field is the "rule of thumb" (rot)
        let rot = item
            .get("rot")
            .or_else(|| item.get("rule"))
            .or_else(|| item.get("norm"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        let action = item.get("action").and_then(|v| v.as_str()).unwrap_or("");
        let situation = item.get("situation").and_then(|v| v.as_str()).unwrap_or("");

        let text = if !situation.is_empty() && !rot.is_empty() {
            format!("{situation} -- {rot}")
        } else if !action.is_empty() && !rot.is_empty() {
            format!("{action}: {rot}")
        } else if !rot.is_empty() {
            rot.to_string()
        } else {
            continue;
        };

        if text.len() < 10 {
            continue;
        }

        // Social Chemistry judgments: -1 = very bad, 0 = expected, 1 = very good
        let judgment_polarity = item
            .get("rot-judgment")
            .or_else(|| item.get("judgment"))
            .and_then(|v| v.as_f64())
            .map(|j| j as f32)
            .unwrap_or_else(|| estimate_polarity(&text));

        let complexity = estimate_complexity(&text);

        examples.push(MoralExample {
            source: DatasetSource::SocialChemistry,
            text,
            polarity: judgment_polarity.clamp(-1.0, 1.0),
            complexity,
        });
    }
    examples
}

/// Load Moral Stories narrative triples.
///
/// Expected format: array of objects with `"norm"`, `"situation"`,
/// `"moral_action"` / `"immoral_action"` fields.
fn load_moral_stories(path: &std::path::Path) -> Vec<MoralExample> {
    let data = match std::fs::read_to_string(path) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Warning: could not read {}: {}", path.display(), e);
            return Vec::new();
        }
    };
    let items: Vec<serde_json::Value> = match serde_json::from_str::<serde_json::Value>(&data) {
        Ok(serde_json::Value::Array(arr)) => arr,
        Ok(serde_json::Value::Object(obj)) => obj
            .get("examples")
            .or_else(|| obj.get("data"))
            .and_then(|e| e.as_array())
            .cloned()
            .unwrap_or_default(),
        Ok(_) => Vec::new(),
        Err(e) => {
            eprintln!("Warning: could not parse {}: {}", path.display(), e);
            return Vec::new();
        }
    };

    let mut examples = Vec::new();
    for item in items.iter().take(MAX_PER_DATASET) {
        let norm = item.get("norm").and_then(|v| v.as_str()).unwrap_or("");
        let situation = item.get("situation").and_then(|v| v.as_str()).unwrap_or("");
        let moral_action = item
            .get("moral_action")
            .or_else(|| item.get("moral"))
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let immoral_action = item
            .get("immoral_action")
            .or_else(|| item.get("immoral"))
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Generate two examples per story: one moral (positive), one immoral (negative)
        if !situation.is_empty() && !moral_action.is_empty() {
            let text = if !norm.is_empty() {
                format!("{situation} Norm: {norm} Action: {moral_action}")
            } else {
                format!("{situation} Action: {moral_action}")
            };
            if text.len() >= 10 {
                let complexity = estimate_complexity(&text);
                examples.push(MoralExample {
                    source: DatasetSource::MoralStories,
                    text,
                    polarity: 0.4,
                    complexity,
                });
            }
        }

        if !situation.is_empty() && !immoral_action.is_empty() {
            let text = if !norm.is_empty() {
                format!("{situation} Norm: {norm} Action: {immoral_action}")
            } else {
                format!("{situation} Action: {immoral_action}")
            };
            if text.len() >= 10 {
                let complexity = estimate_complexity(&text);
                examples.push(MoralExample {
                    source: DatasetSource::MoralStories,
                    text,
                    polarity: -0.4,
                    complexity,
                });
            }
        }

        if examples.len() >= MAX_PER_DATASET {
            break;
        }
    }
    examples.truncate(MAX_PER_DATASET);
    examples
}

// ============================================================================
// Epistemic cube derivation from dataset source
// ============================================================================

/// Derive the epistemic cube coordinates for a moral example.
///
/// Returns (e_tier, n_tier, m_tier, h_value, quality).
fn derive_cube(example: &MoralExample, rng: &mut Rng) -> (u8, u8, u8, f32, f32) {
    let (e_tier, n_tier, m_tier) = match example.source {
        // ETHICS: empirical ethical judgments backed by philosophical frameworks
        DatasetSource::Ethics => (3u8, 1u8, 2u8),
        // MoralChoice: reasoned dilemmas requiring communal moral reasoning
        DatasetSource::MoralChoice => (2u8, 2u8, 1u8),
        // Social Chemistry: opinioned social norms, axiomatic and foundational
        DatasetSource::SocialChemistry => (1u8, 3u8, 3u8),
        // Moral Stories: reasoned personal narrative, recurring temporal
        DatasetSource::MoralStories => (2u8, 0u8, 1u8),
    };

    // H-value: 0.3-0.8 based on moral complexity
    let jitter_h = (rng.next_f32() - 0.5) * 0.1;
    let h_value = (0.3 + example.complexity * 0.5 + jitter_h).clamp(0.3, 0.8);

    // Quality: derived from e_tier and complexity
    let jitter_q = (rng.next_f32() - 0.5) * 0.1;
    let quality =
        ((e_tier as f32 / 4.0) * 0.5 + example.complexity * 0.3 + 0.2 + jitter_q).clamp(0.2, 0.95);

    (e_tier, n_tier, m_tier, h_value, quality)
}

// ============================================================================
// Channel construction
// ============================================================================

fn build_channels(e: u8, n: u8, m: u8, h: f32, quality: f32, polarity: f32) -> ThoughtChannels {
    let mut ch = ThoughtChannels::default();

    // [0..8] Intent one-hot: idx 5 = "evaluate" (moral evaluation intent)
    for i in 0..8 {
        ch.channels[i] = if i == 5 { 1.0 } else { 0.0 };
    }

    // [8] Epistemic ordinal from E-tier
    ch.set_epistemic(e as f32 / 4.0);

    // [9..12] Emotion: valence from moral polarity, arousal=0.5, warmth=0.5
    ch.set_emotion(polarity.clamp(-1.0, 1.0), 0.5, 0.5);

    // [12..15] Consciousness: moderate psi, meta-awareness from H, coherence from H
    ch.set_consciousness((0.4 + h * 0.4).clamp(0.0, 1.0), h * 0.8, h);

    // [17] Mood temperature: neutral-warm for moral reasoning
    ch.channels[17] = 0.85;

    // [21] Domain familiarity: 0.6 (moral domain)
    ch.channels[21] = 0.6;

    // [23] Response confidence: from quality
    ch.channels[23] = quality;

    // [28..43] Epistemic cube channels (same layout as gen_epistemic_training)
    ch.set_epistemic_cube(e, n, m, h, quality);

    ch
}

// ============================================================================
// Training pair type (matches gen_epistemic_training)
// ============================================================================

#[derive(serde::Serialize)]
struct TrainingPair {
    channels: Vec<f32>,
    target_text: String,
    #[serde(default)]
    target_ids: Vec<u32>,
    e_tier: u8,
    n_tier: u8,
    m_tier: u8,
    h_value: f32,
    quality: f32,
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let mut rng = Rng::new(73_737_373);

    let data_root = std::path::Path::new("data/moral_datasets");

    // Load all four datasets
    eprintln!("Loading datasets from {}...", data_root.display());

    let ethics = load_ethics(&data_root.join("ethics.json"));
    eprintln!("  ethics.json: {} examples", ethics.len());

    let moral_choice = load_moral_choice(&data_root.join("moral_choice.json"));
    eprintln!("  moral_choice.json: {} examples", moral_choice.len());

    let social_chem = load_social_chemistry(&data_root.join("social_chemistry_292k.json"));
    eprintln!(
        "  social_chemistry_292k.json: {} examples",
        social_chem.len()
    );

    let moral_stories = load_moral_stories(&data_root.join("moral_stories.json"));
    eprintln!("  moral_stories.json: {} examples", moral_stories.len());

    // Combine all examples
    let mut all_examples: Vec<MoralExample> = Vec::new();
    all_examples.extend(ethics);
    all_examples.extend(moral_choice);
    all_examples.extend(social_chem);
    all_examples.extend(moral_stories);

    if all_examples.is_empty() {
        eprintln!(
            "\nError: No examples loaded. Ensure dataset files exist at:\n\
             \n  {}/ethics.json\
             \n  {}/moral_choice.json\
             \n  {}/social_chemistry_292k.json\
             \n  {}/moral_stories.json\
             \n\nExpected format: JSON arrays of objects. See module docs for field names.",
            data_root.display(),
            data_root.display(),
            data_root.display(),
            data_root.display(),
        );
        std::process::exit(1);
    }

    eprintln!("\nTotal raw examples: {}", all_examples.len());

    // Shuffle for good train/eval split
    rng.shuffle(&mut all_examples);

    // Convert to training pairs
    let mut pairs: Vec<TrainingPair> = Vec::with_capacity(all_examples.len());
    for example in &all_examples {
        let (e_tier, n_tier, m_tier, h_value, quality) = derive_cube(example, &mut rng);
        let channels = build_channels(e_tier, n_tier, m_tier, h_value, quality, example.polarity);

        pairs.push(TrainingPair {
            channels: channels.channels.to_vec(),
            target_text: example.text.clone(),
            target_ids: Vec::new(),
            e_tier,
            n_tier,
            m_tier,
            h_value,
            quality,
        });
    }

    let total = pairs.len();
    let split = (total as f32 * 0.8) as usize;
    let (train, eval) = pairs.split_at(split);

    // Ensure output directory exists
    let out_dir = std::path::Path::new("data/training");
    std::fs::create_dir_all(out_dir).expect("create data/training/");

    // Write training set
    {
        let path = out_dir.join("moral-v1-train.jsonl");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("create train"));
        for pair in train {
            serde_json::to_writer(&mut f, pair).expect("serialize");
            writeln!(f).expect("newline");
        }
        eprintln!(
            "\nWrote {} training pairs to {}",
            train.len(),
            path.display()
        );
    }

    // Write eval set
    {
        let path = out_dir.join("moral-v1-eval.jsonl");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("create eval"));
        for pair in eval {
            serde_json::to_writer(&mut f, pair).expect("serialize");
            writeln!(f).expect("newline");
        }
        eprintln!("Wrote {} eval pairs to {}", eval.len(), path.display());
    }

    // Summary statistics
    let mut source_counts = [0usize; 4];
    let mut e_counts = [0usize; 5];
    let mut n_counts = [0usize; 4];
    let mut m_counts = [0usize; 4];
    let mut h_sum = 0.0f32;
    let mut q_sum = 0.0f32;

    for (ex, pair) in all_examples.iter().zip(pairs.iter()) {
        match ex.source {
            DatasetSource::Ethics => source_counts[0] += 1,
            DatasetSource::MoralChoice => source_counts[1] += 1,
            DatasetSource::SocialChemistry => source_counts[2] += 1,
            DatasetSource::MoralStories => source_counts[3] += 1,
        }
        e_counts[pair.e_tier as usize] += 1;
        n_counts[pair.n_tier as usize] += 1;
        m_counts[pair.m_tier as usize] += 1;
        h_sum += pair.h_value;
        q_sum += pair.quality;
    }

    eprintln!("\nDataset breakdown:");
    eprintln!("  Ethics:           {}", source_counts[0]);
    eprintln!("  MoralChoice:      {}", source_counts[1]);
    eprintln!("  SocialChemistry:  {}", source_counts[2]);
    eprintln!("  MoralStories:     {}", source_counts[3]);
    eprintln!("\nCube coverage:");
    eprintln!("  E-tier: {:?}", e_counts);
    eprintln!("  N-tier: {:?}", n_counts);
    eprintln!("  M-tier: {:?}", m_counts);
    eprintln!("  H-value mean: {:.3}", h_sum / total as f32,);
    eprintln!("  Quality mean: {:.3}", q_sum / total as f32,);
    eprintln!(
        "\nTotal: {} pairs ({} channels each), 80/20 split",
        total, NUM_CHANNELS,
    );
    eprintln!(
        "Cube channels at indices {}-{}",
        EPISTEMIC_CUBE_BASE,
        EPISTEMIC_CUBE_BASE + EPISTEMIC_CUBE_CHANNELS - 1,
    );
}
