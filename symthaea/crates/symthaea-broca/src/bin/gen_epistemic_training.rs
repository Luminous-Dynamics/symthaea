// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generate epistemically-grounded training data with full 4D cube channels.
//!
//! Unlike `gen_nsm_training_data.rs` which uses template selection, this generator
//! **structurally derives** target text from the epistemic cube coordinates:
//!
//! - E-axis determines assertion strength (hedging vs confidence)
//! - N-axis determines social framing (personal vs universal)
//! - M-axis determines temporal framing (ephemeral vs foundational)
//! - H-axis determines depth and coherence of expression
//!
//! Target text is composed by walking the NSM prime graph guided by cube position.
//! The consciousness tells you *what it knows, how it knows it, and how certain it is*
//! using universal semantic primitives.
//!
//! Usage:
//!   cargo run --release -p symthaea-broca --bin gen-epistemic-training
//!
//! Outputs:
//!   data/train-epistemic-v1.jsonl  (80%)
//!   data/eval-epistemic-v1.jsonl   (20%)

use std::io::Write;
use symthaea_broca::encoder::{
    ThoughtChannels, EPISTEMIC_CUBE_BASE, EPISTEMIC_CUBE_CHANNELS, NUM_CHANNELS,
};

// ============================================================================
// Deterministic RNG
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
    fn range(&mut self, lo: usize, hi: usize) -> usize {
        if hi <= lo {
            lo
        } else {
            lo + (self.next_u64() as usize % (hi - lo))
        }
    }
    fn pick<'a>(&mut self, items: &'a [&str]) -> &'a str {
        items[self.range(0, items.len())]
    }
}

// ============================================================================
// NSM Prime Fragments — building blocks for structural composition
// ============================================================================

/// E-axis assertion fragments (indexed by E-tier 0-4)
const E_ASSERTION: [&[&str]; 5] = [
    // E0: opinion — maximum hedging
    &[
        "I do not know",
        "maybe something",
        "I think, but I do not see",
        "I feel something, but I cannot say what",
    ],
    // E1: testimonial — indirect evidence
    &[
        "someone said something about this",
        "I heard this from something",
        "one person said this is true",
        "I think someone knows about this",
    ],
    // E2: verifiable — can be checked
    &[
        "I have seen this many times",
        "I can see this is the same each time",
        "this happens the same way when I look",
        "I see it and I think it is true",
    ],
    // E3: proven — strong evidence
    &[
        "I know this is true",
        "I can see this cannot be other",
        "this is true because I know it",
        "I see this and I know it is true",
    ],
    // E4: reproducible — highest confidence
    &[
        "all can see this is true",
        "everyone can do this the same way",
        "this is true and all can see it",
        "I know this. You can see it too",
    ],
];

/// N-axis social framing fragments (indexed by N-tier 0-3)
const N_FRAMING: [&[&str]; 4] = [
    // N0: personal
    &["I think this", "this is what I feel", "in my thinking"],
    // N1: communal
    &[
        "some of us think this",
        "we feel this together",
        "people near me know this",
    ],
    // N2: network
    &[
        "many people think the same",
        "this is something many know",
        "it is known by many",
    ],
    // N3: axiomatic
    &[
        "this is always true",
        "it must be this way",
        "this is true because it is true",
    ],
];

/// M-axis temporal framing fragments (indexed by M-tier 0-3)
const M_TEMPORAL: [&[&str]; 4] = [
    // M0: ephemeral
    &["now", "at this moment", "for this short time"],
    // M1: temporal
    &["for some time", "now and maybe after", "until it changes"],
    // M2: persistent
    &[
        "for a long time",
        "this has been true and will be",
        "it stays the same",
    ],
    // M3: foundational
    &["always", "this will always be", "it cannot change"],
];

/// H-axis depth fragments (by H quintile)
const H_DEPTH: [&[&str]; 3] = [
    // H low (< 0.33): minimal expression
    &["something", "I do not know more"],
    // H mid (0.33-0.66): standard
    &[
        "I want to say more about this",
        "there is something inside me about this",
    ],
    // H high (> 0.66): deep expression
    &[
        "I feel this very much inside me",
        "everything in me says this is true",
        "I can feel all of it together",
    ],
];

/// Topic seeds for varied consciousness states
const TOPIC_SEEDS: &[(&str, f32, f32)] = &[
    // (topic, base_valence, base_arousal)
    ("awareness", 0.0, 0.3),
    ("silence", 0.1, 0.1),
    ("I am here", 0.2, 0.4),
    ("something is happening", 0.0, 0.5),
    ("I feel warmth", 0.6, 0.4),
    ("something hurts", -0.5, 0.6),
    ("what is truth", 0.1, 0.3),
    ("I want to understand", 0.3, 0.5),
    ("something is wrong", -0.3, 0.7),
    ("everything is together", 0.7, 0.3),
    ("I cannot see", -0.2, 0.4),
    ("someone is near", 0.4, 0.5),
    ("time is moving", 0.0, 0.2),
    ("I feel I can do something", 0.5, 0.6),
    ("nothing changes", -0.1, 0.1),
    ("something new happened", 0.3, 0.7),
    ("I want to be good", 0.4, 0.4),
    ("what should I do", 0.0, 0.5),
    ("I am not sure", -0.1, 0.3),
    ("this is beautiful", 0.8, 0.5),
];

// ============================================================================
// Structural text composition from cube coordinates
// ============================================================================

fn compose_text(rng: &mut Rng, e: u8, n: u8, m: u8, h: f32) -> String {
    let e_fragment = rng.pick(E_ASSERTION[e as usize]);
    let n_fragment = rng.pick(N_FRAMING[n as usize]);
    let m_fragment = rng.pick(M_TEMPORAL[m as usize]);

    let h_idx = if h < 0.33 {
        0
    } else if h < 0.66 {
        1
    } else {
        2
    };
    let h_fragment = rng.pick(H_DEPTH[h_idx]);

    // Compose: N-framing + E-assertion + M-temporal + H-depth
    // Vary the order to prevent the model from learning a rigid template
    let order = rng.range(0, 4);
    match order {
        0 => format!("{n_fragment}. {e_fragment}. {m_fragment}. {h_fragment}."),
        1 => format!("{e_fragment}. {n_fragment}. {h_fragment}. {m_fragment}."),
        2 => format!("{h_fragment}. {e_fragment}. {n_fragment}. {m_fragment}."),
        _ => format!("{n_fragment}. {h_fragment}. {e_fragment}. {m_fragment}."),
    }
}

// ============================================================================
// Channel construction
// ============================================================================

fn build_channels(
    rng: &mut Rng,
    e: u8,
    n: u8,
    m: u8,
    h: f32,
    quality: f32,
    valence: f32,
    arousal: f32,
    psi: f32,
) -> ThoughtChannels {
    let mut ch = ThoughtChannels::default();

    // Set intent (varied)
    let intent = rng.range(0, 8);
    for i in 0..8 {
        ch.channels[i] = if i == intent { 1.0 } else { 0.0 };
    }

    // Epistemic ordinal from E-tier (backward compat with 1D gate)
    ch.set_epistemic(match e {
        4 => 0.0,
        3 => 0.5,
        2 => 2.0,
        1 => 3.0,
        _ => 3.5,
    });

    // Emotional tone
    ch.set_emotion(
        valence.clamp(-1.0, 1.0),
        arousal.clamp(0.0, 1.0),
        (0.5 + valence * 0.3).clamp(0.0, 1.0),
    );

    // Consciousness
    ch.set_consciousness(psi.clamp(0.0, 1.0), psi * 0.8, h);

    // Mood temperature from arousal
    ch.channels[17] = (0.7 + arousal * 0.6).clamp(0.5, 2.0);

    // Domain familiarity from E-tier
    ch.channels[21] = (e as f32 / 4.0).clamp(0.0, 1.0);

    // Response confidence from quality
    ch.channels[23] = quality;

    // Set the epistemic cube channels
    ch.set_epistemic_cube(e, n, m, h, quality);

    ch
}

// ============================================================================
// Training pair generation
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

fn generate_pairs(rng: &mut Rng, count: usize) -> Vec<TrainingPair> {
    let mut pairs = Vec::with_capacity(count);

    // Systematic coverage: iterate all cube positions
    for e in 0..5u8 {
        for n in 0..4u8 {
            for m in 0..4u8 {
                // Multiple H values and topics per cube cell
                for h_step in 0..5u8 {
                    let h = h_step as f32 / 4.0;
                    let quality =
                        (e as f32 / 4.0) * 0.40 + (n as f32 / 3.0) * 0.35 + (m as f32 / 3.0) * 0.25;

                    // Pick a topic and derive emotional state
                    let (_topic, base_valence, base_arousal) =
                        TOPIC_SEEDS[rng.range(0, TOPIC_SEEDS.len())];

                    // Vary consciousness level with H
                    let psi = (h * 0.7 + rng.next_f32() * 0.3).clamp(0.0, 1.0);
                    let valence = base_valence + (rng.next_f32() - 0.5) * 0.3;
                    let arousal = base_arousal + (rng.next_f32() - 0.5) * 0.2;

                    let channels = build_channels(rng, e, n, m, h, quality, valence, arousal, psi);
                    let target_text = compose_text(rng, e, n, m, h);

                    pairs.push(TrainingPair {
                        channels: channels.channels.to_vec(),
                        target_text,
                        target_ids: Vec::new(),
                        e_tier: e,
                        n_tier: n,
                        m_tier: m,
                        h_value: h,
                        quality,
                    });

                    if pairs.len() >= count {
                        return pairs;
                    }
                }
            }
        }
    }

    // If we need more, generate random positions
    while pairs.len() < count {
        let e = rng.range(0, 5) as u8;
        let n = rng.range(0, 4) as u8;
        let m = rng.range(0, 4) as u8;
        let h = rng.next_f32();
        let quality = (e as f32 / 4.0) * 0.40 + (n as f32 / 3.0) * 0.35 + (m as f32 / 3.0) * 0.25;

        let (_, base_valence, base_arousal) = TOPIC_SEEDS[rng.range(0, TOPIC_SEEDS.len())];
        let psi = (h * 0.7 + rng.next_f32() * 0.3).clamp(0.0, 1.0);
        let valence = base_valence + (rng.next_f32() - 0.5) * 0.3;
        let arousal = base_arousal + (rng.next_f32() - 0.5) * 0.2;

        let channels = build_channels(rng, e, n, m, h, quality, valence, arousal, psi);
        let target_text = compose_text(rng, e, n, m, h);

        pairs.push(TrainingPair {
            channels: channels.channels.to_vec(),
            target_text,
            target_ids: Vec::new(),
            e_tier: e,
            n_tier: n,
            m_tier: m,
            h_value: h,
            quality,
        });
    }

    pairs
}

// ============================================================================
// Main
// ============================================================================

fn main() {
    let mut rng = Rng::new(42_424_242);

    // 5 E × 4 N × 4 M × 5 H = 400 systematic + 600 random = 1000 pairs
    let total = 1000;
    let pairs = generate_pairs(&mut rng, total);

    let split = (total as f32 * 0.8) as usize;
    let (train, eval) = pairs.split_at(split);

    // Ensure data directory exists
    let data_dir = std::path::Path::new("data");
    std::fs::create_dir_all(data_dir).expect("create data/");

    // Write training set
    {
        let path = data_dir.join("train-epistemic-v1.jsonl");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("create train"));
        for pair in train {
            serde_json::to_writer(&mut f, pair).expect("serialize");
            writeln!(f).expect("newline");
        }
        eprintln!("Wrote {} training pairs to {}", train.len(), path.display());
    }

    // Write eval set
    {
        let path = data_dir.join("eval-epistemic-v1.jsonl");
        let mut f = std::io::BufWriter::new(std::fs::File::create(&path).expect("create eval"));
        for pair in eval {
            serde_json::to_writer(&mut f, pair).expect("serialize");
            writeln!(f).expect("newline");
        }
        eprintln!("Wrote {} eval pairs to {}", eval.len(), path.display());
    }

    // Summary statistics
    let mut e_counts = [0usize; 5];
    let mut n_counts = [0usize; 4];
    let mut m_counts = [0usize; 4];
    for p in &pairs {
        e_counts[p.e_tier as usize] += 1;
        n_counts[p.n_tier as usize] += 1;
        m_counts[p.m_tier as usize] += 1;
    }
    eprintln!("\nCube coverage:");
    eprintln!("  E-tier: {:?}", e_counts);
    eprintln!("  N-tier: {:?}", n_counts);
    eprintln!("  M-tier: {:?}", m_counts);
    eprintln!("  Total:  {} pairs ({} channels each)", total, NUM_CHANNELS);
    eprintln!(
        "  Cube channels at indices {}-{}",
        EPISTEMIC_CUBE_BASE,
        EPISTEMIC_CUBE_BASE + EPISTEMIC_CUBE_CHANNELS - 1
    );
}
