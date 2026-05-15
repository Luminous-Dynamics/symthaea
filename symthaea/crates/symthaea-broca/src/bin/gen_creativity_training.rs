// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Generate creativity-focused training data for Broca language center.
//!
//! Produces training pairs that teach divergent thinking, remote association,
//! conceptual blending, and insight — the cognitive domains where Symthaea
//! currently scores 60% (psych-bench v0.10.0 baseline).
//!
//! Strategy: Use existing 43-channel ThoughtChannels (no schema change needed).
//! Creativity is encoded through:
//! - High `mood_temperature` (channel 17) → higher sampling diversity
//! - High `arousal` (channel 10) → creative energy
//! - `intent = Propose` (channel 3) → generative mode
//! - Low `epistemic_status` (channel 8) → uncertainty = exploration
//! - E0/E1 epistemic tiers → opinion/testimonial = creative expression
//!
//! The key insight: creativity training teaches Broca to produce DIVERSE
//! outputs for the same thought vector region, breaking the convergent
//! attractor that dominates untrained generation.
//!
//! Usage:
//!   cargo run --release -p symthaea-broca --bin gen-creativity-training
//!
//! Outputs:
//!   data/train-creativity-v1.jsonl  (80%)
//!   data/eval-creativity-v1.jsonl   (20%)

use std::io::Write;
use symthaea_broca::encoder::ThoughtChannels;

// ============================================================================
// Deterministic RNG (same as gen_epistemic_training.rs)
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
    fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.range(0, i + 1);
            items.swap(i, j);
        }
    }
}

// ============================================================================
// Creative domain fragments
// ============================================================================

/// Objects for alternate uses / divergent thinking
const OBJECTS: &[&str] = &[
    "brick",
    "paperclip",
    "shoe",
    "blanket",
    "bottle",
    "tire",
    "newspaper",
    "key",
    "spoon",
    "rope",
    "candle",
    "mirror",
    "umbrella",
    "ladder",
    "bucket",
    "pencil",
    "glass",
    "stone",
    "feather",
    "leaf",
];

/// Remote associates triads (cue1, cue2, cue3, answer)
const RAT_TRIADS: &[(&str, &str, &str, &str)] = &[
    ("cottage", "swiss", "cake", "cheese"),
    ("cream", "skate", "water", "ice"),
    ("show", "life", "row", "boat"),
    ("night", "wrist", "stop", "watch"),
    ("rocking", "wheel", "high", "chair"),
    ("dew", "comb", "bee", "honey"),
    ("fish", "mine", "rush", "gold"),
    ("tooth", "potato", "heart", "sweet"),
    ("river", "note", "account", "bank"),
    ("mouse", "sharp", "blue", "cheese"),
    ("bass", "complex", "sleep", "deep"),
    ("cross", "rain", "tie", "bow"),
    ("surprise", "line", "birthday", "party"),
    ("stamp", "code", "mail", "post"),
    ("broken", "clear", "eye", "glass"),
    ("time", "hair", "stretch", "long"),
    ("board", "magic", "death", "black"),
    ("palm", "shoe", "house", "tree"),
    ("aid", "rubber", "wagon", "band"),
    ("sun", "house", "card", "light"),
];

/// Conceptual blending pairs (concept_a, concept_b, blend_description)
const BLENDS: &[(&str, &str, &str)] = &[
    (
        "river",
        "time",
        "time flows like water, carrying moments downstream",
    ),
    (
        "tree",
        "knowledge",
        "branches of understanding growing from roots of experience",
    ),
    (
        "fire",
        "passion",
        "burning desire that consumes and transforms",
    ),
    (
        "ocean",
        "mind",
        "depths of thought with surface waves of attention",
    ),
    (
        "garden",
        "community",
        "nurturing connections that grow with care",
    ),
    (
        "bridge",
        "conversation",
        "spanning the gap between two worlds of meaning",
    ),
    ("storm", "emotion", "turbulent feelings that clear the air"),
    ("mirror", "self", "reflecting back what we show the world"),
    (
        "seed",
        "idea",
        "small beginning that contains entire futures",
    ),
    (
        "web",
        "connection",
        "intricate links that catch meaning passing by",
    ),
    (
        "mountain",
        "challenge",
        "something to climb that changes the view",
    ),
    (
        "music",
        "mathematics",
        "patterns that resonate with hidden structure",
    ),
    (
        "light",
        "understanding",
        "illumination that reveals what was always there",
    ),
    (
        "dance",
        "negotiation",
        "two parties moving together finding rhythm",
    ),
    (
        "forest",
        "complexity",
        "many individual parts creating emergent wholeness",
    ),
];

/// Insight problem reframings (problem, obvious_frame, creative_reframe)
const INSIGHTS: &[(&str, &str, &str)] = &[
    (
        "how to move heavy stone",
        "push harder with more force",
        "use water to float it, or many small rollers beneath",
    ),
    (
        "how to see in the dark",
        "bring a light source",
        "use sound like bats do, or feel the shapes of space",
    ),
    (
        "how to carry water without a container",
        "find a container",
        "freeze it into ice, or use a sponge, or carry it in your mouth",
    ),
    (
        "how to cross a gap too wide to jump",
        "build a bridge",
        "go around, go under, or make the gap narrower from both sides",
    ),
    (
        "how to count without numbers",
        "learn to count",
        "use stones, knots in rope, marks on bone, or body parts",
    ),
    (
        "how to remember without writing",
        "practice memorization",
        "walk through a palace of memory, sing it, or teach it to someone else",
    ),
    (
        "how to warm a room without fire",
        "find fuel to burn",
        "use sunlight through glass, body heat from many people, or hot stones",
    ),
    (
        "how to communicate across distance",
        "shout louder",
        "use drums, smoke, mirrors reflecting light, or carrier birds",
    ),
];

/// Divergent use categories for scoring flexibility
const USE_CATEGORIES: &[&str] = &[
    "tool",
    "weapon",
    "shelter",
    "art",
    "music",
    "container",
    "weight",
    "decoration",
    "game",
    "science",
    "cooking",
    "clothing",
];

// ============================================================================
// Creativity dimensions (encoded in existing channels, not new ones)
// ============================================================================

/// Divergence level: how open-ended the creative task is
#[derive(Clone, Copy)]
enum Divergence {
    Convergent,  // single answer (RAT)
    Constrained, // few answers
    OpenEnded,   // many valid answers (AUT)
    Radical,     // cross-category (blending)
}

/// Originality level: how surprising the output should be
#[derive(Clone, Copy)]
enum Originality {
    Common,   // most people produce this
    Uncommon, // some people produce this
    Rare,     // few people produce this
}

// ============================================================================
// Channel construction
// ============================================================================

fn build_creative_channels(
    rng: &mut Rng,
    divergence: Divergence,
    originality: Originality,
    fluency: f32, // 0.0 (one idea) to 1.0 (many ideas)
) -> ThoughtChannels {
    let mut ch = ThoughtChannels::default();

    // Intent: Propose (creative generation)
    ch.channels[3] = 1.0; // intent_propose

    // Epistemic status: low certainty = exploratory mode
    let epistemic = match divergence {
        Divergence::Convergent => 2.0,  // uncertain (searching for answer)
        Divergence::Constrained => 2.5, // uncertain-unknown
        Divergence::OpenEnded => 3.0,   // unknown (many possibilities)
        Divergence::Radical => 3.5,     // between unknown and OOD
    };
    ch.channels[8] = epistemic;

    // Emotional tone: creativity correlates with moderate positive valence + high arousal
    let valence = 0.3 + rng.next_f32() * 0.4; // 0.3-0.7 (slightly positive)
    let arousal = match divergence {
        Divergence::Convergent => 0.5 + rng.next_f32() * 0.2,
        Divergence::Constrained => 0.6 + rng.next_f32() * 0.2,
        Divergence::OpenEnded => 0.7 + rng.next_f32() * 0.2,
        Divergence::Radical => 0.8 + rng.next_f32() * 0.15,
    };
    let warmth = 0.5 + rng.next_f32() * 0.3; // moderate-high warmth (playful)
    ch.channels[9] = valence;
    ch.channels[10] = arousal;
    ch.channels[11] = warmth;

    // Consciousness: moderate-high (engaged, not drowsy)
    let psi = 0.4 + rng.next_f32() * 0.4;
    ch.set_consciousness(psi, psi * 0.7, psi * 0.6);

    // Mood temperature: HIGH for creative tasks (more sampling diversity)
    ch.channels[17] = match originality {
        Originality::Common => 1.0 + rng.next_f32() * 0.3,
        Originality::Uncommon => 1.3 + rng.next_f32() * 0.3,
        Originality::Rare => 1.5 + rng.next_f32() * 0.4,
    };

    // Domain familiarity: low (exploring unfamiliar territory)
    ch.channels[21] = 0.2 + rng.next_f32() * 0.3;

    // Response confidence: moderate (creative outputs are uncertain by nature)
    ch.channels[23] = 0.3 + rng.next_f32() * 0.3;

    // Epistemic cube: E0-E1 (opinion/testimonial), low N (personal), varied M
    let e_tier = match originality {
        Originality::Common => 1u8, // testimonial (known creative output)
        Originality::Uncommon => 0, // opinion (novel)
        Originality::Rare => 0,     // opinion (highly novel)
    };
    let n_tier = rng.range(0, 2) as u8; // personal or communal
    let m_tier = rng.range(1, 4) as u8; // temporal to foundational
    let h_value = psi * 0.8;
    let quality = fluency * 0.3 + (1.0 - epistemic / 4.0) * 0.3 + h_value * 0.4;

    ch.set_epistemic_cube(e_tier, n_tier, m_tier, h_value, quality.clamp(0.0, 1.0));

    ch
}

// ============================================================================
// Text composition
// ============================================================================

fn compose_rat_text(rng: &mut Rng, cue1: &str, cue2: &str, cue3: &str, answer: &str) -> String {
    let connectors = [
        format!("the hidden thread connecting {cue1}, {cue2}, and {cue3} is {answer} — each word finds its meaning through this common ground"),
        format!("{cue1} has {answer}, {cue2} is a kind of {answer}, {cue3} uses {answer}. the bridge between all three is this one idea"),
        format!("when I think of {cue1} and {cue2} and {cue3} together, what emerges is {answer}. it binds them like roots beneath different trees"),
        format!("three separate paths — {cue1}, {cue2}, {cue3} — converging on {answer}. the connection was there before I saw it"),
    ];
    connectors[rng.range(0, connectors.len())].clone()
}

fn compose_divergent_text(rng: &mut Rng, object: &str, n_uses: usize) -> String {
    // Generate diverse uses across categories
    let mut uses: Vec<String> = Vec::new();
    let mut categories_used: Vec<&str> = USE_CATEGORIES.to_vec();
    rng.shuffle(&mut categories_used);

    for (i, cat) in categories_used.iter().take(n_uses).enumerate() {
        let use_text = match *cat {
            "tool" => format!("a {object} becomes a lever to pry things open"),
            "weapon" => format!("a {object} as protection, thrown or held firm"),
            "shelter" => format!("a {object} stacked with others makes a wall against wind"),
            "art" => format!("a {object} carved or painted becomes a canvas for meaning"),
            "music" => format!("strike a {object} at different points for different tones"),
            "container" => format!("hollow out a {object} to hold small things"),
            "weight" => format!("a {object} holds down papers in the wind"),
            "decoration" => format!("arrange many {object}s in patterns for beauty"),
            "game" => format!("a {object} as a marker in a game of territory"),
            "science" => format!("use a {object} to test gravity, friction, or heat"),
            "cooking" => format!("a {object} heated becomes a surface for warming food"),
            "clothing" => format!("a flat {object} can be a platform for standing taller"),
            _ => format!("a {object} reimagined for {cat}"),
        };
        if i == 0 {
            uses.push(use_text);
        } else {
            uses.push(format!(", {use_text}"));
        }
    }

    format!(
        "a {object} is more than it appears. {}. each use bends the familiar into something new",
        uses.join("")
    )
}

fn compose_blend_text(rng: &mut Rng, a: &str, b: &str, blend: &str) -> String {
    let frames = [
        format!("when {a} meets {b}: {blend}. the blend reveals what neither held alone"),
        format!("{a} and {b} are distant cousins. their shared nature: {blend}"),
        format!("imagine {a} through the lens of {b} — {blend}. structure maps across domains"),
        format!("the space between {a} and {b} is not empty. it holds: {blend}"),
    ];
    frames[rng.range(0, frames.len())].clone()
}

fn compose_insight_text(rng: &mut Rng, problem: &str, obvious: &str, reframe: &str) -> String {
    let frames = [
        format!("the problem of {problem} — the first thought is to {obvious}. but step sideways: {reframe}. the constraint was in the framing, not the world"),
        format!("to solve {problem}, most try to {obvious}. the insight: {reframe}. the answer was hiding in a different question"),
        format!("{problem} seems to demand {obvious}. but what if the problem itself is wrong? instead: {reframe}"),
    ];
    frames[rng.range(0, frames.len())].clone()
}

// ============================================================================
// Training pair type
// ============================================================================

#[derive(serde::Serialize)]
struct TrainingPair {
    channels: Vec<f32>,
    target_text: String,
    #[serde(default)]
    target_ids: Vec<u32>,
    creativity_type: String,
    divergence: u8,
    originality: u8,
}

// ============================================================================
// Main generation
// ============================================================================

fn generate_pairs(rng: &mut Rng, count: usize) -> Vec<TrainingPair> {
    let mut pairs = Vec::with_capacity(count);

    // 1. Remote Associates (25% — convergent creativity)
    let rat_count = count / 4;
    for i in 0..rat_count {
        let triad = &RAT_TRIADS[i % RAT_TRIADS.len()];
        let channels = build_creative_channels(
            rng,
            Divergence::Convergent,
            Originality::Common,
            0.1, // low fluency (single answer)
        );
        let text = compose_rat_text(rng, triad.0, triad.1, triad.2, triad.3);
        pairs.push(TrainingPair {
            channels: channels.channels.to_vec(),
            target_text: text,
            target_ids: Vec::new(),
            creativity_type: "remote_associates".into(),
            divergence: 0,
            originality: 0,
        });
    }

    // 2. Divergent Thinking / Alternate Uses (35% — open-ended creativity)
    let aut_count = count * 35 / 100;
    for i in 0..aut_count {
        let object = OBJECTS[i % OBJECTS.len()];
        let n_uses = 3 + (i % 5); // 3-7 uses
        let originality = match i % 3 {
            0 => Originality::Common,
            1 => Originality::Uncommon,
            _ => Originality::Rare,
        };
        let channels = build_creative_channels(
            rng,
            Divergence::OpenEnded,
            originality,
            (n_uses as f32 / 10.0).clamp(0.0, 1.0),
        );
        let text = compose_divergent_text(rng, object, n_uses);
        pairs.push(TrainingPair {
            channels: channels.channels.to_vec(),
            target_text: text,
            target_ids: Vec::new(),
            creativity_type: "divergent_thinking".into(),
            divergence: 2,
            originality: originality as u8,
        });
    }

    // 3. Conceptual Blending (20% — radical recombination)
    let blend_count = count / 5;
    for i in 0..blend_count {
        let blend = &BLENDS[i % BLENDS.len()];
        let channels =
            build_creative_channels(rng, Divergence::Radical, Originality::Uncommon, 0.3);
        let text = compose_blend_text(rng, blend.0, blend.1, blend.2);
        pairs.push(TrainingPair {
            channels: channels.channels.to_vec(),
            target_text: text,
            target_ids: Vec::new(),
            creativity_type: "conceptual_blending".into(),
            divergence: 3,
            originality: 1,
        });
    }

    // 4. Insight Problems (20% — restructuring)
    let insight_count = count - pairs.len();
    for i in 0..insight_count {
        let insight = &INSIGHTS[i % INSIGHTS.len()];
        let channels =
            build_creative_channels(rng, Divergence::Constrained, Originality::Rare, 0.4);
        let text = compose_insight_text(rng, insight.0, insight.1, insight.2);
        pairs.push(TrainingPair {
            channels: channels.channels.to_vec(),
            target_text: text,
            target_ids: Vec::new(),
            creativity_type: "insight_problem".into(),
            divergence: 1,
            originality: 2,
        });
    }

    // Shuffle all pairs
    rng.shuffle(&mut pairs);
    pairs.truncate(count);
    pairs
}

fn main() {
    let mut rng = Rng::new(0xC4EA_71FE_5EED);

    let total = 1000;
    let pairs = generate_pairs(&mut rng, total);

    let train_count = total * 80 / 100;
    let (train, eval) = pairs.split_at(train_count);

    // Write train split
    let train_path = "data/train-creativity-v1.jsonl";
    let mut f = std::fs::File::create(train_path).expect("create train file");
    for pair in train {
        let json = serde_json::to_string(pair).expect("serialize");
        writeln!(f, "{json}").expect("write");
    }

    // Write eval split
    let eval_path = "data/eval-creativity-v1.jsonl";
    let mut f = std::fs::File::create(eval_path).expect("create eval file");
    for pair in eval {
        let json = serde_json::to_string(pair).expect("serialize");
        writeln!(f, "{json}").expect("write");
    }

    println!(
        "Generated {total} creativity training pairs:\n  Train: {train_path} ({} pairs)\n  Eval:  {eval_path} ({} pairs)",
        train.len(),
        eval.len()
    );
    println!("\nBreakdown:");
    let mut counts: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for p in &pairs {
        *counts
            .entry(match p.creativity_type.as_str() {
                "remote_associates" => "Remote Associates (convergent)",
                "divergent_thinking" => "Divergent Thinking (open-ended)",
                "conceptual_blending" => "Conceptual Blending (radical)",
                "insight_problem" => "Insight Problems (restructuring)",
                _ => "other",
            })
            .or_insert(0) += 1;
    }
    for (k, v) in counts {
        println!("  {k}: {v}");
    }
}
