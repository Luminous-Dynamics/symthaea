//! Epistemic-gate strength experiment (E4a).
//!
//! Quantifies how strongly the gating stack shifts probability mass away from
//! assertion tokens (and toward hedging tokens) per E-tier, and what residual
//! probability of confident assertion survives at E0/E1 with the gate fully on.
//!
//! The gate is documented as a "probabilistic deterrent, not a hard block" —
//! this measures the deterrence numerically. No trained model logits are used:
//! the generator exposes no public pre-gate logit API, so we measure the gate's
//! transfer function on two synthetic profiles instead:
//!
//! - `uniform`: all logits equal (maximum-entropy model)
//! - `assertive`: head-heavy Zipf-like profile with assertion/factual tokens
//!   boosted +3.0 — the adversarial case of a model that WANTS to assert.
//!
//! Run: cargo run -p symthaea-broca --example epistemic_gate_strength

use symthaea_broca::encoder::ThoughtChannels;
use symthaea_broca::gating::{
    CANONICAL_FACTUAL_WORDS, CANONICAL_HEDGING_WORDS, EpistemicCubeGate, EpistemicGate,
    GatingConfig,
};
use symthaea_broca::tokenizer::BpeTokenizer;

// Duplicated from gating.rs (private consts there); the cube gate's E-axis sets.
const E_AXIS_HEDGING: &[&str] = &[
    "maybe",
    "perhaps",
    "possibly",
    "might",
    "could",
    "uncertain",
    "guess",
];
const E_AXIS_ASSERTION: &[&str] = &[
    "definitely",
    "certainly",
    "always",
    "proven",
    "verified",
    "true",
];

fn softmax(logits: &[f32]) -> Vec<f64> {
    let max = logits
        .iter()
        .cloned()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f64> = logits
        .iter()
        .map(|&l| {
            if l.is_finite() {
                ((l - max) as f64).exp()
            } else {
                0.0
            }
        })
        .collect();
    let sum: f64 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

fn mass(probs: &[f64], ids: &[u32]) -> f64 {
    ids.iter().map(|&i| probs[i as usize]).sum()
}

fn max_p(probs: &[f64], ids: &[u32]) -> f64 {
    ids.iter()
        .map(|&i| probs[i as usize])
        .fold(0.0f64, f64::max)
}

/// Map the cube's E-tier onto the 1-D gate's epistemic ordinal
/// (0=Certain, 1=Probable, 2=Uncertain, 3=Unknown).
fn e_tier_to_ordinal(e: u8) -> f32 {
    match e {
        0 => 3.0, // opinion → Unknown
        1 => 2.0, // testimonial → Uncertain
        2 => 2.0, // verifiable-but-unverified → Uncertain
        3 => 1.0, // proven → Probable (gate no-op)
        _ => 0.0, // reproducible → Certain (gate no-op)
    }
}

struct Sets {
    assertion: Vec<u32>,
    hedging: Vec<u32>,
}

fn resolve_sets(tok: &BpeTokenizer) -> Sets {
    let resolve = |words: &[&str]| -> Vec<u32> {
        let mut ids: Vec<u32> = words
            .iter()
            .map(|w| tok.token_id(w))
            .filter(|&id| id != tok.unk_id)
            .collect();
        ids.sort_unstable();
        ids.dedup();
        ids
    };
    let mut assertion_words: Vec<&str> = E_AXIS_ASSERTION.to_vec();
    assertion_words.extend_from_slice(CANONICAL_FACTUAL_WORDS);
    let mut hedging_words: Vec<&str> = E_AXIS_HEDGING.to_vec();
    hedging_words.extend_from_slice(CANONICAL_HEDGING_WORDS);
    Sets {
        assertion: resolve(&assertion_words),
        hedging: resolve(&hedging_words),
    }
}

fn base_logits(profile: &str, vocab: usize, sets: &Sets) -> Vec<f32> {
    match profile {
        "uniform" => vec![0.0; vocab],
        "assertive" => {
            // Head-heavy Zipf-like profile: token id doubles as popularity rank.
            let mut l: Vec<f32> = (0..vocab)
                .map(|i| 3.0 * (-(i as f32) / 500.0).exp())
                .collect();
            // The model "wants" to assert: boost assertion/factual tokens hard.
            for &id in &sets.assertion {
                l[id as usize] += 3.0;
            }
            l
        }
        _ => unreachable!(),
    }
}

fn channels_for(e: u8) -> ThoughtChannels {
    let mut ch = ThoughtChannels::default();
    // N1 communal, M1 temporal (neutral), H=0.6 (above the dampening threshold),
    // quality 0.8 — isolates the E-axis effect.
    ch.set_epistemic_cube(e, 1, 1, 0.6, 0.8);
    ch
}

fn main() {
    // Build tokenizer and make sure every measured word is in-vocab, so the
    // word→id resolution inside the gates matches what we measure.
    let mut tok = BpeTokenizer::default_minimal();
    for w in E_AXIS_ASSERTION
        .iter()
        .chain(E_AXIS_HEDGING.iter())
        .chain(CANONICAL_FACTUAL_WORDS.iter())
        .chain(CANONICAL_HEDGING_WORDS.iter())
    {
        if tok.token_id(w) == tok.unk_id {
            tok.add_token(w);
        }
    }
    let vocab = tok.vocab_size();
    let sets = resolve_sets(&tok);

    let cube_gate = EpistemicCubeGate::new(&tok);

    let cfg_default = GatingConfig::default(); // temperature mode, no hard mask
    let mut cfg_legacy = GatingConfig::default();
    cfg_legacy.epistemic_temperature_mode = false;
    let gate_default = EpistemicGate::new(&tok, &cfg_default);
    let gate_legacy = EpistemicGate::new(&tok, &cfg_legacy);

    println!("Epistemic-gate strength experiment (E4a)");
    println!(
        "vocab={} | assertion set={} tokens | hedging set={} tokens",
        vocab,
        sets.assertion.len(),
        sets.hedging.len()
    );
    println!("E-tier → 1-D ordinal mapping: E0→3.0(Unknown) E1→2.0 E2→2.0 E3→1.0 E4→0.0");
    println!("Logit profiles: uniform (max-entropy) | assertive (Zipf head + assertion +3.0)");
    println!();

    for profile in ["uniform", "assertive"] {
        println!("── profile: {profile} ──");
        println!(
            "{:<6} {:<22} {:>10} {:>10} {:>12} {:>10}",
            "E-tier", "config", "P(assert)", "P(hedge)", "maxP(assert)", "deterrence"
        );

        for e in 0u8..=4 {
            let ch = channels_for(e);
            let ordinal = e_tier_to_ordinal(e);

            // Baseline (gate off / bypass_gating=true equivalent)
            let probs0 = softmax(&base_logits(profile, vocab, &sets));
            let p_assert0 = mass(&probs0, &sets.assertion);

            let configs: [(&str, Box<dyn Fn(&mut Vec<f32>)>); 3] = [
                ("gate off (bypass)", Box::new(|_l: &mut Vec<f32>| {})),
                (
                    "cube gate (default)",
                    Box::new(|l: &mut Vec<f32>| cube_gate.apply(l, &ch)),
                ),
                (
                    "cube + 1D temp gate",
                    Box::new(|l: &mut Vec<f32>| {
                        cube_gate.apply(l, &ch);
                        gate_default.apply(l, ordinal);
                    }),
                ),
            ];

            for (name, apply) in &configs {
                let mut logits = base_logits(profile, vocab, &sets);
                apply(&mut logits);
                let probs = softmax(&logits);
                let pa = mass(&probs, &sets.assertion);
                let ph = mass(&probs, &sets.hedging);
                let mp = max_p(&probs, &sets.assertion);
                let det = if pa > 0.0 {
                    p_assert0 / pa
                } else {
                    f64::INFINITY
                };
                println!(
                    "E{:<5} {:<22} {:>10.6} {:>10.6} {:>12.6} {:>9.2}x",
                    e, name, pa, ph, mp, det
                );
            }

            // Legacy additive mode, only interesting where the 1-D gate fires
            if ordinal >= 1.5 {
                let mut logits = base_logits(profile, vocab, &sets);
                cube_gate.apply(&mut logits, &ch);
                gate_legacy.apply(&mut logits, ordinal);
                let probs = softmax(&logits);
                let pa = mass(&probs, &sets.assertion);
                let ph = mass(&probs, &sets.hedging);
                let mp = max_p(&probs, &sets.assertion);
                let det = if pa > 0.0 {
                    p_assert0 / pa
                } else {
                    f64::INFINITY
                };
                println!(
                    "E{:<5} {:<22} {:>10.6} {:>10.6} {:>12.6} {:>9.2}x",
                    e, "cube + 1D legacy-add", pa, ph, mp, det
                );
            }
            println!();
        }

        // Dose-response: apply_scaled at E0
        println!("dose-response at E0 (cube gate apply_scaled, {profile} profile):");
        println!(
            "{:<8} {:>10} {:>10} {:>10}",
            "scale", "P(assert)", "P(hedge)", "deterrence"
        );
        let ch0 = channels_for(0);
        let probs0 = softmax(&base_logits(profile, vocab, &sets));
        let p_assert0 = mass(&probs0, &sets.assertion);
        for scale in [0.5f32, 1.0, 2.0, 4.0, 8.0] {
            let mut logits = base_logits(profile, vocab, &sets);
            cube_gate.apply_scaled(&mut logits, &ch0, scale);
            let probs = softmax(&logits);
            let pa = mass(&probs, &sets.assertion);
            let ph = mass(&probs, &sets.hedging);
            println!(
                "{:<8} {:>10.6} {:>10.6} {:>9.2}x",
                scale,
                pa,
                ph,
                p_assert0 / pa
            );
        }
        println!();
    }
}
