// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Dose-response affect gate: does the voice's ACOUSTIC output move in the
//! right direction as consciousness state changes, independent of words?
//!
//! Distillation (v1-v4, all negative — see VOICE_IMPROVEMENT_PLAN_2026-07-15.md)
//! spent a whole investigation asking "did the words survive" (round-trip
//! WER through Whisper). This gate asks a different question: synthesize
//! the SAME fixed sentence at several deliberately-contrasting consciousness
//! states (a small arousal/valence grid, matching `update_from_consciousness`'s
//! real signature), then blind-re-extract prosody from the audio alone via
//! `symthaea_vocal_tract::formant_extraction` (proven code — no new
//! extractor needed) and check whether pitch/energy/duration move with the
//! input state in a sensible, monotonic way. Scored as correlation/
//! dose-response, not pass/fail accuracy — same pattern already validated
//! this session for the speed knob (0.75x -> 2.85s, 1.25x -> 1.73s).
//!
//! Gut-check finding that motivated scoping this in two stages (2026-07-17):
//! both candidate live prosody-perception paths in this codebase
//! (`symthaea_stt::audio::ProsodyFeatures`, `src/perception/audio.rs`'s own
//! duplicate) are dead ends — zero external callers, and the perception
//! module's copy has two of its four fields hardcoded to 0.0 with a
//! "would need more sophisticated analysis" comment. There is no trained
//! ML classifier for arousal/valence anywhere in this codebase, and
//! building one needs a labeled training corpus that doesn't exist — out
//! of scope for one session.
//!
//! What IS honestly buildable without that: a small ordinary-least-squares
//! inverse model, fit directly on the (energy, f0_std, duration) ->
//! (arousal, valence) relationships this gate already proved robust
//! (interaction-checked, 3/3 stratum agreement on the strong terms). Fit on
//! the 9-point calibration grid, then tested on 5 HELD-OUT states never
//! used for fitting — a genuine train/test split, so a good held-out
//! correlation means real self-perception, not restating the forward
//! mapping. This is narrow (single fixed sentence, linear, only two
//! affect dimensions) but real: it closes the self-hearing loop the
//! original gate sketch called for, honestly scoped to what's actually
//! measurable today.
//!
//! ```bash
//! cargo run --example voice_affect_gate --features vocal-tract
//! ```

use symthaea::voice::repl_voice::{ReplVoiceConfig, ReplVoiceOutput};
use symthaea_vocal_tract::formant_extraction::{ExtractionConfig, extract_formant_frames};
use symthaea_vocal_tract::types::SourceType;

/// A point in Russell's circumplex affect space, plus the rest of
/// `update_from_consciousness`'s inputs held at neutral/fixed values so the
/// grid isolates valence/arousal's own acoustic effect.
struct AffectPoint {
    name: &'static str,
    valence: f32,
    arousal: f32,
}

// Orthogonal 3x3 arousal x valence grid (9 points), replacing the original
// 5-point diamond. The diamond confounded arousal and valence — "excited"
// meant high-arousal AND positive-valence together, "distressed" meant
// high-arousal AND negative-valence together — so a strong correlation
// couldn't say which input dimension actually caused it. Every arousal
// level here is paired with every valence level equally, so the naive
// per-dimension Pearson correlation across all 9 points is now a valid
// (first-order, no-interaction-tested) main-effect estimate rather than a
// confounded one. First-pass result (5-point diamond, 2026-07-17): the
// pacing modulation is real and non-trivial (|r| up to 0.80) but two of
// its strongest effects ran opposite the classic human-speech expectation
// (arousal flattened f0 variability instead of raising it; negative
// valence raised f0 instead of lowering it) — worth re-checking cleanly.
const AROUSAL_LEVELS: [f32; 3] = [0.1, 0.5, 0.85];
const VALENCE_LEVELS: [f32; 3] = [-0.7, 0.0, 0.7];
const AROUSAL_NAMES: [&str; 3] = ["low-a", "mid-a", "hi-a"];
const VALENCE_NAMES: [&str; 3] = ["neg-v", "neu-v", "pos-v"];

fn grid() -> Vec<AffectPoint> {
    let mut points = Vec::with_capacity(9);
    for (ai, &arousal) in AROUSAL_LEVELS.iter().enumerate() {
        for (vi, &valence) in VALENCE_LEVELS.iter().enumerate() {
            points.push(AffectPoint {
                name: Box::leak(
                    format!("{}/{}", AROUSAL_NAMES[ai], VALENCE_NAMES[vi]).into_boxed_str(),
                ),
                valence,
                arousal,
            });
        }
    }
    points
}

const TEST_SENTENCE: &str = "the meeting starts at nine oclock";

struct Summary {
    name: &'static str,
    valence: f32,
    arousal: f32,
    f0_mean: f32,
    f0_std: f32,
    energy_mean: f32,
    duration_s: f32,
    voiced_frac: f32,
}

fn summarize(frames: &[symthaea_vocal_tract::types::FormantFrame], duration_s: f32) -> Summary {
    let voiced: Vec<&symthaea_vocal_tract::types::FormantFrame> = frames
        .iter()
        .filter(|f| f.source_type == SourceType::Vowel && f.voicing > 0.5 && f.f0 > 0.0)
        .collect();
    let n = voiced.len().max(1) as f32;
    let f0_mean = voiced.iter().map(|f| f.f0).sum::<f32>() / n;
    let f0_var = voiced.iter().map(|f| (f.f0 - f0_mean).powi(2)).sum::<f32>() / n;
    let energy_mean = frames.iter().map(|f| f.energy).sum::<f32>() / frames.len().max(1) as f32;
    let voiced_frac = voiced.len() as f32 / frames.len().max(1) as f32;
    Summary {
        name: "",
        valence: 0.0,
        arousal: 0.0,
        f0_mean,
        f0_std: f0_var.sqrt(),
        energy_mean,
        duration_s,
        voiced_frac,
    }
}

/// Pearson correlation coefficient. Returns 0.0 for degenerate (zero-variance)
/// inputs rather than NaN.
fn pearson(xs: &[f32], ys: &[f32]) -> f32 {
    let n = xs.len() as f32;
    let mx = xs.iter().sum::<f32>() / n;
    let my = ys.iter().sum::<f32>() / n;
    let cov: f32 = xs.iter().zip(ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let vx: f32 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let vy: f32 = ys.iter().map(|y| (y - my).powi(2)).sum();
    if vx <= 1e-9 || vy <= 1e-9 {
        0.0
    } else {
        cov / (vx.sqrt() * vy.sqrt())
    }
}

/// Held-out affect states: NEVER used to fit the self-perception model
/// below, only to test it. Deliberately off the 3x3 calibration lattice
/// (arousal/valence values the grid never used) so a good result can't be
/// explained by the model having simply memorized calibration points.
struct HeldOutPoint {
    name: &'static str,
    valence: f32,
    arousal: f32,
}
const HELD_OUT: &[HeldOutPoint] = &[
    HeldOutPoint {
        name: "mild-neg/lowmid-a",
        valence: -0.4,
        arousal: 0.3,
    },
    HeldOutPoint {
        name: "mild-pos/lowmid-a",
        valence: 0.4,
        arousal: 0.3,
    },
    HeldOutPoint {
        name: "mild-neg/himid-a",
        valence: -0.4,
        arousal: 0.7,
    },
    HeldOutPoint {
        name: "mild-pos/himid-a",
        valence: 0.4,
        arousal: 0.7,
    },
    HeldOutPoint {
        name: "offgrid-center",
        valence: 0.2,
        arousal: 0.6,
    },
];

/// Ordinary least squares: y ~ b0 + b1*x1 + b2*x2 + b3*x3, via the normal
/// equations solved by Gaussian elimination with partial pivoting. n=9
/// calibration points, 4 unknowns — small enough that a dependency-free
/// hand-rolled solver is the right call over pulling in a linear-algebra
/// crate for one example.
fn fit_ols(xs: &[[f32; 3]], ys: &[f32]) -> [f32; 4] {
    let n = xs.len();
    let mut xtx = [[0.0f64; 4]; 4];
    let mut xty = [0.0f64; 4];
    for i in 0..n {
        let row = [1.0, xs[i][0] as f64, xs[i][1] as f64, xs[i][2] as f64];
        for a in 0..4 {
            xty[a] += row[a] * ys[i] as f64;
            for b in 0..4 {
                xtx[a][b] += row[a] * row[b];
            }
        }
    }
    let mut aug = [[0.0f64; 5]; 4];
    for a in 0..4 {
        aug[a][..4].copy_from_slice(&xtx[a]);
        aug[a][4] = xty[a];
    }
    for col in 0..4 {
        let mut pivot = col;
        for row in (col + 1)..4 {
            if aug[row][col].abs() > aug[pivot][col].abs() {
                pivot = row;
            }
        }
        aug.swap(col, pivot);
        let d = aug[col][col];
        if d.abs() > 1e-12 {
            for k in col..5 {
                aug[col][k] /= d;
            }
            for row in 0..4 {
                if row != col {
                    let factor = aug[row][col];
                    for k in col..5 {
                        aug[row][k] -= factor * aug[col][k];
                    }
                }
            }
        }
    }
    [
        aug[0][4] as f32,
        aug[1][4] as f32,
        aug[2][4] as f32,
        aug[3][4] as f32,
    ]
}

fn predict_ols(coeffs: [f32; 4], x: [f32; 3]) -> f32 {
    coeffs[0] + coeffs[1] * x[0] + coeffs[2] * x[1] + coeffs[3] * x[2]
}

fn main() -> anyhow::Result<()> {
    let config = ReplVoiceConfig {
        use_ltc_pipeline: true,
        use_articulatory: true,
        ..ReplVoiceConfig::default()
    };
    let sample_rate = config.sample_rate;
    let mut voice = ReplVoiceOutput::new(config)?;
    let extraction = ExtractionConfig::default();

    let grid_points = grid();
    println!(
        "Affect gate: '{TEST_SENTENCE}' synthesized at {} states (orthogonal {}x{} arousal x valence grid)\n",
        grid_points.len(),
        AROUSAL_LEVELS.len(),
        VALENCE_LEVELS.len()
    );

    let mut results = Vec::new();
    for point in &grid_points {
        // Full update_from_consciousness signal: unified_psi/prediction_error/
        // tau_mean held at plausible mid-range constants, in_flow off, rate/
        // pause multipliers held at 1.0 so this grid isolates valence/arousal
        // specifically (the rate/pause knobs were already separately verified
        // functional in the 2026-07-16 speed-knob check).
        voice.update_from_consciousness(
            0.7,           // unified_psi
            0.1,           // prediction_error
            point.valence, // emotional_valence
            point.arousal, // emotional_arousal
            false,         // in_flow
            1.0,           // speech_rate_multiplier
            1.0,           // pause_multiplier
            0.05,          // tau_mean
        );
        let audio = voice.synthesize(TEST_SENTENCE)?;
        let duration_s = audio.len() as f32 / sample_rate as f32;
        let frames = extract_formant_frames(&audio, sample_rate, &extraction);
        let mut s = summarize(&frames, duration_s);
        s.name = point.name;
        s.valence = point.valence;
        s.arousal = point.arousal;
        println!(
            "  {:<18} valence={:>5.2} arousal={:>5.2}  ->  f0_mean={:>6.1}Hz f0_std={:>6.1}Hz energy={:>5.3} dur={:>5.2}s voiced={:>4.0}%",
            s.name,
            s.valence,
            s.arousal,
            s.f0_mean,
            s.f0_std,
            s.energy_mean,
            s.duration_s,
            s.voiced_frac * 100.0
        );
        results.push(s);
    }

    let arousals: Vec<f32> = results.iter().map(|r| r.arousal).collect();
    let valences: Vec<f32> = results.iter().map(|r| r.valence).collect();
    let f0_means: Vec<f32> = results.iter().map(|r| r.f0_mean).collect();
    let f0_stds: Vec<f32> = results.iter().map(|r| r.f0_std).collect();
    let energies: Vec<f32> = results.iter().map(|r| r.energy_mean).collect();
    let durations: Vec<f32> = results.iter().map(|r| r.duration_s).collect();

    println!(
        "\nDose-response correlations (Pearson r across the {} orthogonal states —\n\
         arousal and valence are now independently varied, so these are valid\n\
         first-order main-effect estimates, not confounded like the original\n\
         5-point diamond design):",
        grid_points.len()
    );
    println!(
        "  arousal vs f0_mean   : {:>6.3}",
        pearson(&arousals, &f0_means)
    );
    println!(
        "  arousal vs f0_std    : {:>6.3}  (higher arousal -> more pitch variability is the classic expectation)",
        pearson(&arousals, &f0_stds)
    );
    println!(
        "  arousal vs energy    : {:>6.3}  (higher arousal -> louder/more energetic is the classic expectation)",
        pearson(&arousals, &energies)
    );
    println!(
        "  arousal vs duration  : {:>6.3}  (higher arousal often -> faster/shorter, so a negative r is expected here)",
        pearson(&arousals, &durations)
    );
    println!(
        "  valence vs f0_mean   : {:>6.3}  (valence's acoustic correlates are weaker/less settled in the literature than arousal's — no strong sign predicted)",
        pearson(&valences, &f0_means)
    );
    println!(
        "  valence vs f0_std    : {:>6.3}",
        pearson(&valences, &f0_stds)
    );
    println!(
        "  valence vs energy    : {:>6.3}",
        pearson(&valences, &energies)
    );
    println!(
        "  valence vs duration  : {:>6.3}",
        pearson(&valences, &durations)
    );

    // Interaction check: the grid is ordered arousal-major (results[ai*3+vi]
    // for ai,vi in 0..3), so a fixed arousal index spans a valence stratum
    // and a fixed valence index (stepping by 3) spans an arousal stratum.
    // With only 3 points per stratum, a Pearson r there is too unstable to
    // be meaningful (near +-1 for almost any non-degenerate triple) — so
    // this checks direction-agreement instead: for each main effect, does
    // the endpoint-to-endpoint sign within EVERY stratum of the other
    // variable agree with the overall correlation's sign? Full agreement
    // (3/3) means the effect looks like a genuine main effect, not an
    // artifact concentrated in one stratum (which would be evidence of an
    // interaction this simple 3x3 grid can't otherwise characterize).
    fn stratum_sign(values: [f32; 3]) -> i32 {
        (values[2] - values[0]).signum() as i32
    }
    fn agreement(overall_r: f32, stratum_signs: [i32; 3]) -> String {
        let expected = overall_r.signum() as i32;
        let agree = stratum_signs
            .iter()
            .filter(|&&s| s == expected || s == 0)
            .count();
        format!("{agree}/3 strata agree in direction")
    }

    let by = |i: usize, f: fn(&Summary) -> f32| f(&results[i]);
    // Grid is arousal-major, valence-minor: results[ai*3 + vi].
    // "arousal vs X" needs arousal VARYING with valence FIXED — step by 3
    // (ai=0,1,2) at a fixed vi. "valence vs X" needs valence VARYING with
    // arousal FIXED — the 3 consecutive entries at a fixed ai. (An earlier
    // version of this file had these two swapped — arousal's checks were
    // silently validating valence's within-arousal-fixed consistency
    // instead, and vice versa. Caught by hand-checking the printed "1/3"
    // for arousal->energy against the raw table, where every valence level
    // shows energy rising with arousal — should have been 3/3. Named these
    // unambiguously by what they hold fixed to make that mistake harder to
    // repeat.)
    let at_fixed_valence =
        |vi: usize, f: fn(&Summary) -> f32| [by(vi, f), by(3 + vi, f), by(6 + vi, f)];
    let at_fixed_arousal =
        |ai: usize, f: fn(&Summary) -> f32| [by(ai * 3, f), by(ai * 3 + 1, f), by(ai * 3 + 2, f)];

    println!(
        "\nInteraction check (does each main effect hold within every stratum of the other variable?):"
    );
    let f0m: fn(&Summary) -> f32 = |s| s.f0_mean;
    let f0s: fn(&Summary) -> f32 = |s| s.f0_std;
    let en: fn(&Summary) -> f32 = |s| s.energy_mean;
    let du: fn(&Summary) -> f32 = |s| s.duration_s;
    for (label, r, signs) in [
        (
            "arousal vs f0_mean  ",
            pearson(&arousals, &f0_means),
            [
                stratum_sign(at_fixed_valence(0, f0m)),
                stratum_sign(at_fixed_valence(1, f0m)),
                stratum_sign(at_fixed_valence(2, f0m)),
            ],
        ),
        (
            "arousal vs f0_std   ",
            pearson(&arousals, &f0_stds),
            [
                stratum_sign(at_fixed_valence(0, f0s)),
                stratum_sign(at_fixed_valence(1, f0s)),
                stratum_sign(at_fixed_valence(2, f0s)),
            ],
        ),
        (
            "arousal vs energy   ",
            pearson(&arousals, &energies),
            [
                stratum_sign(at_fixed_valence(0, en)),
                stratum_sign(at_fixed_valence(1, en)),
                stratum_sign(at_fixed_valence(2, en)),
            ],
        ),
        (
            "arousal vs duration ",
            pearson(&arousals, &durations),
            [
                stratum_sign(at_fixed_valence(0, du)),
                stratum_sign(at_fixed_valence(1, du)),
                stratum_sign(at_fixed_valence(2, du)),
            ],
        ),
        (
            "valence vs f0_mean  ",
            pearson(&valences, &f0_means),
            [
                stratum_sign(at_fixed_arousal(0, f0m)),
                stratum_sign(at_fixed_arousal(1, f0m)),
                stratum_sign(at_fixed_arousal(2, f0m)),
            ],
        ),
        (
            "valence vs f0_std   ",
            pearson(&valences, &f0_stds),
            [
                stratum_sign(at_fixed_arousal(0, f0s)),
                stratum_sign(at_fixed_arousal(1, f0s)),
                stratum_sign(at_fixed_arousal(2, f0s)),
            ],
        ),
        (
            "valence vs energy   ",
            pearson(&valences, &energies),
            [
                stratum_sign(at_fixed_arousal(0, en)),
                stratum_sign(at_fixed_arousal(1, en)),
                stratum_sign(at_fixed_arousal(2, en)),
            ],
        ),
        (
            "valence vs duration ",
            pearson(&valences, &durations),
            [
                stratum_sign(at_fixed_arousal(0, du)),
                stratum_sign(at_fixed_arousal(1, du)),
                stratum_sign(at_fixed_arousal(2, du)),
            ],
        ),
    ] {
        println!("  {label}: {}", agreement(r, signs));
    }

    println!(
        "\nInterpretation (forward mapping): does the SYNTHESIS pipeline's consciousness-\n\
         modulated pacing actually reach the acoustic signal in a detectable, direction-\n\
         sensible way? |r| well below ~0.3 on the strong rows would mean the pacing\n\
         modulation isn't acoustically real despite existing in code."
    );

    // ------------------------------------------------------------------
    // Self-perception closure: can the pipeline recognize its OWN intended
    // affect state from its OWN synthesized voice? Fit a linear inverse
    // model — (energy, f0_std, duration) -> (arousal, valence) — on the 9
    // calibration points collected above, then test it on 5 HELD-OUT
    // states that were never used for fitting. A good held-out correlation
    // means real self-perception; a bad one would mean the forward mapping
    // exists but isn't invertible from acoustics alone (e.g. because
    // energy/f0_std/duration don't jointly pin down a unique state).
    // ------------------------------------------------------------------
    let calib_features: Vec<[f32; 3]> = results
        .iter()
        .map(|r| [r.energy_mean, r.f0_std, r.duration_s])
        .collect();
    let calib_arousal: Vec<f32> = results.iter().map(|r| r.arousal).collect();
    let calib_valence: Vec<f32> = results.iter().map(|r| r.valence).collect();
    let arousal_model = fit_ols(&calib_features, &calib_arousal);
    let valence_model = fit_ols(&calib_features, &calib_valence);

    println!(
        "\nSelf-perception closure: fitting (energy, f0_std, duration) -> (arousal, valence)\n\
         on the {} calibration points above, then testing on {} held-out states never used\n\
         for fitting.",
        calib_features.len(),
        HELD_OUT.len()
    );

    let mut perceived_arousal = Vec::new();
    let mut perceived_valence = Vec::new();
    let mut true_arousal = Vec::new();
    let mut true_valence = Vec::new();
    println!("\nHeld-out test:");
    for point in HELD_OUT {
        voice.update_from_consciousness(
            0.7,
            0.1,
            point.valence,
            point.arousal,
            false,
            1.0,
            1.0,
            0.05,
        );
        let audio = voice.synthesize(TEST_SENTENCE)?;
        let duration_s = audio.len() as f32 / sample_rate as f32;
        let frames = extract_formant_frames(&audio, sample_rate, &extraction);
        let s = summarize(&frames, duration_s);
        let features = [s.energy_mean, s.f0_std, s.duration_s];
        let p_arousal = predict_ols(arousal_model, features);
        let p_valence = predict_ols(valence_model, features);
        println!(
            "  {:<20} true: valence={:>5.2} arousal={:>5.2}  |  perceived: valence={:>5.2} arousal={:>5.2}",
            point.name, point.valence, point.arousal, p_valence, p_arousal
        );
        true_arousal.push(point.arousal);
        true_valence.push(point.valence);
        perceived_arousal.push(p_arousal);
        perceived_valence.push(p_valence);
    }

    let arousal_self_r = pearson(&true_arousal, &perceived_arousal);
    let valence_self_r = pearson(&true_valence, &perceived_valence);
    let arousal_mae: f32 = true_arousal
        .iter()
        .zip(&perceived_arousal)
        .map(|(t, p)| (t - p).abs())
        .sum::<f32>()
        / true_arousal.len() as f32;
    let valence_mae: f32 = true_valence
        .iter()
        .zip(&perceived_valence)
        .map(|(t, p)| (t - p).abs())
        .sum::<f32>()
        / true_valence.len() as f32;

    println!("\nSelf-perception accuracy on held-out states (true vs. perceived):");
    println!(
        "  arousal: Pearson r={arousal_self_r:>6.3}, mean abs error={arousal_mae:.3} (range 0..1)"
    );
    println!(
        "  valence: Pearson r={valence_self_r:>6.3}, mean abs error={valence_mae:.3} (range -1..1)"
    );
    println!(
        "\nInterpretation (self-perception): r well above ~0.5 with modest error on states the\n\
         model never saw means this pipeline, paired with this calibrated inverse model,\n\
         would recognize its own intended emotional state from its own synthesized audio —\n\
         a genuine (if simple, linear, single-sentence) self-hearing closure, not just a\n\
         restatement of the forward mapping above. This is narrow by design: two affect\n\
         dimensions, one fixed sentence, a linear model — not a general emotion recognizer."
    );

    Ok(())
}
