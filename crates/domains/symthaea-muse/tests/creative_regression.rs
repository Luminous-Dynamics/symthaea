// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Creative quality regression gate.
//!
//! Locks in the minimum acceptable quality for Symthaea's music synthesis.
//! If any metric drops below `BASELINE - TOLERANCE`, the test fails, alerting
//! that a code change has degraded creative output.
//!
//! # Do not re-baseline from our own output
//!
//! This file used to say "copy the printed scores into the BASELINE_* constants
//! below". That instruction is why four of those constants are ~15% below whatever
//! this system happened to produce in April 2026, and it is a loop: the gate is
//! set from the output, so the output always passes the gate.
//!
//! ```bash
//! cargo test -p symthaea-muse -- print_creative_benchmark --nocapture
//! ```
//!
//! now prints each gate's **headroom** instead of a paste-ready block. Read it
//! before touching a constant. A gate with 55% headroom is not protecting anything.
//!
//! # RETRACTION — what this file used to claim about itself
//!
//! Until 2026-07-31 this header read: *"They catch gross regressions (a synthesis
//! bug that flattens every note to one pitch would destroy melodic coherence)."*
//!
//! **That is false, and it is exactly backwards.** A one-pitch drone scores
//! `melodic_coherence` **0.9141** against the real system's 0.7049 — the named bug
//! *raises* the metric by 29.7%, because the interval table gives unison the single
//! highest probability (`creative_bench.rs:57`), so a melody of nothing but unisons
//! is a melody of maximally-probable intervals. The drone also beats real output on
//! the **composite** (0.7366 vs 0.7349). See
//! `degenerate_drone_outscores_real_output_this_is_the_defect`, which measures this
//! on every run.
//!
//! These metrics are *proxy* measures, not ground truth, and they are weaker than
//! that phrase suggests: they catch some regressions, prefer some others, and are
//! no substitute for listening.
//!
//! # Known defects, found by measurement rather than by reading
//!
//! A 5-metric audit on 2026-07-31 (25 agents, 20 findings, none refuted under
//! adversarial verification) found every metric here defective in at least one way.
//! Full record with file:line evidence:
//! `symthaea/docs/design/CREATIVE_METRIC_VALIDITY_AUDIT_2026-07-31.md`.
//!
//! - **Inverted.** `onset_evenness` is `1 − CV(IOI)`, so 1.0 is metronomic. It
//!   carried a one-sided floor of 0.780 — above the median of real recorded
//!   performance. **Fixed**: now a two-sided band measured from MAESTRO, and it is
//!   one of only two gates that catch the drone.
//! - **Inverted.** `melodic_coherence` is maximized by a single repeated pitch (see
//!   the retraction above). Still gated by a one-sided floor. NOT fixed.
//! - **Saturated.** `form_compliance` returns exactly 1.000 for *any* composition
//!   with an empty analysis window — a structural guarantee, and the reason it reads
//!   1.000 here. NOT fixed.
//! - **Tautological.** `emotional_alignment` has no causal channel from the target
//!   valence/arousal to the composition in the gated path. NOT fixed.
//! - **Inherited.** The `composite` still adds `+0.25 × onset_evenness` under a
//!   one-sided floor, so this file contains two gates assigning opposite sign to the
//!   same quantity. NOT fixed.
//!
//! The lesson generalises: a one-sided threshold on a two-sided quantity is a bug,
//! and a virtuous-sounding name is how it survives review.

use symthaea_muse::{MuseConfig, MusicalState, creative_bench};

/// Regression tolerance: a metric may decline by at most this much before failing.
/// Allows for minor floating-point variance across platforms.
const TOLERANCE: f32 = 0.05;

// ─── Baselines ────────────────────────────────────────────────────────────────
// Established from the initial production implementation (Apr 2026).
// Run `print_creative_benchmark` to regenerate.

// Measured 2026-04-11 (post-rhythm+sparsity fix):
//   melodic=0.755, rhythm=0.933, emotion=0.463, form=0.606, composite=0.704
// Baselines set ~15% below measured to absorb cross-platform + stochastic variance.
const BASELINE_MELODIC_COHERENCE: f32 = 0.64;
const BASELINE_EMOTIONAL_ALIGNMENT: f32 = 0.38;
const BASELINE_FORM_COMPLIANCE: f32 = 0.50;
const BASELINE_COMPOSITE: f32 = 0.59;

/// Onset evenness is NOT a baseline — it is a two-sided band, and it does not
/// come from our own output.
///
/// The four constants above are self-referential: they were set ~15% below what
/// this system happened to produce in April 2026, which makes them regression
/// detectors and nothing more. That is fine for a regression detector, but it is
/// how the old `BASELINE_ONSET_EVENNESS = 0.78` went wrong — the metric is
/// `1 − CV(IOI)`, so a floor on it is a demand for *metronomic* playing, and
/// nobody noticed because "rhythmic regularity" sounds like a virtue.
///
/// These two numbers are the 5th and 95th percentiles of **MAESTRO**, scored by
/// the same production function, grouped to the same unit the gate reads (a mean
/// over `n_compositions`). Real repertoire's median is 0.607. The old 0.78 floor
/// would have failed **90% of real recorded performances**.
///
/// Regenerate with:
/// ```sh
/// cargo run --release -p symthaea-muse --features theory \
///   --example rhythm_gate_calibration -- /opt/datasets/maestro 60
/// ```
const REPERTOIRE_EVENNESS_LOW: f32 = 0.36;
const REPERTOIRE_EVENNESS_HIGH: f32 = 0.82;

// ─── Standard benchmark config ────────────────────────────────────────────────

fn bench_config() -> MuseConfig {
    MuseConfig {
        duration_secs: 4.0,
        max_notes: 16,
        ..MuseConfig::default()
    }
}

// ─── Regression gate ──────────────────────────────────────────────────────────

#[test]
fn creative_quality_regression() {
    let result = creative_bench::run_benchmark(&bench_config(), &MusicalState::default());

    assert!(
        result.mean_melodic_coherence >= BASELINE_MELODIC_COHERENCE - TOLERANCE,
        "melodic coherence REGRESSION: {:.3} < baseline {:.3} - tolerance {:.3}\n{}",
        result.mean_melodic_coherence,
        BASELINE_MELODIC_COHERENCE,
        TOLERANCE,
        result.report(),
    );
    // NOTE: onset evenness is gated separately and two-sidedly — see
    // `onset_evenness_within_repertoire_band` below. It does not belong in this
    // list because every constant here is a one-sided floor.
    assert!(
        result.mean_emotional_alignment >= BASELINE_EMOTIONAL_ALIGNMENT - TOLERANCE,
        "emotional alignment REGRESSION: {:.3} < baseline {:.3} - tolerance {:.3}\n{}",
        result.mean_emotional_alignment,
        BASELINE_EMOTIONAL_ALIGNMENT,
        TOLERANCE,
        result.report(),
    );
    assert!(
        result.mean_form_compliance >= BASELINE_FORM_COMPLIANCE - TOLERANCE,
        "form compliance REGRESSION: {:.3} < baseline {:.3} - tolerance {:.3}\n{}",
        result.mean_form_compliance,
        BASELINE_FORM_COMPLIANCE,
        TOLERANCE,
        result.report(),
    );
    assert!(
        result.mean_composite >= BASELINE_COMPOSITE - TOLERANCE,
        "composite score REGRESSION: {:.3} < baseline {:.3} - tolerance {:.3}\n{}",
        result.mean_composite,
        BASELINE_COMPOSITE,
        TOLERANCE,
        result.report(),
    );
}

/// UN-PARKED 2026-07-31: two-sided, calibrated against real performance, and it
/// runs on every CI run like the other four.
///
/// # History, because the failure mode is the interesting part
///
/// This test spent two days `#[ignore]`d. The reasoning at the time was sound —
/// the gate was a one-sided floor of 0.780 on `1 − CV(IOI)`, which is a demand
/// for *metronomic* playing, so the observed 0.583 was plausibly a musical
/// improvement being penalised by construction. Bumping the constant would have
/// tuned a quality gate to match its own output; deleting it would have lost a
/// real signal. Parking it was the honest third option.
///
/// But `#[ignore]` is not a resting state, it is a slow leak: an ignored test
/// looks like coverage in the file and provides none, and the note said only
/// that "a musical-intent decision" was needed — which nobody was ever going to
/// make on a Tuesday. An external review named this correctly as rot.
///
/// # What un-parked it
///
/// The decision did not need taste after all, only a number from real music.
/// `examples/rhythm_gate_calibration.rs` scores MAESTRO — 60 recital
/// performances, 23,446 windows — through this exact production function and
/// grouped to this exact unit (a mean over `n_compositions`):
///
/// ```text
///                                     p05   median    mean    p95
///   MAESTRO, mean of 5 (gate's unit)  0.356   0.607   0.600   0.824
///   symthaea-muse (ours)                  -       -   0.583      -
/// ```
///
/// So the old floor of 0.780 sat **above the repertoire median**: 90% of real
/// recorded performances would have failed it. And our 0.583 is not a defect —
/// it is 0.024 from the median of real music.
///
/// # Why a band, and why these numbers are not ours
///
/// The band is the repertoire's own 5th–95th percentile. It is two-sided on
/// purpose: this metric is not a virtue, and drifting toward 1.0 (drum-machine
/// rigidity) is as much a regression as drifting toward 0.0 (no pulse). Unlike
/// the four constants above, these numbers do not come from our own output, so
/// this gate cannot be satisfied by moving.
///
/// See the calibration harness for the confounds; the load-bearing one is that
/// grouping *consecutive* repertoire windows keeps within-piece correlation, so
/// the band is wider — more permissive — than independent sampling would give.
#[test]
fn onset_evenness_within_repertoire_band() {
    let result = creative_bench::run_benchmark(&bench_config(), &MusicalState::default());
    let v = result.mean_onset_evenness;
    assert!(
        (REPERTOIRE_EVENNESS_LOW..=REPERTOIRE_EVENNESS_HIGH).contains(&v),
        "onset evenness {v:.3} is outside the band real music occupies \
         [{REPERTOIRE_EVENNESS_LOW:.2}, {REPERTOIRE_EVENNESS_HIGH:.2}] \
         (MAESTRO median 0.607).\n\
         This metric is 1 - CV(IOI): {} \
         DO NOT 'fix' this by widening the band to admit the new value — the \
         band is measured from repertoire, not from us. Re-run \
         examples/rhythm_gate_calibration.rs if you believe the measurement is \
         wrong.\n{}",
        if v > REPERTOIRE_EVENNESS_HIGH {
            "too HIGH means mechanically metronomic, not 'better'."
        } else {
            "too LOW means no detectable pulse."
        },
        result.report(),
    );
}

/// Regression gate for the neural melody mode (CfC-driven generation).
#[test]
fn neural_melody_regression() {
    let config = MuseConfig {
        duration_secs: 4.0,
        max_notes: 16,
        melody_mode: symthaea_muse::MelodyMode::Neural,
        ..MuseConfig::default()
    };
    let result = creative_bench::run_benchmark(&config, &MusicalState::default());

    // Neural mode has its own baseline (lower than classic due to CfC timing).
    const NEURAL_BASELINE_COMPOSITE: f32 = 0.35;
    assert!(
        result.mean_composite >= NEURAL_BASELINE_COMPOSITE - TOLERANCE,
        "neural melody composite REGRESSION: {:.3} < {:.3}\n{}",
        result.mean_composite,
        NEURAL_BASELINE_COMPOSITE - TOLERANCE,
        result.report(),
    );
}

/// Print current benchmark scores, and the *headroom* each gate has.
///
/// Run with:  cargo test -p symthaea-muse -- print_creative_benchmark --nocapture
///
/// This deliberately no longer emits a paste-ready block of new constants. It used
/// to, and that made re-baselining from our own output the path of least resistance
/// — which is how four of these constants came to be "~15% below whatever the system
/// produced in April 2026", a shape that detects regressions and cannot say whether
/// the output is any good. Headroom is printed instead, because a gate with 55%
/// headroom is a gate that will not fire.
///
/// `onset_evenness` is deliberately absent from the update path entirely: its band
/// is measured from repertoire, not from us, and regenerating it means re-running
/// `examples/rhythm_gate_calibration.rs` against MAESTRO.
#[test]
fn print_creative_benchmark() {
    let result = creative_bench::run_benchmark(&bench_config(), &MusicalState::default());
    println!("\n{}", result.report());

    let row = |name: &str, value: f32, floor: f32| {
        let trigger = floor - TOLERANCE;
        let headroom = if value > 0.0 {
            100.0 * (value - trigger) / value
        } else {
            0.0
        };
        println!(
            "  {name:<22} {value:>6.3}   fires below {trigger:>5.3}   headroom {headroom:>5.1}%"
        );
    };
    println!("\nGate headroom — how far each metric must fall before its gate fires:");
    row(
        "melodic_coherence",
        result.mean_melodic_coherence,
        BASELINE_MELODIC_COHERENCE,
    );
    row(
        "emotional_alignment",
        result.mean_emotional_alignment,
        BASELINE_EMOTIONAL_ALIGNMENT,
    );
    row(
        "form_compliance",
        result.mean_form_compliance,
        BASELINE_FORM_COMPLIANCE,
    );
    row("composite", result.mean_composite, BASELINE_COMPOSITE);
    println!(
        "  {:<22} {:>6.3}   two-sided band [{:.2}, {:.2}] measured from MAESTRO",
        "onset_evenness",
        result.mean_onset_evenness,
        REPERTOIRE_EVENNESS_LOW,
        REPERTOIRE_EVENNESS_HIGH,
    );
    println!(
        "\n  A metric reading 1.000 is pinned at its ceiling: it cannot detect any\n  \
         improvement, and only a large regression. Large headroom means the same\n  \
         thing more quietly. Neither is fixed by editing the constant."
    );
}

// ─── Degenerate-input proof ───────────────────────────────────────────────────

/// Build a deliberately terrible "composition": one pitch, perfectly metronomic,
/// constant velocity. This is the exact failure the module header claims the gates
/// catch ("a synthesis bug that flattens every note to one pitch").
fn drone(n: usize, secs: f32) -> symthaea_muse::Composition {
    let step = secs / n as f32;
    symthaea_muse::Composition {
        audio: symthaea_muse::AudioData::F32(Vec::new()),
        sample_rate: 48_000,
        notes: (0..n)
            .map(|i| symthaea_muse::Note {
                frequency: 440.0,
                start_time: i as f32 * step,
                duration: step * 0.9,
                velocity: 0.7,
            })
            .collect(),
        duration_secs: secs,
        section: symthaea_muse::structure::SectionType::Developmental,
    }
}

/// THE BENCHMARK SCORES A ONE-PITCH DRONE ABOVE THE REAL SYSTEM. Measured, locked in.
///
/// ```text
///   composite:          drone 0.7366  >  real output 0.7349
///   melodic_coherence:  drone 0.9141  >  real output 0.7049   (+29.7%)
/// ```
///
/// A characterization test, not an endorsement — read the name. It asserts the
/// *known-bad* current behaviour so that repairing any metric trips it and forces
/// whoever does the repair to update this record deliberately.
///
/// # Scope — stated precisely, because the obvious summary is wrong
///
/// The drone passes **3 of 5** gates, not all of them. It is caught by
/// `onset_evenness` (1.0, above the repertoire band) and by `form_compliance`
/// (0.0). "A drone passes every gate" would be false and this test does not claim
/// it.
///
/// But `form_compliance` catching it is **luck, not protection**: that metric
/// returns exactly 1.000 for *any* composition with an empty analysis window, so
/// chord-stab-then-silence — a more degenerate output than this drone — scores its
/// maximum. It happens to score 0.0 here because a perfectly uniform drone has zero
/// window-to-window variation. A detector that gives the worst possible input full
/// marks is not detecting degeneracy.
///
/// And `onset_evenness` only catches it because the gate was made two-sided on
/// 2026-07-31. Under the previous one-sided floor of 0.780, a metronomic drone's
/// 1.0 would have been welcomed as excellent.
///
/// Established 2026-07-31 by a 5-metric audit (25 agents, 20 findings, none refuted
/// under adversarial verification). Every number here is measured, not predicted.
#[test]
fn degenerate_drone_outscores_real_output_this_is_the_defect() {
    let d = drone(16, 4.0);
    let s = symthaea_muse::creative_bench::CreativeQualityScore::evaluate(
        &d,
        symthaea_aesthetic::ValenceArousal::new(0.5, 0.7),
    );
    let real = creative_bench::run_benchmark(&bench_config(), &MusicalState::default());

    // 1. THE HEADLINE. melodic_coherence does not merely tolerate a one-pitch
    //    collapse — it is MAXIMIZED by it. The interval table gives unison the
    //    single highest probability (0.18, creative_bench.rs:57), so a melody of
    //    nothing but unisons is a melody of maximally-probable intervals. The
    //    module header used to cite this exact bug as the thing the gates catch.
    assert!(
        s.melodic_coherence > real.mean_melodic_coherence,
        "melodic_coherence appears REPAIRED: a one-pitch drone scored {:.4}, no \
         longer above the real system's {:.4}. Update this test and the module \
         header, which documents the defect.",
        s.melodic_coherence,
        real.mean_melodic_coherence,
    );

    // 2. The composite — the gate that is supposed to summarize overall quality —
    //    also rates the drone above real output.
    assert!(
        s.composite > real.mean_composite,
        "composite appears REPAIRED: drone {:.4} no longer outscores real output \
         {:.4}.",
        s.composite,
        real.mean_composite,
    );
    assert!(
        s.composite >= BASELINE_COMPOSITE - TOLERANCE,
        "the composite gate now catches a plain drone ({:.4} < {:.4}) — a real \
         improvement. Update this test.",
        s.composite,
        BASELINE_COMPOSITE - TOLERANCE,
    );

    // 3. Record which two gates DO catch it, so a future change that silently
    //    loses either one is visible.
    assert!(
        s.onset_evenness > REPERTOIRE_EVENNESS_HIGH,
        "drone onset_evenness {:.4} no longer exceeds the band's upper edge — the \
         two-sided band is one of only two gates catching this input.",
        s.onset_evenness,
    );
    assert!(
        s.form_compliance < BASELINE_FORM_COMPLIANCE - TOLERANCE,
        "drone form_compliance {:.4} now passes — note this metric catches the \
         drone by luck, not design (it returns 1.000 for any empty window).",
        s.form_compliance,
    );

    println!(
        "\n  DEGENERATE DRONE (one pitch, metronomic, constant velocity)\n\
         \x20   melodic_coherence   {:.4}  (floor {BASELINE_MELODIC_COHERENCE}) -> {}\n\
         \x20   onset_evenness      {:.4}  (band [{REPERTOIRE_EVENNESS_LOW}, {REPERTOIRE_EVENNESS_HIGH}]) -> {}\n\
         \x20   emotional_alignment {:.4}  (floor {BASELINE_EMOTIONAL_ALIGNMENT}) -> {}\n\
         \x20   form_compliance     {:.4}  (floor {BASELINE_FORM_COMPLIANCE}) -> {}\n\
         \x20   composite           {:.4}  (floor {BASELINE_COMPOSITE}) -> {}\n\
         \x20 real system composite {:.4}",
        s.melodic_coherence,
        pass(s.melodic_coherence >= BASELINE_MELODIC_COHERENCE - TOLERANCE),
        s.onset_evenness,
        pass((REPERTOIRE_EVENNESS_LOW..=REPERTOIRE_EVENNESS_HIGH).contains(&s.onset_evenness)),
        s.emotional_alignment,
        pass(s.emotional_alignment >= BASELINE_EMOTIONAL_ALIGNMENT - TOLERANCE),
        s.form_compliance,
        pass(s.form_compliance >= BASELINE_FORM_COMPLIANCE - TOLERANCE),
        s.composite,
        pass(s.composite >= BASELINE_COMPOSITE - TOLERANCE),
        real.mean_composite,
    );
}

fn pass(b: bool) -> &'static str {
    if b { "PASSES gate" } else { "caught" }
}

// ─── Mechanism checks, independent of any metric ──────────────────────────────

/// The autonomous path's humanization and chord spread are verified STRUCTURALLY,
/// not through the benchmark.
///
/// Four of the five benchmark metrics cannot distinguish a one-pitch drone from
/// real output (see `degenerate_drone_outscores_real_output_this_is_the_defect`),
/// so "the score moved" is not evidence that anything improved. These assertions
/// read the note stream directly and check the two properties the 2026-07-31
/// change actually claims.
#[test]
fn autonomous_path_emits_humanized_non_simultaneous_notes() {
    let comp = symthaea_muse::compose(&bench_config(), &MusicalState::default(), 7);
    let mut onsets: Vec<f32> = comp.notes.iter().map(|n| n.start_time).collect();
    onsets.sort_by(|a, b| a.total_cmp(b));
    assert!(
        onsets.len() >= 6,
        "too few notes to judge: {}",
        onsets.len()
    );

    // 1. NOT ON A GRID. Before humanization every onset was an exact multiple of a
    //    subdivision. Jitter is Gaussian and sub-10ms, so the tell is that onsets
    //    are not all near-exact multiples of the smallest gap.
    let exact_grid = onsets
        .windows(2)
        .filter(|w| {
            let d = w[1] - w[0];
            d > 1e-4 && (d * 1000.0).fract() < 1e-3
        })
        .count();
    assert!(
        exact_grid < onsets.len() / 2,
        "{exact_grid} of {} gaps land on an exact millisecond grid — humanization \
         does not appear to be running on the autonomous path.",
        onsets.len() - 1,
    );

    // 2. CHORDS ARE ROLLED, NOT STACKED. Previously every tone of a chord shared
    //    one `start_time` exactly. Nothing should now be exactly simultaneous.
    let exact_ties = onsets.windows(2).filter(|w| w[1] == w[0]).count();
    assert_eq!(
        exact_ties, 0,
        "{exact_ties} pairs of notes share an identical onset — chord tones are \
         still attacking as one transient, which is the signature this change \
         exists to remove.",
    );

    // 3. The roll must stay SHORT. A spread wide enough to be heard as separate
    //    events would be a different defect, not a fix.
    let widest_cluster = onsets
        .windows(2)
        .map(|w| w[1] - w[0])
        .filter(|&d| d < 0.05)
        .fold(0.0f32, f32::max);
    assert!(
        widest_cluster < 0.05,
        "closely-spaced onsets span {widest_cluster:.4}s; a chord roll must stay \
         well under the ~30ms at which it stops being heard as one chord.",
    );
}
