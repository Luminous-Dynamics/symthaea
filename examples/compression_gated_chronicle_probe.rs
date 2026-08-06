// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Predictive Compression Program C2 -- compression-gated Chronicle.
//!
//! Pre-registered protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md §8
//! (registered BEFORE this harness existed).
//!
//! Does gating episodic-memory writes/priority by `bits_saved_by_update` (loss reduction from
//! a real training update on this episode's own `(input, target)` pair) select meaningfully
//! different, more genuinely replay-useful episodes than the current gate (a hard
//! `psi < psi_threshold=0.3` reject, `crates/domains/symthaea-memory/src/episodic_replay.rs`)?
//!
//! Self-contained: uses `HdcLtcBridge` directly (not the full `CognitiveLoopService`/live
//! `episodic_replay.rs` gate -- zero production-code risk, matching this program's
//! "measure before you gate" discipline at its most conservative). `psi` here is a disclosed
//! proxy (`1 / (1 + pre_loss)`, mapped into the same [0,1] range the real psi/coherence
//! channel occupies), not a reproduction of the live coherence signal -- this harness doesn't
//! run the full cognitive loop, so the real channel isn't available to it.
//!
//! Two standing lessons from this session applied directly:
//! 1. Within-arm correlation is not causation -- the replay-utility metric below is a TRUE
//!    on/off counterfactual (clone the final bridge, replay one candidate episode in one
//!    clone only, compare held-out loss between clones), not an observed correlation.
//! 2. Recall-harm findings are schedule-dependent -- this runs under two structurally
//!    different content schedules before drawing any general conclusion.
//!
//! Run: cargo run --release --example compression_gated_chronicle_probe

use ndarray::Array1;
use std::collections::HashSet;
use symthaea::hdc_ltc_bridge::{HdcLtcBridge, HdcLtcBridgeConfig};
use symthaea_core::hdc::hdc_ltc_unified::NetworkStateSnapshot;

const DT: f32 = 0.02;
const LR: f32 = 0.01;
const DIM: usize = 256;
const PSI_THRESHOLD: f64 = 0.3; // matches EpisodicReplayConfig::default().psi_threshold
const SCHEDULE_LEN: usize = 400;
const MAX_CAUSAL_CHECK_PER_DIRECTION: usize = 5;
/// C2 recalibration (docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md, "C2 recalibration --
/// multi-step compressibility"): number of training steps used to measure sustained
/// learnability of one (start, input, target) triple, replacing the single-step
/// `bits_saved` metric that failed its own manipulation check.
const K_STEPS: usize = 10;

#[derive(Clone)]
struct EpisodeRecord {
    timestamp: u64,
    tier: &'static str,
    psi_proxy: f64,
    pre_loss: f32,
    bits_saved: f32,
    bits_saved_k_step: f32,
    start_snapshot: NetworkStateSnapshot,
    input: Array1<f32>,
    target: Array1<f32>,
}

/// C2 recalibration: loss reduction achievable after `K_STEPS` training steps on a *fixed*
/// `(start, input, target)` triple, in a disposable clone that never affects the caller's real
/// bridge. Always trains from the same `start` snapshot each iteration (not chaining state
/// forward) -- only weights accumulate across the K calls, isolating "how learnable is this one
/// transition" from any state-trajectory effect. `train_step_from` is PURE w.r.t. evolution
/// state by construction, so this is safe to call repeatedly against the same snapshot.
fn k_step_bits_saved(
    bridge: &HdcLtcBridge,
    start: &NetworkStateSnapshot,
    input: &Array1<f32>,
    target: &Array1<f32>,
    pre_loss: f32,
) -> f32 {
    let mut probe = bridge.clone();
    let mut post_loss = pre_loss;
    for _ in 0..K_STEPS {
        post_loss = probe
            .train_step_from(start, input, target, DT, LR)
            .expect("train_step_from");
    }
    // train_step_from reports the loss BEFORE its own gradient step (matches
    // eval_loss_from's pre-update semantics, per test_eval_loss_from_matches_
    // train_step_from_pre_update_loss) -- so after K calls, `post_loss` is the loss going
    // INTO the Kth call, i.e. after K-1 real updates. Do one final pure evaluation to get
    // the loss after all K updates.
    let final_loss = probe.eval_loss_from(start, input, target, DT);
    let _ = post_loss; // superseded by final_loss; kept only to document the semantics above
    if pre_loss > 1e-8 && final_loss > 1e-8 {
        ((pre_loss as f64) / (final_loss as f64)).log2() as f32
    } else {
        0.0
    }
}

/// Highly regular content: a single constant vector, trivially compressible once seen.
fn easy_vector() -> Array1<f32> {
    Array1::from_vec(vec![0.6f32; DIM])
}

/// Structurally varied content: `variant` selects a distinct waveform, deliberately harder to
/// compress than the constant easy vector.
fn hard_vector(variant: usize) -> Array1<f32> {
    let mut v = vec![0.0f32; DIM];
    for (i, slot) in v.iter_mut().enumerate() {
        let phase = (variant as f32) * 1.7 + (i as f32) * 0.13;
        *slot = (phase.sin() * 0.5 + (phase * 2.3).cos() * 0.3).clamp(-1.0, 1.0);
    }
    Array1::from_vec(v)
}

/// Schedule A: strict 1:1 alternation (easy, hard, easy, hard, ...).
fn schedule_alternating(total: usize) -> Vec<(&'static str, Array1<f32>)> {
    (0..total)
        .map(|t| {
            if t % 2 == 0 {
                ("easy", easy_vector())
            } else {
                ("hard", hard_vector(t % 4))
            }
        })
        .collect()
}

/// Schedule B: 1 easy : 3 hard -- structurally different interleaving from Schedule A, per the
/// schedule-dependence lesson.
fn schedule_skewed(total: usize) -> Vec<(&'static str, Array1<f32>)> {
    (0..total)
        .map(|t| {
            if t % 4 == 0 {
                ("easy", easy_vector())
            } else {
                ("hard", hard_vector(t % 4))
            }
        })
        .collect()
}

/// A held-out set, deliberately using `hard_vector` variant indices never seen during
/// training (100+ instead of 0-3) so held-out loss reflects genuine generalization, not
/// memorized training content.
fn held_out_pairs() -> Vec<(Array1<f32>, Array1<f32>)> {
    (0..10)
        .map(|i| (hard_vector(100 + i), hard_vector(200 + i)))
        .collect()
}

fn mean_held_out_loss(bridge: &mut HdcLtcBridge, pairs: &[(Array1<f32>, Array1<f32>)]) -> f32 {
    let start = bridge.snapshot_evolution_state();
    let total: f32 = pairs
        .iter()
        .map(|(input, target)| bridge.eval_loss_from(&start, input, target, DT))
        .sum();
    total / pairs.len() as f32
}

/// Runs one full schedule through a fresh bridge, returning per-step records plus the bridge
/// in its final (fully-trained-on-this-schedule) state, for the causal replay check.
fn run_schedule(schedule: &[(&'static str, Array1<f32>)]) -> (Vec<EpisodeRecord>, HdcLtcBridge) {
    let mut bridge = HdcLtcBridge::new(HdcLtcBridgeConfig::default());
    let mut records = Vec::with_capacity(schedule.len());

    for t in 0..schedule.len().saturating_sub(1) {
        let (tier, input) = &schedule[t];
        let (_, target) = &schedule[t + 1];

        let start = bridge.snapshot_evolution_state();

        let pre_loss = bridge.eval_loss_from(&start, input, target, DT);

        // C2 recalibration: measure sustained learnability BEFORE the real single online
        // step, in a disposable clone of the bridge as it exists right now -- never affects
        // the real bridge's own single-step training below.
        let bits_saved_k_step = k_step_bits_saved(&bridge, &start, input, target, pre_loss);

        let _ = bridge
            .train_step_from(&start, input, target, DT, LR)
            .expect("train_step_from");
        let post_loss = bridge.eval_loss_from(&start, input, target, DT);

        let bits_saved = if pre_loss > 1e-8 && post_loss > 1e-8 {
            ((pre_loss as f64) / (post_loss as f64)).log2() as f32
        } else {
            0.0
        };

        // Advance the LIVE evolution state for real -- a separate concern from the pure
        // training-on-snapshot pass above, matching the cognitive loop's own
        // step-then-train-on-historical-snapshot pattern.
        let _ = bridge.step(input, DT);

        let psi_proxy = (1.0 / (1.0 + pre_loss as f64)).clamp(0.0, 1.0);

        records.push(EpisodeRecord {
            timestamp: t as u64,
            tier,
            psi_proxy,
            pre_loss,
            bits_saved,
            bits_saved_k_step,
            start_snapshot: start,
            input: input.clone(),
            target: target.clone(),
        });
    }

    (records, bridge)
}

/// Rank-based top-half selection by `psi_proxy`, NOT the literal absolute `PSI_THRESHOLD`
/// value: a first run using `psi_proxy >= PSI_THRESHOLD` directly found the threshold never
/// rejected anything (399/399 selected in both schedules) -- `psi_proxy`'s invented [0,1]
/// mapping from `pre_loss` was never calibrated to actually straddle 0.3 for this harness's
/// loss scale, an absolute-threshold miscalibration disclosed here rather than silently
/// patched over. Switching to a rank-based top-K (matching `compression_gate_selected`'s own
/// rank-based design) removes the dependency on getting an arbitrary absolute cutoff right,
/// and still faithfully answers the real question: do these two candidate priority signals
/// select the same episodes at an equal, non-degenerate acceptance rate?
fn current_gate_selected(records: &[EpisodeRecord]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..records.len()).collect();
    idx.sort_by(|&a, &b| {
        records[b]
            .psi_proxy
            .partial_cmp(&records[a].psi_proxy)
            .unwrap()
    });
    idx.into_iter().take(records.len() / 2).collect()
}

/// Rank-based top-K selection by an arbitrary per-episode key -- shared by both the original
/// single-step `bits_saved` gate and the recalibrated `bits_saved_k_step` gate.
fn top_k_by(
    records: &[EpisodeRecord],
    target_count: usize,
    key: impl Fn(&EpisodeRecord) -> f32,
) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..records.len()).collect();
    idx.sort_by(|&a, &b| key(&records[b]).partial_cmp(&key(&records[a])).unwrap());
    idx.into_iter().take(target_count).collect()
}

fn jaccard_overlap(a: &[usize], b: &[usize]) -> f64 {
    let set_a: HashSet<_> = a.iter().collect();
    let set_b: HashSet<_> = b.iter().collect();
    let inter = set_a.intersection(&set_b).count();
    let union = set_a.union(&set_b).count();
    if union == 0 {
        1.0
    } else {
        inter as f64 / union as f64
    }
}

/// True on/off counterfactual: does replaying this ONE candidate episode (an extra training
/// step on its own (start, input, target)) reduce held-out loss relative to not replaying it?
/// Both arms are independent clones of the same final bridge state -- no shared mutation.
fn causal_replay_check(
    final_bridge: &HdcLtcBridge,
    episode: &EpisodeRecord,
    held_out: &[(Array1<f32>, Array1<f32>)],
) -> (f32, f32) {
    let mut replay_arm = final_bridge.clone();
    let mut no_replay_arm = final_bridge.clone();

    let _ = replay_arm
        .train_step_from(
            &episode.start_snapshot,
            &episode.input,
            &episode.target,
            DT,
            LR,
        )
        .expect("train_step_from");

    let replay_loss = mean_held_out_loss(&mut replay_arm, held_out);
    let no_replay_loss = mean_held_out_loss(&mut no_replay_arm, held_out);
    (replay_loss, no_replay_loss)
}

fn run_and_report(schedule_name: &str, schedule: &[(&'static str, Array1<f32>)]) {
    println!(
        "=== Schedule: {schedule_name} ({} steps) ===",
        schedule.len()
    );

    let (records, final_bridge) = run_schedule(schedule);

    let n_easy = records.iter().filter(|r| r.tier == "easy").count();
    let n_hard = records.iter().filter(|r| r.tier == "hard").count();
    println!("  content mix: easy={n_easy} hard={n_hard}");

    let pre_losses: Vec<f32> = records.iter().map(|r| r.pre_loss).collect();
    let bits_saveds: Vec<f32> = records.iter().map(|r| r.bits_saved).collect();
    let psi_proxies: Vec<f64> = records.iter().map(|r| r.psi_proxy).collect();
    let stats = |v: &[f32]| {
        let min = v.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = v.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mean = v.iter().sum::<f32>() / v.len() as f32;
        (min, max, mean)
    };
    let (pl_min, pl_max, pl_mean) = stats(&pre_losses);
    let (bs_min, bs_max, bs_mean) = stats(&bits_saveds);
    let psi_mean = psi_proxies.iter().sum::<f64>() / psi_proxies.len() as f64;
    println!(
        "  pre_loss:   min={pl_min:.6} max={pl_max:.6} mean={pl_mean:.6} (PSI_THRESHOLD={PSI_THRESHOLD} \
         unused now, see current_gate_selected's doc comment)"
    );
    println!("  bits_saved: min={bs_min:.4} max={bs_max:.4} mean={bs_mean:.4}");
    println!("  psi_proxy mean={psi_mean:.6}");

    let current = current_gate_selected(&records);
    println!(
        "  current gate (top-half by psi_proxy): {}/{} selected ({:.1}%)",
        current.len(),
        records.len(),
        100.0 * current.len() as f64 / records.len() as f64
    );

    let bits_saved_k_steps: Vec<f32> = records.iter().map(|r| r.bits_saved_k_step).collect();
    let (bsk_min, bsk_max, bsk_mean) = stats(&bits_saved_k_steps);
    println!(
        "  bits_saved_k_step (K={K_STEPS}): min={bsk_min:.4} max={bsk_max:.4} mean={bsk_mean:.4}"
    );

    let held_out = held_out_pairs();
    let baseline_loss = mean_held_out_loss(&mut final_bridge.clone(), &held_out);
    println!("  baseline held-out loss (no extra replay) = {baseline_loss:.6}");

    fn key_bits_saved(r: &EpisodeRecord) -> f32 {
        r.bits_saved
    }
    fn key_bits_saved_k_step(r: &EpisodeRecord) -> f32 {
        r.bits_saved_k_step
    }
    let metric_variants: [(&str, fn(&EpisodeRecord) -> f32); 2] = [
        ("single-step bits_saved", key_bits_saved),
        ("K-step bits_saved_k_step", key_bits_saved_k_step),
    ];
    for (metric_label, key) in metric_variants {
        println!("  --- compression gate variant: {metric_label} ---");
        let compression = top_k_by(&records, current.len(), key);
        println!(
            "  compression gate (top-{} by {metric_label}, rate-matched to current gate)",
            current.len()
        );

        let overlap = jaccard_overlap(&current, &compression);
        println!("  Jaccard overlap(current, compression) = {overlap:.4}");

        let current_set: HashSet<_> = current.iter().collect();
        let compression_set: HashSet<_> = compression.iter().collect();
        let current_only: Vec<usize> = current
            .iter()
            .copied()
            .filter(|i| !compression_set.contains(i))
            .collect();
        let compression_only: Vec<usize> = compression
            .iter()
            .copied()
            .filter(|i| !current_set.contains(i))
            .collect();
        println!(
            "  disagreement set: current_only={} compression_only={}",
            current_only.len(),
            compression_only.len()
        );

        for (label, sample) in [
            ("current_only", &current_only),
            ("compression_only", &compression_only),
        ] {
            let mut deltas = Vec::new();
            for &idx in sample.iter().take(MAX_CAUSAL_CHECK_PER_DIRECTION) {
                let (replay_loss, no_replay_loss) =
                    causal_replay_check(&final_bridge, &records[idx], &held_out);
                let delta = no_replay_loss - replay_loss; // positive = replay helped
                deltas.push(delta);
                println!(
                    "    [{label}] episode t={} tier={} bits_saved={:.4} bits_saved_k_step={:.4}: \
                     replay_loss={:.6} no_replay_loss={:.6} delta={delta:+.6}",
                    records[idx].timestamp,
                    records[idx].tier,
                    records[idx].bits_saved,
                    records[idx].bits_saved_k_step,
                    replay_loss,
                    no_replay_loss
                );
            }
            if !deltas.is_empty() {
                let mean_delta = deltas.iter().sum::<f32>() / deltas.len() as f32;
                println!(
                    "    [{label}] mean causal replay-utility delta (n={}) = {mean_delta:+.6} \
                     (positive = replay reduced held-out loss)",
                    deltas.len()
                );
            } else {
                println!("    [{label}] no episodes in this direction (empty disagreement set)");
            }
        }
    }
    println!();
}

fn main() {
    println!("Predictive Compression C2 -- compression-gated Chronicle");
    println!("protocol: docs/PREDICTIVE_COMPRESSION_PROGRAM_2026-07-17.md (Experiment C2)");
    println!();

    run_and_report("A: alternating 1:1", &schedule_alternating(SCHEDULE_LEN));
    run_and_report("B: skewed 1:3", &schedule_skewed(SCHEDULE_LEN));

    println!(
        "done. Append results + verdict to the protocol doc (§8, C2 Results), per house \
         convention."
    );
}
