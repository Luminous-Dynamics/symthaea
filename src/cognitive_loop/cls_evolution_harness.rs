// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Shared CLS threshold-evolution harness (Tier 1.2, DISCOVERY_AND_SELF_IMPROVEMENT_PLAN_2026-07-06.md).
//!
//! This is the single implementation of "run N cycles through a REAL
//! `CognitiveLoopService` (never a cheap CfC proxy — a prior proxy attempt was
//! rejected because its ranking didn't transfer to the real loop, see
//! `memory/cls_evolution_finding.md`) and score Phi / FE-reduction /
//! prediction-accuracy / stability for a given `ThresholdPhenotype`". It is
//! shared by three call sites so they can never silently drift apart:
//!   - `examples/evolve_cls.rs` — evolves candidates, scores them on
//!     [`EVOLUTION_INPUTS`].
//!   - `examples/cls_promotion_gate.rs` — re-scores a saved candidate on
//!     disjoint [`FRESH_INPUTS`] to guard against overfitting to the
//!     evolution seeds before it can be promoted.
//!   - `tests/cls_threshold_promotion_e2e.rs` — proves the full
//!     evolve → gate → promote → construct-a-live-service loop.
//!
//! # The promotion path (mirrors the Broca curriculum bridge's shape)
//!
//! 1. `evolve_cls` writes a winning [`ThresholdPhenotype`] to
//!    `<candidate-dir>/candidate-phenotype.json` plus a [`CandidateProvenance`]
//!    sidecar at `<candidate-dir>/provenance.json`. Never touches any path the
//!    live system reads.
//! 2. `cls_promotion_gate` re-evaluates the candidate on [`FRESH_INPUTS`]
//!    (seeds never used during evolution) and, only if fresh fitness clears
//!    the recorded fitness within tolerance, writes
//!    `<candidate-dir>/PROMOTION_READY.json`.
//! 3. `scripts/cls_promote_candidate.sh` requires a human + an explicit
//!    confirm flag, backs up whatever was previously active, and copies the
//!    candidate phenotype to the path `ThresholdOverrides::from_env()` reads
//!    (`SYMTHAEA_THRESHOLD_OVERRIDES_PATH`). No live consumer hot-reloads this
//!    — a process must restart to pick up promoted thresholds, exactly like
//!    Broca checkpoint promotion.
//!
//! # Why `ThresholdPhenotype` JSON loads directly as `ThresholdOverrides`
//!
//! [`symthaea_neuroevolution::ThresholdPhenotype`] and
//! [`super::threshold_overrides::ThresholdOverrides`] declare the exact same
//! 18 fields, in the same order, with the same names and (modulo `Option<T>`)
//! the same types. Serializing a `ThresholdPhenotype` and deserializing the
//! result as a `ThresholdOverrides` round-trips every field into `Some(..)` —
//! no conversion code needed. `threshold_overrides.rs`'s test suite pins this
//! contract down explicitly so future field drift between the two structs is
//! caught at compile/test time rather than silently promoting a partial
//! override set.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use symthaea_neuroevolution::ThresholdPhenotype;

use super::{CognitiveLoopConfig, CognitiveLoopService};

/// Sentences used to *evolve* threshold phenotypes. Never reused by the
/// promotion gate — see [`FRESH_INPUTS`].
pub const EVOLUTION_INPUTS: &[&str] = &[
    "the sun rises over the mountain, warming the valley below",
    "a sudden crash echoes through the darkness",
    "gentle waves lap at the shore as the moon rises",
    "the machine grinds to a halt, alarms blaring",
    "children laughing in the garden after the rain",
    "silence falls across the empty room like snow",
    "new patterns emerge from chaos, self-organizing",
    "the old bridge creaks under the weight of memory",
    "music drifts through the open window at dusk",
    "a warning light blinks red on the console",
    "the forest canopy filters sunlight into emerald shards",
    "thunder rolls across the plains, shaking the earth",
    "a single bird calls across the frozen lake",
    "the reactor core temperature stabilizes at nominal",
    "two strangers share a knowing glance on the train",
];

/// Sentences used only by the re-evaluation gate (and the promotion e2e
/// test), deliberately disjoint from [`EVOLUTION_INPUTS`] so a candidate that
/// merely overfit to the evolution seeds cannot pass the gate.
pub const FRESH_INPUTS: &[&str] = &[
    "frost creeps across the windowpane before dawn",
    "the crowd falls quiet as the curtain rises",
    "a kettle whistles somewhere down the hall",
    "waves of static roll across the broken radio",
    "seedlings push through the cracked pavement",
    "the elevator groans between floors, then stops",
    "starlight scatters across the still black lake",
    "an alarm bell rings twice, then falls silent",
    "the potter's wheel spins a shape out of clay",
    "wind rattles the shutters through the long night",
    "a lantern sways on the porch in the storm",
    "the crowd erupts as the final whistle blows",
    "dust motes drift through a shaft of afternoon light",
    "the compass needle spins, then finally settles north",
    "a single candle gutters in the empty chapel",
];

#[test]
fn evolution_and_fresh_inputs_are_disjoint() {
    for s in EVOLUTION_INPUTS {
        assert!(
            !FRESH_INPUTS.contains(s),
            "FRESH_INPUTS must never overlap EVOLUTION_INPUTS (found {s:?}) — \
             the gate exists to catch overfitting to evolution seeds"
        );
    }
}

/// Fitness of a [`ThresholdPhenotype`] as measured by a REAL CLS run.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ClsFitness {
    pub mean_phi: f64,
    pub fe_reduction: f64,
    pub pred_accuracy: f64,
    pub phi_stability: f64,
    /// Internal consistency score from `evaluate_threshold_fitness` (cheap,
    /// no CLS cycles) — filled in by the caller, not by [`evaluate_with_cls`].
    pub threshold_consistency: f64,
}

impl ClsFitness {
    /// Multi-objective composite for ranking/gating. Weighted sum with
    /// Goodhart guards (see `evolve_cls.rs` module doc); Pareto would be
    /// better but is overkill for small populations / a single gate check.
    pub fn composite(&self) -> f64 {
        self.mean_phi * 0.35
            + self.fe_reduction.max(0.0) * 0.25
            + self.pred_accuracy * 0.20
            + self.phi_stability * 0.10
            + self.threshold_consistency * 0.10
    }
}

/// Evaluate a threshold phenotype by constructing a fresh, REAL
/// `CognitiveLoopService`, applying the phenotype as overrides, and running
/// `cycles` cognitive cycles over `inputs` (cycled if `cycles > inputs.len()`).
///
/// This is deliberately expensive (~1 sec/cycle) — it is the only way to
/// evaluate thresholds against their actual effect on the full consciousness
/// pipeline. Do not replace with a lightweight proxy (see module doc).
pub fn evaluate_with_cls(
    phenotype: &ThresholdPhenotype,
    inputs: &[&str],
    cycles: usize,
) -> ClsFitness {
    let config = CognitiveLoopConfig::default();
    let mut service = match CognitiveLoopService::new(config) {
        Ok(s) => s,
        Err(_) => return ClsFitness::default(),
    };

    service
        .threshold_overrides_mut()
        .apply_from_phenotype(phenotype);

    let mut phi_values = Vec::with_capacity(cycles);
    let mut pe_values = Vec::with_capacity(cycles);

    for i in 0..cycles {
        let input = inputs[i % inputs.len()];
        let result = service.cycle(input);
        let phi = result.metadata.consciousness.consciousness_level;
        phi_values.push(phi);
        pe_values.push(result.prediction_error as f64);
    }

    let n = cycles.max(1) as f64;
    let mean_phi = phi_values.iter().sum::<f64>() / n;

    let fe_reduction = if cycles >= 4 {
        let q = cycles / 4;
        let first_quarter: f64 = pe_values[..q].iter().sum::<f64>() / q as f64;
        let last_quarter: f64 = pe_values[3 * q..].iter().sum::<f64>() / (cycles - 3 * q) as f64;
        if first_quarter > 1e-6 {
            ((first_quarter - last_quarter) / first_quarter).clamp(-1.0, 1.0)
        } else {
            0.0
        }
    } else {
        0.0
    };

    let mean_pe: f64 = pe_values.iter().sum::<f64>() / n;
    let pred_accuracy = (1.0 - mean_pe / 2.0).clamp(0.0, 1.0);

    let phi_var = if cycles > 1 {
        phi_values
            .iter()
            .map(|&p| (p - mean_phi).powi(2))
            .sum::<f64>()
            / (n - 1.0)
    } else {
        0.0
    };
    let phi_stability = 1.0 / (1.0 + 10.0 * phi_var);

    ClsFitness {
        mean_phi,
        fe_reduction,
        pred_accuracy,
        phi_stability,
        threshold_consistency: 0.0, // filled by caller via evaluate_threshold_fitness
    }
}

/// Current git SHA (`git rev-parse HEAD`, trimmed), or `"unknown"` if the
/// command fails (e.g. not in a git checkout). Used for provenance sidecars
/// so a candidate/gate result always names the source tree it was produced
/// from.
pub fn current_git_sha() -> String {
    std::process::Command::new("git")
        .args(["rev-parse", "HEAD"])
        .output()
        .ok()
        .filter(|o| o.status.success())
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "unknown".to_string())
}

/// Provenance sidecar written alongside a candidate `ThresholdPhenotype`.
/// Mirrors the Broca curriculum bridge's manifest files: enough to answer
/// "how was this candidate produced, and how good was it, without re-running
/// evolution".
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CandidateProvenance {
    pub created_at_utc: String,
    pub git_sha: String,
    pub pop_size: usize,
    pub generations: usize,
    pub eval_cycles: usize,
    pub genesis_seed_phrase: String,
    pub evolution_input_count: usize,
    pub default_fitness: ClsFitness,
    pub final_fitness: ClsFitness,
}

/// Result of the re-evaluation gate, written to `PROMOTION_READY.json` only
/// when `passed == true`. The promote script (`scripts/cls_promote_candidate.sh`)
/// refuses to act on a candidate directory that lacks this file or where
/// `passed != true`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromotionReady {
    pub candidate_phenotype_path: String,
    pub created_at_utc: String,
    pub gate_git_sha: String,
    pub recorded_fitness: ClsFitness,
    pub fresh_fitness: ClsFitness,
    pub fresh_input_count: usize,
    pub eval_cycles: usize,
    pub tolerance: f64,
    pub passed: bool,
}

/// Load a `ThresholdPhenotype` candidate JSON file.
pub fn load_candidate_phenotype(path: &std::path::Path) -> Result<ThresholdPhenotype> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("reading candidate phenotype at {}", path.display()))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("parsing candidate phenotype at {}", path.display()))
}

/// Load a [`CandidateProvenance`] sidecar JSON file.
pub fn load_provenance(path: &std::path::Path) -> Result<CandidateProvenance> {
    let contents = std::fs::read_to_string(path)
        .with_context(|| format!("reading provenance at {}", path.display()))?;
    serde_json::from_str(&contents)
        .with_context(|| format!("parsing provenance at {}", path.display()))
}
