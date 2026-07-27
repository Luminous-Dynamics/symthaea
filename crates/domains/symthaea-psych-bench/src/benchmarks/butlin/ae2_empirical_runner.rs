// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Real, backend-gated empirical runner for the Butlin **AE-2 (Embodiment)**
//! indicator only — step 5 of `symthaea/docs/BUTLIN_PR_B_RUNNER_PLAN_2026-07-27.md`'s
//! sequencing, and the first genuinely new empirical row in this campaign.
//! Scoped to AE-2 alone by explicit direction: `HOT-3`/`PP-1` and `GWT-3` are
//! deliberately deferred to a later, separate run, once this first one has
//! actually been reviewed. This module's job is not to "produce support" —
//! it is to answer seven narrower questions before `resolve_outcome()` ever
//! sees the measured effect:
//!
//! 1. Can the declared ablation actually be applied? → `target_hook_fired_fraction`
//!    (`module_timings_us.embodied_cognition`, mirroring the same real
//!    per-module timing signal `GWT-3`/`RPT-2` already use).
//! 2. Can the positive control actually be applied? → a pure data-mutation
//!    sanity check on this module's own probe (`ae2_probe`), not a live run
//!    (the design's control is `Instrumentation`-purpose, not a stimulus).
//! 3. Can the sham actually be applied? → `sham_lever_fired_fraction`
//!    (`module_timings_us.predictive_processing`, HOT-1's real ablation lever
//!    reused here as an unrelated disruptive control).
//! 4. Do instrumentation counters prove those hooks fired, independent of
//!    whether the probe moved? → the hook-fired fractions above, computed
//!    entirely separately from `embodied_agency` itself.
//! 5. Does the intervention change only the intended mechanism? → the
//!    specificity health panel (`HealthPanelEntry`), a small preregistered
//!    set of UNRELATED fields (HOT-2/AST-1/GWT-4/PP-1&HOT-3/GWT-3/RPT-2's own
//!    signals) checked for baseline-vs-target-ablated drift.
//! 6. Is the AE-2 signal numerically usable in all arms? → finite-value
//!    checks on `embodied_agency` across baseline/target/sham.
//! 7. Does the live registry identity match the frozen design? →
//!    `check_identity_against_registry` against the real, live
//!    `ablation::ablation_specs()` row, not a cached assumption.
//!
//! **The health-panel tolerance is a disclosed, uncalibrated first-pass
//! choice** (50% relative deviation, floored at 0.05) — real calibration
//! needs a baseline-variance study across seeds, which is the explicit
//! *next* step after this one (repeated seeds), not this one.
//!
//! **This run may not find `Supported`.** A qualified `Inconclusive` (e.g.
//! because the mutation hook turns out absent or nonspecific) is just as
//! valid a first empirical milestone as a positive finding, provided the
//! failure is diagnosed accurately — that is what `RuntimeQualification`'s
//! named fields are for.

use super::ablation::{ablation_specs, build_loop, run_downstream_benchmark};
use super::qualification_design::{ControlPurpose, planned_designs};
use super::qualification_runtime::{
    RuntimeQualification, check_identity_against_registry, resolve_outcome,
};
use super::report::{EvidenceOutcome, SupportTier, classify_ablation};
use symthaea::cognitive_loop::CycleMetadata;

const NUM_CYCLES: usize = 200;
const WARMUP: usize = 20;

/// First-pass, uncalibrated specificity tolerance — see module doc comment.
const HEALTH_PANEL_RELATIVE_TOLERANCE: f64 = 0.5;
const HEALTH_PANEL_MIN_FLOOR: f64 = 0.05;

/// Same stimulus content as `ablation::measure_indicator`'s cross-cycle loop
/// (that function's array is private to that module) — kept identical so
/// every arm sees equivalent input structure.
const STIMULI: [&str; 10] = [
    "The quick brown fox jumps over the lazy dog",
    "A neural network learns to predict sequences",
    "Consciousness emerges from integrated information",
    "Working memory maintains active representations",
    "Prediction errors drive learning and adaptation",
    "Social cognition requires mental model tracking",
    "The hippocampus consolidates episodic memories",
    "Attention selects relevant information for processing",
    "Free energy principle explains perception and action",
    "Temporal binding creates unified experience",
];

/// This module's own AE-2 probe — reads `embodied_agency` with no
/// transformation. Factored out so the positive control tests THIS exact
/// function, not a copy of its logic.
fn ae2_probe(metadata: &CycleMetadata) -> f64 {
    metadata.embodied.embodied_agency
}

fn mean(samples: &[CycleMetadata], extract: fn(&CycleMetadata) -> f64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    samples.iter().map(extract).sum::<f64>() / samples.len() as f64
}

/// Fraction of sampled cycles where `extract` (a `module_timings_us` field)
/// is nonzero — the real "did this module actually execute" signal, same
/// pattern `GWT-3`/`RPT-2` already use in `ablation.rs`.
fn hook_fired_fraction(samples: &[CycleMetadata], extract: fn(&CycleMetadata) -> u64) -> f64 {
    if samples.is_empty() {
        return 0.0;
    }
    let fired = samples.iter().filter(|m| extract(m) > 0).count();
    fired as f64 / samples.len() as f64
}

fn within_tolerance(baseline: f64, other: f64) -> bool {
    let floor = baseline.abs().max(HEALTH_PANEL_MIN_FLOOR);
    (other - baseline).abs() <= HEALTH_PANEL_RELATIVE_TOLERANCE * floor
}

/// One entry in the specificity health panel: a field UNRELATED to AE-2
/// that should stay put when the AE-2 target lever is applied. Continuous
/// fields compare the mean via `within_tolerance`; module-timing fields
/// compare the fired-*fraction* the same way (see
/// `module_activity_health_entry` for why "stay put" is the right question,
/// not "stays nonzero").
#[derive(Debug, Clone)]
pub struct HealthPanelEntry {
    pub field: &'static str,
    pub baseline: f64,
    pub target_ablated: f64,
    pub within_tolerance: bool,
}

fn continuous_health_entry(
    field: &'static str,
    baseline: &[CycleMetadata],
    ablated: &[CycleMetadata],
    extract: fn(&CycleMetadata) -> f64,
) -> HealthPanelEntry {
    let b = mean(baseline, extract);
    let a = mean(ablated, extract);
    HealthPanelEntry {
        field,
        baseline: b,
        target_ablated: a,
        within_tolerance: within_tolerance(b, a),
    }
}

/// Specificity check for a module-timing field: the fired-*fraction* must
/// stay put between arms. Deliberately does NOT require the fraction be
/// high in both arms — `cross_modal_binding` is never enabled by
/// `ConsciousnessProfile::Standard` at all (confirmed against
/// `config/consciousness.rs`'s `Standard` arm, which lists `enable_gwt`/
/// `enable_embodied_cognition` but not `enable_cross_modal_binding`), so its
/// honest baseline fraction is 0.0 -- a module that's off in BOTH arms is
/// exactly as specific as one that's on in both, and a naive "both > 0.9"
/// rule would misreport that as a specificity failure (caught in the first
/// real run of this module, 2026-07-27 -- see the empirical bundle).
fn module_activity_health_entry(
    field: &'static str,
    baseline: &[CycleMetadata],
    ablated: &[CycleMetadata],
    extract: fn(&CycleMetadata) -> u64,
) -> HealthPanelEntry {
    let b = hook_fired_fraction(baseline, extract);
    let a = hook_fired_fraction(ablated, extract);
    HealthPanelEntry {
        field,
        baseline: b,
        target_ablated: a,
        within_tolerance: within_tolerance(b, a),
    }
}

/// One entry in the broader diagnostic snapshot: descriptive only, never
/// used to compute `intervention_specificity_passed` or any other
/// qualification field. Exists so a future reviewer can spot an unexpected
/// collateral change without retroactively moving the qualification gate —
/// the health panel stays frozen to its 6 preregistered entries regardless
/// of what shows up here.
#[derive(Debug, Clone)]
pub struct DiagnosticEntry {
    pub field: &'static str,
    pub baseline: f64,
    pub target_ablated: f64,
}

fn diagnostic_entry(
    field: &'static str,
    baseline: &[CycleMetadata],
    ablated: &[CycleMetadata],
    extract: fn(&CycleMetadata) -> f64,
) -> DiagnosticEntry {
    DiagnosticEntry {
        field,
        baseline: mean(baseline, extract),
        target_ablated: mean(ablated, extract),
    }
}

fn diagnostic_bool_fraction_entry(
    field: &'static str,
    baseline: &[CycleMetadata],
    ablated: &[CycleMetadata],
    extract: fn(&CycleMetadata) -> bool,
) -> DiagnosticEntry {
    let frac = |samples: &[CycleMetadata]| -> f64 {
        if samples.is_empty() {
            return 0.0;
        }
        samples.iter().filter(|m| extract(m)).count() as f64 / samples.len() as f64
    };
    DiagnosticEntry {
        field,
        baseline: frac(baseline),
        target_ablated: frac(ablated),
    }
}

fn run_arm(
    mutator: Option<fn(&mut symthaea::cognitive_loop::CognitiveLoopConfig)>,
) -> Vec<CycleMetadata> {
    let mut service =
        build_loop(mutator).expect("AE-2 empirical runner: CognitiveLoopService must build");
    let mut samples = Vec::with_capacity(NUM_CYCLES.saturating_sub(WARMUP));
    for i in 0..NUM_CYCLES {
        let input = STIMULI[i % STIMULI.len()];
        let result = service.cycle(input);
        if i >= WARMUP {
            samples.push(result.metadata);
        }
    }
    samples
}

/// Full exported record of one AE-2 empirical run — every field the design
/// review asked for: exact identities, hook-execution counters, pre/post
/// manipulated-field values, the specificity health panel, the raw AE-2
/// signal, the downstream behavioral metric, every `RuntimeQualification`
/// field, and the final `EvidenceOutcome`.
#[derive(Debug, Clone)]
pub struct Ae2EmpiricalRun {
    pub config_hash: String,
    pub seed_identity: String,
    pub target_lever_name: &'static str,
    pub sham_lever_name: &'static str,
    pub functional_benchmark: &'static str,
    pub num_cycles: usize,
    pub warmup: usize,

    pub baseline_embodied_agency: f64,
    pub target_ablated_embodied_agency: f64,
    pub sham_embodied_agency: f64,

    /// Fraction of sampled cycles with `module_timings_us.embodied_cognition
    /// > 0` in the BASELINE arm (expected ~1.0).
    pub baseline_hook_fired_fraction: f64,
    /// Same field, TARGET-ABLATED arm (expected ~0.0 — proves the
    /// intervention fired, independent of whether the probe moved).
    pub target_hook_fired_fraction: f64,
    /// Fraction of sampled cycles with `module_timings_us.predictive_processing
    /// == 0` in the SHAM arm (expected ~1.0 — proves the sham's OWN lever
    /// fired, independent of the AE-2 signal).
    pub sham_lever_fired_fraction: f64,

    pub health_panel: Vec<HealthPanelEntry>,

    pub baseline_benchmark_accuracy: f64,
    pub target_ablated_benchmark_accuracy: f64,

    /// `Err` iff this run's declared identity doesn't match the live
    /// `ablation_specs()` registry right now — a hard failure, reported
    /// separately from `qualification` (see `QualificationRunError`).
    pub identity_check: Result<(), String>,

    pub positive_control_zero_read_correct: bool,
    pub positive_control_restore_read_correct: bool,
    /// The design's declared purpose for this positive control —
    /// `Instrumentation` here, NOT `MechanisticResponsiveness`. Exported
    /// explicitly so a reader can't silently read "positive control passed"
    /// as "the probe responds to a real change in action-outcome
    /// contingency" — it only proves this module's own field-reading code
    /// has no hidden transform.
    pub positive_control_purpose: ControlPurpose,

    /// Broader, NON-gating diagnostic fields — descriptive only, see
    /// `DiagnosticEntry`.
    pub diagnostic_snapshot: Vec<DiagnosticEntry>,

    /// Known scope limitations of THIS run, preserved as part of the
    /// evidence record rather than left as prose that could drift from the
    /// code. Not exhaustive, but each entry names a specific, checkable gap.
    pub known_limitations: Vec<&'static str>,

    pub qualification: RuntimeQualification,
    pub outcome: EvidenceOutcome,
}

impl Ae2EmpiricalRun {
    /// States precisely WHAT OBJECT is causally supported (or isn't) by
    /// this run's `outcome` — deliberately narrower than the bare
    /// `EvidenceOutcome` label, which says nothing about scope on its own.
    /// A `CausallySupported` result here means the AE-2 INTERNAL PROBE
    /// SIGNAL, not the broader theoretical capacity AE-2 claims, and
    /// certainly not consciousness.
    pub fn claim_scope_note(&self) -> &'static str {
        match self.outcome {
            EvidenceOutcome::Supported(SupportTier::CausallySupported) => {
                "Causal support is scoped to the AE-2 INTERNAL PROBE SIGNAL only: the \
                 embodied-cognition ablation causally eliminated embodied_agency while the sham \
                 and measured unrelated state remained stable. No degradation was detected on the \
                 current downstream proxy benchmark (a ceiling effect), so functional \
                 embodied-agency consequences -- and a fortiori any claim about consciousness -- \
                 remain UNESTABLISHED."
            }
            EvidenceOutcome::Supported(SupportTier::FunctionallySupported) => {
                "Both the internal AE-2 probe signal AND the downstream functional benchmark \
                 degraded under the target ablation, while the sham left both intact -- broader \
                 support than a probe-only finding, but still scoped to this proxy benchmark's \
                 construct, not to consciousness."
            }
            EvidenceOutcome::Supported(SupportTier::Observed) => {
                "A live signal was measured and passed quality checks, but no qualified ablation \
                 comparison exists for this run -- an observational claim only, not causal."
            }
            EvidenceOutcome::Supported(SupportTier::ArchitecturalOnly) => {
                "The mechanism exists and is wired in; no live signal was measured this run -- an \
                 architectural claim only, not empirical."
            }
            EvidenceOutcome::NotDemonstrated => {
                "The run qualified (every runtime-qualification check passed), but no effect was \
                 observed on the AE-2 probe -- a genuine null result, not evidence against \
                 embodiment."
            }
            EvidenceOutcome::Contradicted => {
                "The measured effect moved opposite the predicted direction -- treat as a red flag \
                 requiring investigation, not evidence for or against AE-2."
            }
            EvidenceOutcome::Inconclusive => {
                "This run does not qualify as evidence either way -- see \
                 `qualification.failure_reasons()` for which check(s) failed."
            }
        }
    }
}

/// Run the real, four-arm AE-2 empirical experiment. The only
/// backend-touching function in this module — everything else here is
/// pure post-processing over its output.
pub fn run_ae2_empirical() -> Ae2EmpiricalRun {
    let design = *planned_designs()
        .iter()
        .find(|d| d.indicator == "AE-2")
        .expect("AE-2 design must exist in planned_designs()");
    let target_lever_name = design.target_lever;
    let functional_benchmark = design.functional_benchmark;

    let specs = ablation_specs();
    let ae2_spec = specs
        .iter()
        .find(|s| s.target_indicator == "AE-2")
        .expect("AE-2 row must exist in ablation_specs()");
    let hot1_spec = specs
        .iter()
        .find(|s| s.target_indicator == "HOT-1")
        .expect("HOT-1 row (AE-2's sham lever) must exist in ablation_specs()");
    let sham_lever_name = hot1_spec.name;

    // Question 7: identity check against the LIVE registry, up front.
    let identity_check_result =
        check_identity_against_registry(&design, ae2_spec.name, ae2_spec.downstream_benchmark);
    let identity_and_config_match = identity_check_result.is_ok();

    // The three live arms.
    let baseline_samples = run_arm(None);
    let target_samples = run_arm(Some(ae2_spec.config_mutator));
    let sham_samples = run_arm(Some(hot1_spec.config_mutator));

    // Question 6: is the AE-2 signal numerically usable in all arms?
    let baseline_embodied_agency = mean(&baseline_samples, ae2_probe);
    let target_ablated_embodied_agency = mean(&target_samples, ae2_probe);
    let sham_embodied_agency = mean(&sham_samples, ae2_probe);
    let probe_signal_usable = baseline_embodied_agency.is_finite()
        && target_ablated_embodied_agency.is_finite()
        && sham_embodied_agency.is_finite();

    // Question 1 + 4: did the target ablation actually fire?
    let baseline_hook_fired_fraction = hook_fired_fraction(&baseline_samples, |m| {
        m.module_timings_us.embodied_cognition
    });
    let target_hook_fired_fraction =
        hook_fired_fraction(&target_samples, |m| m.module_timings_us.embodied_cognition);
    let intervention_applied =
        baseline_hook_fired_fraction > 0.9 && target_hook_fired_fraction < 0.1;

    // Question 3 + 4 (sham half): did the sham's OWN lever actually fire?
    let sham_lever_fired_fraction =
        1.0 - hook_fired_fraction(&sham_samples, |m| m.module_timings_us.predictive_processing);
    let sham_lever_fired = sham_lever_fired_fraction > 0.9;
    // Specificity half of the sham: it must NOT also collapse the AE-2 signal.
    let sham_behaved_as_expected =
        sham_lever_fired && within_tolerance(baseline_embodied_agency, sham_embodied_agency);

    // Question 5: specificity health panel over UNRELATED signals.
    let mut health_panel = vec![
        continuous_health_entry(
            "meta_cognitive_accuracy",
            &baseline_samples,
            &target_samples,
            |m| m.quality.meta_cognitive_accuracy as f64,
        ),
        continuous_health_entry(
            "attention_schema_focus",
            &baseline_samples,
            &target_samples,
            |m| m.attention.attention_schema_focus as f64,
        ),
        continuous_health_entry(
            "phi_attention_weight",
            &baseline_samples,
            &target_samples,
            |m| m.attention.phi_attention_weight as f64,
        ),
        continuous_health_entry(
            "actual_effective_lr",
            &baseline_samples,
            &target_samples,
            |m| m.actual_effective_lr as f64,
        ),
    ];
    health_panel.push(module_activity_health_entry(
        "module_timings.gwt_still_executes",
        &baseline_samples,
        &target_samples,
        |m| m.module_timings_us.gwt,
    ));
    health_panel.push(module_activity_health_entry(
        "module_timings.cross_modal_binding_still_executes",
        &baseline_samples,
        &target_samples,
        |m| m.module_timings_us.cross_modal_binding,
    ));
    let intervention_specificity_passed = health_panel.iter().all(|h| h.within_tolerance);

    // Question 2: positive control -- pure data-mutation sanity check on
    // this module's OWN probe, reusing one real captured baseline sample.
    // Not a live run: the design's control purpose is Instrumentation.
    let (positive_control_zero_read_correct, positive_control_restore_read_correct) =
        match baseline_samples.last() {
            Some(sample) => {
                let original = ae2_probe(sample);
                let mut zeroed = sample.clone();
                zeroed.embodied.embodied_agency = 0.0;
                let zero_read_correct = ae2_probe(&zeroed) == 0.0;
                let mut restored = zeroed.clone();
                restored.embodied.embodied_agency = original;
                let restore_read_correct = ae2_probe(&restored) == original;
                (zero_read_correct, restore_read_correct)
            }
            None => (false, false),
        };
    let positive_control_effect_observed =
        positive_control_zero_read_correct && positive_control_restore_read_correct;

    // Broader, non-gating diagnostic snapshot -- descriptive only, never
    // consulted by intervention_specificity_passed.
    let diagnostic_snapshot = vec![
        diagnostic_entry(
            "embodied.body_phi_modulation",
            &baseline_samples,
            &target_samples,
            |m| m.embodied.body_phi_modulation,
        ),
        diagnostic_entry(
            "embodied.body_valence",
            &baseline_samples,
            &target_samples,
            |m| m.embodied.body_valence as f64,
        ),
        diagnostic_entry(
            "embodied.body_arousal",
            &baseline_samples,
            &target_samples,
            |m| m.embodied.body_arousal as f64,
        ),
        diagnostic_entry(
            "embodied.embodied_phi_modulation",
            &baseline_samples,
            &target_samples,
            |m| m.embodied.embodied_phi_modulation,
        ),
        diagnostic_entry(
            "embodied.affective_valence",
            &baseline_samples,
            &target_samples,
            |m| m.embodied.affective_valence as f64,
        ),
        diagnostic_entry(
            "embodied.affective_arousal",
            &baseline_samples,
            &target_samples,
            |m| m.embodied.affective_arousal as f64,
        ),
        diagnostic_entry(
            "embodied.affect_consciousness_valence",
            &baseline_samples,
            &target_samples,
            |m| m.embodied.affect_consciousness_valence as f64,
        ),
        diagnostic_entry(
            "embodied.affect_consciousness_arousal",
            &baseline_samples,
            &target_samples,
            |m| m.embodied.affect_consciousness_arousal as f64,
        ),
        diagnostic_entry(
            "embodied.mood_temperature",
            &baseline_samples,
            &target_samples,
            |m| m.embodied.mood_temperature as f64,
        ),
        diagnostic_entry(
            "embodied.somatic_stress",
            &baseline_samples,
            &target_samples,
            |m| m.embodied.somatic_stress,
        ),
        diagnostic_entry(
            "attention.attention_fatigue",
            &baseline_samples,
            &target_samples,
            |m| m.attention.attention_fatigue as f64,
        ),
        diagnostic_entry(
            "attention.attention_prediction_accuracy",
            &baseline_samples,
            &target_samples,
            |m| m.attention.attention_prediction_accuracy as f64,
        ),
        diagnostic_entry(
            "attention.psi_attention_avg",
            &baseline_samples,
            &target_samples,
            |m| m.attention.psi_attention_avg as f64,
        ),
        diagnostic_entry(
            "attention.gwt_coalition_size",
            &baseline_samples,
            &target_samples,
            |m| m.attention.gwt_coalition_size as f64,
        ),
        diagnostic_bool_fraction_entry(
            "attention.gwt_broadcast_fraction",
            &baseline_samples,
            &target_samples,
            |m| m.attention.gwt_broadcast,
        ),
        diagnostic_bool_fraction_entry(
            "surprise_triggered_fraction",
            &baseline_samples,
            &target_samples,
            |m| m.surprise_triggered,
        ),
        diagnostic_bool_fraction_entry(
            "prefrontal_veto_fraction",
            &baseline_samples,
            &target_samples,
            |m| m.prefrontal_veto,
        ),
    ];

    let known_limitations = vec![
        "Positive control is INSTRUMENTATION-level only (confirms this module's own field-reading \
         code has no hidden transform), not a mechanistic-responsiveness control over real \
         action-outcome contingency -- see qualification_design.rs's AE-2 positive_control.purpose.",
        "A single sham (disable_predictive_processing) establishes specificity against one \
         alternative perturbation, not general specificity against generic module-count, cycle-timing, \
         or state-competition effects. A second matched perturbation has not yet been run.",
        "Single seed (fixed genesis phrase), single fixed 10-sentence stimulus rotation. Not yet \
         replicated across fresh seeds or a structurally different stimulus schedule.",
        "The downstream WorM::SpatialUpdating benchmark showed a ceiling effect (1.0 in both arms) \
         under this proxy-ablation config. No discriminating external functional task has been run, \
         so functional (not just internal-probe) causal support remains unestablished.",
        "The initial health-panel specificity check used an incorrect 'must be active in both arms' \
         rule that flagged cross_modal_binding (never enabled by ConsciousnessProfile::Standard) as \
         a false specificity failure; corrected to a 'fraction must not change' rule before this \
         result was accepted -- see symthaea/docs/BUTLIN_AE2_FIRST_EMPIRICAL_RESULT_2026-07-27.md \
         for the original failed run's numbers.",
        "embodied_agency is documented as already 0.0 when embodied cognition is disabled -- the \
         lever and probe may be tightly/structurally coupled by construction. This run establishes \
         the lever->probe causal link and rules out the tested sham, but does not by itself \
         establish that the module confers the broader theoretical capacity AE-2 claims (embodied \
         action-outcome contingency learning).",
    ];

    // Downstream behavioral metric -- reuses the existing dispatch, not a
    // second bespoke benchmark.
    let (baseline_benchmark_accuracy, target_ablated_benchmark_accuracy) =
        run_downstream_benchmark(ae2_spec);

    let mut qualification = RuntimeQualification::from_static_design(&design);
    qualification.intervention_applied = intervention_applied;
    qualification.intervention_specificity_passed = intervention_specificity_passed;
    qualification.positive_control_effect_observed = positive_control_effect_observed;
    qualification.sham_behaved_as_expected = sham_behaved_as_expected;
    qualification.probe_signal_usable = probe_signal_usable;
    qualification.identity_and_config_match = identity_and_config_match;

    let classification = classify_ablation(
        baseline_embodied_agency,
        target_ablated_embodied_agency,
        baseline_benchmark_accuracy,
        target_ablated_benchmark_accuracy,
    );
    let outcome = resolve_outcome(
        &qualification,
        classification.indicator_dropped,
        classification.benchmark_degraded,
    );

    let config_hash = {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        format!("{target_lever_name}:{sham_lever_name}:{functional_benchmark}").hash(&mut hasher);
        format!("{:x}", hasher.finish())
    };

    Ae2EmpiricalRun {
        config_hash,
        seed_identity: "ablation-matrix-deterministic".to_string(),
        target_lever_name,
        sham_lever_name,
        functional_benchmark,
        num_cycles: NUM_CYCLES,
        warmup: WARMUP,
        baseline_embodied_agency,
        target_ablated_embodied_agency,
        sham_embodied_agency,
        baseline_hook_fired_fraction,
        target_hook_fired_fraction,
        sham_lever_fired_fraction,
        health_panel,
        baseline_benchmark_accuracy,
        target_ablated_benchmark_accuracy,
        identity_check: identity_check_result.map_err(|e| e.to_string()),
        positive_control_zero_read_correct,
        positive_control_restore_read_correct,
        positive_control_purpose: design.positive_control.purpose,
        diagnostic_snapshot,
        known_limitations,
        qualification,
        outcome,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The first genuine empirical evidence produced by this campaign.
    /// Deliberately asserts only WIRING correctness (identity match,
    /// AE-2's known static eligibility) -- NOT the scientific outcome.
    /// Run with `--nocapture` to read the full bundle; per explicit
    /// direction, this is where the campaign stops for human review, not
    /// where it silently asserts `Supported`.
    #[test]
    fn ae2_first_empirical_run_produces_a_complete_evidence_bundle() {
        let run = run_ae2_empirical();

        assert!(
            run.identity_check.is_ok(),
            "wiring correctness: AE-2's frozen design must match the live ablation_specs() \
             registry (a genuine mismatch here would be a real bug, not a scientific finding): \
             {:?}",
            run.identity_check
        );
        assert!(
            run.qualification.static_design_qualifies,
            "AE-2 is one of the four rows already known to pass static_design_qualifies()"
        );
        assert_eq!(
            run.health_panel.len(),
            6,
            "expected exactly 6 preregistered health-panel entries"
        );
        assert_eq!(
            run.positive_control_purpose,
            ControlPurpose::Instrumentation,
            "AE-2's positive control must remain declared as Instrumentation-purpose -- if this \
             changes, claim_scope_note()'s text needs to change with it"
        );

        println!("\n=== AE-2 first empirical run: full evidence bundle ===\n{run:#?}\n");
        println!("=== Outcome: {:?} ===", run.outcome);
        println!("=== Claim scope: {} ===", run.claim_scope_note());
        println!(
            "=== Known limitations ({}) ===",
            run.known_limitations.len()
        );
        for (i, limitation) in run.known_limitations.iter().enumerate() {
            println!("  {}. {limitation}", i + 1);
        }
        println!(
            "=== Diagnostic snapshot ({} entries, non-gating) ===",
            run.diagnostic_snapshot.len()
        );
        for d in &run.diagnostic_snapshot {
            println!(
                "  {}: baseline={:.6} target_ablated={:.6}",
                d.field, d.baseline, d.target_ablated
            );
        }
    }
}
