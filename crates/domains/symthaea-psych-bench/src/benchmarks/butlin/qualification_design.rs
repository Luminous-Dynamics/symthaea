// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Static experimental-design declarations for the Butlin Probe Qualification
//! campaign (see `BUTLIN_PROBE_QUALIFICATION_V1_PLAN_2026-07-27.md`, repo
//! root). **No experiment runs here** — this module declares, as data, which
//! intervention/positive-control/sham/benchmark/probe each of the 12
//! directly-ablated indicators uses, and validates that the declaration is
//! internally honest before any campaign is allowed to run against it.
//!
//! This module has grown from "declare the dependency structure" into a real
//! **construct-validity audit** of the existing ablation matrix, by checking
//! every claim against the actual code in `ablation.rs::measure_indicator`/
//! `measure_activity_fraction`/`extract_indicator_score` rather than trusting
//! prose descriptions (including this module's own earlier drafts). The
//! findings are summarized here because they materially change what the
//! current 12-row matrix can honestly claim:
//!
//! 1. **Shared intervention ⇒ causal dependence; shared benchmark ⇒
//!    functional-evidence dependence; shared raw probe field ⇒ observational
//!    dependence. Three different claims, never collapsed into one boolean.**
//!    `QualificationDesign` exposes independent `shares_*` queries (target,
//!    sham, target↔sham cross-role, positive-control id, positive-control
//!    *protocol*, and now probe *signal*) plus `causally_independent_of`/
//!    `functionally_independent_of`/`fully_independent_of`.
//! 2. **GWT-2 and GWT-3 share the identical lever** (`enable_gwt = false`) —
//!    two outcomes of one causal unit, not two causal replications; they
//!    also deliberately share one sham.
//! 3. **Six of the twelve indicators share `WorM::N-back`** as their
//!    downstream benchmark.
//! 4. **Every one of the 12 shams reuses another indicator's real target
//!    lever** — legitimate, but must be declared via `reuses_target_of`
//!    (an explicit `Option<&'static str>` owner, not just a bool), which
//!    `validate_designs` checks for both omission *and* incorrectness.
//! 5. **Target↔sham cross-role overlap is real and distinct from same-role
//!    sharing** — e.g. HOT-1's sham is textually AST-1's own target lever,
//!    a dependency neither `shares_target_intervention` nor
//!    `shares_sham_intervention` alone would catch.
//! 6. **HOT-3 and PP-1's positive controls share an instrumentation
//!    protocol** (pinning `actual_effective_lr`) despite distinct
//!    `PositiveControlId`s.
//! 7. **HOT-3 and PP-1's *indicator probes themselves* read the literal same
//!    raw field** (`metadata.actual_effective_lr`, confirmed directly against
//!    `extract_indicator_score`'s `"PP-1" | "HOT-3" => metadata.
//!    actual_effective_lr as f64` arm) — a stronger, distinct claim from
//!    (6): they don't just share a *control* mechanism, they share the
//!    *observed signal itself*. Whatever distinguishes "predictive
//!    processing drives learning" (PP-1) from "belief updating from action
//!    outcomes" (HOT-3) as theoretical claims, both currently cash out to
//!    "is this one field nonzero" — their apparent independence as two of
//!    the paper's 14 indicators rests entirely on their *different targeted
//!    interventions* producing different patterns on an *identical* measured
//!    quantity, not on measuring different things. Declared via
//!    `probe_group`/`shares_probe_signal`; this is the only confirmed
//!    raw-field duplicate among the 12 (verified by reading every match arm
//!    in `extract_indicator_score` and every special-cased branch in
//!    `measure_indicator` — no other pair shares a raw field).
//! 8. **Several positive controls are not achievable, discriminating, or
//!    even well-typed as "positive controls" at all** — this is the most
//!    consequential finding. Per-indicator status (`ControlPurpose`/
//!    `ControlReadiness` on `PositiveControlPlan`, see `planned_designs()`
//!    for full rationale on each):
//!    - **RPT-1**: the original "feed a frozen input" idea doesn't test
//!      recurrence at all — the real formula needs ≥2 distinct-input
//!      centroids or it hard-returns `0.0`, so a single-input run reads
//!      `0.0` regardless of whether CfC is healthy or broken. Reclassified
//!      as `ControlPurpose::DegenerateGuardTest` (formula-verified: the
//!      guard correctly returns `0.0` in both arms) rather than left
//!      mislabeled as a working positive control —
//!      `PositiveControlPlan::control_design_qualifies()` returns `false`
//!      for it by construction, so a future runner can't mistake "a control
//!      exists" for "the probe is qualified."
//!    - **GWT-2**'s original control ("force capacity to 1") doesn't violate
//!      the real predicate (`size > 0 && size < 1000`) — corrected to force
//!      `size = 0` directly, formula-verified against the real predicate.
//!      **But its probe is still `ExecutionProxy`** — a design review of the
//!      5 nominally-qualifying rows found the original single combined
//!      "qualifies" check didn't account for this, letting `GWT-2` report
//!      as equally qualified to the 4 genuinely `DirectMeasure` rows. Fixed
//!      by splitting `control_design_qualifies()` (control credibility
//!      only) from `QualificationDesign::static_design_qualifies()` (control
//!      credibility AND probe validity together) — `GWT-2` passes the
//!      former, fails the latter.
//!    - **GWT-3, RPT-2**: flagged as **construct-validity concerns**, not
//!      merely unverified controls — both indicators' real signals are
//!      `module_timings_us.{gwt,cross_modal_binding} > 0`, i.e. "did this
//!      module execute at all," which is an `ExecutionProxy` for the
//!      theoretical claim (global broadcast; cross-modal binding quality),
//!      not a direct or even graded behavioral measure of it. A control that
//!      changes this boolean would validate the timing proxy, not the
//!      theoretical mapping — this module deliberately does not manufacture
//!      one just to fill the field.
//!    - **RPT-2, HOT-1, AST-1, AE-1**: their originally-drafted controls
//!      assume a custom-input/override hook that `measure_indicator`'s real
//!      signature doesn't expose (it hardcodes a fixed 10-sentence rotation
//!      with no override point) — marked `Unverified`, not `FormulaVerified`.
//!    - **GWT-4, HOT-2 (field only, injection path unconfirmed), HOT-3,
//!      PP-1, AE-2**: formula-verified against real formulas in
//!      `extract_indicator_score`. **None of the 5 `FormulaVerified` controls
//!      have had their manipulation's real-world *achievability* — an actual
//!      exposed mutation hook, not just formula correctness — independently
//!      confirmed.** That's why the readiness variant is named
//!      `FormulaVerified`, not `Verified`: the earlier name overclaimed what
//!      had actually been checked, and would have read as stronger evidence
//!      than it is once serialized into an evidence bundle. See
//!      `BUTLIN_PR_B_RUNNER_PLAN_2026-07-27.md`'s `RuntimeQualification` for
//!      the achievability/specificity checks a real run still needs.
//!    Only 5 of 12 (`GWT-2, GWT-4, HOT-3, PP-1, AE-2`) have a
//!    `FormulaVerified` positive control with a non-`DegenerateGuardTest`
//!    purpose (`control_design_qualifies()`); combined with `probe_validity`,
//!    only 4 (`GWT-4, HOT-3, PP-1, AE-2`) currently pass
//!    `static_design_qualifies()` — and even those 4 are static claims only,
//!    not runtime-proven ones.
//!
//! **Contract this places on any future PR B (the runner, not built yet):**
//! a row whose `static_design_qualifies()` is `false`, or (once a runner
//! exists) whose `RuntimeQualification::qualifies_run()` is `false`, must
//! not be interpreted as `NotDemonstrated`/`CausallySupported`/
//! `FunctionallySupported` from an ablation null/positive result — it must be
//! reported as inconclusive at the qualification layer, before ever reaching
//! `report.rs`'s own (already-correct) `Inconclusive` gate. This module
//! cannot enforce that itself (no runner exists to enforce it against), so
//! it's recorded here as an explicit requirement, not assumed self-evident.
//!
//! Always compiled (no `symthaea-backend` feature needed) — this is pure
//! declarative metadata, not a call into `symthaea::cognitive_loop`. The one
//! place this module *does* reach into the backend-gated `ablation` module
//! is a `#[cfg(feature = "symthaea-backend")]` test confirming the
//! declarations here haven't drifted from the real `ablation_specs()` table.

use std::collections::HashMap;

/// The direction a probe's raw value is expected to move under a comparison,
/// or `Invariant` when the comparison is expected to produce *no* difference
/// (e.g. `RPT-1`'s degenerate-guard test, where both arms should read
/// identically) — not every comparison in this module is directional.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EffectDirection {
    /// The probe should read lower under the comparison's "off"/intervention
    /// side relative to its baseline side.
    Decrease,
    /// The probe should read higher under the comparison's "off"/intervention
    /// side relative to its baseline side.
    Increase,
    /// The comparison is expected to produce no meaningful difference at all
    /// — used for structural/guard checks, not real effect claims.
    Invariant,
}

/// Which two conditions an `ExpectedEffect` compares. Naming the comparison
/// explicitly prevents a runner from computing the right sign against the
/// wrong pair of conditions — `expected_direction: Decrease` alone doesn't
/// say what's being compared to what.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Comparison {
    /// The indicator's targeted ablation vs. baseline (the paper's actual
    /// causal claim).
    TargetedAblationVsBaseline,
    /// A positive control's manipulated condition vs. the ordinary,
    /// unmanipulated baseline.
    PositiveControlVsBaseline,
    /// The two extremes of a pin-style positive control against each other
    /// directly (e.g. pin a field to a nonzero constant, then to zero, and
    /// compare those two pinned readings) — distinct from
    /// `PositiveControlVsBaseline`, which compares the control condition
    /// against an *unmanipulated* run.
    ZeroPinnedVsNonzeroPinned,
    /// A stimulus-responsiveness control specifically contrasting an
    /// adversarial/surprising input stream against a predictable one.
    AdversarialVsPredictableStimulus,
    /// A sham lever's effect on the probe of the *other* indicator whose
    /// real target it happens to be (confirms the sham is a live,
    /// working intervention elsewhere).
    ShamVsBaselineOnOwnRealTarget,
    /// A structural/guard-branch check where both compared conditions are
    /// expected to produce the identical (degenerate) reading — see
    /// `EffectDirection::Invariant`.
    DegenerateSingleInputBothArms,
}

/// A fully-specified expected effect: what's being measured, what two
/// conditions are compared, which way it should move (or `Invariant` if it
/// shouldn't), and why.
#[derive(Debug, Clone, Copy)]
pub struct ExpectedEffect {
    pub metric: &'static str,
    pub comparison: Comparison,
    pub direction: EffectDirection,
    pub rationale: &'static str,
}

/// What role a positive control actually plays, since not everything
/// originally drafted as a "positive control" turned out to validate the
/// measurement path at all once checked against the real formula.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlPurpose {
    /// Inject/construct a known change directly into the measured signal or
    /// its immediate controlling field. Validates the *measurement path*
    /// only — proves the reporting plumbing works, nothing about real
    /// operation.
    Instrumentation,
    /// Feed a real, controlled input *stimulus* through the actual
    /// cognitive pipeline and observe whether the probe's real,
    /// pipeline-computed output responds. Stronger than Instrumentation
    /// (exercises real computation on the input side).
    StimulusResponsiveness,
    /// A legitimate system-level manipulation, distinct from the targeted
    /// ablation lever, expected to change the property under real cognitive
    /// operation — validates the *interpretation*, not just the plumbing.
    MechanisticResponsiveness,
    /// Confirms a degenerate/guard branch in the real formula fires
    /// correctly (e.g. `if centroids.len() < 2 { return 0.0 }`) — this is
    /// **not** a positive control for the underlying phenomenon and must
    /// never be treated as one. `control_design_qualifies()` always returns
    /// `false` for this purpose, regardless of `readiness`.
    DegenerateGuardTest,
}

/// Whether a control (positive control or sham) has actually been checked
/// against something real yet, and what that check found.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ControlReadiness {
    /// Declared as a design intent; not yet examined against real code or a
    /// health panel at all.
    Proposed,
    /// Examined against the real formula/API, but not confirmed to actually
    /// work (e.g. the manipulation may not be achievable via the current
    /// measurement surface, or its effect on the real predicate is unclear).
    Unverified,
    /// Confirmed, by reading the real formula/predicate directly, that IF
    /// the described manipulation were applied, the formula would produce
    /// the claimed effect. **Deliberately not named `Verified`** — that name
    /// was found to overclaim: it reads as "this control works," but what's
    /// actually been checked is narrower. Two things remain unconfirmed even
    /// at this level: (1) whether the real harness actually exposes a
    /// mutation hook to apply the manipulation at all (achievability), and
    /// (2) whether applying it would touch only the intended field or also
    /// perturb unrelated state (specificity). Both require running
    /// something, which is exactly what a future `RuntimeQualification`
    /// (see `BUTLIN_PR_B_RUNNER_PLAN_2026-07-27.md`) is for — this variant
    /// only ever licenses "formula-level reasoning checks out," never
    /// "this control has been run."
    FormulaVerified,
    /// Confirmed, by reading the real formula/predicate directly, to NOT
    /// produce the claimed effect — a known-broken control, kept in the
    /// record rather than silently deleted so the finding isn't lost.
    Invalid,
}

/// Stable identity for a positive control, distinct from its free-text
/// `description`. A future runner needs a deterministic value to execute
/// against and evidence bundles need a stable value to record.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositiveControlId(pub &'static str);

/// A planned positive control for one indicator. Declared, not executed.
#[derive(Debug, Clone, Copy)]
pub struct PositiveControlPlan {
    /// Unique per indicator — no two entries in `planned_designs()` share
    /// this (see `test_no_two_indicators_share_a_positive_control_id`).
    pub id: PositiveControlId,
    pub purpose: ControlPurpose,
    pub readiness: ControlReadiness,
    /// Which underlying instrumentation *protocol* this control uses,
    /// distinct from `id`. Two indicators can have distinct unique
    /// invocation IDs while still sharing the same underlying mechanism
    /// (HOT-3 and PP-1 both pin `actual_effective_lr`).
    pub protocol_group: &'static str,
    /// Free-text description for human review before implementation.
    pub description: &'static str,
    pub expected_effect: ExpectedEffect,
}

impl PositiveControlPlan {
    /// Whether the *control itself* is credibly designed — `false` whenever
    /// `purpose` is `DegenerateGuardTest` (structurally can never qualify,
    /// regardless of readiness) or `readiness` isn't `FormulaVerified`.
    ///
    /// Deliberately narrower than "is the whole probe eligible for
    /// interpretation" — this method only answers the control-design
    /// question. A control can pass this and the probe still not be
    /// eligible, if the *probe itself* is a coarse execution proxy (see
    /// `QualificationDesign::static_design_qualifies`, which combines this
    /// with `probe_validity`). Splitting these was a real fix: the earlier,
    /// single combined method let `GWT-2` (a genuinely credible control on
    /// an `ExecutionProxy` probe) report as equally qualified to `GWT-4`/
    /// `HOT-3`/`PP-1`/`AE-2` (all `DirectMeasure`), which conflated two
    /// different questions.
    pub fn control_design_qualifies(&self) -> bool {
        !matches!(self.purpose, ControlPurpose::DegenerateGuardTest)
            && self.readiness == ControlReadiness::FormulaVerified
    }
}

/// Which dimension(s) a sham is deliberately matched to its target
/// intervention on, so "unrelated configuration toggle" is never silently
/// treated as equivalent to "matched negative control." Declared here as a
/// design commitment; PR B/the health panel is what actually measures
/// whether the match holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MatchedDimension {
    ComputeCost,
    RuntimeLatency,
    StateDisruption,
    ParameterCount,
    NoiseMagnitude,
    DisabledSubsystemCount,
}

/// How directly a probe measures the theoretical property it's named for,
/// vs. proxying for it through something coarser. `ExecutionProxy` entries
/// (module-ran boolean checks) are real findings from this audit, not
/// invented pessimism — see the module doc's GWT-3/RPT-2 discussion.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProbeValidity {
    /// The formula reads a field that IS (or is very close to) the
    /// theoretical quantity itself (e.g. GWT-4 reading `phi_attention_weight`
    /// deviation directly).
    DirectMeasure,
    /// A graded, real behavioral signal that stands in for the theoretical
    /// property without being a literal reading of it (e.g. RPT-1's
    /// cross-input centroid-distance discrimination measure).
    BehavioralProxy,
    /// A coarse "did this subsystem execute at all" boolean/fraction,
    /// standing in for a theoretical claim about the *quality* or *kind* of
    /// what that subsystem did, not just whether it ran.
    ExecutionProxy,
    /// Not yet classified against the real formula.
    Unverified,
}

/// A planned sham/unrelated-ablation control for one indicator. Reusing
/// another indicator's real target lever as a sham is legitimate, but the
/// reuse must be acknowledged explicitly and correctly — see
/// `reuses_target_of`.
#[derive(Debug, Clone, Copy)]
pub struct ShamControlPlan {
    pub lever: &'static str,
    pub group: &'static str,
    pub rationale: &'static str,
    pub matched_dimensions: &'static [MatchedDimension],
    pub expected_effect: ExpectedEffect,
    /// Ceiling on acceptable general performance damage before this sham is
    /// disqualified as "too disruptive to be a specificity control."
    /// Deliberately `None` throughout v1 — no common health panel exists
    /// yet to measure against.
    pub maximum_allowed_global_impairment: Option<f64>,
    /// `Some(indicator_id)` naming the *other* indicator whose real
    /// `target_lever` this sham reuses (by lever-name equality), or `None`
    /// if this sham's lever is not, in fact, anyone else's real target.
    pub reuses_target_of: Option<&'static str>,
    pub readiness: ControlReadiness,
}

/// One indicator's full experimental-design declaration.
#[derive(Debug, Clone, Copy)]
pub struct QualificationDesign {
    pub indicator: &'static str,
    pub target_lever: &'static str,
    pub target_lever_group: &'static str,
    pub positive_control: PositiveControlPlan,
    pub sham: ShamControlPlan,
    pub functional_benchmark: &'static str,
    pub functional_benchmark_group: &'static str,
    /// Which real field/computation this indicator's own score is read
    /// from (e.g. `"actual_effective_lr"`), for human review.
    pub probe_metric: &'static str,
    /// Group ID for the probe signal ("observational source"). Two
    /// indicators sharing this value read the literal identical raw field —
    /// their apparent independence rests entirely on their different
    /// targeted interventions producing different patterns on that one
    /// signal, not on measuring different things. HOT-3/PP-1 is the only
    /// confirmed case in v1 (see module doc, finding 7).
    pub probe_group: &'static str,
    pub probe_validity: ProbeValidity,
    pub expected_effect: ExpectedEffect,
}

impl QualificationDesign {
    /// Same causal intervention lever (a "causal unit" dependency).
    pub fn shares_target_intervention(&self, other: &Self) -> bool {
        self.target_lever_group == other.target_lever_group
    }

    /// Same downstream benchmark (a "functional endpoint" dependency).
    pub fn shares_functional_benchmark(&self, other: &Self) -> bool {
        self.functional_benchmark_group == other.functional_benchmark_group
    }

    /// Same sham intervention.
    pub fn shares_sham_intervention(&self, other: &Self) -> bool {
        self.sham.group == other.sham.group
    }

    /// **Cross-role** overlap: `self`'s sham is the same intervention as
    /// `other`'s target (or vice versa) — since every sham in v1's design
    /// reuses another indicator's real target, two indicators can be
    /// evidentially dependent through this path even when neither same-role
    /// check fires (e.g. HOT-1's sham is AST-1's real target).
    pub fn shares_intervention_cross_role(&self, other: &Self) -> bool {
        self.sham.group == other.target_lever_group || self.target_lever_group == other.sham.group
    }

    /// Same positive control *invocation* (unique `id`).
    pub fn shares_positive_control(&self, other: &Self) -> bool {
        self.positive_control.id == other.positive_control.id
    }

    /// Same underlying positive-control *protocol*, even with distinct
    /// `id`s (HOT-3/PP-1's shared pinned-learning-rate instrumentation).
    pub fn shares_positive_control_protocol(&self, other: &Self) -> bool {
        self.positive_control.protocol_group == other.positive_control.protocol_group
    }

    /// Same underlying raw observed field/computation — an *observational*
    /// dependency distinct from all the intervention-side ones above. Two
    /// indicators sharing this read the literal identical signal (HOT-3/
    /// PP-1's confirmed `actual_effective_lr` case).
    pub fn shares_probe_signal(&self, other: &Self) -> bool {
        self.probe_group == other.probe_group
    }

    pub fn causally_independent_of(&self, other: &Self) -> bool {
        !self.shares_target_intervention(other)
    }

    pub fn functionally_independent_of(&self, other: &Self) -> bool {
        !self.shares_functional_benchmark(other)
    }

    /// Whether `self` and `other` share *no* evidence dependency at all —
    /// causal, functional, sham, cross-role, positive-control/protocol, or
    /// probe-signal. The method a campaign report should actually use
    /// before presenting two results as independent corroboration.
    pub fn fully_independent_of(&self, other: &Self) -> bool {
        self.causally_independent_of(other)
            && self.functionally_independent_of(other)
            && !self.shares_sham_intervention(other)
            && !self.shares_intervention_cross_role(other)
            && !self.shares_positive_control(other)
            && !self.shares_positive_control_protocol(other)
            && !self.shares_probe_signal(other)
    }

    pub fn dependency_of(&self, other: &Self) -> EvidenceDependency {
        EvidenceDependency {
            shared_target: self.shares_target_intervention(other),
            shared_sham: self.shares_sham_intervention(other),
            shared_intervention_cross_role: self.shares_intervention_cross_role(other),
            shared_positive_control: self.shares_positive_control(other),
            shared_positive_control_protocol: self.shares_positive_control_protocol(other),
            shared_benchmark: self.shares_functional_benchmark(other),
            shared_probe_signal: self.shares_probe_signal(other),
        }
    }

    /// Whether this row's **complete static design** — control credibility
    /// AND probe validity together — is eligible for interpretation.
    /// Stricter than `positive_control.control_design_qualifies()` alone:
    /// a row with a genuinely credible control can still fail here if its
    /// probe is a coarse `ExecutionProxy`/`Unverified` proxy rather than a
    /// real measure of the theoretical property (`GWT-2`'s case — its
    /// control is `FormulaVerified`, but this method still returns `false`
    /// for it).
    ///
    /// Whether `ProbeValidity::BehavioralProxy` should qualify is a
    /// deliberate policy call, not a certainty — a graded real behavioral
    /// signal is weaker evidence than a direct measure but stronger than a
    /// boolean execution check; included here as "eligible" for now, but
    /// this is a reasonable place to tighten later if a `BehavioralProxy`
    /// row's results turn out unconvincing in practice.
    ///
    /// Even `true` here is still only a **static** claim — see
    /// `BUTLIN_PR_B_RUNNER_PLAN_2026-07-27.md`'s `RuntimeQualification` for
    /// why a passing static design does not by itself mean any given run
    /// actually qualifies (achievability and specificity can only be
    /// established by running something).
    pub fn static_design_qualifies(&self) -> bool {
        self.positive_control.control_design_qualifies()
            && matches!(
                self.probe_validity,
                ProbeValidity::DirectMeasure | ProbeValidity::BehavioralProxy
            )
    }
}

/// Which evidence dimensions two `QualificationDesign`s share. A struct of
/// independent booleans, not a single collapsed verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceDependency {
    pub shared_target: bool,
    pub shared_sham: bool,
    pub shared_intervention_cross_role: bool,
    pub shared_positive_control: bool,
    pub shared_positive_control_protocol: bool,
    pub shared_benchmark: bool,
    pub shared_probe_signal: bool,
}

impl EvidenceDependency {
    pub fn is_fully_independent(&self) -> bool {
        !(self.shared_target
            || self.shared_sham
            || self.shared_intervention_cross_role
            || self.shared_positive_control
            || self.shared_positive_control_protocol
            || self.shared_benchmark
            || self.shared_probe_signal)
    }
}

/// A structural problem found by `validate_designs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DesignViolation {
    ShamMatchesTarget {
        indicator: &'static str,
    },
    PositiveControlMatchesTarget {
        indicator: &'static str,
    },
    UndeclaredSharedIntervention {
        indicator_a: &'static str,
        indicator_b: &'static str,
    },
    UndeclaredSharedBenchmark {
        indicator_a: &'static str,
        indicator_b: &'static str,
    },
    ShamSharesTargetGroup {
        indicator: &'static str,
    },
    ShamTargetReuseUndeclared {
        indicator: &'static str,
        sham_lever: &'static str,
        actual_owner: &'static str,
    },
    ShamTargetReuseMismatch {
        indicator: &'static str,
        sham_lever: &'static str,
        actual_owner: &'static str,
        declared_owner: &'static str,
    },
    ShamFalselyClaimsTargetReuse {
        indicator: &'static str,
        sham_lever: &'static str,
        declared_owner: &'static str,
    },
}

impl std::fmt::Display for DesignViolation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DesignViolation::ShamMatchesTarget { indicator } => write!(
                f,
                "{indicator}: sham_lever is identical to target_lever — not a sham"
            ),
            DesignViolation::PositiveControlMatchesTarget { indicator } => write!(
                f,
                "{indicator}: Mechanistic positive control description matches target_lever — \
                 not independent of the targeted ablation"
            ),
            DesignViolation::UndeclaredSharedIntervention {
                indicator_a,
                indicator_b,
            } => write!(
                f,
                "{indicator_a} and {indicator_b} share target_lever by name but declare \
                 different target_lever_group values"
            ),
            DesignViolation::UndeclaredSharedBenchmark {
                indicator_a,
                indicator_b,
            } => write!(
                f,
                "{indicator_a} and {indicator_b} share functional_benchmark by name but declare \
                 different functional_benchmark_group values"
            ),
            DesignViolation::ShamSharesTargetGroup { indicator } => write!(
                f,
                "{indicator}: sham.group matches its own target_lever_group"
            ),
            DesignViolation::ShamTargetReuseUndeclared {
                indicator,
                sham_lever,
                actual_owner,
            } => write!(
                f,
                "{indicator}: sham_lever {sham_lever:?} is {actual_owner}'s real target lever, \
                 but reuses_target_of is None"
            ),
            DesignViolation::ShamTargetReuseMismatch {
                indicator,
                sham_lever,
                actual_owner,
                declared_owner,
            } => write!(
                f,
                "{indicator}: sham_lever {sham_lever:?} is actually {actual_owner}'s real \
                 target, but reuses_target_of declares {declared_owner:?}"
            ),
            DesignViolation::ShamFalselyClaimsTargetReuse {
                indicator,
                sham_lever,
                declared_owner,
            } => write!(
                f,
                "{indicator}: reuses_target_of claims {declared_owner:?}, but sham_lever \
                 {sham_lever:?} is not anyone's real target lever"
            ),
        }
    }
}

/// Validate a full set of qualification designs against the invariants this
/// module exists to enforce.
pub fn validate_designs(designs: &[QualificationDesign]) -> Vec<DesignViolation> {
    let mut violations = Vec::new();

    for d in designs {
        if d.sham.lever == d.target_lever {
            violations.push(DesignViolation::ShamMatchesTarget {
                indicator: d.indicator,
            });
        }
        if d.sham.group == d.target_lever_group {
            violations.push(DesignViolation::ShamSharesTargetGroup {
                indicator: d.indicator,
            });
        }
        if d.positive_control.purpose == ControlPurpose::MechanisticResponsiveness
            && d.positive_control.description == d.target_lever
        {
            violations.push(DesignViolation::PositiveControlMatchesTarget {
                indicator: d.indicator,
            });
        }
    }

    let mut lever_groups: HashMap<&'static str, &'static str> = HashMap::new();
    let mut benchmark_groups: HashMap<&'static str, &'static str> = HashMap::new();
    for d in designs {
        if let Some(&existing) = lever_groups.get(d.target_lever) {
            if existing != d.target_lever_group {
                if let Some(earlier) = designs
                    .iter()
                    .find(|o| o.target_lever == d.target_lever && o.indicator != d.indicator)
                {
                    violations.push(DesignViolation::UndeclaredSharedIntervention {
                        indicator_a: earlier.indicator,
                        indicator_b: d.indicator,
                    });
                }
            }
        } else {
            lever_groups.insert(d.target_lever, d.target_lever_group);
        }

        if let Some(&existing) = benchmark_groups.get(d.functional_benchmark) {
            if existing != d.functional_benchmark_group {
                if let Some(earlier) = designs.iter().find(|o| {
                    o.functional_benchmark == d.functional_benchmark && o.indicator != d.indicator
                }) {
                    violations.push(DesignViolation::UndeclaredSharedBenchmark {
                        indicator_a: earlier.indicator,
                        indicator_b: d.indicator,
                    });
                }
            }
        } else {
            benchmark_groups.insert(d.functional_benchmark, d.functional_benchmark_group);
        }
    }

    let target_owner: HashMap<&'static str, &'static str> = designs
        .iter()
        .map(|d| (d.target_lever, d.indicator))
        .collect();
    for d in designs {
        let actual_owner = target_owner
            .get(d.sham.lever)
            .copied()
            .filter(|&owner| owner != d.indicator);
        match (actual_owner, d.sham.reuses_target_of) {
            (Some(actual), Some(declared)) if actual != declared => {
                violations.push(DesignViolation::ShamTargetReuseMismatch {
                    indicator: d.indicator,
                    sham_lever: d.sham.lever,
                    actual_owner: actual,
                    declared_owner: declared,
                });
            }
            (Some(actual), None) => {
                violations.push(DesignViolation::ShamTargetReuseUndeclared {
                    indicator: d.indicator,
                    sham_lever: d.sham.lever,
                    actual_owner: actual,
                });
            }
            (None, Some(declared)) => {
                violations.push(DesignViolation::ShamFalselyClaimsTargetReuse {
                    indicator: d.indicator,
                    sham_lever: d.sham.lever,
                    declared_owner: declared,
                });
            }
            _ => {}
        }
    }

    violations
}

/// A declared shared-evidence relationship between indicators, for
/// human-readable reporting.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SharedGroup {
    Intervention {
        group: &'static str,
        indicators: Vec<&'static str>,
    },
    FunctionalBenchmark {
        group: &'static str,
        indicators: Vec<&'static str>,
    },
    ProbeSignal {
        group: &'static str,
        indicators: Vec<&'static str>,
    },
}

/// Derive every shared-evidence group with more than one member.
pub fn shared_groups(designs: &[QualificationDesign]) -> Vec<SharedGroup> {
    let mut by_lever_group: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
    let mut by_benchmark_group: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
    let mut by_probe_group: HashMap<&'static str, Vec<&'static str>> = HashMap::new();
    for d in designs {
        by_lever_group
            .entry(d.target_lever_group)
            .or_default()
            .push(d.indicator);
        by_benchmark_group
            .entry(d.functional_benchmark_group)
            .or_default()
            .push(d.indicator);
        by_probe_group
            .entry(d.probe_group)
            .or_default()
            .push(d.indicator);
    }

    let mut groups = Vec::new();
    let mut lever_keys: Vec<_> = by_lever_group.keys().copied().collect();
    lever_keys.sort_unstable();
    for group in lever_keys {
        let indicators = by_lever_group.remove(group).unwrap();
        if indicators.len() > 1 {
            groups.push(SharedGroup::Intervention { group, indicators });
        }
    }
    let mut benchmark_keys: Vec<_> = by_benchmark_group.keys().copied().collect();
    benchmark_keys.sort_unstable();
    for group in benchmark_keys {
        let indicators = by_benchmark_group.remove(group).unwrap();
        if indicators.len() > 1 {
            groups.push(SharedGroup::FunctionalBenchmark { group, indicators });
        }
    }
    let mut probe_keys: Vec<_> = by_probe_group.keys().copied().collect();
    probe_keys.sort_unstable();
    for group in probe_keys {
        let indicators = by_probe_group.remove(group).unwrap();
        if indicators.len() > 1 {
            groups.push(SharedGroup::ProbeSignal { group, indicators });
        }
    }
    groups
}

/// The 12 directly-ablated indicators' qualification designs (excludes GWT-1,
/// a derived aggregate, and HOT-4, which already has an internal
/// responsiveness test). See the module doc comment for the full audit
/// summary this data encodes.
pub fn planned_designs() -> Vec<QualificationDesign> {
    use Comparison::*;
    use ControlPurpose::*;
    use ControlReadiness::*;
    use EffectDirection::*;
    use MatchedDimension::DisabledSubsystemCount;
    // Not glob-imported: ProbeValidity::Unverified collides with
    // ControlReadiness::Unverified above (both used in this function) --
    // qualified explicitly at each of the 12 use sites below instead.

    let target_effect = |metric: &'static str, rationale: &'static str| ExpectedEffect {
        metric,
        comparison: TargetedAblationVsBaseline,
        direction: Decrease,
        rationale,
    };

    let sham = |lever: &'static str,
                group: &'static str,
                rationale: &'static str,
                reuses_target_of: &'static str,
                metric: &'static str| ShamControlPlan {
        lever,
        group,
        rationale,
        matched_dimensions: &[DisabledSubsystemCount],
        expected_effect: ExpectedEffect {
            metric,
            comparison: ShamVsBaselineOnOwnRealTarget,
            direction: Decrease,
            rationale: "confirms the sham lever is a live, working intervention on the \
                        indicator whose real target it is",
        },
        maximum_allowed_global_impairment: None,
        reuses_target_of: Some(reuses_target_of),
        readiness: Proposed,
    };

    vec![
        QualificationDesign {
            indicator: "RPT-1",
            target_lever: "disable_cfc_recurrence",
            target_lever_group: "cfc_recurrence",
            positive_control: PositiveControlPlan {
                id: PositiveControlId("rpt1_single_input_degenerate_case"),
                purpose: DegenerateGuardTest,
                readiness: FormulaVerified,
                protocol_group: "single_repeated_input_degenerate_check",
                description: "The real formula needs >=2 distinct-input centroids or it \
                              hard-returns 0.0 regardless of CfC health -- verified this branch \
                              fires correctly (reads 0.0 in both baseline and ablated configs \
                              when collapsed to one repeated input), but this proves the guard \
                              works, NOT that the probe can detect healthy vs. broken \
                              recurrence. A true positive control needs a lower-level hook this \
                              measurement API doesn't expose -- disclosed as an open design gap, \
                              not solved here.",
                expected_effect: ExpectedEffect {
                    metric: "rpt1_centroid_pairwise_distance",
                    comparison: DegenerateSingleInputBothArms,
                    direction: Invariant,
                    rationale: "both arms structurally read exactly 0.0 -- this is a guard-\
                                branch check, not a directional claim",
                },
            },
            sham: sham(
                "disable_metacognition",
                "metacognition",
                "reuses HOT-2's real target as an unrelated disruptive control for RPT-1",
                "HOT-2",
                "meta_cognitive_accuracy",
            ),
            functional_benchmark: "WorM::N-back",
            functional_benchmark_group: "worm_nback",
            probe_metric: "rpt1_centroid_pairwise_distance",
            probe_group: "rpt1_centroid_pairwise_distance",
            probe_validity: ProbeValidity::BehavioralProxy,
            expected_effect: target_effect(
                "rpt1_centroid_pairwise_distance",
                "full CfC recurrence should discriminate distinct inputs into distant \
                 centroids; a 1-neuron bottleneck should collapse them",
            ),
        },
        QualificationDesign {
            indicator: "RPT-2",
            target_lever: "disable_cross_modal_binding",
            target_lever_group: "cross_modal_binding",
            positive_control: PositiveControlPlan {
                id: PositiveControlId("rpt2_degenerate_channels"),
                purpose: StimulusResponsiveness,
                readiness: Unverified,
                protocol_group: "degenerate_channel_stimulus",
                description: "The real signal is a boolean 'did the cross_modal_binding module \
                              execute this cycle' (module_timings_us.cross_modal_binding > 0), \
                              not a graded binding-quality measure, AND measure_indicator's real \
                              signature has no exposed override point for custom channel \
                              content -- neither achievability nor discriminating power is \
                              confirmed.",
                expected_effect: ExpectedEffect {
                    metric: "cross_modal_binding_module_timing",
                    comparison: PositiveControlVsBaseline,
                    direction: Decrease,
                    rationale: "unverified -- see description",
                },
            },
            sham: sham(
                "disable_attention_schema",
                "attention_schema",
                "reuses AST-1's real target as an unrelated disruptive control for RPT-2",
                "AST-1",
                "attention_schema_focus",
            ),
            functional_benchmark: "WorM::ChangeDetection",
            functional_benchmark_group: "worm_changedetection",
            probe_metric: "cross_modal_binding_module_timing",
            probe_group: "cross_modal_binding_module_timing",
            probe_validity: ProbeValidity::ExecutionProxy,
            expected_effect: target_effect(
                "cross_modal_binding_module_timing",
                "module_timings_us.cross_modal_binding measures whether the module executed, \
                 an execution proxy for binding quality, not binding quality itself",
            ),
        },
        QualificationDesign {
            indicator: "GWT-2",
            target_lever: "disable_gwt_capacity",
            target_lever_group: "gwt_enable",
            positive_control: PositiveControlPlan {
                id: PositiveControlId("gwt2_forced_zero_coalition"),
                purpose: Instrumentation,
                readiness: FormulaVerified,
                protocol_group: "gwt2_coalition_size_override",
                description: "CORRECTED: the original 'force capacity to 1' does not violate \
                              the real predicate (size > 0 && size < 1000) -- size 1 still \
                              satisfies it. Corrected: force gwt_coalition_size to 0 directly, \
                              which genuinely violates the `> 0` half -- verified against the \
                              real predicate in ablation.rs.",
                expected_effect: ExpectedEffect {
                    metric: "gwt_coalition_size_bounded_activity",
                    comparison: ZeroPinnedVsNonzeroPinned,
                    direction: Decrease,
                    rationale: "forcing size=0 fails `size > 0`, dropping the activity fraction \
                                toward 0.0",
                },
            },
            sham: sham(
                "disable_online_learning",
                "online_learning",
                "shared with GWT-3 -- both GWT outcomes of the one enable_gwt causal unit use \
                 the identical sham so their specificity evidence is comparable",
                "HOT-3",
                "actual_effective_lr",
            ),
            functional_benchmark: "WorM::N-back",
            functional_benchmark_group: "worm_nback",
            probe_metric: "gwt_coalition_size_bounded_activity",
            probe_group: "gwt_coalition_size_bounded_activity",
            probe_validity: ProbeValidity::ExecutionProxy,
            expected_effect: target_effect(
                "gwt_coalition_size_bounded_activity",
                "when enable_gwt=false, coalition_size is always 0, failing the non-empty half \
                 of the bounded-capacity check",
            ),
        },
        QualificationDesign {
            indicator: "GWT-3",
            target_lever: "disable_gwt_broadcast",
            target_lever_group: "gwt_enable",
            positive_control: PositiveControlPlan {
                id: PositiveControlId("gwt3_forced_no_broadcast"),
                purpose: StimulusResponsiveness,
                readiness: Unverified,
                protocol_group: "gwt3_broadcast_flag_override",
                description: "CONSTRUCT-VALIDITY CONCERN (not merely unverified): the real \
                              signal is a boolean 'did the gwt module execute this cycle' \
                              (module_timings_us.gwt > 0), which measures module EXECUTION, not \
                              whether a global broadcast specifically occurred. Forcing a \
                              broadcast flag off may not stop the module from running/timing \
                              itself at all. Validating this control would only confirm the \
                              timing proxy, not the theoretical broadcast claim -- a real fix \
                              needs a genuine broadcast-event/fan-out/uptake field, which this \
                              module does not manufacture just to fill this slot.",
                expected_effect: ExpectedEffect {
                    metric: "gwt_module_timing_activity",
                    comparison: PositiveControlVsBaseline,
                    direction: Decrease,
                    rationale: "unverified and construct-validity-limited -- see description",
                },
            },
            sham: sham(
                "disable_online_learning",
                "online_learning",
                "shared with GWT-2 -- both GWT outcomes of the one enable_gwt causal unit use \
                 the identical sham so their specificity evidence is comparable",
                "HOT-3",
                "actual_effective_lr",
            ),
            functional_benchmark: "WorM::N-back",
            functional_benchmark_group: "worm_nback",
            probe_metric: "gwt_module_timing_activity",
            probe_group: "gwt_module_timing_activity",
            probe_validity: ProbeValidity::ExecutionProxy,
            expected_effect: target_effect(
                "gwt_module_timing_activity",
                "module_timings_us.gwt measures whether the module executed, an execution \
                 proxy for global broadcast, not broadcast fan-out/reach itself",
            ),
        },
        QualificationDesign {
            indicator: "GWT-4",
            target_lever: "disable_phi_attention",
            target_lever_group: "phi_attention",
            positive_control: PositiveControlPlan {
                id: PositiveControlId("gwt4_pinned_neutral_weight"),
                purpose: Instrumentation,
                readiness: FormulaVerified,
                protocol_group: "gwt4_phi_attention_weight_pin",
                description: "pin phi_attention_weight to 1.0 (neutral) and confirm the \
                              deviation probe reads exactly 0.0, then unpin and confirm nonzero \
                              -- verified directly against the real formula \
                              ((weight - 1.0).abs().min(1.0)) in extract_indicator_score.",
                expected_effect: ExpectedEffect {
                    metric: "phi_attention_weight_deviation",
                    comparison: ZeroPinnedVsNonzeroPinned,
                    direction: Decrease,
                    rationale: "pinning to the neutral value directly zeroes the deviation \
                                formula",
                },
            },
            sham: sham(
                "disable_embodied_cognition",
                "embodied_cognition",
                "reuses AE-2's real target as an unrelated disruptive control for GWT-4",
                "AE-2",
                "embodied_agency_field",
            ),
            functional_benchmark: "WorM::SpatialUpdating",
            functional_benchmark_group: "worm_spatialupdating",
            probe_metric: "phi_attention_weight_deviation",
            probe_group: "phi_attention_weight_deviation",
            probe_validity: ProbeValidity::DirectMeasure,
            expected_effect: target_effect(
                "phi_attention_weight_deviation",
                "when enable_phi_attention=false, the weight stays exactly at its neutral \
                 value (1.0), zeroing the deviation reading",
            ),
        },
        QualificationDesign {
            indicator: "HOT-1",
            target_lever: "disable_predictive_processing",
            target_lever_group: "predictive_processing",
            positive_control: PositiveControlPlan {
                id: PositiveControlId("hot1_adversarial_input"),
                purpose: StimulusResponsiveness,
                readiness: Unverified,
                protocol_group: "hot1_adversarial_input_stimulus",
                description: "measure_indicator's real signature hardcodes a fixed 10-sentence \
                              rotation with no override point exposed for a custom 'adversarial' \
                              input stream -- not confirmed achievable via the current \
                              measurement API without a code change.",
                expected_effect: ExpectedEffect {
                    metric: "prediction_error_variance_across_inputs",
                    comparison: AdversarialVsPredictableStimulus,
                    direction: Decrease,
                    rationale: "unverified -- see description",
                },
            },
            sham: sham(
                "disable_attention_schema",
                "attention_schema",
                "reuses AST-1's real target as an unrelated disruptive control for HOT-1",
                "AST-1",
                "attention_schema_focus",
            ),
            functional_benchmark: "CogBench::TwoStep",
            functional_benchmark_group: "cogbench_twostep",
            probe_metric: "prediction_error_variance_across_inputs",
            probe_group: "prediction_error_variance_across_inputs",
            probe_validity: ProbeValidity::BehavioralProxy,
            expected_effect: target_effect(
                "prediction_error_variance_across_inputs",
                "variance of prediction_error across distinct inputs should collapse toward 0 \
                 if nothing is genuinely being predicted differently",
            ),
        },
        QualificationDesign {
            indicator: "HOT-2",
            target_lever: "disable_metacognition",
            target_lever_group: "metacognition",
            positive_control: PositiveControlPlan {
                id: PositiveControlId("hot2_injected_wrong_confidence"),
                purpose: Instrumentation,
                readiness: Unverified,
                protocol_group: "hot2_confidence_injection",
                description: "meta_cognitive_accuracy is a real field (verified in \
                              extract_indicator_score), but whether it has an exposed injection \
                              hook for a 'known-wrong confidence signal' hasn't been traced.",
                expected_effect: ExpectedEffect {
                    metric: "meta_cognitive_accuracy",
                    comparison: PositiveControlVsBaseline,
                    direction: Decrease,
                    rationale: "field confirmed real; injection mechanism unverified",
                },
            },
            sham: sham(
                "disable_cfc_recurrence",
                "cfc_recurrence",
                "reuses RPT-1's real target as an unrelated disruptive control for HOT-2",
                "RPT-1",
                "rpt1_centroid_pairwise_distance",
            ),
            functional_benchmark: "WorM::N-back",
            functional_benchmark_group: "worm_nback",
            probe_metric: "meta_cognitive_accuracy",
            probe_group: "meta_cognitive_accuracy",
            probe_validity: ProbeValidity::DirectMeasure,
            expected_effect: target_effect(
                "meta_cognitive_accuracy",
                "metadata.quality.meta_cognitive_accuracy directly reads confidence calibration \
                 quality",
            ),
        },
        QualificationDesign {
            indicator: "HOT-3",
            target_lever: "disable_online_learning",
            target_lever_group: "online_learning",
            positive_control: PositiveControlPlan {
                id: PositiveControlId("hot3_pinned_learning_rate"),
                purpose: Instrumentation,
                readiness: FormulaVerified,
                protocol_group: "pinned_learning_rate_instrumentation",
                description: "pin actual_effective_lr to a known nonzero constant and confirm \
                              the probe reads it back, then to zero -- verified directly against \
                              extract_indicator_score's `\"PP-1\" | \"HOT-3\" => metadata.\
                              actual_effective_lr as f64` arm.",
                expected_effect: ExpectedEffect {
                    metric: "actual_effective_lr",
                    comparison: ZeroPinnedVsNonzeroPinned,
                    direction: Decrease,
                    rationale: "the probe reads this exact field with no transformation",
                },
            },
            sham: sham(
                "disable_cross_modal_binding",
                "cross_modal_binding",
                "reuses RPT-2's real target as an unrelated disruptive control for HOT-3",
                "RPT-2",
                "cross_modal_binding_module_timing",
            ),
            functional_benchmark: "CogBench::InstrumentalLearning",
            functional_benchmark_group: "cogbench_instrumentallearning",
            probe_metric: "actual_effective_lr",
            // Shared with PP-1 -- see module doc, finding 7. This is the
            // one confirmed raw-probe-signal duplicate among all 12.
            probe_group: "actual_effective_lr",
            probe_validity: ProbeValidity::DirectMeasure,
            expected_effect: target_effect(
                "actual_effective_lr",
                "when disable_online_learning is active, effective LR should read 0.0; \
                 NOTE: this is the identical raw field PP-1 also reads (see module doc) -- \
                 whether HOT-3 and PP-1 can be distinguished at all depends on their different \
                 interventions producing different patterns on this one shared signal",
            ),
        },
        QualificationDesign {
            indicator: "AST-1",
            target_lever: "disable_attention_schema",
            target_lever_group: "attention_schema",
            positive_control: PositiveControlPlan {
                id: PositiveControlId("ast1_fixed_attention_input"),
                purpose: StimulusResponsiveness,
                readiness: Unverified,
                protocol_group: "ast1_fixed_upstream_input_stimulus",
                description: "assumes an override point for the attention-focus signal's \
                              upstream input that hasn't been confirmed to exist \
                              (attention_schema_focus is computed in \
                              cycle_late_consciousness/monitors.rs, not traced fully here).",
                expected_effect: ExpectedEffect {
                    metric: "attention_schema_focus",
                    comparison: PositiveControlVsBaseline,
                    direction: Decrease,
                    rationale: "unverified -- see description",
                },
            },
            sham: sham(
                "disable_phi_attention",
                "phi_attention",
                "reuses GWT-4's real target as an unrelated disruptive control for AST-1",
                "GWT-4",
                "phi_attention_weight_deviation",
            ),
            functional_benchmark: "WorM::N-back",
            functional_benchmark_group: "worm_nback",
            probe_metric: "attention_schema_focus",
            probe_group: "attention_schema_focus",
            probe_validity: ProbeValidity::DirectMeasure,
            expected_effect: target_effect(
                "attention_schema_focus",
                "metadata.attention.attention_schema_focus directly reads the attention-schema \
                 signal (with a 0.01 non-zero fallback)",
            ),
        },
        QualificationDesign {
            indicator: "PP-1",
            target_lever: "disable_prediction_learning",
            target_lever_group: "prediction_learning",
            positive_control: PositiveControlPlan {
                id: PositiveControlId("pp1_pinned_learning_rate"),
                purpose: Instrumentation,
                readiness: FormulaVerified,
                protocol_group: "pinned_learning_rate_instrumentation",
                description: "pin actual_effective_lr to a known nonzero constant and confirm \
                              the probe reflects it -- same underlying instrumentation protocol \
                              as HOT-3's (shares_positive_control_protocol is true for this \
                              pair; shares_positive_control, comparing only the unique id, is \
                              false).",
                expected_effect: ExpectedEffect {
                    metric: "actual_effective_lr",
                    comparison: ZeroPinnedVsNonzeroPinned,
                    direction: Decrease,
                    rationale: "the probe reads this exact field with no transformation",
                },
            },
            sham: sham(
                "disable_embodied_cognition",
                "embodied_cognition",
                "reuses AE-2's real target as an unrelated disruptive control for PP-1",
                "AE-2",
                "embodied_agency_field",
            ),
            functional_benchmark: "WorM::N-back",
            functional_benchmark_group: "worm_nback",
            probe_metric: "actual_effective_lr",
            // Shared with HOT-3 -- see module doc, finding 7.
            probe_group: "actual_effective_lr",
            probe_validity: ProbeValidity::DirectMeasure,
            expected_effect: target_effect(
                "actual_effective_lr",
                "when disable_prediction_learning is active (learning_threshold=f32::MAX), \
                 effective LR should read 0.0; NOTE: identical raw field to HOT-3, see module doc",
            ),
        },
        QualificationDesign {
            indicator: "AE-1",
            target_lever: "disable_trajectory_planning",
            target_lever_group: "trajectory_planning",
            positive_control: PositiveControlPlan {
                id: PositiveControlId("ae1_repeated_identical_input"),
                purpose: StimulusResponsiveness,
                readiness: Unverified,
                protocol_group: "ae1_repeated_input_stimulus",
                description: "the real measure_indicator cross-cycle loop always advances \
                              through the fixed 10-sentence rotation via `i % inputs.len()` with \
                              no exposed way to force the same sentence every cycle -- \
                              achievability unconfirmed.",
                expected_effect: ExpectedEffect {
                    metric: "fep_action_diversity_count",
                    comparison: PositiveControlVsBaseline,
                    direction: Decrease,
                    rationale: "unverified -- see description",
                },
            },
            sham: sham(
                "disable_gwt_capacity",
                "gwt_enable",
                "reuses GWT-2/GWT-3's real target as an unrelated disruptive control for AE-1",
                "GWT-2",
                "gwt_coalition_size_bounded_activity",
            ),
            functional_benchmark: "CogBench::TwoStep",
            functional_benchmark_group: "cogbench_twostep",
            probe_metric: "fep_action_diversity_count",
            probe_group: "fep_action_diversity_count",
            probe_validity: ProbeValidity::BehavioralProxy,
            expected_effect: target_effect(
                "fep_action_diversity_count",
                "count of distinct FEP actions (0=exploit,1=consolidate,2=explore,3=tighten) \
                 seen across distinct inputs, normalized by 4",
            ),
        },
        QualificationDesign {
            indicator: "AE-2",
            target_lever: "disable_embodied_cognition",
            target_lever_group: "embodied_cognition",
            positive_control: PositiveControlPlan {
                id: PositiveControlId("ae2_zeroed_embodied_agency_field"),
                purpose: Instrumentation,
                readiness: FormulaVerified,
                protocol_group: "ae2_embodied_agency_field_zero",
                description: "directly zero the embodied_agency field in CycleMetadata and \
                              confirm the probe reports zero, then restore and confirm nonzero \
                              -- verified directly against extract_indicator_score \
                              (metadata.embodied.embodied_agency, documented as already 0.0 \
                              when disabled).",
                expected_effect: ExpectedEffect {
                    metric: "embodied_agency_field",
                    comparison: ZeroPinnedVsNonzeroPinned,
                    direction: Decrease,
                    rationale: "the probe reads this exact field with no transformation",
                },
            },
            sham: sham(
                "disable_predictive_processing",
                "predictive_processing",
                "reuses HOT-1's real target as an unrelated disruptive control for AE-2",
                "HOT-1",
                "prediction_error_variance_across_inputs",
            ),
            functional_benchmark: "WorM::SpatialUpdating",
            functional_benchmark_group: "worm_spatialupdating",
            probe_metric: "embodied_agency_field",
            probe_group: "embodied_agency_field",
            probe_validity: ProbeValidity::DirectMeasure,
            expected_effect: target_effect(
                "embodied_agency_field",
                "metadata.embodied.embodied_agency is documented as already 0.0 when embodied \
                 cognition is disabled",
            ),
        },
    ]
}

/// The exact 12 indicator IDs `planned_designs()` must cover.
const EXPECTED_INDICATOR_IDS: [&str; 12] = [
    "AE-1", "AE-2", "AST-1", "GWT-2", "GWT-3", "GWT-4", "HOT-1", "HOT-2", "HOT-3", "PP-1", "RPT-1",
    "RPT-2",
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planned_designs_cover_all_twelve_indicators() {
        let designs = planned_designs();
        let mut ids: Vec<&str> = designs.iter().map(|d| d.indicator).collect();
        ids.sort_unstable();
        assert_eq!(ids, EXPECTED_INDICATOR_IDS.to_vec());
    }

    #[test]
    fn test_planned_designs_pass_validation() {
        let violations = validate_designs(&planned_designs());
        assert!(violations.is_empty(), "found: {violations:?}");
    }

    #[test]
    fn test_no_sham_equals_its_own_target() {
        for d in planned_designs() {
            assert_ne!(d.sham.lever, d.target_lever, "{}", d.indicator);
        }
    }

    // ── Dimension-specific dependency semantics ─────────────────────────

    #[test]
    fn test_gwt2_gwt3_share_target_but_not_probe_signal() {
        let designs = planned_designs();
        let gwt2 = designs.iter().find(|d| d.indicator == "GWT-2").unwrap();
        let gwt3 = designs.iter().find(|d| d.indicator == "GWT-3").unwrap();
        assert!(gwt2.shares_target_intervention(gwt3));
        assert!(!gwt2.causally_independent_of(gwt3));
        assert!(gwt2.shares_functional_benchmark(gwt3));
        assert!(gwt2.shares_sham_intervention(gwt3));
        assert!(
            !gwt2.shares_probe_signal(gwt3),
            "GWT-2 and GWT-3 read different fields (coalition size vs. module timing) despite \
             sharing a causal unit"
        );
        assert!(!gwt2.fully_independent_of(gwt3));
    }

    #[test]
    fn test_hot3_and_pp1_share_probe_signal() {
        // The confirmed finding: HOT-3 and PP-1 read the literal same raw
        // field, a stronger claim than sharing a positive-control protocol.
        let designs = planned_designs();
        let hot3 = designs.iter().find(|d| d.indicator == "HOT-3").unwrap();
        let pp1 = designs.iter().find(|d| d.indicator == "PP-1").unwrap();
        assert!(hot3.shares_probe_signal(pp1));
        assert!(hot3.shares_positive_control_protocol(pp1));
        assert!(
            !hot3.shares_positive_control(pp1),
            "ids must remain distinct"
        );
        assert!(
            hot3.causally_independent_of(pp1),
            "different target levers -- causal evidence remains independent"
        );
        assert!(
            !hot3.fully_independent_of(pp1),
            "shared probe signal blocks full independence"
        );
        let dep = hot3.dependency_of(pp1);
        assert!(dep.shared_probe_signal);
        assert!(dep.shared_positive_control_protocol);
        assert!(!dep.is_fully_independent());
    }

    #[test]
    fn test_only_hot3_and_pp1_share_a_probe_signal() {
        // Completeness check: no OTHER pair among the 12 accidentally
        // shares a probe_group.
        let designs = planned_designs();
        for a in &designs {
            for b in &designs {
                if a.indicator == b.indicator {
                    continue;
                }
                let is_known_pair = (a.indicator == "HOT-3" && b.indicator == "PP-1")
                    || (a.indicator == "PP-1" && b.indicator == "HOT-3");
                assert_eq!(
                    a.shares_probe_signal(b),
                    is_known_pair,
                    "{} vs {}: unexpected probe-signal sharing state",
                    a.indicator,
                    b.indicator
                );
            }
        }
    }

    #[test]
    fn test_cross_role_overlap_hot1_sham_is_ast1_target() {
        let designs = planned_designs();
        let hot1 = designs.iter().find(|d| d.indicator == "HOT-1").unwrap();
        let ast1 = designs.iter().find(|d| d.indicator == "AST-1").unwrap();
        assert!(!hot1.shares_target_intervention(ast1));
        assert!(!hot1.shares_sham_intervention(ast1));
        assert!(hot1.shares_intervention_cross_role(ast1));
        assert!(!hot1.fully_independent_of(ast1));
    }

    #[test]
    fn test_independent_pair_reports_fully_independent_on_every_dimension() {
        let designs = planned_designs();
        let rpt2 = designs.iter().find(|d| d.indicator == "RPT-2").unwrap();
        let hot2 = designs.iter().find(|d| d.indicator == "HOT-2").unwrap();
        assert!(rpt2.fully_independent_of(hot2));
        assert!(rpt2.dependency_of(hot2).is_fully_independent());
    }

    #[test]
    fn test_dependency_methods_are_symmetric() {
        let designs = planned_designs();
        for a in &designs {
            for b in &designs {
                assert_eq!(
                    a.shares_target_intervention(b),
                    b.shares_target_intervention(a)
                );
                assert_eq!(
                    a.shares_functional_benchmark(b),
                    b.shares_functional_benchmark(a)
                );
                assert_eq!(a.shares_sham_intervention(b), b.shares_sham_intervention(a));
                assert_eq!(
                    a.shares_intervention_cross_role(b),
                    b.shares_intervention_cross_role(a)
                );
                assert_eq!(a.shares_positive_control(b), b.shares_positive_control(a));
                assert_eq!(
                    a.shares_positive_control_protocol(b),
                    b.shares_positive_control_protocol(a)
                );
                assert_eq!(a.shares_probe_signal(b), b.shares_probe_signal(a));
                assert_eq!(a.fully_independent_of(b), b.fully_independent_of(a));
            }
        }
    }

    #[test]
    fn test_no_two_indicators_share_a_positive_control_id() {
        let designs = planned_designs();
        for a in &designs {
            for b in &designs {
                if a.indicator != b.indicator {
                    assert!(
                        !a.shares_positive_control(b),
                        "{} / {}",
                        a.indicator,
                        b.indicator
                    );
                }
            }
        }
    }

    // ── Control readiness / purpose ──────────────────────────────────────

    #[test]
    fn test_rpt1_positive_control_does_not_qualify_by_design() {
        let designs = planned_designs();
        let rpt1 = designs.iter().find(|d| d.indicator == "RPT-1").unwrap();
        assert_eq!(
            rpt1.positive_control.purpose,
            ControlPurpose::DegenerateGuardTest
        );
        assert!(!rpt1.positive_control.control_design_qualifies());
        assert!(!rpt1.static_design_qualifies());
    }

    #[test]
    fn test_exactly_five_control_designs_currently_qualify() {
        // Control-design credibility alone (purpose + readiness) --
        // deliberately NOT the same question as whether the whole probe is
        // eligible, see test_exactly_four_rows_pass_static_design below.
        let designs = planned_designs();
        let qualifying: Vec<&str> = designs
            .iter()
            .filter(|d| d.positive_control.control_design_qualifies())
            .map(|d| d.indicator)
            .collect();
        let mut sorted = qualifying.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec!["AE-2", "GWT-2", "GWT-4", "HOT-3", "PP-1"],
            "got: {qualifying:?}"
        );
    }

    #[test]
    fn test_exactly_four_rows_pass_static_design_qualifies() {
        // The corrected honest headline finding, after splitting control-
        // design credibility from probe validity: GWT-2's control is
        // FormulaVerified, but its probe is ExecutionProxy, so it fails this
        // stricter combined check even though it passes the control-only one.
        let designs = planned_designs();
        let qualifying: Vec<&str> = designs
            .iter()
            .filter(|d| d.static_design_qualifies())
            .map(|d| d.indicator)
            .collect();
        let mut sorted = qualifying.clone();
        sorted.sort_unstable();
        assert_eq!(
            sorted,
            vec!["AE-2", "GWT-4", "HOT-3", "PP-1"],
            "got: {qualifying:?}"
        );
    }

    #[test]
    fn test_gwt2_control_qualifies_but_static_design_does_not() {
        let designs = planned_designs();
        let gwt2 = designs.iter().find(|d| d.indicator == "GWT-2").unwrap();
        assert!(
            gwt2.positive_control.control_design_qualifies(),
            "GWT-2's control is genuinely FormulaVerified"
        );
        assert!(
            !gwt2.static_design_qualifies(),
            "but GWT-2's probe is ExecutionProxy, so the combined check must fail"
        );
    }

    #[test]
    fn test_construct_validity_concerns_are_execution_proxies() {
        let designs = planned_designs();
        for id in ["GWT-3", "RPT-2", "GWT-2"] {
            let d = designs.iter().find(|d| d.indicator == id).unwrap();
            assert_eq!(d.probe_validity, ProbeValidity::ExecutionProxy, "{id}");
        }
    }

    #[test]
    fn test_all_shams_are_proposed_not_formula_verified() {
        for d in planned_designs() {
            assert_eq!(
                d.sham.readiness,
                ControlReadiness::Proposed,
                "{}",
                d.indicator
            );
        }
    }

    // ── Shared-group reporting ───────────────────────────────────────────

    #[test]
    fn test_shared_groups_reports_all_three_known_cases() {
        let groups = shared_groups(&planned_designs());
        assert!(groups.iter().any(
            |g| matches!(g, SharedGroup::Intervention { group, indicators }
            if *group == "gwt_enable" && indicators.len() == 2)
        ));
        assert!(groups.iter().any(
            |g| matches!(g, SharedGroup::FunctionalBenchmark { group, indicators }
                if *group == "worm_nback" && indicators.len() == 6)
        ));
        assert!(groups.iter().any(
            |g| matches!(g, SharedGroup::ProbeSignal { group, indicators }
            if *group == "actual_effective_lr" && indicators.len() == 2)
        ));
    }

    // ── Validator: synthetic corruption tests ───────────────────────────

    #[test]
    fn test_validate_designs_catches_sham_matching_target() {
        let mut designs = planned_designs();
        let broken = designs.iter_mut().find(|d| d.indicator == "RPT-1").unwrap();
        broken.sham.lever = broken.target_lever;
        assert!(
            validate_designs(&designs)
                .contains(&DesignViolation::ShamMatchesTarget { indicator: "RPT-1" })
        );
    }

    #[test]
    fn test_validate_designs_catches_sham_sharing_target_group() {
        let mut designs = planned_designs();
        let broken = designs.iter_mut().find(|d| d.indicator == "RPT-1").unwrap();
        broken.sham.lever = "some_other_name_same_mechanism";
        broken.sham.group = broken.target_lever_group;
        assert!(
            validate_designs(&designs)
                .contains(&DesignViolation::ShamSharesTargetGroup { indicator: "RPT-1" })
        );
    }

    #[test]
    fn test_validate_designs_catches_undeclared_shared_intervention() {
        let mut designs = planned_designs();
        let broken = designs.iter_mut().find(|d| d.indicator == "AE-1").unwrap();
        broken.target_lever = "disable_gwt_broadcast";
        broken.target_lever_group = "trajectory_planning";
        assert!(
            validate_designs(&designs)
                .iter()
                .any(|v| matches!(v, DesignViolation::UndeclaredSharedIntervention { .. }))
        );
    }

    #[test]
    fn test_validate_designs_catches_undeclared_shared_benchmark() {
        let mut designs = planned_designs();
        let broken = designs.iter_mut().find(|d| d.indicator == "RPT-2").unwrap();
        broken.functional_benchmark = "WorM::N-back";
        broken.functional_benchmark_group = "some_inconsistent_group";
        assert!(
            validate_designs(&designs)
                .iter()
                .any(|v| matches!(v, DesignViolation::UndeclaredSharedBenchmark { .. }))
        );
    }

    #[test]
    fn test_planned_designs_all_declare_correct_sham_reuse_ownership() {
        let designs = planned_designs();
        let target_owner: std::collections::HashMap<&str, &str> = designs
            .iter()
            .map(|d| (d.target_lever, d.indicator))
            .collect();
        let mut reused_count = 0;
        for d in &designs {
            if let Some(&owner) = target_owner.get(d.sham.lever) {
                assert_ne!(owner, d.indicator);
                reused_count += 1;
                assert_eq!(d.sham.reuses_target_of, Some(owner), "{}", d.indicator);
            }
        }
        assert_eq!(reused_count, 12);
    }

    #[test]
    fn test_validate_designs_catches_undeclared_sham_reuse() {
        let mut designs = planned_designs();
        let broken = designs.iter_mut().find(|d| d.indicator == "RPT-1").unwrap();
        broken.sham.reuses_target_of = None;
        assert!(
            validate_designs(&designs).contains(&DesignViolation::ShamTargetReuseUndeclared {
                indicator: "RPT-1",
                sham_lever: "disable_metacognition",
                actual_owner: "HOT-2",
            })
        );
    }

    #[test]
    fn test_validate_designs_catches_wrong_declared_sham_reuse_owner() {
        let mut designs = planned_designs();
        let broken = designs.iter_mut().find(|d| d.indicator == "RPT-1").unwrap();
        broken.sham.reuses_target_of = Some("AE-2");
        assert!(
            validate_designs(&designs).contains(&DesignViolation::ShamTargetReuseMismatch {
                indicator: "RPT-1",
                sham_lever: "disable_metacognition",
                actual_owner: "HOT-2",
                declared_owner: "AE-2",
            })
        );
    }

    #[test]
    fn test_validate_designs_catches_false_claim_of_sham_reuse() {
        let mut designs = planned_designs();
        let broken = designs.iter_mut().find(|d| d.indicator == "RPT-1").unwrap();
        broken.sham.lever = "disable_something_nobody_targets";
        broken.sham.group = "nobody_targets_this";
        assert!(validate_designs(&designs).contains(
            &DesignViolation::ShamFalselyClaimsTargetReuse {
                indicator: "RPT-1",
                sham_lever: "disable_something_nobody_targets",
                declared_owner: "HOT-2",
            }
        ));
    }

    // ── Drift protection against the real ablation_specs() table ────────
    #[cfg(feature = "symthaea-backend")]
    #[test]
    fn qualification_design_matches_ablation_specs() {
        use super::super::ablation::ablation_specs;

        let real_specs = ablation_specs();
        let designs = planned_designs();

        let mut real_ids: Vec<&str> = real_specs.iter().map(|s| s.target_indicator).collect();
        real_ids.sort_unstable();
        real_ids.dedup();
        assert_eq!(real_ids.len(), real_specs.len());
        let mut design_ids: Vec<&str> = designs.iter().map(|d| d.indicator).collect();
        design_ids.sort_unstable();
        assert_eq!(design_ids, real_ids);

        for real in &real_specs {
            let design = designs
                .iter()
                .find(|d| d.indicator == real.target_indicator)
                .unwrap_or_else(|| {
                    panic!(
                        "no matching planned_designs() entry for {:?}",
                        real.target_indicator
                    )
                });
            assert_eq!(design.target_lever, real.name, "{}", real.target_indicator);
            assert_eq!(
                design.functional_benchmark, real.downstream_benchmark,
                "{}",
                real.target_indicator
            );
        }
    }
}
