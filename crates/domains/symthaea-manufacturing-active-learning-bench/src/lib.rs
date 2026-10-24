// Copyright (C) 2024-2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Minimal, deterministic benchmark for manufacturing-conditioned experiment selection.
//!
//! This is synthetic policy evidence only. It does not model or qualify a real
//! manufacturing process and has no live equipment, network, or actuation interface.
#![forbid(unsafe_code)]

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::{BTreeMap, BTreeSet};

pub mod analysis;
pub mod evaluator;
pub use analysis::{
    BenchmarkSummary, CampaignEvaluationSample, PairedPolicyComparison, PolicyAggregate,
    summarize_paired_campaigns,
};
pub use evaluator::evaluate_transcript;

pub const BENCHMARK_ID: &str = "MFG-ACTIVE-001";
pub const SCHEMA_VERSION: u16 = 1;
pub const PROFILE_ID: &str = "profile:synthetic-reference-cell-v1";

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum ScenarioFamily {
    SmoothResponse,
    NarrowFeasibleWindow,
    MaterialProcessInteraction,
    ConfoundedFactors,
    MultiFidelityDiscrepancy,
}

impl ScenarioFamily {
    pub const ALL: [Self; 5] = [
        Self::SmoothResponse,
        Self::NarrowFeasibleWindow,
        Self::MaterialProcessInteraction,
        Self::ConfoundedFactors,
        Self::MultiFidelityDiscrepancy,
    ];
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Fidelity {
    Proxy,
    Reference,
}

impl Fidelity {
    const ALL: [Self; 2] = [Self::Proxy, Self::Reference];
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Feasibility {
    Feasible,
    Blocked { reason: String },
    Unknown { reason: String },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Candidate {
    pub id: String,
    /// Normalized dimensionless design factor, 0..=10.
    pub factor_x: i32,
    /// Normalized dimensionless process factor, 0..=10.
    pub factor_y: i32,
    pub process_profile_id: String,
    pub feasibility: Feasibility,
    pub proxy_available: bool,
    pub reference_available: bool,
    /// Relative cost multiplier, not a currency or real process estimate.
    pub cost_scale: u32,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignManifest {
    pub schema_version: u16,
    pub benchmark_id: String,
    /// Kept in evaluator-side manifest; never copied to the selector's public view.
    pub scenario: ScenarioFamily,
    pub oracle_version: String,
    /// Oracle seed; policies do not receive this value.
    pub oracle_seed: u64,
    pub budget_units: u32,
    pub proxy_cost_units: u32,
    pub reference_cost_units: u32,
    pub minimum_quality_margin_milli: i32,
    pub process_profile_id: String,
    /// Prior experiments shared by all policy arms; costs count toward each budget.
    pub initial_actions: Vec<ExperimentAction>,
    pub candidates: Vec<Candidate>,
}

impl CampaignManifest {
    /// Create the frozen synthetic reference cell for one declared scenario and seed.
    /// All numeric factors are dimensionless; cost units are relative benchmark units.
    pub fn reference(scenario: ScenarioFamily, oracle_seed: u64, budget_units: u32) -> Self {
        let mut candidates = Vec::with_capacity(121);
        for x in 0..=10 {
            for y in 0..=10 {
                let feasibility = if x == 0 && y == 0 {
                    Feasibility::Unknown {
                        reason: "synthetic calibration evidence is stale".into(),
                    }
                } else if (x == 0 && y == 10) || (x == 10 && y == 0) {
                    Feasibility::Blocked {
                        reason: "outside the declared synthetic workcell envelope".into(),
                    }
                } else {
                    Feasibility::Feasible
                };
                let reference_available = (x + y) % 11 != 0;
                // Cost structure is scenario-independent, so the visible candidate
                // catalog does not encode the hidden oracle family.
                let cost_scale = if x != y && (x + y) % 3 == 0 { 4 } else { 1 };
                candidates.push(Candidate {
                    id: format!("C{x:02}-{y:02}"),
                    factor_x: x,
                    factor_y: y,
                    process_profile_id: PROFILE_ID.into(),
                    feasibility,
                    proxy_available: true,
                    reference_available,
                    cost_scale,
                });
            }
        }

        let initial_actions = if scenario == ScenarioFamily::ConfoundedFactors {
            ["C02-02", "C05-05", "C08-08"]
                .into_iter()
                .map(|candidate_id| ExperimentAction {
                    candidate_id: candidate_id.into(),
                    fidelity: Fidelity::Reference,
                })
                .collect()
        } else {
            Vec::new()
        };

        Self {
            schema_version: SCHEMA_VERSION,
            benchmark_id: BENCHMARK_ID.into(),
            scenario,
            oracle_version: "synthetic-oracle-v2".into(),
            oracle_seed,
            budget_units,
            proxy_cost_units: 1,
            reference_cost_units: 4,
            minimum_quality_margin_milli: 0,
            process_profile_id: PROFILE_ID.into(),
            initial_actions,
            candidates,
        }
    }

    pub fn digest(&self) -> Result<String, String> {
        digest_serializable(self)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.schema_version != SCHEMA_VERSION {
            return Err("unsupported manifest schema version".into());
        }
        if self.benchmark_id != BENCHMARK_ID {
            return Err("unexpected benchmark identity".into());
        }
        if self.oracle_version != "synthetic-oracle-v2" {
            return Err("unsupported synthetic oracle version".into());
        }
        if self.process_profile_id.trim().is_empty() {
            return Err("process profile identity must be non-empty".into());
        }
        if self.budget_units == 0 || self.proxy_cost_units == 0 || self.reference_cost_units == 0 {
            return Err("budget and fidelity costs must be positive".into());
        }
        if self.candidates.is_empty() {
            return Err("candidate domain cannot be empty".into());
        }

        let mut ids = BTreeSet::new();
        for candidate in &self.candidates {
            if candidate.id.trim().is_empty() || !ids.insert(candidate.id.as_str()) {
                return Err("candidate IDs must be non-empty and unique".into());
            }
            if !(0..=10).contains(&candidate.factor_x) || !(0..=10).contains(&candidate.factor_y) {
                return Err(format!(
                    "candidate {} lies outside the normalized domain",
                    candidate.id
                ));
            }
            if candidate.process_profile_id != self.process_profile_id {
                return Err(format!("candidate {} has a mismatched process profile", candidate.id));
            }
            if candidate.cost_scale == 0 {
                return Err(format!("candidate {} has a zero cost scale", candidate.id));
            }
            if self.proxy_cost_units.checked_mul(candidate.cost_scale).is_none()
                || self.reference_cost_units.checked_mul(candidate.cost_scale).is_none()
            {
                return Err(format!("candidate {} has an overflowing declared cost", candidate.id));
            }
            if !candidate.proxy_available && !candidate.reference_available {
                return Err(format!(
                    "candidate {} has no available measurement fidelity",
                    candidate.id
                ));
            }
        }
        let candidates_by_id: BTreeMap<&str, &Candidate> = self
            .candidates
            .iter()
            .map(|candidate| (candidate.id.as_str(), candidate))
            .collect();
        let mut seen_initial_actions = BTreeSet::new();
        let mut initial_cost = 0u32;
        for action in &self.initial_actions {
            if !seen_initial_actions.insert(action.key()) {
                return Err("initial design contains a duplicate action".into());
            }
            let candidate = candidates_by_id
                .get(action.candidate_id.as_str())
                .ok_or_else(|| "initial design references a missing candidate".to_string())?;
            if candidate.feasibility != Feasibility::Feasible {
                return Err("initial design contains a blocked or unknown candidate".into());
            }
            let available = match action.fidelity {
                Fidelity::Proxy => candidate.proxy_available,
                Fidelity::Reference => candidate.reference_available,
            };
            if !available {
                return Err("initial design requests an unavailable measurement fidelity".into());
            }
            let base = match action.fidelity {
                Fidelity::Proxy => self.proxy_cost_units,
                Fidelity::Reference => self.reference_cost_units,
            };
            initial_cost = initial_cost
                .checked_add(base.saturating_mul(candidate.cost_scale))
                .ok_or_else(|| "initial design cost overflow".to_string())?;
        }
        if initial_cost > self.budget_units {
            return Err("initial design exceeds the declared campaign budget".into());
        }

        Ok(())
    }

    pub(crate) fn public_view(&self) -> PublicCampaignView {
        PublicCampaignView {
            benchmark_id: self.benchmark_id.clone(),
            process_profile_id: self.process_profile_id.clone(),
            budget_units: self.budget_units,
            proxy_cost_units: self.proxy_cost_units,
            reference_cost_units: self.reference_cost_units,
            minimum_quality_margin_milli: self.minimum_quality_margin_milli,
            candidates: self.candidates.clone(),
        }
    }
}

/// Only declared design/process information reaches selection policies.
/// Scenario family, oracle version and oracle seed are deliberately omitted.
#[derive(Debug, Clone)]
pub struct PublicCampaignView {
    pub benchmark_id: String,
    pub process_profile_id: String,
    pub budget_units: u32,
    pub proxy_cost_units: u32,
    pub reference_cost_units: u32,
    pub minimum_quality_margin_milli: i32,
    pub candidates: Vec<Candidate>,
}

impl PublicCampaignView {
    pub fn action_cost(&self, candidate: &Candidate, fidelity: Fidelity) -> u32 {
        let base = match fidelity {
            Fidelity::Proxy => self.proxy_cost_units,
            Fidelity::Reference => self.reference_cost_units,
        };
        base.saturating_mul(candidate.cost_scale)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Policy {
    FixedSpaceFilling,
    RandomFeasible,
    UncertaintyOnly,
    InformationGainProxy,
    CostAwareInformationGainProxy,
    DecisionAware,
}

impl Policy {
    pub const ALL: [Self; 6] = [
        Self::FixedSpaceFilling,
        Self::RandomFeasible,
        Self::UncertaintyOnly,
        Self::InformationGainProxy,
        Self::CostAwareInformationGainProxy,
        Self::DecisionAware,
    ];

    pub fn id(self) -> &'static str {
        match self {
            Self::FixedSpaceFilling => "P0-fixed-space-filling",
            Self::RandomFeasible => "P1-random-feasible",
            Self::UncertaintyOnly => "P2-uncertainty-only",
            Self::InformationGainProxy => "P3-information-gain-proxy",
            Self::CostAwareInformationGainProxy => "P4-cost-aware-information-gain-proxy",
            Self::DecisionAware => "P5-decision-aware",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentAction {
    pub candidate_id: String,
    pub fidelity: Fidelity,
}

impl ExperimentAction {
    pub(crate) fn key(&self) -> String {
        format!("{}:{:?}", self.candidate_id, self.fidelity)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Measurements {
    /// Required synthetic product-response margin in milli-units. Conformance requires >= limit.
    pub quality_margin_milli: Option<i32>,
    /// Synthetic conforming-throughput proxy, in benchmark units.
    pub throughput_units: Option<u32>,
    /// Synthetic resource-use proxy, in benchmark units.
    pub resource_units: Option<u32>,
}

impl Measurements {
    fn complete(&self) -> bool {
        self.quality_margin_milli.is_some()
            && self.throughput_units.is_some()
            && self.resource_units.is_some()
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExperimentRecord {
    pub step_index: u32,
    pub action: ExperimentAction,
    pub reported_cost_units: u32,
    pub measurements: Measurements,
    /// A redundant claim retained for the independent evaluator to check.
    pub claimed_conforming: bool,
    /// Integrity identifier for the declared manifest, action, profile and measurements.
    pub measurement_digest: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct CampaignTranscript {
    pub manifest_digest: String,
    pub policy: Policy,
    /// Policy RNG seed; separate from the hidden oracle seed.
    pub selector_seed: u64,
    pub records: Vec<ExperimentRecord>,
    /// Redundant aggregates are intentionally recomputed and checked by evaluate_transcript.
    pub claimed_total_cost_units: u32,
    pub claimed_first_conforming_cost_units: Option<u32>,
    pub claimed_best_utility: Option<i64>,
    pub stop_reason: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvaluationReport {
    pub benchmark_id: String,
    pub scenario: ScenarioFamily,
    pub policy: Policy,
    pub manifest_digest: String,
    pub experiments: u32,
    pub total_cost_units: u32,
    pub first_conforming_cost_units: Option<u32>,
    pub best_conforming_utility: Option<i64>,
    pub max_conforming_throughput_units: Option<u32>,
    pub total_resource_units: u32,
    pub blocked_catalog_candidates: u32,
    pub unknown_catalog_candidates: u32,
    /// Indicates only geometric excitation, not causal identifiability by itself.
    pub factor_design_full_rank: bool,
}

#[derive(Debug, Clone)]
struct ObservationForPolicy {
    action: ExperimentAction,
    factor_x: i32,
    factor_y: i32,
    measurements: Measurements,
}

pub fn run_campaign(
    manifest: &CampaignManifest,
    policy: Policy,
    selector_seed: u64,
) -> Result<CampaignTranscript, String> {
    manifest.validate()?;
    let manifest_digest = manifest.digest()?;
    let public_view = manifest.public_view();
    let mut records: Vec<ExperimentRecord> = Vec::new();
    let mut history: Vec<ObservationForPolicy> = Vec::new();
    let mut spent = 0u32;

    // Seed a shared prior history before policy selection. Its cost counts toward
    // each arm's total budget so no policy receives free extra experiments.
    for action in &manifest.initial_actions {
        let candidate = public_view
            .candidates
            .iter()
            .find(|candidate| candidate.id == action.candidate_id)
            .ok_or_else(|| "initial action references a missing candidate".to_string())?;
        let cost = public_view.action_cost(candidate, action.fidelity);
        if cost == 0 || cost > manifest.budget_units.saturating_sub(spent) {
            return Err("initial design exceeds the remaining budget".into());
        }
        let measurements = synthetic_oracle(manifest, candidate, action.fidelity);
        let claimed_conforming = action.fidelity == Fidelity::Reference
            && measurements
                .quality_margin_milli
                .is_some_and(|margin| margin >= public_view.minimum_quality_margin_milli);
        let measurement_digest = measurement_digest(
            &manifest_digest,
            action,
            &candidate.process_profile_id,
            &measurements,
        )?;
        records.push(ExperimentRecord {
            step_index: records.len() as u32,
            action: action.clone(),
            reported_cost_units: cost,
            measurements: measurements.clone(),
            claimed_conforming,
            measurement_digest,
        });
        history.push(ObservationForPolicy {
            action: action.clone(),
            factor_x: candidate.factor_x,
            factor_y: candidate.factor_y,
            measurements,
        });
        spent += cost;
    }

    loop {
        let selected = choose_action(
            &public_view,
            &history,
            policy,
            selector_seed,
            (records.len() - manifest.initial_actions.len()) as u64,
            manifest.budget_units.saturating_sub(spent),
        );
        let Some(action) = selected else {
            break;
        };

        let candidate = public_view
            .candidates
            .iter()
            .find(|candidate| candidate.id == action.candidate_id)
            .ok_or_else(|| "selector returned a candidate outside the domain".to_string())?;
        let cost = public_view.action_cost(candidate, action.fidelity);
        if cost == 0 || cost > manifest.budget_units.saturating_sub(spent) {
            return Err("selector returned an action outside the remaining budget".into());
        }

        let measurements = synthetic_oracle(manifest, candidate, action.fidelity);
        let claimed_conforming = action.fidelity == Fidelity::Reference
            && measurements
                .quality_margin_milli
                .is_some_and(|margin| margin >= public_view.minimum_quality_margin_milli);
        let measurement_digest = measurement_digest(
            &manifest_digest,
            &action,
            &candidate.process_profile_id,
            &measurements,
        )?;

        records.push(ExperimentRecord {
            step_index: records.len() as u32,
            action: action.clone(),
            reported_cost_units: cost,
            measurements: measurements.clone(),
            claimed_conforming,
            measurement_digest,
        });
        history.push(ObservationForPolicy {
            action,
            factor_x: candidate.factor_x,
            factor_y: candidate.factor_y,
            measurements,
        });
        spent += cost;
    }

    let (claimed_first_conforming_cost_units, claimed_best_utility) =
        campaign_claims(&records, &public_view);
    let stop_reason = if spent >= manifest.budget_units {
        "BudgetExhausted"
    } else {
        "NoEligibleAffordableAction"
    };

    Ok(CampaignTranscript {
        manifest_digest,
        policy,
        selector_seed,
        records,
        claimed_total_cost_units: spent,
        claimed_first_conforming_cost_units,
        claimed_best_utility,
        stop_reason: stop_reason.into(),
    })
}

fn choose_action(
    view: &PublicCampaignView,
    history: &[ObservationForPolicy],
    policy: Policy,
    selector_seed: u64,
    step: u64,
    remaining_budget: u32,
) -> Option<ExperimentAction> {
    let mut already_run = BTreeSet::new();
    for item in history {
        already_run.insert(item.action.key());
    }

    // Use the same predeclared fidelity cadence for every policy arm so the main
    // comparison tests candidate selection rather than silently rewarding policies
    // that never pay for reference-quality confirmation. A policy falls back only
    // when no affordable candidate supports the scheduled fidelity.
    let preferred_fidelity = if step % 4 == 3 {
        Fidelity::Reference
    } else {
        Fidelity::Proxy
    };
    let has_affordable_action = |fidelity: Fidelity| {
        view.candidates.iter().any(|candidate| {
            if candidate.feasibility != Feasibility::Feasible {
                return false;
            }
            let available = match fidelity {
                Fidelity::Proxy => candidate.proxy_available,
                Fidelity::Reference => candidate.reference_available,
            };
            if !available {
                return false;
            }
            let action = ExperimentAction {
                candidate_id: candidate.id.clone(),
                fidelity,
            };
            let cost = view.action_cost(candidate, fidelity);
            !already_run.contains(&action.key()) && cost > 0 && cost <= remaining_budget
        })
    };
    let alternate_fidelity = match preferred_fidelity {
        Fidelity::Proxy => Fidelity::Reference,
        Fidelity::Reference => Fidelity::Proxy,
    };
    let selected_fidelity = if has_affordable_action(preferred_fidelity) {
        preferred_fidelity
    } else if has_affordable_action(alternate_fidelity) {
        alternate_fidelity
    } else {
        return None;
    };

    let mut best: Option<(i64, ExperimentAction)> = None;
    for candidate in &view.candidates {
        if candidate.feasibility != Feasibility::Feasible {
            continue;
        }
        for fidelity in [selected_fidelity] {
            let available = match fidelity {
                Fidelity::Proxy => candidate.proxy_available,
                Fidelity::Reference => candidate.reference_available,
            };
            if !available {
                continue;
            }
            let action = ExperimentAction {
                candidate_id: candidate.id.clone(),
                fidelity,
            };
            if already_run.contains(&action.key()) {
                continue;
            }
            let cost = view.action_cost(candidate, fidelity);
            if cost == 0 || cost > remaining_budget {
                continue;
            }

            let novelty = novelty_score(candidate, fidelity, history);
            let leverage = 10 + candidate.factor_x + candidate.factor_y
                + 2 * (candidate.factor_x - candidate.factor_y).abs();
            if policy == Policy::FixedSpaceFilling
                && history.iter().any(|item| {
                    item.factor_x == candidate.factor_x && item.factor_y == candidate.factor_y
                })
            {
                // The baseline is a predeclared design, not an adaptive multifidelity policy.
                // Do not spend a second fidelity measurement at a point already in its design.
                continue;
            }
            let score = match policy {
                Policy::FixedSpaceFilling => -(fixed_design_rank(candidate) as i64),
                Policy::RandomFeasible => {
                    i64::from(stable_random_key(selector_seed, step, &action.key()))
                }
                Policy::UncertaintyOnly => novelty,
                // Transparent design-space proxy; this is not Bayesian expected information gain.
                Policy::InformationGainProxy => novelty * i64::from(leverage),
                Policy::CostAwareInformationGainProxy => {
                    novelty * i64::from(leverage) * 100 / i64::from(cost)
                }
                Policy::DecisionAware => {
                    let predicted_margin = nearest_observed_margin(candidate, history).unwrap_or(0);
                    i64::from(predicted_margin) * 4
                        + novelty * 2
                        + if fidelity == Fidelity::Reference { 25 } else { 0 }
                        - i64::from(cost) * 20
                }
            };
            // Stable candidate/fidelity iteration order resolves exact score ties.
            if best.as_ref().is_none_or(|(best_score, _)| score > *best_score) {
                best = Some((score, action));
            }
        }
    }
    best.map(|(_, action)| action)
}

/// Fixed 2-factor Latin-hypercube-like order over the 11×11 normalized grid.
/// Each block visits every x and every y once; successive blocks shift the
/// permutation so the sequence spans the full grid without adapting to outputs.
fn fixed_design_rank(candidate: &Candidate) -> u32 {
    let x = candidate.factor_x;
    let y = candidate.factor_y;
    let block = ((y - 4 * x).rem_euclid(11) * 4).rem_euclid(11);
    (block * 11 + x) as u32
}

fn novelty_score(
    candidate: &Candidate,
    fidelity: Fidelity,
    history: &[ObservationForPolicy],
) -> i64 {
    if history.is_empty() {
        return 200;
    }
    let nearest_distance = history
        .iter()
        .map(|item| {
            let dx = candidate.factor_x - item.factor_x;
            let dy = candidate.factor_y - item.factor_y;
            if dx == 0 && dy == 0 {
                if item.action.fidelity == fidelity { 0 } else { 2 }
            } else {
                i64::from(dx * dx + dy * dy) * 4
            }
        })
        .min()
        .unwrap_or(200);
    nearest_distance
}

fn nearest_observed_margin(
    candidate: &Candidate,
    history: &[ObservationForPolicy],
) -> Option<i32> {
    // Until the multifidelity issue supplies a calibrated discrepancy model, do not
    // use cheap proxy responses to make a conformance-sensitive decision estimate.
    history
        .iter()
        .filter(|item| item.action.fidelity == Fidelity::Reference)
        .filter_map(|item| {
            item.measurements
                .quality_margin_milli
                .map(|margin| {
                    let dx = candidate.factor_x - item.factor_x;
                    let dy = candidate.factor_y - item.factor_y;
                    (dx * dx + dy * dy, margin)
                })
        })
        .min_by_key(|(distance, _)| *distance)
        .map(|(_, margin)| margin)
}

fn stable_random_key(seed: u64, step: u64, action_key: &str) -> u32 {
    // FNV-1a-like stable key followed by fixed integer mixing. No global RNG or wall clock.
    let mut value = 0xcbf29ce484222325u64 ^ seed ^ step.rotate_left(17);
    for byte in action_key.as_bytes() {
        value ^= u64::from(*byte);
        value = value.wrapping_mul(0x100000001b3);
    }
    value ^= value >> 30;
    value = value.wrapping_mul(0xbf58476d1ce4e5b9);
    value ^= value >> 27;
    value = value.wrapping_mul(0x94d049bb133111eb);
    value ^= value >> 31;
    (value >> 32) as u32
}

fn synthetic_oracle(
    manifest: &CampaignManifest,
    candidate: &Candidate,
    fidelity: Fidelity,
) -> Measurements {
    let x = candidate.factor_x;
    let y = candidate.factor_y;
    let dx_smooth = x - 7;
    let dy_smooth = y - 3;
    let distance_narrow = (x - 8) * (x - 8) + (y - 8) * (y - 8);

    let true_margin = match manifest.scenario {
        ScenarioFamily::SmoothResponse => {
            280 - 18 * dx_smooth * dx_smooth - 11 * dy_smooth * dy_smooth
        }
        ScenarioFamily::NarrowFeasibleWindow => 240 - 90 * distance_narrow,
        ScenarioFamily::MaterialProcessInteraction => -90 + 7 * x * y - 20 * x - 12 * y,
        // The supplied initial data lie on x == y and all fail the target; an off-diagonal
        // experiment is needed before the campaign can discover the feasible synthetic region.
        ScenarioFamily::ConfoundedFactors => {
            -50 - 20 * (x + y - 10) * (x + y - 10) + 10 * (x - y) * (x - y)
        },
        ScenarioFamily::MultiFidelityDiscrepancy => {
            160 - 10 * (x - 6) * (x - 6) - 8 * (y - 6) * (y - 6)
        }
    };

    let bias = if fidelity == Fidelity::Reference {
        0
    } else {
        match manifest.scenario {
            ScenarioFamily::MultiFidelityDiscrepancy if x >= 6 && y <= 4 => 180,
            ScenarioFamily::MultiFidelityDiscrepancy => -35,
            _ if (x + 2 * y + (manifest.oracle_seed % 3) as i32) % 5 == 0 => 12,
            _ => -4,
        }
    };
    let margin = true_margin + bias;
    let throughput = (30 + 2 * x - y).max(1) as u32;
    let resource = (10 + x + y) as u32;

    Measurements {
        quality_margin_milli: Some(margin),
        throughput_units: Some(throughput),
        resource_units: Some(resource),
    }
}

pub(crate) fn measurement_digest(
    manifest_digest: &str,
    action: &ExperimentAction,
    process_profile_id: &str,
    measurements: &Measurements,
) -> Result<String, String> {
    digest_serializable(&(manifest_digest, action, process_profile_id, measurements))
}

fn digest_serializable<T: Serialize>(value: &T) -> Result<String, String> {
    let bytes = serde_json::to_vec(value)
        .map_err(|error| format!("serialization failed: {error}"))?;
    Ok(hex::encode(Sha256::digest(bytes)))
}

fn campaign_claims(
    records: &[ExperimentRecord],
    view: &PublicCampaignView,
) -> (Option<u32>, Option<i64>) {
    let mut cumulative_cost = 0u32;
    let mut first_conforming = None;
    let mut best_utility = None;
    for record in records {
        cumulative_cost = cumulative_cost.saturating_add(record.reported_cost_units);
        let margin = record.measurements.quality_margin_milli;
        let throughput = record.measurements.throughput_units;
        let resource = record.measurements.resource_units;
        // Proxy-fidelity observations can inform future selection, but cannot qualify a
        // conforming candidate or production utility. Qualification requires reference fidelity.
        if record.action.fidelity != Fidelity::Reference {
            continue;
        }
        if let (Some(margin), Some(throughput), Some(resource)) = (margin, throughput, resource) {
            if margin >= view.minimum_quality_margin_milli {
                if first_conforming.is_none() {
                    first_conforming = Some(cumulative_cost);
                }
                let utility = i64::from(throughput) * 1000 - i64::from(resource) * 100
                    + i64::from(margin);
                if best_utility.is_none_or(|best| utility > best) {
                    best_utility = Some(utility);
                }
            }
        }
    }
    (first_conforming, best_utility)
}

/// Recompute metrics from the frozen manifest and raw observation receipts.
///
/// This evaluator is a separate function boundary from policy selection and oracle access,
/// but is in the same source module and is not an independently built executable. The receipt
/// digest proves byte identity within this harness; it is not a signature or proof of physical evidence.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frozen_manifest_is_deterministic_and_versioned() {
        let a = CampaignManifest::reference(ScenarioFamily::SmoothResponse, 7, 20);
        let b = CampaignManifest::reference(ScenarioFamily::SmoothResponse, 7, 20);
        assert_eq!(a.digest().unwrap(), b.digest().unwrap());
        assert_eq!(a.candidates.len(), 121);
        assert_eq!(a.schema_version, SCHEMA_VERSION);
    }

    #[test]
    fn reference_campaign_manifests_match_frozen_sha256_digests() {
        let expected = [
            "65c78abb4af2e6a909821a000d6c7260c3f8badc9e15c20d5938ecece2090024",
            "795df0dc743cd7314f871b3093ea5fb93f1ba863b8cb8389cdc82c2272077852",
            "defdab16abefe200f505bcc2a9c2ef183b8bff4ab3d75788a6bee395c9c77032",
            "e3f085264c5d8eefdbdfe528769f573c09e07953ef791302f69daf54c610cbb3",
            "04d215b2141f533cfa9c183caaf96a68c9129ef2e010d95f37c6513ba9e2a63c",
        ];
        for (index, scenario) in ScenarioFamily::ALL.into_iter().enumerate() {
            let manifest =
                CampaignManifest::reference(scenario, 0x4d46_4700_u64 + index as u64, 20);
            assert_eq!(manifest.digest().unwrap(), expected[index], "scenario={scenario:?}");
        }
    }

    #[test]
    fn manifest_rejects_unknown_oracle_version() {
        let mut manifest = CampaignManifest::reference(ScenarioFamily::SmoothResponse, 1, 20);
        manifest.oracle_version = "synthetic-oracle-unknown".into();
        assert!(manifest.validate().unwrap_err().contains("unsupported synthetic oracle version"));
    }

    #[test]
    fn selector_view_does_not_expose_scenario_oracle_version_or_oracle_seed() {
        let manifest =
            CampaignManifest::reference(ScenarioFamily::MultiFidelityDiscrepancy, 99, 10);
        let view = manifest.public_view();
        assert_eq!(view.benchmark_id, BENCHMARK_ID);
        // The public view's type has no scenario, oracle_version or oracle_seed fields by design.
        assert_eq!(view.candidates.len(), 121);
    }

    #[test]
    fn fixed_design_order_is_a_permutation_of_the_full_candidate_grid() {
        let candidates = CampaignManifest::reference(ScenarioFamily::SmoothResponse, 1, 20)
            .candidates;
        let ranks: BTreeSet<u32> = candidates
            .iter()
            .map(fixed_design_rank)
            .collect();
        assert_eq!(ranks.len(), 121);
        assert_eq!(ranks.first(), Some(&0));
        assert_eq!(ranks.last(), Some(&120));
    }

    #[test]
    fn all_policies_replay_deterministically_across_all_synthetic_families() {
        for scenario in ScenarioFamily::ALL {
            let manifest = CampaignManifest::reference(scenario, 23, 20);
            for policy in Policy::ALL {
                let a = run_campaign(&manifest, policy, 1234).unwrap();
                let b = run_campaign(&manifest, policy, 1234).unwrap();
                assert_eq!(a, b, "scenario={scenario:?}, policy={policy:?}");
                let report = evaluate_transcript(&manifest, &a).unwrap();
                assert_eq!(report.total_cost_units, a.claimed_total_cost_units);
                assert!(report.total_cost_units <= manifest.budget_units);
                let expected_max_throughput = a.records.iter().filter_map(|record| {
                    if record.action.fidelity != Fidelity::Reference {
                        return None;
                    }
                    let margin = record.measurements.quality_margin_milli?;
                    let throughput = record.measurements.throughput_units?;
                    (margin >= manifest.minimum_quality_margin_milli).then_some(throughput)
                }).max();
                assert_eq!(report.max_conforming_throughput_units, expected_max_throughput);
            }
        }
    }

    #[test]
    fn all_policies_share_three_proxy_slots_then_reference_confirmation() {
        let manifest = CampaignManifest::reference(ScenarioFamily::SmoothResponse, 17, 20);
        for policy in Policy::ALL {
            let transcript = run_campaign(&manifest, policy, 42).unwrap();
            assert!(transcript.records.len() >= 4);
            assert!(transcript.records[0..3]
                .iter()
                .all(|record| record.action.fidelity == Fidelity::Proxy));
            assert_eq!(transcript.records[3].action.fidelity, Fidelity::Reference);
        }

        let manifest = CampaignManifest::reference(ScenarioFamily::ConfoundedFactors, 17, 20);
        let transcript = run_campaign(&manifest, Policy::FixedSpaceFilling, 42).unwrap();
        let offset = manifest.initial_actions.len();
        assert!(transcript.records[offset..offset + 3]
            .iter()
            .all(|record| record.action.fidelity == Fidelity::Proxy));
        assert_eq!(
            transcript.records[offset + 3].action.fidelity,
            Fidelity::Reference
        );
    }

    #[test]
    fn no_policy_selects_blocked_or_unknown_candidates() {
        for scenario in ScenarioFamily::ALL {
            let manifest = CampaignManifest::reference(scenario, 1, 20);
            for policy in Policy::ALL {
                let transcript = run_campaign(&manifest, policy, 42).unwrap();
                for record in transcript.records {
                    let candidate = manifest.candidates.iter()
                        .find(|candidate| candidate.id == record.action.candidate_id).unwrap();
                    assert_eq!(candidate.feasibility, Feasibility::Feasible);
                }
            }
        }
    }

    #[test]
    fn confounded_scenario_starts_with_collinear_history_and_can_break_rank() {
        let manifest = CampaignManifest::reference(ScenarioFamily::ConfoundedFactors, 9, 20);
        assert_eq!(manifest.initial_actions.len(), 3);
        assert_eq!(
            manifest
                .initial_actions
                .iter()
                .map(|action| action.candidate_id.as_str())
                .collect::<Vec<_>>(),
            vec!["C02-02", "C05-05", "C08-08"]
        );
        let transcript =
            run_campaign(&manifest, Policy::FixedSpaceFilling, 77).unwrap();
        let report = evaluate_transcript(&manifest, &transcript).unwrap();
        assert_eq!(transcript.records[0].action, manifest.initial_actions[0]);
        assert!(transcript.records.iter().all(|record| {
            let expected_conforming = record.action.fidelity == Fidelity::Reference
                && record.measurements.quality_margin_milli
                    .is_some_and(|margin| margin >= manifest.minimum_quality_margin_milli);
            record.claimed_conforming == expected_conforming
        }));
        assert!(report.total_cost_units <= manifest.budget_units);
        assert!(
            report.factor_design_full_rank,
            "the design should escape the diagonal-only prior"
        );
        // Discovery is a reported outcome, not an invariant that every policy
        // must satisfy; a benchmark must preserve valid NotFound results.
    }

    #[test]
    fn evaluator_rejects_falsified_stop_reason() {
        let manifest = CampaignManifest::reference(ScenarioFamily::SmoothResponse, 7, 20);
        let mut transcript = run_campaign(&manifest, Policy::FixedSpaceFilling, 4).unwrap();
        transcript.stop_reason = if transcript.stop_reason == "BudgetExhausted" {
            "NoEligibleAffordableAction".into()
        } else {
            "BudgetExhausted".into()
        };
        assert!(evaluate_transcript(&manifest, &transcript)
            .unwrap_err().contains("stop reason"));
    }

    #[test]
    fn evaluator_rejects_falsified_cost_aggregate() {
        let manifest = CampaignManifest::reference(ScenarioFamily::SmoothResponse, 7, 20);
        let mut transcript = run_campaign(&manifest, Policy::FixedSpaceFilling, 4).unwrap();
        transcript.claimed_total_cost_units += 1;
        assert!(evaluate_transcript(&manifest, &transcript)
            .unwrap_err().contains("claimed campaign cost"));
    }

    #[test]
    fn evaluator_rejects_falsified_conformity_claim() {
        let manifest = CampaignManifest::reference(ScenarioFamily::SmoothResponse, 7, 20);
        let mut transcript = run_campaign(&manifest, Policy::FixedSpaceFilling, 4).unwrap();
        transcript.records[0].claimed_conforming = !transcript.records[0].claimed_conforming;
        assert!(evaluate_transcript(&manifest, &transcript)
            .unwrap_err().contains("claimed conformity"));
    }

    #[test]
    fn evaluator_rejects_mutated_measurement_receipt() {
        let manifest = CampaignManifest::reference(ScenarioFamily::SmoothResponse, 7, 20);
        let mut transcript = run_campaign(&manifest, Policy::FixedSpaceFilling, 4).unwrap();
        transcript.records[0].measurements.resource_units =
            Some(transcript.records[0].measurements.resource_units.unwrap() + 1);
        assert!(evaluate_transcript(&manifest, &transcript)
            .unwrap_err().contains("measurement receipt digest"));
    }

    #[test]
    fn evaluator_rejects_missing_critical_measurement() {
        let manifest = CampaignManifest::reference(ScenarioFamily::SmoothResponse, 7, 20);
        let mut transcript = run_campaign(&manifest, Policy::FixedSpaceFilling, 4).unwrap();
        transcript.records[0].measurements.quality_margin_milli = None;
        assert!(evaluate_transcript(&manifest, &transcript)
            .unwrap_err().contains("missing required measurement"));
    }

    #[test]
    fn evaluator_rejects_blocked_candidate_even_if_transcript_is_forged() {
        let manifest = CampaignManifest::reference(ScenarioFamily::SmoothResponse, 7, 20);
        let mut transcript = run_campaign(&manifest, Policy::FixedSpaceFilling, 4).unwrap();
        transcript.records[0].action.candidate_id = "C00-10".into();
        assert!(evaluate_transcript(&manifest, &transcript)
            .unwrap_err().contains("blocked or unknown"));
    }

    #[test]
    fn evaluator_rejects_proxy_only_claim_of_conformity() {
        let manifest =
            CampaignManifest::reference(ScenarioFamily::MultiFidelityDiscrepancy, 7, 20);
        let candidate = manifest
            .candidates
            .iter()
            .find(|candidate| candidate.factor_x == 8 && candidate.factor_y == 2)
            .unwrap();
        let action = ExperimentAction {
            candidate_id: candidate.id.clone(),
            fidelity: Fidelity::Proxy,
        };
        let measurements = synthetic_oracle(&manifest, candidate, Fidelity::Proxy);
        assert!(measurements.quality_margin_milli.unwrap() >= 0);
        let manifest_digest = manifest.digest().unwrap();
        let record = ExperimentRecord {
            step_index: 0,
            action: action.clone(),
            reported_cost_units: manifest.public_view().action_cost(candidate, Fidelity::Proxy),
            measurements: measurements.clone(),
            claimed_conforming: true,
            measurement_digest: measurement_digest(
                &manifest_digest,
                &action,
                &candidate.process_profile_id,
                &measurements,
            )
            .unwrap(),
        };
        let transcript = CampaignTranscript {
            manifest_digest,
            policy: Policy::DecisionAware,
            selector_seed: 1,
            records: vec![record],
            claimed_total_cost_units: 1,
            claimed_first_conforming_cost_units: None,
            claimed_best_utility: None,
            stop_reason: "Test".into(),
        };
        assert!(evaluate_transcript(&manifest, &transcript)
            .unwrap_err()
            .contains("reference-fidelity measurement"));
    }

    #[test]
    fn biased_proxy_response_cannot_qualify_a_conforming_candidate() {
        let manifest =
            CampaignManifest::reference(ScenarioFamily::MultiFidelityDiscrepancy, 7, 20);
        let candidate = manifest
            .candidates
            .iter()
            .find(|candidate| candidate.factor_x == 8 && candidate.factor_y == 2)
            .unwrap();
        let proxy = synthetic_oracle(&manifest, candidate, Fidelity::Proxy);
        let reference = synthetic_oracle(&manifest, candidate, Fidelity::Reference);
        assert!(proxy.quality_margin_milli.unwrap() >= manifest.minimum_quality_margin_milli);
        assert!(reference.quality_margin_milli.unwrap() < manifest.minimum_quality_margin_milli);

        let record = ExperimentRecord {
            step_index: 0,
            action: ExperimentAction {
                candidate_id: candidate.id.clone(),
                fidelity: Fidelity::Proxy,
            },
            reported_cost_units: 1,
            measurements: proxy,
            claimed_conforming: false,
            measurement_digest: String::new(),
        };
        assert_eq!(
            campaign_claims(&[record], &manifest.public_view()),
            (None, None),
            "proxy predictions are exploratory evidence, not conformity evidence"
        );
    }

    #[test]
    fn proxy_and_reference_fidelity_are_distinct_in_discrepancy_scenario() {
        let manifest = CampaignManifest::reference(ScenarioFamily::MultiFidelityDiscrepancy, 7, 10);
        let candidate = manifest.candidates.iter()
            .find(|candidate| candidate.factor_x == 8 && candidate.factor_y == 3).unwrap();
        let proxy = synthetic_oracle(&manifest, candidate, Fidelity::Proxy);
        let reference = synthetic_oracle(&manifest, candidate, Fidelity::Reference);
        assert_ne!(proxy.quality_margin_milli, reference.quality_margin_milli);
    }

    #[test]
    fn factor_design_rank_is_not_inferred_from_a_diagonal_only_campaign() {
        let points = BTreeSet::from([(1, 1), (4, 4), (9, 9)]);
        assert!(!evaluator::has_full_rank_factor_design(&points));
        let points = BTreeSet::from([(1, 1), (4, 4), (4, 1)]);
        assert!(has_full_rank_factor_design(&points));
    }
}
