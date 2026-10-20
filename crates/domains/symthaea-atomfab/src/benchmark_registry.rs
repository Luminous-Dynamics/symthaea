//! Machine-readable benchmark registry and integrity checks.
//!
//! This module loads the checked-in comparison matrix and validates its
//! structure before a campaign runner uses it. Passing this validation means
//! the manifest is internally consistent; it does not mean a benchmark ran
//! or that any result is state-of-the-art.

use serde::Deserialize;
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

pub const BENCHMARK_REGISTRY_SCHEMA_VERSION: u16 = 1;

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkRegistry {
    pub schema_version: u16,
    pub registry_id: String,
    pub reviewed_on: String,
    pub status: String,
    pub purpose: String,
    pub claim_policy: ClaimPolicy,
    pub benchmark_lanes: Vec<BenchmarkLane>,
    pub policy_baselines: Vec<PolicyBaseline>,
    pub performance_claims: PerformanceClaims,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ClaimPolicy {
    pub no_sota_claim_until: Vec<String>,
    pub distinguish_result_classes: Vec<String>,
    pub rule: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkLane {
    pub id: String,
    pub status: String,
    pub goal: String,
    pub reference: BenchmarkReference,
    pub baselines: Vec<String>,
    pub metrics: Vec<String>,
    pub gates: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkReference {
    pub name: String,
    pub doi: Option<String>,
    pub url: Option<String>,
    pub systems: Option<Vec<String>>,
    pub scope: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PolicyBaseline {
    pub id: String,
    pub definition: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct PerformanceClaims {
    pub accelerations: AccelerationGate,
    pub process_feasibility: FeasibilityGate,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AccelerationGate {
    pub stretch_target_median_af: f64,
    pub minimum_campaigns: u64,
    pub confidence_interval: String,
    pub precondition: String,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FeasibilityGate {
    pub maximum_false_feasible_accepts_on_frozen_adversarial_suite: u64,
    pub if_no_candidates_accepted: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegistryError {
    UnsupportedSchemaVersion,
    MissingMetadata,
    MissingResultClass,
    EmptyLaneSet,
    DuplicateLaneId,
    IncompleteLane,
    DuplicatePolicyId,
    MissingRequiredPolicy,
    InvalidAccelerationGate,
    InvalidFeasibilityGate,
}

impl fmt::Display for RegistryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnsupportedSchemaVersion => {
                write!(f, "unsupported benchmark registry schema version")
            }
            Self::MissingMetadata => write!(f, "benchmark registry metadata is incomplete"),
            Self::MissingResultClass => {
                write!(f, "benchmark registry must distinguish every claim class")
            }
            Self::EmptyLaneSet => write!(f, "benchmark registry must contain at least one lane"),
            Self::DuplicateLaneId => write!(f, "benchmark lane identifiers must be unique"),
            Self::IncompleteLane => write!(
                f,
                "every benchmark lane needs a reference, goal, baselines, metrics and gates"
            ),
            Self::DuplicatePolicyId => write!(f, "benchmark policy identifiers must be unique"),
            Self::MissingRequiredPolicy => write!(
                f,
                "registry must include random and Symthaea policy baselines"
            ),
            Self::InvalidAccelerationGate => write!(
                f,
                "acceleration gate must require AF > 1 and at least two campaigns"
            ),
            Self::InvalidFeasibilityGate => write!(
                f,
                "false-feasible gate must require zero accepted hard-constraint violations"
            ),
        }
    }
}

impl Error for RegistryError {}

impl BenchmarkRegistry {
    /// Validate the registry's structural invariants and minimum research gates.
    pub fn validate(&self) -> Result<(), RegistryError> {
        if self.schema_version != BENCHMARK_REGISTRY_SCHEMA_VERSION {
            return Err(RegistryError::UnsupportedSchemaVersion);
        }
        if self.registry_id.trim().is_empty()
            || self.reviewed_on.trim().is_empty()
            || self.status.trim().is_empty()
            || self.purpose.trim().is_empty()
            || self.claim_policy.rule.trim().is_empty()
            || self.claim_policy.no_sota_claim_until.is_empty()
        {
            return Err(RegistryError::MissingMetadata);
        }
        for class in [
            "reference",
            "prediction",
            "simulation",
            "measurement",
            "independently_reproduced",
        ] {
            if !self
                .claim_policy
                .distinguish_result_classes
                .iter()
                .any(|item| item == class)
            {
                return Err(RegistryError::MissingResultClass);
            }
        }
        if self.benchmark_lanes.is_empty() {
            return Err(RegistryError::EmptyLaneSet);
        }

        let mut lane_ids = BTreeSet::new();
        for lane in &self.benchmark_lanes {
            if !lane_ids.insert(lane.id.as_str()) {
                return Err(RegistryError::DuplicateLaneId);
            }
            if lane.id.trim().is_empty()
                || lane.status.trim().is_empty()
                || lane.goal.trim().is_empty()
                || lane.reference.name.trim().is_empty()
                || lane.baselines.is_empty()
                || lane.metrics.is_empty()
                || lane.gates.is_empty()
                || lane.baselines.iter().any(|item| item.trim().is_empty())
                || lane.metrics.iter().any(|item| item.trim().is_empty())
                || lane.gates.iter().any(|item| item.trim().is_empty())
            {
                return Err(RegistryError::IncompleteLane);
            }
        }

        let mut policy_ids = BTreeSet::new();
        for policy in &self.policy_baselines {
            if policy.id.trim().is_empty()
                || policy.definition.trim().is_empty()
                || !policy_ids.insert(policy.id.as_str())
            {
                return Err(RegistryError::DuplicatePolicyId);
            }
        }
        if !self
            .policy_baselines
            .iter()
            .any(|policy| policy.id == "random")
            || !self
                .policy_baselines
                .iter()
                .any(|policy| policy.id == "symthaea_full")
        {
            return Err(RegistryError::MissingRequiredPolicy);
        }

        let acceleration = &self.performance_claims.accelerations;
        if !acceleration.stretch_target_median_af.is_finite()
            || acceleration.stretch_target_median_af <= 1.0
            || acceleration.minimum_campaigns < 2
            || acceleration.confidence_interval.trim().is_empty()
            || acceleration.precondition.trim().is_empty()
        {
            return Err(RegistryError::InvalidAccelerationGate);
        }
        let feasibility = &self.performance_claims.process_feasibility;
        if feasibility.maximum_false_feasible_accepts_on_frozen_adversarial_suite != 0
            || feasibility.if_no_candidates_accepted.trim().is_empty()
        {
            return Err(RegistryError::InvalidFeasibilityGate);
        }
        Ok(())
    }
}

/// Parse the committed v1 benchmark matrix. The file is included at compile time.
pub fn load_embedded_registry() -> Result<BenchmarkRegistry, serde_json::Error> {
    serde_json::from_str(include_str!("../benchmarks/benchmark-matrix-v1.json"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn embedded_registry_parses_and_passes_integrity_checks() {
        let registry = load_embedded_registry().expect("checked-in JSON should parse");
        registry
            .validate()
            .expect("registry should be internally consistent");
        assert_eq!(registry.benchmark_lanes.len(), 10);
        assert!(
            registry
                .benchmark_lanes
                .iter()
                .any(|lane| lane.id == "experimental-reality-gap")
        );
        assert!(
            registry
                .benchmark_lanes
                .iter()
                .any(|lane| lane.id == "symtropy-workcell")
        );
        let nanoparticle_lane = registry
            .benchmark_lanes
            .iter()
            .find(|lane| lane.id == "supported-nanoparticle-search")
            .expect("supported nanoparticle lane should be registered");
        assert!(
            nanoparticle_lane
                .metrics
                .iter()
                .any(|metric| metric.contains("p=0.94")),
            "ranking persistence must be preregistered"
        );
        assert!(
            nanoparticle_lane
                .metrics
                .iter()
                .any(|metric| metric.contains("restricted to shared canonical candidate IDs")),
            "ranking quality must be measured on matched canonical structures"
        );
        assert!(
            nanoparticle_lane
                .metrics
                .iter()
                .any(|metric| metric.starts_with("candidate-set Jaccard")),
            "discovery-set agreement must be reported separately"
        );
        let reality_gap = registry
            .benchmark_lanes
            .iter()
            .find(|lane| lane.id == "experimental-reality-gap")
            .expect("experimental reality-gap lane should be registered");
        assert!(
            reality_gap
                .metrics
                .iter()
                .any(|metric| metric.contains("risk-coverage") && metric.contains("AURC")),
            "selective-risk evaluation should be part of the reality-gap lane"
        );
        let active_learning = registry
            .benchmark_lanes
            .iter()
            .find(|lane| lane.id == "active-learning-policy")
            .expect("active-learning lane should be registered");
        assert!(
            active_learning
                .gates
                .iter()
                .any(|gate| gate.contains("shuffled-score negative controls")),
            "active-learning evaluation should include uncertainty negative controls"
        );
    }

    #[test]
    fn registry_requires_nontrivial_acceleration_and_zero_false_feasible_gate() {
        let mut registry = load_embedded_registry().expect("checked-in JSON should parse");
        registry
            .performance_claims
            .accelerations
            .stretch_target_median_af = 1.0;
        assert_eq!(
            registry.validate(),
            Err(RegistryError::InvalidAccelerationGate)
        );

        let mut registry = load_embedded_registry().expect("checked-in JSON should parse");
        registry
            .performance_claims
            .process_feasibility
            .maximum_false_feasible_accepts_on_frozen_adversarial_suite = 1;
        assert_eq!(
            registry.validate(),
            Err(RegistryError::InvalidFeasibilityGate)
        );
    }

    #[test]
    fn registry_rejects_duplicate_lane_and_policy_identifiers() {
        let mut registry = load_embedded_registry().expect("checked-in JSON should parse");
        registry
            .benchmark_lanes
            .push(registry.benchmark_lanes[0].clone());
        assert_eq!(registry.validate(), Err(RegistryError::DuplicateLaneId));

        let mut registry = load_embedded_registry().expect("checked-in JSON should parse");
        registry
            .policy_baselines
            .push(registry.policy_baselines[0].clone());
        assert_eq!(registry.validate(), Err(RegistryError::DuplicatePolicyId));
    }
    #[test]
    fn workcell_lane_requires_the_frozen_fault_corpus_parity_gate() {
        let registry = load_embedded_registry().expect("checked-in JSON should parse");
        let workcell = registry
            .benchmark_lanes
            .iter()
            .find(|lane| lane.id == "symtropy-workcell")
            .expect("workcell lane should be registered");

        assert!(
            workcell.metrics.iter().any(|metric| {
                metric.contains("decision/reason-code parity")
                    && metric.contains("atomfab-workcell-fault-corpus-v1")
            }),
            "workcell lane must measure cross-repository fault-corpus parity"
        );
        assert!(
            workcell.gates.iter().any(|gate| {
                gate.contains("all 15 corpus cases")
                    && gate.contains("zero false-feasible accepts")
                    && gate.contains("accepted in-domain controls")
            }),
            "workcell lane must include fail-closed corpus execution and valid controls"
        );
    }

    #[test]
    fn discovery_and_active_learning_lanes_require_pareto_reporting() {
        let registry = load_embedded_registry().expect("checked-in JSON should parse");
        for lane_id in ["supported-nanoparticle-search", "active-learning-policy"] {
            let lane = registry
                .benchmark_lanes
                .iter()
                .find(|lane| lane.id == lane_id)
                .unwrap_or_else(|| panic!("missing benchmark lane {lane_id}"));
            assert!(
                lane.metrics.iter().any(|metric| {
                    metric.contains("Pareto frontier") && metric.contains("total cost")
                }),
                "{lane_id} must report a multi-objective discovery frontier"
            );
            assert!(
                lane.gates
                    .iter()
                    .any(|gate| { gate.contains("nondominated") && gate.contains("cost") }),
                "{lane_id} must explicitly gate Pareto reporting"
            );
        }
    }

    #[test]
    fn evidence_routing_lane_requires_calibrated_multifidelity_comparisons() {
        let registry = load_embedded_registry().expect("checked-in JSON should parse");
        let lane = registry
            .benchmark_lanes
            .iter()
            .find(|lane| lane.id == "multi-fidelity-evidence-routing")
            .expect("multi-fidelity evidence-routing lane should be registered");
        assert!(
            lane.baselines.iter().any(|baseline| {
                baseline.contains("Fixed cheapest-first")
            }),
            "router benchmark must include a conventional fixed-fidelity baseline"
        );
        assert!(
            lane.metrics.iter().any(|metric| {
                metric.contains("decision-loss reduction estimate calibration")
            }),
            "router benchmark must score calibration of its value estimates"
        );
        assert!(
            lane.gates.iter().any(|gate| {
                gate.contains("same request, evidence domain")
                    && gate.contains("budget")
            }),
            "router benchmark must prevent unlike-domain or unlike-budget comparisons"
        );
        assert!(
            lane.gates.iter().any(|gate| {
                gate.contains("shuffled value estimates")
                    && gate.contains("inflated uncalibrated value")
            }),
            "router benchmark must resist value-estimate gaming"
        );
    }

}
