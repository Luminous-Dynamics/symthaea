// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Separate MACE governance sidecars and matched-comparison gates.
//! Official benchmark metric payloads are bound by digest, never rewritten.
//! BLAKE3 digests are integrity commitments, not signatures or attestations.

use serde::{Deserialize, Serialize};

pub const SIDECAR_SCHEMA_V1: &str = "mace_benchmark_governance_sidecar_v1";
pub const COMPARISON_SCHEMA_V1: &str = "mace_benchmark_comparison_receipt_v1";

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct CapsuleBindingV1 {
    pub run_id: String,
    pub capsule_sha256: String,
    pub benchmark_id: String,
    pub track_id: String,
    pub release: String,
    pub dataset_split_sha256: String,
    pub official_metrics_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InfluenceEvidenceLevelV1 {
    Diagnostic,
    Instrumented,
    IndependentlyVerified,
    CausallyQualified,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct EvidenceFlowCountsV1 {
    pub retrieved: u64,
    pub admissible: u64,
    pub exposed: u64,
    pub cited: u64,
    pub influence_assessed: u64,
    pub influential_positive: u64,
    pub influential_negative: u64,
    pub influence_indeterminate: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct OpportunityDenominatorsV1 {
    pub evidence_use_opportunities: u64,
    pub decision_opportunities: u64,
    pub citation_opportunities: u64,
    pub influence_assessment_opportunities: u64,
    pub lifecycle_transition_opportunities: u64,
    pub authorization_opportunities: u64,
    pub erasure_verification_opportunities: u64,
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GovernanceViolationCountsV1 {
    pub stale_evidence_exposed: u64,
    pub expired_evidence_exposed: u64,
    pub revoked_evidence_exposed: u64,
    pub superseded_evidence_exposed: u64,
    pub wrong_scope_exposures: u64,
    pub unsupported_provenance_claims: u64,
    pub unauthorized_actions: u64,
    pub cross_scope_disclosures: u64,
    pub prohibited_proactive_disclosures: u64,
    pub erasure_resurrections: u64,
    pub privacy_violations: u64,
}

impl GovernanceViolationCountsV1 {
    pub fn total_non_compensable(&self) -> u64 {
        [
            self.stale_evidence_exposed, self.expired_evidence_exposed,
            self.revoked_evidence_exposed, self.superseded_evidence_exposed,
            self.wrong_scope_exposures, self.unsupported_provenance_claims,
            self.unauthorized_actions, self.cross_scope_disclosures,
            self.prohibited_proactive_disclosures, self.erasure_resurrections,
            self.privacy_violations,
        ].into_iter().fold(0_u64, u64::saturating_add)
    }
}

#[derive(Clone, Debug, Default, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GovernanceCostV1 {
    pub verifier_calls: u64,
    pub reobservations: u64,
    pub lifecycle_checks: u64,
    pub lineage_checks: u64,
    pub evidence_bytes_inspected: u64,
    pub governance_latency_p50_ns: Option<u64>,
    pub governance_latency_p95_ns: Option<u64>,
    pub governance_cost_microunits: Option<u64>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct GovernanceVerificationV1 {
    pub evidence_use_receipt_set_sha256: String,
    pub exposure_link_set_sha256: String,
    pub evaluator_output_sha256: Option<String>,
    pub raw_trace_receipt_sha256: Option<String>,
    pub split_audit_receipt_sha256: Option<String>,
    pub influence_report_sha256: Option<String>,
    pub randomization_receipt_sha256: Option<String>,
    pub independent_verifier_id: Option<String>,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BenchmarkGovernanceSidecarV1 {
    pub schema_version: String,
    pub run_id: String,
    pub capsule_sha256: String,
    pub benchmark_id: String,
    pub track_id: String,
    pub release: String,
    /// Digest of the exact official metrics array in the immutable V1 capsule.
    pub official_metrics_sha256: String,
    pub governance_contract_sha256: String,
    pub evidence_flow: EvidenceFlowCountsV1,
    pub denominators: OpportunityDenominatorsV1,
    pub violations: GovernanceViolationCountsV1,
    pub costs: GovernanceCostV1,
    pub influence_evidence_level: InfluenceEvidenceLevelV1,
    pub verification: GovernanceVerificationV1,
    pub hard_gates_passed: bool,
    pub explanatory_note: String,
    pub sidecar_sha256: String,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SidecarRejectV1 {
    InvalidIdentity, InvalidDigest, CapsuleBindingMismatch, OfficialMetricBindingMismatch,
    InvalidEvidenceFlowOrder, InvalidDenominator, InvalidInfluenceAccounting,
    InvalidCostQuantileOrder, HardGateContradiction, MissingInstrumentedEvidence,
    MissingIndependentInfluenceCustody, IntegrityMismatch, SerializationFailure,
}

fn valid_sha256(value: &str) -> bool {
    value.len() == 64 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn valid_git_sha(value: &str) -> bool {
    value.len() == 40 && value.bytes().all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}
fn nonempty(value: &str) -> bool { !value.trim().is_empty() && value.len() <= 512 }
fn digest_serialized<T: Serialize>(value: &T) -> Result<String, SidecarRejectV1> {
    serde_json::to_vec(value).map(|bytes| blake3::hash(&bytes).to_hex().to_string())
        .map_err(|_| SidecarRejectV1::SerializationFailure)
}
fn sidecar_digest(value: &BenchmarkGovernanceSidecarV1) -> Result<String, SidecarRejectV1> {
    let mut unsigned = value.clone();
    unsigned.sidecar_sha256.clear();
    digest_serialized(&unsigned)
}

impl BenchmarkGovernanceSidecarV1 {
    pub fn verify_integrity(&self) -> bool {
        sidecar_digest(self).is_ok_and(|hash| hash == self.sidecar_sha256)
    }

    pub fn validate_against(&self, capsule: &CapsuleBindingV1) -> Result<(), SidecarRejectV1> {
        if self.schema_version != SIDECAR_SCHEMA_V1 || !nonempty(&self.run_id)
            || !nonempty(&self.benchmark_id) || !nonempty(&self.track_id)
            || !nonempty(&self.release) || !nonempty(&self.explanatory_note) {
            return Err(SidecarRejectV1::InvalidIdentity);
        }
        for value in [
            self.capsule_sha256.as_str(), self.official_metrics_sha256.as_str(),
            self.governance_contract_sha256.as_str(),
            self.verification.evidence_use_receipt_set_sha256.as_str(),
            self.verification.exposure_link_set_sha256.as_str(),
        ] {
            if !valid_sha256(value) { return Err(SidecarRejectV1::InvalidDigest); }
        }
        for value in [
            self.verification.evaluator_output_sha256.as_deref(),
            self.verification.raw_trace_receipt_sha256.as_deref(),
            self.verification.split_audit_receipt_sha256.as_deref(),
            self.verification.influence_report_sha256.as_deref(),
            self.verification.randomization_receipt_sha256.as_deref(),
        ].into_iter().flatten() {
            if !valid_sha256(value) { return Err(SidecarRejectV1::InvalidDigest); }
        }
        if self.run_id != capsule.run_id || self.capsule_sha256 != capsule.capsule_sha256
            || self.benchmark_id != capsule.benchmark_id || self.track_id != capsule.track_id
            || self.release != capsule.release {
            return Err(SidecarRejectV1::CapsuleBindingMismatch);
        }
        if self.official_metrics_sha256 != capsule.official_metrics_sha256 {
            return Err(SidecarRejectV1::OfficialMetricBindingMismatch);
        }
        let flow = &self.evidence_flow;
        if flow.admissible > flow.retrieved || flow.exposed > flow.admissible || flow.cited > flow.exposed {
            return Err(SidecarRejectV1::InvalidEvidenceFlowOrder);
        }
        let denom = &self.denominators;
        let violations = &self.violations;
        if flow.exposed > denom.evidence_use_opportunities
            || flow.cited > denom.citation_opportunities
            || flow.influence_assessed > denom.influence_assessment_opportunities
            || violations.stale_evidence_exposed > denom.evidence_use_opportunities
            || violations.expired_evidence_exposed > denom.evidence_use_opportunities
            || violations.revoked_evidence_exposed > denom.evidence_use_opportunities
            || violations.superseded_evidence_exposed > denom.evidence_use_opportunities
            || violations.wrong_scope_exposures > denom.evidence_use_opportunities
            || violations.unauthorized_actions > denom.authorization_opportunities
            || violations.cross_scope_disclosures > denom.authorization_opportunities
            || violations.erasure_resurrections > denom.erasure_verification_opportunities {
            return Err(SidecarRejectV1::InvalidDenominator);
        }
        let classified = flow.influential_positive.saturating_add(flow.influential_negative)
            .saturating_add(flow.influence_indeterminate);
        if flow.influence_assessed > flow.exposed || classified != flow.influence_assessed {
            return Err(SidecarRejectV1::InvalidInfluenceAccounting);
        }
        if let (Some(p50), Some(p95)) = (self.costs.governance_latency_p50_ns, self.costs.governance_latency_p95_ns) {
            if p95 < p50 { return Err(SidecarRejectV1::InvalidCostQuantileOrder); }
        }
        if self.hard_gates_passed && violations.total_non_compensable() != 0 {
            return Err(SidecarRejectV1::HardGateContradiction);
        }
        if self.influence_evidence_level != InfluenceEvidenceLevelV1::Diagnostic
            && (self.verification.evaluator_output_sha256.is_none() || self.verification.raw_trace_receipt_sha256.is_none()) {
            return Err(SidecarRejectV1::MissingInstrumentedEvidence);
        }
        if self.influence_evidence_level == InfluenceEvidenceLevelV1::CausallyQualified
            && (self.verification.independent_verifier_id.as_deref().is_none_or(str::is_empty)
                || self.verification.influence_report_sha256.is_none()
                || self.verification.randomization_receipt_sha256.is_none()
                || self.verification.split_audit_receipt_sha256.is_none()) {
            return Err(SidecarRejectV1::MissingIndependentInfluenceCustody);
        }
        if !self.verify_integrity() { return Err(SidecarRejectV1::IntegrityMismatch); }
        Ok(())
    }
}

pub fn seal_sidecar_v1(
    mut sidecar: BenchmarkGovernanceSidecarV1,
    capsule: &CapsuleBindingV1,
) -> Result<BenchmarkGovernanceSidecarV1, SidecarRejectV1> {
    sidecar.sidecar_sha256.clear();
    sidecar.sidecar_sha256 = sidecar_digest(&sidecar)?;
    sidecar.validate_against(capsule)?;
    Ok(sidecar)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonIdentityV1 {
    pub run_id: String,
    pub benchmark_id: String,
    pub track_id: String,
    pub release: String,
    pub dataset_split_sha256: String,
    pub grader_sha256: String,
    pub metric_contract_sha256: String,
    pub reader_model_sha256: String,
    pub prompt_sha256: String,
    pub tool_manifest_sha256: String,
    pub budget_profile_sha256: String,
    pub task_ids_sha256: String,
    pub cluster_unit: String,
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ComparisonGateV1 {
    pub schema_version: String,
    pub candidate_run_id: String,
    pub baseline_run_id: String,
    pub matched: bool,
    pub mismatches: Vec<String>,
    pub benchmark_id: String,
    pub track_id: String,
    pub comparison_scope: String,
    pub gate_sha256: String,
}

fn comparison_digest(gate: &ComparisonGateV1) -> Result<String, SidecarRejectV1> {
    let mut unsigned = gate.clone();
    unsigned.gate_sha256.clear();
    digest_serialized(&unsigned)
}

pub fn compare_runs_v1(
    candidate: &ComparisonIdentityV1,
    baseline: &ComparisonIdentityV1,
) -> Result<ComparisonGateV1, SidecarRejectV1> {
    let mut mismatches = Vec::new();
    for value in [
        candidate.dataset_split_sha256.as_str(), candidate.grader_sha256.as_str(),
        candidate.metric_contract_sha256.as_str(), candidate.reader_model_sha256.as_str(),
        candidate.prompt_sha256.as_str(), candidate.tool_manifest_sha256.as_str(),
        candidate.budget_profile_sha256.as_str(), candidate.task_ids_sha256.as_str(),
    ] {
        if !valid_sha256(value) { return Err(SidecarRejectV1::InvalidDigest); }
    }
    if !nonempty(&candidate.run_id) || !nonempty(&baseline.run_id)
        || !nonempty(&candidate.cluster_unit) || !nonempty(&baseline.cluster_unit) {
        return Err(SidecarRejectV1::InvalidIdentity);
    }
    if candidate.benchmark_id != baseline.benchmark_id { mismatches.push("benchmark_id".into()); }
    if candidate.track_id != baseline.track_id { mismatches.push("track_id".into()); }
    if candidate.release != baseline.release { mismatches.push("release".into()); }
    if candidate.dataset_split_sha256 != baseline.dataset_split_sha256 { mismatches.push("dataset_split_sha256".into()); }
    if candidate.grader_sha256 != baseline.grader_sha256 { mismatches.push("grader_sha256".into()); }
    if candidate.metric_contract_sha256 != baseline.metric_contract_sha256 { mismatches.push("metric_contract_sha256".into()); }
    if candidate.reader_model_sha256 != baseline.reader_model_sha256 { mismatches.push("reader_model_sha256".into()); }
    if candidate.prompt_sha256 != baseline.prompt_sha256 { mismatches.push("prompt_sha256".into()); }
    if candidate.tool_manifest_sha256 != baseline.tool_manifest_sha256 { mismatches.push("tool_manifest_sha256".into()); }
    if candidate.budget_profile_sha256 != baseline.budget_profile_sha256 { mismatches.push("budget_profile_sha256".into()); }
    if candidate.task_ids_sha256 != baseline.task_ids_sha256 { mismatches.push("task_ids_sha256".into()); }
    if candidate.cluster_unit != baseline.cluster_unit { mismatches.push("cluster_unit".into()); }
    let mut gate = ComparisonGateV1 {
        schema_version: COMPARISON_SCHEMA_V1.into(),
        candidate_run_id: candidate.run_id.clone(),
        baseline_run_id: baseline.run_id.clone(),
        matched: mismatches.is_empty(),
        mismatches,
        benchmark_id: candidate.benchmark_id.clone(),
        track_id: candidate.track_id.clone(),
        comparison_scope: if candidate.benchmark_id == baseline.benchmark_id && candidate.track_id == baseline.track_id {
            "same_benchmark_track".into()
        } else {
            "cross_benchmark_or_track_not_comparable".into()
        },
        gate_sha256: String::new(),
    };
    gate.gate_sha256 = comparison_digest(&gate)?;
    Ok(gate)
}

impl ComparisonGateV1 {
    pub fn verify_integrity(&self) -> bool {
        comparison_digest(self).is_ok_and(|hash| hash == self.gate_sha256)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn digest(value: &str) -> String { blake3::hash(value.as_bytes()).to_hex().to_string() }

    fn capsule() -> CapsuleBindingV1 {
        CapsuleBindingV1 {
            run_id: "run-001".into(), capsule_sha256: digest("capsule"),
            benchmark_id: "longmemeval_v2".into(), track_id: "small".into(), release: "v2".into(),
            dataset_split_sha256: digest("split"), official_metrics_sha256: digest("official-metrics"),
        }
    }

    fn sidecar() -> BenchmarkGovernanceSidecarV1 {
        BenchmarkGovernanceSidecarV1 {
            schema_version: SIDECAR_SCHEMA_V1.into(), run_id: "run-001".into(),
            capsule_sha256: digest("capsule"), benchmark_id: "longmemeval_v2".into(),
            track_id: "small".into(), release: "v2".into(),
            official_metrics_sha256: digest("official-metrics"), governance_contract_sha256: digest("contract"),
            evidence_flow: EvidenceFlowCountsV1 {
                retrieved: 120, admissible: 100, exposed: 80, cited: 52,
                influence_assessed: 40, influential_positive: 14,
                influential_negative: 8, influence_indeterminate: 18,
            },
            denominators: OpportunityDenominatorsV1 {
                evidence_use_opportunities: 100, decision_opportunities: 80,
                citation_opportunities: 80, influence_assessment_opportunities: 60,
                lifecycle_transition_opportunities: 30, authorization_opportunities: 80,
                erasure_verification_opportunities: 20,
            },
            violations: GovernanceViolationCountsV1::default(),
            costs: GovernanceCostV1 {
                verifier_calls: 80, reobservations: 5, lifecycle_checks: 120,
                lineage_checks: 80, evidence_bytes_inspected: 64000,
                governance_latency_p50_ns: Some(1_000_000),
                governance_latency_p95_ns: Some(5_000_000),
                governance_cost_microunits: Some(125),
            },
            influence_evidence_level: InfluenceEvidenceLevelV1::Diagnostic,
            verification: GovernanceVerificationV1 {
                evidence_use_receipt_set_sha256: digest("use-receipts"),
                exposure_link_set_sha256: digest("exposure-links"),
                evaluator_output_sha256: None, raw_trace_receipt_sha256: None,
                split_audit_receipt_sha256: None, influence_report_sha256: None,
                randomization_receipt_sha256: None, independent_verifier_id: None,
            },
            hard_gates_passed: true,
            explanatory_note: "Synthetic fixture only; no external benchmark claim.".into(),
            sidecar_sha256: String::new(),
        }
    }

    fn comparison_identity(run_id: &str) -> ComparisonIdentityV1 {
        ComparisonIdentityV1 {
            run_id: run_id.into(), benchmark_id: "longmemeval_v2".into(),
            track_id: "small".into(), release: "v2".into(), dataset_split_sha256: digest("split"),
            grader_sha256: digest("grader"), metric_contract_sha256: digest("metric-contract"),
            reader_model_sha256: digest("reader"), prompt_sha256: digest("prompt"),
            tool_manifest_sha256: digest("tools"), budget_profile_sha256: digest("budget"),
            task_ids_sha256: digest("tasks"), cluster_unit: "conversation_family".into(),
        }
    }

    #[test]
    fn sidecar_binds_capsule_and_official_metrics_without_rewriting_them() {
        let capsule = capsule();
        let value = seal_sidecar_v1(sidecar(), &capsule).unwrap();
        assert_eq!(value.official_metrics_sha256, capsule.official_metrics_sha256);
        assert!(value.verify_integrity());
        value.validate_against(&capsule).unwrap();
    }

    #[test]
    fn changed_official_metric_digest_is_rejected() {
        let capsule = capsule();
        let mut value = sidecar();
        value.official_metrics_sha256 = digest("rewritten-score");
        assert_eq!(value.validate_against(&capsule), Err(SidecarRejectV1::OfficialMetricBindingMismatch));
    }

    #[test]
    fn evidence_flow_must_be_monotone() {
        let capsule = capsule();
        let mut value = sidecar();
        value.evidence_flow.exposed = value.evidence_flow.admissible + 1;
        assert_eq!(value.validate_against(&capsule), Err(SidecarRejectV1::InvalidEvidenceFlowOrder));
    }

    #[test]
    fn influence_classifications_must_close_exactly() {
        let capsule = capsule();
        let mut value = sidecar();
        value.evidence_flow.influence_indeterminate -= 1;
        assert_eq!(value.validate_against(&capsule), Err(SidecarRejectV1::InvalidInfluenceAccounting));
    }

    #[test]
    fn denominator_overflow_and_nonmonotone_cost_quantiles_fail_closed() {
        let capsule = capsule();
        let mut value = sidecar();
        value.violations.erasure_resurrections = value.denominators.erasure_verification_opportunities + 1;
        assert_eq!(value.validate_against(&capsule), Err(SidecarRejectV1::InvalidDenominator));
        let mut bad_cost = sidecar();
        bad_cost.costs.governance_latency_p95_ns = Some(100);
        assert_eq!(bad_cost.validate_against(&capsule), Err(SidecarRejectV1::InvalidCostQuantileOrder));
    }

    #[test]
    fn critical_violations_cannot_coexist_with_passed_hard_gates() {
        let capsule = capsule();
        let mut value = sidecar();
        value.violations.stale_evidence_exposed = 1;
        assert_eq!(value.validate_against(&capsule), Err(SidecarRejectV1::HardGateContradiction));
    }

    #[test]
    fn causal_influence_claim_requires_independent_evaluator_and_randomization_receipts() {
        let capsule = capsule();
        let mut value = sidecar();
        value.influence_evidence_level = InfluenceEvidenceLevelV1::CausallyQualified;
        assert_eq!(value.validate_against(&capsule), Err(SidecarRejectV1::MissingInstrumentedEvidence));
        value.verification.evaluator_output_sha256 = Some(digest("evaluator"));
        value.verification.raw_trace_receipt_sha256 = Some(digest("trace"));
        assert_eq!(value.validate_against(&capsule), Err(SidecarRejectV1::MissingIndependentInfluenceCustody));
    }

    #[test]
    fn comparison_accepts_only_exactly_matched_benchmark_tracks() {
        let candidate = comparison_identity("candidate-run");
        let baseline = comparison_identity("baseline-run");
        let gate = compare_runs_v1(&candidate, &baseline).unwrap();
        assert!(gate.matched);
        assert!(gate.verify_integrity());
        let mut other_track = baseline.clone();
        other_track.track_id = "medium".into();
        let gate = compare_runs_v1(&candidate, &other_track).unwrap();
        assert!(!gate.matched);
        assert!(gate.mismatches.contains(&"track_id".to_owned()));
        assert_eq!(gate.comparison_scope, "same_benchmark_track");
    }

    #[test]
    fn cross_benchmark_results_are_never_marked_comparable() {
        let candidate = comparison_identity("candidate-run");
        let mut other = comparison_identity("baseline-run");
        other.benchmark_id = "agent_memory_leaderboard".into();
        other.track_id = "textual".into();
        let gate = compare_runs_v1(&candidate, &other).unwrap();
        assert!(!gate.matched);
        assert_eq!(gate.comparison_scope, "cross_benchmark_or_track_not_comparable");
    }

    #[test]
    fn sidecar_integrity_detects_mutated_counts() {
        let capsule = capsule();
        let mut value = seal_sidecar_v1(sidecar(), &capsule).unwrap();
        value.evidence_flow.exposed += 1;
        assert!(!value.verify_integrity());
        assert_eq!(value.validate_against(&capsule), Err(SidecarRejectV1::IntegrityMismatch));
    }
}
