//! Raw-byte-verified REA Evidence admission before planner provenance projection.
//!
//! REA Evidence semantic identity and planner origin classification are distinct.
//! This adapter verifies exact Evidence JSON bytes and semantic IDs, then carries
//! caller-declared lineage into the existing planner API without authenticating it.

use serde::Serialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};
use symthaea_rea_compat::{EvidenceError, canonical_jcs_bytes, sha256_hex, verify_evidence_json};
use symthaea_rea_investigator::{
    Authorization, EvidenceOrigin, EvidenceProvenance, ExperimentProposal, Hypothesis,
    InvestigationPlan, PlannerConfig, PlannerError, TargetImpact, plan_next_experiment,
};

const LINEAGE_SCHEMA: &str = "luminous.rea.planner-lineage-declaration.v0";
const GOLDEN_EVIDENCE: &[u8] = include_bytes!(
    "../../../crates/domains/symthaea-rea-compat/tests/fixtures/rea-evidence-v1.json"
);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum DeclaredEvidenceOrigin {
    IndependentObservation,
    IndependentDerivation,
    ModelGenerated,
    Synthetic,
    ExternalReference,
}

impl DeclaredEvidenceOrigin {
    /// Preserve the caller's declaration in the report, but do not let an
    /// unverified declaration qualify as a planner grounding root. The existing
    /// planner has no CallerDeclaredUnverified variant, so v0 maps both
    /// caller-declared observation classes to its non-grounding ExternalReference
    /// category. A future verified attestation policy can add a separate,
    /// explicit qualification path.
    fn to_unverified_planner_origin(self) -> EvidenceOrigin {
        match self {
            Self::IndependentObservation | Self::IndependentDerivation => {
                EvidenceOrigin::ExternalReference
            }
            Self::ModelGenerated => EvidenceOrigin::ModelGenerated,
            Self::Synthetic => EvidenceOrigin::Synthetic,
            Self::ExternalReference => EvidenceOrigin::ExternalReference,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum OriginQualification {
    CallerDeclaredUnverified,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum RawBindingKind {
    IndividuallyVerifiedRawJson,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct LineageDeclaration {
    /// Must match the Evidence ID computed from the admitted raw bytes.
    pub evidence_id: String,
    /// Caller-supplied classification; never promoted to authenticated origin.
    pub declared_origin: DeclaredEvidenceOrigin,
    /// Typed derivation ancestry, separate from Evidence v1 semantic identity.
    pub parent_evidence_ids: Vec<String>,
    /// Caller-declared family key for downstream grouping; not proof of independence.
    pub source_family_id: String,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmittedRecordSummary {
    pub evidence_id: String,
    pub semantic_sha256: String,
    pub raw_file_sha256: String,
    pub raw_binding: RawBindingKind,
    pub declared_origin: DeclaredEvidenceOrigin,
    pub origin_qualification: OriginQualification,
    pub parent_evidence_ids: Vec<String>,
    pub source_family_id: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
pub struct AdmissionClaimCeiling {
    pub raw_bytes_admitted: bool,
    pub semantic_evidence_identity_verified: bool,
    pub producer_authenticated: bool,
    pub caller_origin_authenticated: bool,
    pub source_independence_authenticated: bool,
    pub observation_truth_proven: bool,
    pub issuer_history_completeness_proven: bool,
}

impl AdmissionClaimCeiling {
    fn verified_raw_identity() -> Self {
        Self {
            raw_bytes_admitted: true,
            semantic_evidence_identity_verified: true,
            producer_authenticated: false,
            caller_origin_authenticated: false,
            source_independence_authenticated: false,
            observation_truth_proven: false,
            issuer_history_completeness_proven: false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct AdmissionReport {
    pub schema: String,
    pub lineage_manifest_sha256: String,
    pub records: Vec<AdmittedRecordSummary>,
    pub claim_ceiling: AdmissionClaimCeiling,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AdmissionErrorKind {
    RawEvidenceRejected,
    DuplicateEvidenceId,
    MissingLineageDeclaration,
    UnknownLineageEvidence,
    DuplicateLineageDeclaration,
    DuplicateParent,
    SelfParent,
    DanglingParent,
    ParentCycle,
    InvalidOriginParentShape,
    EmptySourceFamily,
    MissingHypothesisEvidence,
    CommonModeIndependentRoot,
    PlannerRejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AdmissionError {
    pub kind: AdmissionErrorKind,
    pub message: String,
}

impl AdmissionError {
    fn new(kind: AdmissionErrorKind, message: impl Into<String>) -> Self {
        Self { kind, message: message.into() }
    }
}

impl std::fmt::Display for AdmissionError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{:?}: {}", self.kind, self.message)
    }
}

impl std::error::Error for AdmissionError {}

impl From<EvidenceError> for AdmissionError {
    fn from(error: EvidenceError) -> Self {
        Self::new(
            AdmissionErrorKind::RawEvidenceRejected,
            format!("REA raw Evidence admission failed: {error}"),
        )
    }
}

impl From<PlannerError> for AdmissionError {
    fn from(error: PlannerError) -> Self {
        Self::new(
            AdmissionErrorKind::PlannerRejected,
            format!("existing REA planner rejected its inputs: {error}"),
        )
    }
}

#[derive(Debug)]
struct AdmittedRecord {
    evidence_id: String,
    semantic_sha256: String,
    raw_file_sha256: String,
    declared_origin: DeclaredEvidenceOrigin,
    parent_evidence_ids: Vec<String>,
    source_family_id: String,
}

#[derive(Debug)]
pub struct VerifiedEvidenceCatalog {
    records: BTreeMap<String, AdmittedRecord>,
    lineage_manifest_sha256: String,
}

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedPlannerRun {
    pub plan: InvestigationPlan,
    pub admission: AdmissionReport,
    /// Canonical digest of the exact planner inputs and admitted lineage/raw digests.
    /// This binds a request; it is not a planner-run receipt or signature.
    pub planner_request_sha256: String,
}

impl VerifiedEvidenceCatalog {
    /// Admit individual Evidence JSON records, then bind their independently
    /// declared lineage sidecar to the IDs calculated from those exact bytes.
    ///
    /// The API intentionally accepts bytes, not caller-provided Evidence IDs.
    pub fn admit_raw_records(
        raw_records: Vec<Vec<u8>>,
        lineage: Vec<LineageDeclaration>,
    ) -> Result<Self, AdmissionError> {
        if raw_records.is_empty() {
            return Err(AdmissionError::new(
                AdmissionErrorKind::RawEvidenceRejected,
                "at least one raw Evidence record is required",
            ));
        }

        let mut verified = BTreeMap::new();
        for raw in raw_records {
            let item = verify_evidence_json(&raw)?;
            let raw_file_sha256 = item.raw_file_sha256.clone().ok_or_else(|| {
                AdmissionError::new(
                    AdmissionErrorKind::RawEvidenceRejected,
                    "REA verifier did not return an exact raw-file digest",
                )
            })?;
            let id = item.evidence_id.clone();
            if verified.insert(
                id.clone(),
                (item.semantic_sha256, raw_file_sha256),
            ).is_some() {
                return Err(AdmissionError::new(
                    AdmissionErrorKind::DuplicateEvidenceId,
                    format!("duplicate verified Evidence ID {id}"),
                ));
            }
        }

        Self::bind_lineage(verified, lineage)
    }

    fn bind_lineage(
        verified: BTreeMap<String, (String, String)>,
        lineage: Vec<LineageDeclaration>,
    ) -> Result<Self, AdmissionError> {
        let mut declarations = BTreeMap::<String, LineageDeclaration>::new();
        for declaration in lineage {
            let id = declaration.evidence_id.clone();
            if !verified.contains_key(&id) {
                return Err(AdmissionError::new(
                    AdmissionErrorKind::UnknownLineageEvidence,
                    format!("lineage declaration refers to unadmitted Evidence ID {id}"),
                ));
            }
            if declarations.insert(id.clone(), declaration).is_some() {
                return Err(AdmissionError::new(
                    AdmissionErrorKind::DuplicateLineageDeclaration,
                    format!("duplicate lineage declaration for {id}"),
                ));
            }
        }
        if declarations.len() != verified.len() {
            let missing: Vec<&str> = verified
                .keys()
                .filter(|id| !declarations.contains_key(*id))
                .map(String::as_str)
                .collect();
            return Err(AdmissionError::new(
                AdmissionErrorKind::MissingLineageDeclaration,
                format!("missing lineage declarations for verified Evidence: {missing:?}"),
            ));
        }

        let mut records = BTreeMap::new();
        for (id, (semantic_sha256, raw_file_sha256)) in verified {
            let declaration = declarations.remove(&id).expect("cardinality checked");
            if declaration.source_family_id.trim().is_empty() {
                return Err(AdmissionError::new(
                    AdmissionErrorKind::EmptySourceFamily,
                    format!("source_family_id must be non-empty for {id}"),
                ));
            }

            let mut parent_set = HashSet::new();
            for parent in &declaration.parent_evidence_ids {
                if parent == &id {
                    return Err(AdmissionError::new(
                        AdmissionErrorKind::SelfParent,
                        format!("Evidence {id} cannot be its own derivation parent"),
                    ));
                }
                if !parent_set.insert(parent.as_str()) {
                    return Err(AdmissionError::new(
                        AdmissionErrorKind::DuplicateParent,
                        format!("Evidence {id} repeats parent {parent}"),
                    ));
                }
            }

            match declaration.declared_origin {
                DeclaredEvidenceOrigin::IndependentObservation
                    if !declaration.parent_evidence_ids.is_empty() =>
                {
                    return Err(AdmissionError::new(
                        AdmissionErrorKind::InvalidOriginParentShape,
                        format!("independent observation {id} cannot declare derivation parents"),
                    ));
                }
                DeclaredEvidenceOrigin::IndependentDerivation
                    if declaration.parent_evidence_ids.is_empty() =>
                {
                    return Err(AdmissionError::new(
                        AdmissionErrorKind::InvalidOriginParentShape,
                        format!("independent derivation {id} requires at least one parent"),
                    ));
                }
                _ => {}
            }

            records.insert(
                id.clone(),
                AdmittedRecord {
                    evidence_id: id,
                    semantic_sha256,
                    raw_file_sha256,
                    declared_origin: declaration.declared_origin,
                    parent_evidence_ids: declaration.parent_evidence_ids,
                    source_family_id: declaration.source_family_id,
                },
            );
        }

        for record in records.values() {
            for parent in &record.parent_evidence_ids {
                if !records.contains_key(parent) {
                    return Err(AdmissionError::new(
                        AdmissionErrorKind::DanglingParent,
                        format!(
                            "Evidence {} declares parent {parent}, which is not in the admitted catalog",
                            record.evidence_id
                        ),
                    ));
                }
            }
        }

        validate_acyclic_parent_graph(&records)?;
        let lineage_manifest_sha256 = lineage_manifest_digest(&records)?;

        Ok(Self { records, lineage_manifest_sha256 })
    }

    /// Return a report that keeps byte/semantic verification separate from
    /// the caller-declared origin and source-family claims.
    pub fn report(&self) -> AdmissionReport {
        let records = self.records.values().map(|record| AdmittedRecordSummary {
            evidence_id: record.evidence_id.clone(),
            semantic_sha256: record.semantic_sha256.clone(),
            raw_file_sha256: record.raw_file_sha256.clone(),
            raw_binding: RawBindingKind::IndividuallyVerifiedRawJson,
            declared_origin: record.declared_origin,
            origin_qualification: OriginQualification::CallerDeclaredUnverified,
            parent_evidence_ids: record.parent_evidence_ids.clone(),
            source_family_id: record.source_family_id.clone(),
        }).collect();

        AdmissionReport {
            schema: "luminous.rea.verified-planner-admission-report.v0".to_owned(),
            lineage_manifest_sha256: self.lineage_manifest_sha256.clone(),
            records,
            claim_ceiling: AdmissionClaimCeiling::verified_raw_identity(),
        }
    }


    pub fn planner_request_digest(
        &self,
        hypotheses: &[Hypothesis],
        experiments: &[ExperimentProposal],
        config: PlannerConfig,
    ) -> Result<String, AdmissionError> {
        fn authorization(value: Authorization) -> &'static str {
            match value {
                Authorization::Unknown => "unknown",
                Authorization::Denied => "denied",
                Authorization::Authorized => "authorized",
            }
        }
        fn impact(value: TargetImpact) -> &'static str {
            match value {
                TargetImpact::ReadOnlyObservation => "read_only_observation",
                TargetImpact::TemporaryConfiguration => "temporary_configuration",
                TargetImpact::ServiceRestart => "service_restart",
                TargetImpact::DataMutation => "data_mutation",
                TargetImpact::PhysicalActuation => "physical_actuation",
            }
        }

        let payload = json!({
            "schema": "luminous.rea.verified-planner-request.v0",
            "lineage_manifest_sha256": self.lineage_manifest_sha256,
            "admitted_evidence": self.records.values().map(|record| json!({
                "evidence_id": record.evidence_id,
                "semantic_sha256": record.semantic_sha256,
                "raw_file_sha256": record.raw_file_sha256,
                "declared_origin": record.declared_origin,
                "parent_evidence_ids": record.parent_evidence_ids,
                "source_family_id": record.source_family_id,
            })).collect::<Vec<_>>(),
            "hypotheses": hypotheses.iter().map(|hypothesis| json!({
                "id": hypothesis.id,
                "prior_probability": hypothesis.prior_probability,
                "supporting_evidence_ids": hypothesis.supporting_evidence_ids,
                "contradicting_evidence_ids": hypothesis.contradicting_evidence_ids,
            })).collect::<Vec<_>>(),
            "experiments": experiments.iter().map(|experiment| json!({
                "id": experiment.id,
                "outcomes": experiment.outcomes,
                "likelihoods": experiment.likelihoods.iter().map(|row| json!({
                    "hypothesis_id": row.hypothesis_id,
                    "outcome_probabilities": row.outcome_probabilities,
                })).collect::<Vec<_>>(),
                "authorization": authorization(experiment.authorization),
                "reversible": experiment.reversible,
                "destructive": experiment.destructive,
                "target_impact": impact(experiment.target_impact),
                "estimated_time_seconds": experiment.estimated_time_seconds,
                "estimated_energy_joules": experiment.estimated_energy_joules,
                "estimated_risk": experiment.estimated_risk,
            })).collect::<Vec<_>>(),
            "planner_config": {
                "max_time_seconds": config.max_time_seconds,
                "max_energy_joules": config.max_energy_joules,
                "max_risk": config.max_risk,
                "cost_weights": {
                    "time": config.cost_weights.time,
                    "energy": config.cost_weights.energy,
                    "risk": config.cost_weights.risk,
                },
                "max_target_impact": impact(config.max_target_impact),
                "minimum_information_gain_bits": config.minimum_information_gain_bits,
            }
        });
        let canonical = canonical_jcs_bytes(&payload).map_err(|error| {
            AdmissionError::new(
                AdmissionErrorKind::RawEvidenceRejected,
                format!("planner request canonicalization failed: {error}"),
            )
        })?;
        Ok(sha256_hex(&canonical))
    }

    /// Project verified IDs into the existing planner type. Caller-declared
    /// observation/derivation labels are deliberately downgraded to
    /// ExternalReference until a separate attestation policy qualifies them.
    fn planner_provenance_projection(&self) -> Vec<EvidenceProvenance> {
        self.records.values().map(|record| EvidenceProvenance {
            evidence_id: record.evidence_id.clone(),
            origin: record.declared_origin.to_unverified_planner_origin(),
            parent_evidence_ids: record.parent_evidence_ids.clone(),
        }).collect()
    }


    fn validate_hypothesis_source_family_independence(
        &self,
        hypotheses: &[Hypothesis],
    ) -> Result<(), AdmissionError> {
        fn collect_roots(
            id: &str,
            records: &BTreeMap<String, AdmittedRecord>,
            family_roots: &mut BTreeMap<String, String>,
            visited: &mut BTreeSet<String>,
        ) -> Result<(), AdmissionError> {
            if !visited.insert(id.to_owned()) {
                return Ok(());
            }
            let record = records.get(id).ok_or_else(|| {
                AdmissionError::new(
                    AdmissionErrorKind::MissingHypothesisEvidence,
                    format!("support closure references unadmitted Evidence {id}"),
                )
            })?;
            match record.declared_origin {
                DeclaredEvidenceOrigin::IndependentObservation => {
                    if let Some(existing_root) = family_roots
                        .insert(record.source_family_id.clone(), record.evidence_id.clone())
                    {
                        if existing_root != record.evidence_id {
                            return Err(AdmissionError::new(
                                AdmissionErrorKind::CommonModeIndependentRoot,
                                format!(
                                    "hypothesis support includes distinct independent-root IDs {existing_root} and {} from the same caller-declared source family '{}'; independent corroboration cannot be inferred from tool/record count",
                                    record.evidence_id, record.source_family_id
                                ),
                            ));
                        }
                    }
                }
                DeclaredEvidenceOrigin::IndependentDerivation => {
                    for parent in &record.parent_evidence_ids {
                        collect_roots(parent, records, family_roots, visited)?;
                    }
                }
                DeclaredEvidenceOrigin::ModelGenerated
                | DeclaredEvidenceOrigin::Synthetic
                | DeclaredEvidenceOrigin::ExternalReference => {}
            }
            Ok(())
        }

        for hypothesis in hypotheses {
            let mut family_roots = BTreeMap::new();
            let mut visited = BTreeSet::new();
            for support_id in &hypothesis.supporting_evidence_ids {
                collect_roots(
                    support_id,
                    &self.records,
                    &mut family_roots,
                    &mut visited,
                )?;
            }
        }
        Ok(())
    }

    fn validate_hypothesis_references(
        &self,
        hypotheses: &[Hypothesis],
    ) -> Result<(), AdmissionError> {
        for hypothesis in hypotheses {
            for id in hypothesis
                .supporting_evidence_ids
                .iter()
                .chain(hypothesis.contradicting_evidence_ids.iter())
            {
                if !self.records.contains_key(id) {
                    return Err(AdmissionError::new(
                        AdmissionErrorKind::MissingHypothesisEvidence,
                        format!("hypothesis {} references Evidence {id} not admitted from raw bytes", hypothesis.id),
                    ));
                }
            }
        }
        self.validate_hypothesis_source_family_independence(hypotheses)?;
        Ok(())
    }
}

/// Run the existing point-EIG planner using only Evidence IDs from an
/// exact-raw-byte-admitted catalog. This does not improve the planner's
/// caller-declared origin trust model; the returned report keeps that limit.
pub fn plan_next_experiment_from_verified_evidence(
    catalog: &VerifiedEvidenceCatalog,
    hypotheses: &[Hypothesis],
    experiments: &[ExperimentProposal],
    config: PlannerConfig,
) -> Result<VerifiedPlannerRun, AdmissionError> {
    catalog.validate_hypothesis_references(hypotheses)?;
    let planner_request_sha256 =
        catalog.planner_request_digest(hypotheses, experiments, config)?;
    let evidence = catalog.planner_provenance_projection();
    let plan = plan_next_experiment(&evidence, hypotheses, experiments, config)?;
    Ok(VerifiedPlannerRun {
        plan,
        admission: catalog.report(),
        planner_request_sha256,
    })
}

fn validate_acyclic_parent_graph(
    records: &BTreeMap<String, AdmittedRecord>,
) -> Result<(), AdmissionError> {
    fn visit(
        id: &str,
        records: &BTreeMap<String, AdmittedRecord>,
        visiting: &mut BTreeSet<String>,
        visited: &mut BTreeSet<String>,
    ) -> Result<(), AdmissionError> {
        if visited.contains(id) {
            return Ok(());
        }
        if !visiting.insert(id.to_owned()) {
            return Err(AdmissionError::new(
                AdmissionErrorKind::ParentCycle,
                format!("cycle detected in lineage at Evidence {id}"),
            ));
        }
        let record = records.get(id).ok_or_else(|| {
            AdmissionError::new(
                AdmissionErrorKind::DanglingParent,
                format!("lineage graph references missing Evidence {id}"),
            )
        })?;
        for parent in &record.parent_evidence_ids {
            visit(parent, records, visiting, visited)?;
        }
        visiting.remove(id);
        visited.insert(id.to_owned());
        Ok(())
    }

    let mut visiting = BTreeSet::new();
    let mut visited = BTreeSet::new();
    for id in records.keys() {
        visit(id, records, &mut visiting, &mut visited)?;
    }
    Ok(())
}

fn lineage_manifest_digest(
    records: &BTreeMap<String, AdmittedRecord>,
) -> Result<String, AdmissionError> {
    let payload = json!({
        "schema": LINEAGE_SCHEMA,
        "records": records.values().map(|record| json!({
            "evidence_id": record.evidence_id,
            "declared_origin": record.declared_origin,
            "parent_evidence_ids": record.parent_evidence_ids,
            "source_family_id": record.source_family_id,
        })).collect::<Vec<_>>()
    });
    let canonical = canonical_jcs_bytes(&payload).map_err(|error| {
        AdmissionError::new(
            AdmissionErrorKind::RawEvidenceRejected,
            format!("lineage manifest canonicalization failed: {error}"),
        )
    })?;
    Ok(sha256_hex(&canonical))
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use symthaea_rea_compat::compute_evidence_id;
    use symthaea_rea_investigator::{AbstentionReason, CostWeights};

    fn golden_id() -> String {
        verify_evidence_json(GOLDEN_EVIDENCE).unwrap().evidence_id
    }

    fn declared(id: String, origin: DeclaredEvidenceOrigin, parents: Vec<String>) -> LineageDeclaration {
        LineageDeclaration {
            evidence_id: id,
            declared_origin: origin,
            parent_evidence_ids: parents,
            source_family_id: "synthetic-test-family".to_owned(),
        }
    }

    fn one_record(origin: DeclaredEvidenceOrigin) -> VerifiedEvidenceCatalog {
        VerifiedEvidenceCatalog::admit_raw_records(
            vec![GOLDEN_EVIDENCE.to_vec()],
            vec![declared(golden_id(), origin, Vec::new())],
        ).unwrap()
    }

    fn raw_record_with_operation(operation: &str) -> Vec<u8> {
        let mut value: Value = serde_json::from_slice(GOLDEN_EVIDENCE).unwrap();
        value["operation"] = json!(operation);
        value["evidence_id"] = json!("ev_".to_owned() + &"0".repeat(64));
        value["evidence_id"] = json!(compute_evidence_id(&value).unwrap());
        serde_json::to_vec(&value).unwrap()
    }

    fn evidence_id(bytes: &[u8]) -> String {
        verify_evidence_json(bytes).unwrap().evidence_id
    }

    #[test]
    fn raw_bytes_bind_to_verified_semantic_evidence_id() {
        let catalog = one_record(DeclaredEvidenceOrigin::IndependentObservation);
        let report = catalog.report();
        assert!(report.claim_ceiling.raw_bytes_admitted);
        assert!(report.claim_ceiling.semantic_evidence_identity_verified);
        assert!(!report.claim_ceiling.producer_authenticated);
        assert!(!report.claim_ceiling.caller_origin_authenticated);
        assert!(!report.claim_ceiling.source_independence_authenticated);
        assert!(!report.claim_ceiling.observation_truth_proven);
        assert_eq!(report.records[0].evidence_id, golden_id());
        assert_eq!(
            report.records[0].raw_file_sha256,
            symthaea_rea_compat::sha256_hex(GOLDEN_EVIDENCE)
        );
    }

    #[test]
    fn caller_declared_independent_observation_is_not_reported_as_authenticated() {
        let catalog = one_record(DeclaredEvidenceOrigin::IndependentObservation);
        let record = &catalog.report().records[0];
        assert_eq!(record.declared_origin, DeclaredEvidenceOrigin::IndependentObservation);
        assert_eq!(record.origin_qualification, OriginQualification::CallerDeclaredUnverified);
        assert!(!catalog.report().claim_ceiling.source_independence_authenticated);
    }

    #[test]
    fn forged_evidence_id_in_raw_bytes_is_rejected() {
        let mut value: Value = serde_json::from_slice(GOLDEN_EVIDENCE).unwrap();
        value["evidence_id"] = json!("ev_".to_owned() + &"f".repeat(64));
        let raw = serde_json::to_vec(&value).unwrap();
        let result = VerifiedEvidenceCatalog::admit_raw_records(
            vec![raw],
            vec![declared(
                "ev_".to_owned() + &"f".repeat(64),
                DeclaredEvidenceOrigin::IndependentObservation,
                Vec::new(),
            )],
        );
        assert!(result.is_err());
    }

    #[test]
    fn duplicate_json_keys_reject_before_semantic_admission() {
        let text = std::str::from_utf8(GOLDEN_EVIDENCE).unwrap();
        let replaced = text.replacen(
            "\"evidence_id\": \"ev_a4e49bdca4cbf80c2ec8e4d7eba116af403bc33a4937ffb7fca00b6aab6590c4\",",
            "\"evidence_id\": \"ev_a4e49bdca4cbf80c2ec8e4d7eba116af403bc33a4937ffb7fca00b6aab6590c4\",\\n  \"evidence_id\": \"ev_a4e49bdca4cbf80c2ec8e4d7eba116af403bc33a4937ffb7fca00b6aab6590c4\",",
            1,
        );
        assert_ne!(replaced, text);
        let result = verify_evidence_json(replaced.as_bytes());
        assert!(result.is_err());
    }

    #[test]
    fn trailing_json_data_is_rejected() {
        let mut raw = GOLDEN_EVIDENCE.to_vec();
        raw.extend_from_slice(b" {}");
        assert!(verify_evidence_json(&raw).is_err());
    }

    #[test]
    fn raw_lexical_identity_is_distinct_from_semantic_identity() {
        let compact = {
            let value: Value = serde_json::from_slice(GOLDEN_EVIDENCE).unwrap();
            serde_json::to_vec(&value).unwrap()
        };
        let first = verify_evidence_json(GOLDEN_EVIDENCE).unwrap();
        let second = verify_evidence_json(&compact).unwrap();
        assert_eq!(first.evidence_id, second.evidence_id);
        assert_ne!(first.raw_file_sha256, second.raw_file_sha256);
    }

    #[test]
    fn lineage_manifest_digest_is_stable_under_input_order() {
        let first_raw = raw_record_with_operation("inspect_one");
        let second_raw = raw_record_with_operation("inspect_two");
        let first_id = evidence_id(&first_raw);
        let second_id = evidence_id(&second_raw);

        let input_a = vec![first_raw.clone(), second_raw.clone()];
        let declarations_a = vec![
            declared(first_id.clone(), DeclaredEvidenceOrigin::IndependentObservation, vec![]),
            declared(second_id.clone(), DeclaredEvidenceOrigin::IndependentObservation, vec![]),
        ];
        let input_b = vec![second_raw.clone(), first_raw.clone()];
        let declarations_b = vec![
            declared(second_id, DeclaredEvidenceOrigin::IndependentObservation, vec![]),
            declared(first_id, DeclaredEvidenceOrigin::IndependentObservation, vec![]),
        ];
        let a = VerifiedEvidenceCatalog::admit_raw_records(input_a, declarations_a).unwrap();
        let b = VerifiedEvidenceCatalog::admit_raw_records(input_b, declarations_b).unwrap();
        assert_eq!(a.report().lineage_manifest_sha256, b.report().lineage_manifest_sha256);
    }

    #[test]
    fn undeclared_or_unadmitted_lineage_records_reject() {
        let result = VerifiedEvidenceCatalog::admit_raw_records(
            vec![GOLDEN_EVIDENCE.to_vec()],
            Vec::new(),
        );
        assert_eq!(result.unwrap_err().kind, AdmissionErrorKind::MissingLineageDeclaration);

        let result = VerifiedEvidenceCatalog::admit_raw_records(
            vec![GOLDEN_EVIDENCE.to_vec()],
            vec![declared(
                "ev_".to_owned() + &"f".repeat(64),
                DeclaredEvidenceOrigin::IndependentObservation,
                Vec::new(),
            )],
        );
        assert_eq!(result.unwrap_err().kind, AdmissionErrorKind::UnknownLineageEvidence);
    }

    #[test]
    fn independent_observation_must_be_a_root_and_derivation_must_have_parents() {
        let id = golden_id();
        let raw = GOLDEN_EVIDENCE.to_vec();
        let result = VerifiedEvidenceCatalog::admit_raw_records(
            vec![raw.clone()],
            vec![declared(id.clone(), DeclaredEvidenceOrigin::IndependentObservation, vec![id.clone()])],
        );
        assert_eq!(result.unwrap_err().kind, AdmissionErrorKind::SelfParent);

        let result = VerifiedEvidenceCatalog::admit_raw_records(
            vec![raw],
            vec![declared(id, DeclaredEvidenceOrigin::IndependentDerivation, Vec::new())],
        );
        assert_eq!(result.unwrap_err().kind, AdmissionErrorKind::InvalidOriginParentShape);
    }

    #[test]
    fn dangling_parent_duplicate_parent_and_cycle_reject() {
        let first = raw_record_with_operation("first");
        let second = raw_record_with_operation("second");
        let first_id = evidence_id(&first);
        let second_id = evidence_id(&second);

        let dangling = VerifiedEvidenceCatalog::admit_raw_records(
            vec![first.clone()],
            vec![declared(
                first_id.clone(),
                DeclaredEvidenceOrigin::IndependentDerivation,
                vec![second_id.clone()],
            )],
        );
        assert_eq!(dangling.unwrap_err().kind, AdmissionErrorKind::DanglingParent);

        let duplicate = VerifiedEvidenceCatalog::admit_raw_records(
            vec![first.clone(), second.clone()],
            vec![
                declared(first_id.clone(), DeclaredEvidenceOrigin::IndependentDerivation, vec![second_id.clone(), second_id.clone()]),
                declared(second_id.clone(), DeclaredEvidenceOrigin::IndependentObservation, vec![]),
            ],
        );
        assert_eq!(duplicate.unwrap_err().kind, AdmissionErrorKind::DuplicateParent);

        let cycle = VerifiedEvidenceCatalog::admit_raw_records(
            vec![first, second],
            vec![
                declared(first_id.clone(), DeclaredEvidenceOrigin::IndependentDerivation, vec![second_id.clone()]),
                declared(second_id, DeclaredEvidenceOrigin::IndependentDerivation, vec![first_id]),
            ],
        );
        assert_eq!(cycle.unwrap_err().kind, AdmissionErrorKind::ParentCycle);
    }

    #[test]
    fn hypothesis_cannot_reference_an_unadmitted_evidence_id() {
        let catalog = one_record(DeclaredEvidenceOrigin::IndependentObservation);
        let hypothesis = Hypothesis {
            id: "h1".to_owned(),
            prior_probability: 1.0,
            supporting_evidence_ids: vec!["ev_".to_owned() + &"f".repeat(64)],
            contradicting_evidence_ids: Vec::new(),
        };
        let result = plan_next_experiment_from_verified_evidence(
            &catalog,
            &[hypothesis],
            &[],
            PlannerConfig::default(),
        );
        assert_eq!(result.unwrap_err().kind, AdmissionErrorKind::MissingHypothesisEvidence);
    }

    #[test]
    fn synthetic_only_evidence_does_not_ground_the_existing_production_planner() {
        let catalog = one_record(DeclaredEvidenceOrigin::Synthetic);
        let hypothesis = Hypothesis {
            id: "h1".to_owned(),
            prior_probability: 1.0,
            supporting_evidence_ids: vec![golden_id()],
            contradicting_evidence_ids: Vec::new(),
        };
        let result = plan_next_experiment_from_verified_evidence(
            &catalog,
            &[hypothesis],
            &[],
            PlannerConfig::default(),
        ).unwrap();
        assert_eq!(result.plan.abstention, Some(AbstentionReason::NoIndependentlyGroundedHypothesis));
        assert!(!result.admission.claim_ceiling.caller_origin_authenticated);
    }

    #[test]
    fn admitted_caller_label_is_preserved_but_claim_ceiling_stays_unverified() {
        let catalog = one_record(DeclaredEvidenceOrigin::IndependentObservation);
        let hypothesis = Hypothesis {
            id: "h1".to_owned(),
            prior_probability: 1.0,
            supporting_evidence_ids: vec![golden_id()],
            contradicting_evidence_ids: Vec::new(),
        };
        let result = plan_next_experiment_from_verified_evidence(
            &catalog,
            &[hypothesis],
            &[],
            PlannerConfig::default(),
        ).unwrap();
        assert_eq!(
            result.plan.abstention,
            Some(AbstentionReason::NoIndependentlyGroundedHypothesis),
            "caller-declared origin must not ground a production planner hypothesis"
        );
        assert!(!result.plan.claim_ceiling.evidence_producer_authenticated);
        assert!(!result.admission.claim_ceiling.caller_origin_authenticated);
        assert!(!result.admission.claim_ceiling.source_independence_authenticated);
        assert_eq!(
            result.admission.records[0].origin_qualification,
            OriginQualification::CallerDeclaredUnverified
        );
    }

    #[test]
    fn empty_source_family_rejects() {
        let result = VerifiedEvidenceCatalog::admit_raw_records(
            vec![GOLDEN_EVIDENCE.to_vec()],
            vec![LineageDeclaration {
                evidence_id: golden_id(),
                declared_origin: DeclaredEvidenceOrigin::IndependentObservation,
                parent_evidence_ids: vec![],
                source_family_id: "  ".to_owned(),
            }],
        );
        assert_eq!(result.unwrap_err().kind, AdmissionErrorKind::EmptySourceFamily);
    }

    #[test]
    fn distinct_roots_from_one_declared_common_mode_family_do_not_count_as_independent() {
        let first = raw_record_with_operation("same-provider-observation-one");
        let second = raw_record_with_operation("same-provider-observation-two");
        let first_id = evidence_id(&first);
        let second_id = evidence_id(&second);
        assert_ne!(first_id, second_id);

        let catalog = VerifiedEvidenceCatalog::admit_raw_records(
            vec![first, second],
            vec![
                declared(
                    first_id.clone(),
                    DeclaredEvidenceOrigin::IndependentObservation,
                    Vec::new(),
                ),
                declared(
                    second_id.clone(),
                    DeclaredEvidenceOrigin::IndependentObservation,
                    Vec::new(),
                ),
            ],
        )
        .unwrap();

        let hypothesis = Hypothesis {
            id: "h1".to_owned(),
            prior_probability: 1.0,
            supporting_evidence_ids: vec![first_id, second_id],
            contradicting_evidence_ids: Vec::new(),
        };
        let result = plan_next_experiment_from_verified_evidence(
            &catalog,
            &[hypothesis],
            &[],
            PlannerConfig::default(),
        );
        assert_eq!(
            result.unwrap_err().kind,
            AdmissionErrorKind::CommonModeIndependentRoot
        );
    }

    #[test]
    fn same_root_reused_through_multiple_derived_support_paths_is_not_double_counted() {
        let root = raw_record_with_operation("single-root");
        let derivation_one = raw_record_with_operation("derived-one");
        let derivation_two = raw_record_with_operation("derived-two");
        let root_id = evidence_id(&root);
        let derivation_one_id = evidence_id(&derivation_one);
        let derivation_two_id = evidence_id(&derivation_two);

        let catalog = VerifiedEvidenceCatalog::admit_raw_records(
            vec![root, derivation_one, derivation_two],
            vec![
                declared(
                    root_id.clone(),
                    DeclaredEvidenceOrigin::IndependentObservation,
                    Vec::new(),
                ),
                declared(
                    derivation_one_id.clone(),
                    DeclaredEvidenceOrigin::IndependentDerivation,
                    vec![root_id.clone()],
                ),
                declared(
                    derivation_two_id.clone(),
                    DeclaredEvidenceOrigin::IndependentDerivation,
                    vec![root_id],
                ),
            ],
        )
        .unwrap();

        let hypothesis = Hypothesis {
            id: "h1".to_owned(),
            prior_probability: 1.0,
            supporting_evidence_ids: vec![derivation_one_id, derivation_two_id],
            contradicting_evidence_ids: Vec::new(),
        };
        assert!(catalog
            .validate_hypothesis_source_family_independence(&[hypothesis])
            .is_ok());
    }


    #[test]
    fn planner_request_digest_is_deterministic_and_binds_cost_config() {
        let catalog = one_record(DeclaredEvidenceOrigin::IndependentObservation);
        let hypothesis = Hypothesis {
            id: "h1".to_owned(),
            prior_probability: 1.0,
            supporting_evidence_ids: vec![golden_id()],
            contradicting_evidence_ids: Vec::new(),
        };
        let config = PlannerConfig::default();
        let first = catalog
            .planner_request_digest(&[hypothesis.clone()], &[], config)
            .unwrap();
        let repeated = catalog
            .planner_request_digest(&[hypothesis.clone()], &[], config)
            .unwrap();
        assert_eq!(first, repeated);

        let mut changed = config;
        changed.cost_weights = CostWeights {
            time: config.cost_weights.time + 0.25,
            energy: config.cost_weights.energy,
            risk: config.cost_weights.risk,
        };
        let changed_digest = catalog
            .planner_request_digest(&[hypothesis], &[], changed)
            .unwrap();
        assert_ne!(first, changed_digest);
    }

    #[test]
    fn planner_run_exposes_request_binding_without_claiming_run_attestation() {
        let catalog = one_record(DeclaredEvidenceOrigin::IndependentObservation);
        let hypothesis = Hypothesis {
            id: "h1".to_owned(),
            prior_probability: 1.0,
            supporting_evidence_ids: vec![golden_id()],
            contradicting_evidence_ids: Vec::new(),
        };
        let result = plan_next_experiment_from_verified_evidence(
            &catalog,
            &[hypothesis],
            &[],
            PlannerConfig::default(),
        )
        .unwrap();
        assert_eq!(result.planner_request_sha256.len(), 64);
        assert!(result.planner_request_sha256.bytes().all(|byte| byte.is_ascii_hexdigit()));
    }

}
