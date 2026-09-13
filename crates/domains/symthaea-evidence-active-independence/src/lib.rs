// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Exact-use composition of lifecycle-current verifier profiles and
//! relation-aware common-cause separation.
//!
//! A persisted `Separated` report is evidence, not authority. This crate
//! recomputes separation from the exact bound graph/completeness/policy and
//! mints a non-serializable active qualification only when both lifecycle
//! capabilities were assessed at the exact same use-time.

#![deny(unsafe_code)]

use serde::{Deserialize, Serialize};
use symthaea_evidence_independence_completeness::{
    RelationAwareIndependencePolicy, RelationAwareIndependenceReport, RelationCompletenessMap,
    assess_relation_aware_independence,
};
use symthaea_evidence_independence_graph::{FaultDomainGraph, IndependenceStatus};
use symthaea_evidence_independence_lifecycle::ActiveVerifierProfile;
use symthaea_evidence_independence_provenance::VersionedVerifierProfileRecord;

pub const ACTIVE_INDEPENDENCE_COMPOSITION_SCHEMA_V1: &str =
    "symthaea.assurance.active-independence-composition.v1";

const QUALIFICATION_DIGEST_DOMAIN: &[u8] =
    b"symthaea.assurance.active-independence-qualification.digest.v1\0";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActiveIndependenceDisposition {
    Invalid,
    Blocked,
    Qualified,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ActiveIndependenceIssue {
    InvalidLeftRecord,
    InvalidRightRecord,
    InvalidGraph,
    InvalidCompletenessMap,
    InvalidPolicy,
    LeftRecordDigestMismatch,
    RightRecordDigestMismatch,
    LeftRecordGraphBindingMismatch,
    RightRecordGraphBindingMismatch,
    LeftRecordCompletenessBindingMismatch,
    RightRecordCompletenessBindingMismatch,
    LeftActiveGraphBindingMismatch,
    RightActiveGraphBindingMismatch,
    LeftActiveCompletenessBindingMismatch,
    RightActiveCompletenessBindingMismatch,
    LeftActiveAssessmentTimeMismatch { observed: u64, required: u64 },
    RightActiveAssessmentTimeMismatch { observed: u64, required: u64 },
    RelationAssessmentInvalid,
    RelationAssessmentCorrelated,
    RelationAssessmentIndeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ActiveIndependenceReport {
    pub schema_version: String,
    pub disposition: ActiveIndependenceDisposition,
    pub left_profile_id: String,
    pub right_profile_id: String,
    pub left_verifier_ref: String,
    pub right_verifier_ref: String,
    pub left_record_digest: Option<String>,
    pub right_record_digest: Option<String>,
    pub left_provenance_attestation_digest: String,
    pub right_provenance_attestation_digest: String,
    pub left_lifecycle_digest: String,
    pub right_lifecycle_digest: String,
    pub graph_digest: Option<String>,
    pub relation_completeness_digest: Option<String>,
    pub policy_digest: Option<String>,
    pub relation_status: IndependenceStatus,
    pub left_verification_at_ms: u64,
    pub right_verification_at_ms: u64,
    pub use_at_ms: u64,
    pub issues: Vec<ActiveIndependenceIssue>,
    pub qualification_digest: Option<String>,
}

impl ActiveIndependenceReport {
    pub const fn grants_physical_authority(&self) -> bool {
        false
    }

    pub const fn universal_independence_established(&self) -> bool {
        false
    }
}

/// Runtime-only proof that two exact verifier-profile lineages were both
/// lifecycle-current at one exact use-time and were recomputed as separated
/// under one exact graph/completeness/policy revision.
///
/// Deliberately not serializable/deserializable. Persist the report as evidence,
/// then recompute this capability at the point of use.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveIndependenceQualification {
    qualification_digest: String,
    left_profile_id: String,
    right_profile_id: String,
    left_verifier_ref: String,
    right_verifier_ref: String,
    left_record_digest: String,
    right_record_digest: String,
    left_provenance_attestation_digest: String,
    right_provenance_attestation_digest: String,
    left_lifecycle_digest: String,
    right_lifecycle_digest: String,
    graph_digest: String,
    relation_completeness_digest: String,
    policy_digest: String,
    left_verification_at_ms: u64,
    right_verification_at_ms: u64,
    use_at_ms: u64,
}

impl ActiveIndependenceQualification {
    pub fn qualification_digest(&self) -> &str {
        &self.qualification_digest
    }

    pub fn left_profile_id(&self) -> &str {
        &self.left_profile_id
    }

    pub fn right_profile_id(&self) -> &str {
        &self.right_profile_id
    }

    pub fn left_verifier_ref(&self) -> &str {
        &self.left_verifier_ref
    }

    pub fn right_verifier_ref(&self) -> &str {
        &self.right_verifier_ref
    }

    pub fn left_record_digest(&self) -> &str {
        &self.left_record_digest
    }

    pub fn right_record_digest(&self) -> &str {
        &self.right_record_digest
    }

    pub fn left_provenance_attestation_digest(&self) -> &str {
        &self.left_provenance_attestation_digest
    }

    pub fn right_provenance_attestation_digest(&self) -> &str {
        &self.right_provenance_attestation_digest
    }

    pub fn left_lifecycle_digest(&self) -> &str {
        &self.left_lifecycle_digest
    }

    pub fn right_lifecycle_digest(&self) -> &str {
        &self.right_lifecycle_digest
    }

    pub fn graph_digest(&self) -> &str {
        &self.graph_digest
    }

    pub fn relation_completeness_digest(&self) -> &str {
        &self.relation_completeness_digest
    }

    pub fn policy_digest(&self) -> &str {
        &self.policy_digest
    }

    pub const fn left_verification_at_ms(&self) -> u64 {
        self.left_verification_at_ms
    }

    pub const fn right_verification_at_ms(&self) -> u64 {
        self.right_verification_at_ms
    }

    pub const fn use_at_ms(&self) -> u64 {
        self.use_at_ms
    }

    pub const fn grants_physical_authority(&self) -> bool {
        false
    }

    /// This capability establishes separation only under the exact reviewed
    /// graph/completeness/policy revision it binds. It is not a universal claim
    /// that no unknown common cause exists outside that evidence model.
    pub const fn universal_independence_established(&self) -> bool {
        false
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ActiveIndependenceAssessment {
    pub report: ActiveIndependenceReport,
    pub relation_report: RelationAwareIndependenceReport,
    active: Option<ActiveIndependenceQualification>,
}

impl ActiveIndependenceAssessment {
    pub fn active(&self) -> Option<&ActiveIndependenceQualification> {
        self.active.as_ref()
    }

    pub fn into_active(self) -> Option<ActiveIndependenceQualification> {
        self.active
    }
}

#[allow(clippy::too_many_arguments)]
pub fn compose_active_independence(
    left_record: &VersionedVerifierProfileRecord,
    right_record: &VersionedVerifierProfileRecord,
    left_active: &ActiveVerifierProfile,
    right_active: &ActiveVerifierProfile,
    graph: &FaultDomainGraph,
    completeness: &RelationCompletenessMap,
    policy: &RelationAwareIndependencePolicy,
    use_at_ms: u64,
) -> ActiveIndependenceAssessment {
    let left_record_digest = left_record.canonical_digest();
    let right_record_digest = right_record.canonical_digest();
    let graph_digest = graph.canonical_digest();
    let completeness_digest = completeness.canonical_digest();
    let policy_digest = policy.canonical_digest();

    let mut issues = Vec::new();
    if !left_record.validate() {
        issues.push(ActiveIndependenceIssue::InvalidLeftRecord);
    }
    if !right_record.validate() {
        issues.push(ActiveIndependenceIssue::InvalidRightRecord);
    }
    if graph.validate().is_err() {
        issues.push(ActiveIndependenceIssue::InvalidGraph);
    }
    if !completeness.validate() {
        issues.push(ActiveIndependenceIssue::InvalidCompletenessMap);
    }
    if !policy.validate() {
        issues.push(ActiveIndependenceIssue::InvalidPolicy);
    }

    if left_record_digest.as_deref() != Some(left_active.record_digest()) {
        issues.push(ActiveIndependenceIssue::LeftRecordDigestMismatch);
    }
    if right_record_digest.as_deref() != Some(right_active.record_digest()) {
        issues.push(ActiveIndependenceIssue::RightRecordDigestMismatch);
    }

    if graph_digest.as_deref() != Some(left_record.graph_digest.as_str()) {
        issues.push(ActiveIndependenceIssue::LeftRecordGraphBindingMismatch);
    }
    if graph_digest.as_deref() != Some(right_record.graph_digest.as_str()) {
        issues.push(ActiveIndependenceIssue::RightRecordGraphBindingMismatch);
    }
    if completeness_digest.as_deref()
        != Some(left_record.relation_completeness_digest.as_str())
    {
        issues.push(ActiveIndependenceIssue::LeftRecordCompletenessBindingMismatch);
    }
    if completeness_digest.as_deref()
        != Some(right_record.relation_completeness_digest.as_str())
    {
        issues.push(ActiveIndependenceIssue::RightRecordCompletenessBindingMismatch);
    }

    if graph_digest.as_deref() != Some(left_active.graph_digest()) {
        issues.push(ActiveIndependenceIssue::LeftActiveGraphBindingMismatch);
    }
    if graph_digest.as_deref() != Some(right_active.graph_digest()) {
        issues.push(ActiveIndependenceIssue::RightActiveGraphBindingMismatch);
    }
    if completeness_digest.as_deref() != Some(left_active.relation_completeness_digest()) {
        issues.push(ActiveIndependenceIssue::LeftActiveCompletenessBindingMismatch);
    }
    if completeness_digest.as_deref() != Some(right_active.relation_completeness_digest()) {
        issues.push(ActiveIndependenceIssue::RightActiveCompletenessBindingMismatch);
    }

    if left_active.assessed_at_ms() != use_at_ms {
        issues.push(ActiveIndependenceIssue::LeftActiveAssessmentTimeMismatch {
            observed: left_active.assessed_at_ms(),
            required: use_at_ms,
        });
    }
    if right_active.assessed_at_ms() != use_at_ms {
        issues.push(ActiveIndependenceIssue::RightActiveAssessmentTimeMismatch {
            observed: right_active.assessed_at_ms(),
            required: use_at_ms,
        });
    }

    let relation_report = assess_relation_aware_independence(
        &left_record.profile,
        &right_record.profile,
        graph,
        completeness,
        policy,
    );
    match relation_report.status {
        IndependenceStatus::Invalid => {
            issues.push(ActiveIndependenceIssue::RelationAssessmentInvalid)
        }
        IndependenceStatus::Correlated => {
            issues.push(ActiveIndependenceIssue::RelationAssessmentCorrelated)
        }
        IndependenceStatus::Indeterminate => {
            issues.push(ActiveIndependenceIssue::RelationAssessmentIndeterminate)
        }
        IndependenceStatus::Separated => {}
    }

    let structurally_invalid = issues.iter().any(is_structural_issue);
    let blocked = issues.iter().any(is_blocking_issue);
    let disposition = if structurally_invalid {
        ActiveIndependenceDisposition::Invalid
    } else if blocked {
        ActiveIndependenceDisposition::Blocked
    } else {
        ActiveIndependenceDisposition::Qualified
    };

    let active = if disposition == ActiveIndependenceDisposition::Qualified {
        let left_record_digest = left_record_digest.clone().expect("validated left record");
        let right_record_digest = right_record_digest
            .clone()
            .expect("validated right record");
        let graph_digest = graph_digest.clone().expect("validated graph");
        let relation_completeness_digest = completeness_digest
            .clone()
            .expect("validated completeness map");
        let policy_digest = policy_digest.clone().expect("validated policy");
        let qualification_digest = qualification_digest(
            left_record,
            right_record,
            left_active,
            right_active,
            &left_record_digest,
            &right_record_digest,
            &graph_digest,
            &relation_completeness_digest,
            &policy_digest,
            use_at_ms,
        );
        Some(ActiveIndependenceQualification {
            qualification_digest,
            left_profile_id: left_record.profile_id.clone(),
            right_profile_id: right_record.profile_id.clone(),
            left_verifier_ref: left_record.profile.verifier_ref.clone(),
            right_verifier_ref: right_record.profile.verifier_ref.clone(),
            left_record_digest,
            right_record_digest,
            left_provenance_attestation_digest: left_active
                .provenance_attestation_digest()
                .to_string(),
            right_provenance_attestation_digest: right_active
                .provenance_attestation_digest()
                .to_string(),
            left_lifecycle_digest: left_active.lifecycle_digest().to_string(),
            right_lifecycle_digest: right_active.lifecycle_digest().to_string(),
            graph_digest,
            relation_completeness_digest,
            policy_digest,
            left_verification_at_ms: left_active.verification_at_ms(),
            right_verification_at_ms: right_active.verification_at_ms(),
            use_at_ms,
        })
    } else {
        None
    };

    let qualification_digest = active
        .as_ref()
        .map(|qualification| qualification.qualification_digest.clone());

    ActiveIndependenceAssessment {
        report: ActiveIndependenceReport {
            schema_version: ACTIVE_INDEPENDENCE_COMPOSITION_SCHEMA_V1.into(),
            disposition,
            left_profile_id: left_record.profile_id.clone(),
            right_profile_id: right_record.profile_id.clone(),
            left_verifier_ref: left_record.profile.verifier_ref.clone(),
            right_verifier_ref: right_record.profile.verifier_ref.clone(),
            left_record_digest,
            right_record_digest,
            left_provenance_attestation_digest: left_active
                .provenance_attestation_digest()
                .to_string(),
            right_provenance_attestation_digest: right_active
                .provenance_attestation_digest()
                .to_string(),
            left_lifecycle_digest: left_active.lifecycle_digest().to_string(),
            right_lifecycle_digest: right_active.lifecycle_digest().to_string(),
            graph_digest,
            relation_completeness_digest: completeness_digest,
            policy_digest,
            relation_status: relation_report.status,
            left_verification_at_ms: left_active.verification_at_ms(),
            right_verification_at_ms: right_active.verification_at_ms(),
            use_at_ms,
            issues,
            qualification_digest,
        },
        relation_report,
        active,
    }
}

#[allow(clippy::too_many_arguments)]
fn qualification_digest(
    left_record: &VersionedVerifierProfileRecord,
    right_record: &VersionedVerifierProfileRecord,
    left_active: &ActiveVerifierProfile,
    right_active: &ActiveVerifierProfile,
    left_record_digest: &str,
    right_record_digest: &str,
    graph_digest: &str,
    completeness_digest: &str,
    policy_digest: &str,
    use_at_ms: u64,
) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(QUALIFICATION_DIGEST_DOMAIN);
    push_field(&mut hasher, ACTIVE_INDEPENDENCE_COMPOSITION_SCHEMA_V1);
    // Left/right ordering is intentionally preserved so downstream consumers can
    // assign distinct semantic roles (for example campaign vs obligation review).
    push_field(&mut hasher, &left_record.profile_id);
    push_field(&mut hasher, &left_record.profile.verifier_ref);
    push_field(&mut hasher, left_record_digest);
    push_field(&mut hasher, left_active.provenance_attestation_digest());
    push_field(&mut hasher, left_active.lifecycle_digest());
    push_u64(&mut hasher, left_active.verification_at_ms());
    push_field(&mut hasher, &right_record.profile_id);
    push_field(&mut hasher, &right_record.profile.verifier_ref);
    push_field(&mut hasher, right_record_digest);
    push_field(&mut hasher, right_active.provenance_attestation_digest());
    push_field(&mut hasher, right_active.lifecycle_digest());
    push_u64(&mut hasher, right_active.verification_at_ms());
    push_field(&mut hasher, graph_digest);
    push_field(&mut hasher, completeness_digest);
    push_field(&mut hasher, policy_digest);
    push_field(&mut hasher, "separated");
    push_u64(&mut hasher, use_at_ms);
    format!("blake3:{}", hasher.finalize().to_hex())
}

fn is_structural_issue(issue: &ActiveIndependenceIssue) -> bool {
    matches!(
        issue,
        ActiveIndependenceIssue::InvalidLeftRecord
            | ActiveIndependenceIssue::InvalidRightRecord
            | ActiveIndependenceIssue::InvalidGraph
            | ActiveIndependenceIssue::InvalidCompletenessMap
            | ActiveIndependenceIssue::InvalidPolicy
            | ActiveIndependenceIssue::LeftRecordDigestMismatch
            | ActiveIndependenceIssue::RightRecordDigestMismatch
            | ActiveIndependenceIssue::LeftRecordGraphBindingMismatch
            | ActiveIndependenceIssue::RightRecordGraphBindingMismatch
            | ActiveIndependenceIssue::LeftRecordCompletenessBindingMismatch
            | ActiveIndependenceIssue::RightRecordCompletenessBindingMismatch
            | ActiveIndependenceIssue::LeftActiveGraphBindingMismatch
            | ActiveIndependenceIssue::RightActiveGraphBindingMismatch
            | ActiveIndependenceIssue::LeftActiveCompletenessBindingMismatch
            | ActiveIndependenceIssue::RightActiveCompletenessBindingMismatch
            | ActiveIndependenceIssue::RelationAssessmentInvalid
    )
}

fn is_blocking_issue(issue: &ActiveIndependenceIssue) -> bool {
    matches!(
        issue,
        ActiveIndependenceIssue::LeftActiveAssessmentTimeMismatch { .. }
            | ActiveIndependenceIssue::RightActiveAssessmentTimeMismatch { .. }
            | ActiveIndependenceIssue::RelationAssessmentCorrelated
            | ActiveIndependenceIssue::RelationAssessmentIndeterminate
    )
}

fn push_field(hasher: &mut blake3::Hasher, value: &str) {
    hasher.update(&(value.len() as u64).to_be_bytes());
    hasher.update(value.as_bytes());
}

fn push_u64(hasher: &mut blake3::Hasher, value: u64) {
    hasher.update(&value.to_be_bytes());
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::{Signer, SigningKey};
    use symthaea_evidence_independence_completeness::{
        AxisRelationRequirement, RELATION_AWARE_POLICY_SCHEMA_V1, RELATION_COMPLETENESS_SCHEMA_V1,
        RelationCompletenessClaim, RelationCompletenessState,
    };
    use symthaea_evidence_independence_graph::{
        FAULT_DOMAIN_GRAPH_SCHEMA_V1, FaultDomainEdge, FaultDomainNode, FaultDomainNodeKind,
        FaultDomainRelation, IndependenceAxis,
    };
    use symthaea_evidence_independence_lifecycle::{
        PROFILE_LIFECYCLE_LEDGER_SCHEMA_V1, ProfileLifecycleLedger, assess_profile_lifecycle,
    };
    use symthaea_evidence_independence_provenance::{
        PROFILE_ATTESTATION_SCHEMA_V1, PROFILE_PROVENANCE_POLICY_SCHEMA_V1,
        PROFILE_RECORD_SCHEMA_V1, ProfileAttestationEnvelope, ProfileClaimScope,
        ProfileIssuerKeyRecord, ProfileProvenancePolicy, verify_profile_provenance,
    };
    use symthaea_evidence_verifier_diversity::VerifierFaultDomainProfile;

    struct Fixture {
        left_record: VersionedVerifierProfileRecord,
        right_record: VersionedVerifierProfileRecord,
        left_active: ActiveVerifierProfile,
        right_active: ActiveVerifierProfile,
        graph: FaultDomainGraph,
        completeness: RelationCompletenessMap,
        policy: RelationAwareIndependencePolicy,
        use_at_ms: u64,
    }

    fn profile(verifier: &str, org: &str) -> VerifierFaultDomainProfile {
        VerifierFaultDomainProfile {
            verifier_ref: verifier.into(),
            organization_domain: org.into(),
            review_process_domain: format!("process:{verifier}"),
            toolchain_domain: format!("tool:{verifier}"),
            evidence_source_domain: format!("source:{verifier}"),
            evidence_refs: vec![format!("profile:{verifier}")],
        }
    }

    fn node(id: &str, kind: FaultDomainNodeKind) -> FaultDomainNode {
        FaultDomainNode {
            node_id: id.into(),
            kind,
            lineage_complete: false,
            evidence_refs: vec![format!("node:{id}")],
        }
    }

    fn edge(child: &str, ancestor: &str) -> FaultDomainEdge {
        FaultDomainEdge {
            child_node_id: child.into(),
            ancestor_node_id: ancestor.into(),
            relation: FaultDomainRelation::ControlledBy,
            evidence_refs: vec![format!("edge:{child}:{ancestor}")],
        }
    }

    fn complete(node_id: &str) -> RelationCompletenessClaim {
        RelationCompletenessClaim {
            node_id: node_id.into(),
            relation: FaultDomainRelation::ControlledBy,
            state: RelationCompletenessState::Complete,
            evidence_refs: vec![format!("complete:{node_id}")],
        }
    }

    fn fixture(shared_root: bool, omit_root_completeness: bool) -> Fixture {
        let left_root = if shared_root { "control:shared" } else { "control:a" };
        let right_root = if shared_root { "control:shared" } else { "control:b" };
        let mut nodes = vec![
            node("org:a", FaultDomainNodeKind::Organization),
            node("org:b", FaultDomainNodeKind::Organization),
            node(left_root, FaultDomainNodeKind::ControlPlane),
        ];
        if right_root != left_root {
            nodes.push(node(right_root, FaultDomainNodeKind::ControlPlane));
        }
        let graph = FaultDomainGraph {
            schema_version: FAULT_DOMAIN_GRAPH_SCHEMA_V1.into(),
            graph_id: if shared_root { "graph:shared" } else { "graph:split" }.into(),
            nodes,
            edges: vec![edge("org:a", left_root), edge("org:b", right_root)],
            evidence_refs: vec!["review:graph".into()],
        };
        let graph_digest = graph.canonical_digest().expect("valid graph");

        let mut claims = vec![complete("org:a"), complete("org:b")];
        if !omit_root_completeness {
            claims.push(complete(left_root));
            if right_root != left_root {
                claims.push(complete(right_root));
            }
        }
        let completeness = RelationCompletenessMap {
            schema_version: RELATION_COMPLETENESS_SCHEMA_V1.into(),
            map_id: "complete:org-control".into(),
            graph_digest: graph_digest.clone(),
            claims,
            evidence_refs: vec!["review:completeness".into()],
        };
        let completeness_digest = completeness.canonical_digest().expect("valid completeness");
        let policy = RelationAwareIndependencePolicy {
            schema_version: RELATION_AWARE_POLICY_SCHEMA_V1.into(),
            policy_id: "policy:org-control".into(),
            axis_requirements: vec![AxisRelationRequirement {
                axis: IndependenceAxis::Organization,
                required_relations: vec![FaultDomainRelation::ControlledBy],
            }],
            require_global_separation: false,
            max_ancestry_depth: 8,
            evidence_refs: vec!["review:policy".into()],
        };

        let left_record = record(
            "profile:a",
            1,
            profile("verifier:a", "org:a"),
            &graph_digest,
            &completeness_digest,
        );
        let right_record = record(
            "profile:b",
            1,
            profile("verifier:b", "org:b"),
            &graph_digest,
            &completeness_digest,
        );

        let signing_key = SigningKey::from_bytes(&[7u8; 32]);
        let provenance_policy = provenance_policy(&signing_key);
        let left_provenance = authenticated(&left_record, &provenance_policy, &signing_key, 21);
        let right_provenance = authenticated(&right_record, &provenance_policy, &signing_key, 22);
        let use_at_ms = 200;
        let left_active = active(&left_record, &left_provenance, "ledger:a", 100, use_at_ms);
        let right_active = active(&right_record, &right_provenance, "ledger:b", 110, use_at_ms);

        Fixture {
            left_record,
            right_record,
            left_active,
            right_active,
            graph,
            completeness,
            policy,
            use_at_ms,
        }
    }

    fn record(
        profile_id: &str,
        revision: u64,
        profile: VerifierFaultDomainProfile,
        graph_digest: &str,
        completeness_digest: &str,
    ) -> VersionedVerifierProfileRecord {
        VersionedVerifierProfileRecord {
            schema_version: PROFILE_RECORD_SCHEMA_V1.into(),
            profile_id: profile_id.into(),
            revision,
            profile,
            effective_from_ms: 1,
            effective_until_ms: None,
            graph_digest: graph_digest.into(),
            relation_completeness_digest: completeness_digest.into(),
            evidence_refs: vec![format!("record:{profile_id}")],
        }
    }

    fn scopes() -> Vec<ProfileClaimScope> {
        vec![
            ProfileClaimScope::ProfileIdentity,
            ProfileClaimScope::FaultDomainAssignment,
            ProfileClaimScope::GraphBinding,
            ProfileClaimScope::RelationCompletenessBinding,
        ]
    }

    fn provenance_policy(signing_key: &SigningKey) -> ProfileProvenancePolicy {
        ProfileProvenancePolicy {
            schema_version: PROFILE_PROVENANCE_POLICY_SCHEMA_V1.into(),
            policy_id: "provenance-policy:v1".into(),
            sequence: 1,
            issued_at_ms: 1,
            expires_at_ms: 1_000,
            required_scopes: scopes(),
            trusted_keys: vec![ProfileIssuerKeyRecord {
                key_id: "issuer:v1".into(),
                public_key_ed25519_hex: hex::encode(signing_key.verifying_key().as_bytes()),
                valid_from_ms: 1,
                valid_until_ms: None,
                revoked_at_ms: None,
                allowed_scopes: scopes(),
                evidence_refs: vec!["issuer-review:v1".into()],
            }],
            evidence_refs: vec!["policy-review:v1".into()],
        }
    }

    fn authenticated(
        record: &VersionedVerifierProfileRecord,
        policy: &ProfileProvenancePolicy,
        signing_key: &SigningKey,
        issued_at_ms: u64,
    ) -> symthaea_evidence_independence_provenance::AuthenticatedProfileProvenance {
        let mut envelope = ProfileAttestationEnvelope {
            schema_version: PROFILE_ATTESTATION_SCHEMA_V1.into(),
            record_digest: record.canonical_digest().expect("record digest"),
            policy_digest: policy.canonical_digest().expect("policy digest"),
            issuer_key_id: "issuer:v1".into(),
            issuer_public_key_ed25519_hex: hex::encode(signing_key.verifying_key().as_bytes()),
            issued_at_ms,
            expires_at_ms: None,
            scopes: scopes(),
            nonce_blake3_hex: format!("{:064x}", issued_at_ms),
            signature_ed25519_hex: "00".repeat(64),
        };
        let bytes = envelope.canonical_unsigned_bytes().expect("canonical unsigned bytes");
        envelope.signature_ed25519_hex = hex::encode(signing_key.sign(&bytes).to_bytes());
        verify_profile_provenance(record, policy, &envelope)
            .into_authenticated()
            .expect("authenticated provenance")
    }

    fn active(
        record: &VersionedVerifierProfileRecord,
        provenance: &symthaea_evidence_independence_provenance::AuthenticatedProfileProvenance,
        ledger_id: &str,
        verification_at_ms: u64,
        assessed_at_ms: u64,
    ) -> ActiveVerifierProfile {
        let ledger = ProfileLifecycleLedger {
            schema_version: PROFILE_LIFECYCLE_LEDGER_SCHEMA_V1.into(),
            ledger_id: ledger_id.into(),
            profile_id: record.profile_id.clone(),
            record_digest: record.canonical_digest().expect("record digest"),
            events: Vec::new(),
            evidence_refs: vec![format!("lifecycle:{ledger_id}")],
        };
        assess_profile_lifecycle(
            record,
            provenance,
            &ledger,
            verification_at_ms,
            assessed_at_ms,
        )
        .into_active()
        .expect("active profile")
    }

    #[test]
    fn exact_use_separated_lineages_mint_active_qualification() {
        let f = fixture(false, false);
        let assessment = compose_active_independence(
            &f.left_record,
            &f.right_record,
            &f.left_active,
            &f.right_active,
            &f.graph,
            &f.completeness,
            &f.policy,
            f.use_at_ms,
        );
        assert_eq!(assessment.report.disposition, ActiveIndependenceDisposition::Qualified);
        assert_eq!(assessment.report.relation_status, IndependenceStatus::Separated);
        let active = assessment.active().expect("active qualification");
        assert_eq!(active.left_verification_at_ms(), 100);
        assert_eq!(active.right_verification_at_ms(), 110);
        assert!(!active.grants_physical_authority());
        assert!(!active.universal_independence_established());
    }

    #[test]
    fn stale_lifecycle_assessment_cannot_be_reused_at_later_time() {
        let f = fixture(false, false);
        let assessment = compose_active_independence(
            &f.left_record,
            &f.right_record,
            &f.left_active,
            &f.right_active,
            &f.graph,
            &f.completeness,
            &f.policy,
            f.use_at_ms + 1,
        );
        assert_eq!(assessment.report.disposition, ActiveIndependenceDisposition::Blocked);
        assert!(assessment.active().is_none());
    }

    #[test]
    fn active_capability_cannot_be_rebound_to_another_profile_record() {
        let f = fixture(false, false);
        let assessment = compose_active_independence(
            &f.left_record,
            &f.right_record,
            &f.right_active,
            &f.left_active,
            &f.graph,
            &f.completeness,
            &f.policy,
            f.use_at_ms,
        );
        assert_eq!(assessment.report.disposition, ActiveIndependenceDisposition::Invalid);
        assert!(assessment.active().is_none());
    }

    #[test]
    fn known_common_control_blocks_active_qualification() {
        let f = fixture(true, false);
        let assessment = compose_active_independence(
            &f.left_record,
            &f.right_record,
            &f.left_active,
            &f.right_active,
            &f.graph,
            &f.completeness,
            &f.policy,
            f.use_at_ms,
        );
        assert_eq!(assessment.report.disposition, ActiveIndependenceDisposition::Blocked);
        assert_eq!(assessment.report.relation_status, IndependenceStatus::Correlated);
        assert!(assessment.active().is_none());
    }

    #[test]
    fn incomplete_required_ancestry_blocks_active_qualification() {
        let f = fixture(false, true);
        let assessment = compose_active_independence(
            &f.left_record,
            &f.right_record,
            &f.left_active,
            &f.right_active,
            &f.graph,
            &f.completeness,
            &f.policy,
            f.use_at_ms,
        );
        assert_eq!(assessment.report.disposition, ActiveIndependenceDisposition::Blocked);
        assert_eq!(assessment.report.relation_status, IndependenceStatus::Indeterminate);
        assert!(assessment.active().is_none());
    }

    #[test]
    fn left_right_role_order_is_bound_into_qualification_identity() {
        let f = fixture(false, false);
        let forward = compose_active_independence(
            &f.left_record,
            &f.right_record,
            &f.left_active,
            &f.right_active,
            &f.graph,
            &f.completeness,
            &f.policy,
            f.use_at_ms,
        );
        let reverse = compose_active_independence(
            &f.right_record,
            &f.left_record,
            &f.right_active,
            &f.left_active,
            &f.graph,
            &f.completeness,
            &f.policy,
            f.use_at_ms,
        );
        assert_ne!(
            forward.active().unwrap().qualification_digest(),
            reverse.active().unwrap().qualification_digest()
        );
    }
}
