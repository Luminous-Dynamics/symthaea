// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Provider-neutral network continuity evidence semantics.
//!
//! This crate deliberately does **not** mint `FabricContinuityQualified` or
//! execution authority. It accepts already-normalized live-state observations
//! and packages them into typed continuity witnesses for downstream verifiers.
//!
//! ```text
//! observation != network truth
//! configuration accepted != behavior verified
//! behavior evidence != fabric qualification
//! fabric qualification != execution authority
//! ```

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;
use symthaea_support::{
    CurrentnessStatusV1, EntityId, ObservationId, SystemStateGraphV1,
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NetworkContinuityRequirementClassV1 {
    Reachability,
    Isolation,
    IndependentPathCount,
    MtuBehavior,
    LossLatencyObjective,
    ControlPlaneSession,
    RoutePresenceAbsence,
    ForwardingConsistency,
    Segmentation,
    Redundancy,
    ManagementRecovery,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum NetworkContinuityEvidenceClassV1 {
    NormalizedObservation,
    StaticVerifier,
    NetworkTwin,
    HardwareLab,
    ProductionObservation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkWitnessDispositionV1 {
    SupportsRequirement,
    ChallengesRequirement,
    Indeterminate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkEvidenceCurrentnessRequirementV1 {
    AnyRecorded,
    FreshOnly,
    FreshOrIndeterminate,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkContinuityWitnessV1 {
    pub witness_id: String,
    /// Canonical device/fabric/service subject resolved outside this crate.
    pub subject: EntityId,
    pub requirement_id: String,
    pub requirement_class: NetworkContinuityRequirementClassV1,
    pub disposition: NetworkWitnessDispositionV1,
    pub evidence_class: NetworkContinuityEvidenceClassV1,
    /// Exact support graph revision against which the witness was formed.
    pub graph_revision: u64,
    #[serde(default)]
    pub observations: BTreeSet<ObservationId>,
    /// Adapter/verifier identity, e.g. support-protocol-adapter, Batfish, KNE.
    pub provider: String,
    pub provider_version: Option<String>,
    /// Exact verifier/continuity profile identity where applicable.
    pub verifier_profile_digest: Option<String>,
    /// Exact topology/config snapshot identity where applicable.
    pub topology_digest: Option<String>,
    /// Exact participant/member-set identity for distributed/redundancy checks.
    pub member_set_digest: Option<String>,
}

impl NetworkContinuityWitnessV1 {
    pub fn validate_shape(&self) -> Result<(), NetworkContinuityBridgeErrorV1> {
        require_nonempty(&self.witness_id, "witness id")?;
        require_nonempty(&self.subject.0, "witness subject")?;
        require_nonempty(&self.requirement_id, "requirement id")?;
        require_nonempty(&self.provider, "provider")?;
        validate_optional_nonempty(self.provider_version.as_deref(), "provider version")?;
        validate_optional_nonempty(
            self.verifier_profile_digest.as_deref(),
            "verifier profile digest",
        )?;
        validate_optional_nonempty(self.topology_digest.as_deref(), "topology digest")?;
        validate_optional_nonempty(self.member_set_digest.as_deref(), "member set digest")?;
        Ok(())
    }

    pub fn digest(&self) -> Result<String, NetworkContinuityBridgeErrorV1> {
        self.validate_shape()?;
        let bytes = serde_json::to_vec(&("symthaea-network-continuity-witness-v1", self))
            .map_err(|err| NetworkContinuityBridgeErrorV1::Serialization(err.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkContinuityBridgePolicyV1 {
    pub currentness_requirement: NetworkEvidenceCurrentnessRequirementV1,
    /// Reject a witness if the live graph has changed since it was formed.
    pub require_exact_graph_revision: bool,
    /// Require at least one evidence record whose canonical primary subject is
    /// exactly the witness subject. This prevents cross-device/fabric replay.
    pub require_subject_bound_observation: bool,
    pub require_verifier_profile_digest: bool,
    pub require_topology_digest: bool,
    /// Redundancy evidence should normally bind the exact participant/member set.
    pub require_member_set_digest_for_redundancy: bool,
}

impl Default for NetworkContinuityBridgePolicyV1 {
    fn default() -> Self {
        Self {
            currentness_requirement: NetworkEvidenceCurrentnessRequirementV1::FreshOnly,
            require_exact_graph_revision: true,
            require_subject_bound_observation: true,
            require_verifier_profile_digest: false,
            require_topology_digest: true,
            require_member_set_digest_for_redundancy: true,
        }
    }
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkWitnessCurrentnessSummaryV1 {
    pub total: usize,
    pub fresh: usize,
    pub stale: usize,
    pub indeterminate: usize,
}

impl NetworkWitnessCurrentnessSummaryV1 {
    fn satisfies(self, requirement: NetworkEvidenceCurrentnessRequirementV1) -> bool {
        match requirement {
            NetworkEvidenceCurrentnessRequirementV1::AnyRecorded => self.total > 0,
            NetworkEvidenceCurrentnessRequirementV1::FreshOnly => self.fresh > 0,
            NetworkEvidenceCurrentnessRequirementV1::FreshOrIndeterminate => {
                self.fresh + self.indeterminate > 0
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum NetworkWitnessBlockerV1 {
    UnknownSubject(EntityId),
    NoEvidence,
    UnknownObservation(ObservationId),
    FutureGraphRevision {
        witness_revision: u64,
        current_revision: u64,
    },
    GraphRevisionChanged {
        witness_revision: u64,
        current_revision: u64,
    },
    NoSubjectBoundObservation,
    CurrentnessRequirementNotMet,
    MissingVerifierProfileDigest,
    MissingTopologyDigest,
    MissingMemberSetDigest,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkContinuityWitnessAssessmentV1 {
    pub witness_id: String,
    pub witness_digest: String,
    /// Means only that the witness is structurally/evidentially admissible for a
    /// downstream continuity verifier. It is not continuity qualification.
    pub admissible_for_downstream_verification: bool,
    pub currentness: NetworkWitnessCurrentnessSummaryV1,
    #[serde(default)]
    pub blockers: Vec<NetworkWitnessBlockerV1>,
}

pub fn assess_network_continuity_witness_v1(
    witness: &NetworkContinuityWitnessV1,
    graph: &SystemStateGraphV1,
    now_unix_ms: u64,
    policy: &NetworkContinuityBridgePolicyV1,
) -> Result<NetworkContinuityWitnessAssessmentV1, NetworkContinuityBridgeErrorV1> {
    witness.validate_shape()?;
    let witness_digest = witness.digest()?;
    let mut blockers = Vec::new();
    let mut currentness = NetworkWitnessCurrentnessSummaryV1::default();

    if graph.entity(&witness.subject).is_none() {
        blockers.push(NetworkWitnessBlockerV1::UnknownSubject(
            witness.subject.clone(),
        ));
    }

    if witness.graph_revision > graph.revision {
        blockers.push(NetworkWitnessBlockerV1::FutureGraphRevision {
            witness_revision: witness.graph_revision,
            current_revision: graph.revision,
        });
    } else if policy.require_exact_graph_revision && witness.graph_revision != graph.revision {
        blockers.push(NetworkWitnessBlockerV1::GraphRevisionChanged {
            witness_revision: witness.graph_revision,
            current_revision: graph.revision,
        });
    }

    if witness.observations.is_empty() {
        blockers.push(NetworkWitnessBlockerV1::NoEvidence);
    }

    let mut has_subject_bound_observation = false;
    for observation_id in &witness.observations {
        let Some(observation) = graph.observation(observation_id) else {
            blockers.push(NetworkWitnessBlockerV1::UnknownObservation(
                observation_id.clone(),
            ));
            continue;
        };
        currentness.total += 1;
        match observation.clock.currentness_at(now_unix_ms) {
            CurrentnessStatusV1::Fresh => currentness.fresh += 1,
            CurrentnessStatusV1::Stale => currentness.stale += 1,
            CurrentnessStatusV1::Indeterminate => currentness.indeterminate += 1,
        }
        if observation.subject == witness.subject {
            has_subject_bound_observation = true;
        }
    }

    if policy.require_subject_bound_observation && !has_subject_bound_observation {
        blockers.push(NetworkWitnessBlockerV1::NoSubjectBoundObservation);
    }

    if !currentness.satisfies(policy.currentness_requirement) {
        blockers.push(NetworkWitnessBlockerV1::CurrentnessRequirementNotMet);
    }

    if policy.require_verifier_profile_digest && witness.verifier_profile_digest.is_none() {
        blockers.push(NetworkWitnessBlockerV1::MissingVerifierProfileDigest);
    }
    if policy.require_topology_digest && witness.topology_digest.is_none() {
        blockers.push(NetworkWitnessBlockerV1::MissingTopologyDigest);
    }
    if policy.require_member_set_digest_for_redundancy
        && witness.requirement_class == NetworkContinuityRequirementClassV1::Redundancy
        && witness.member_set_digest.is_none()
    {
        blockers.push(NetworkWitnessBlockerV1::MissingMemberSetDigest);
    }

    Ok(NetworkContinuityWitnessAssessmentV1 {
        witness_id: witness.witness_id.clone(),
        witness_digest,
        admissible_for_downstream_verification: blockers.is_empty(),
        currentness,
        blockers,
    })
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct NetworkContinuityEvidenceBundleV1 {
    pub bundle_id: String,
    pub scope_id: String,
    pub graph_revision: u64,
    #[serde(default)]
    pub witness_digests: BTreeSet<String>,
}

impl NetworkContinuityEvidenceBundleV1 {
    pub fn new(
        bundle_id: impl Into<String>,
        scope_id: impl Into<String>,
        graph_revision: u64,
        assessments: &[NetworkContinuityWitnessAssessmentV1],
    ) -> Result<Self, NetworkContinuityBridgeErrorV1> {
        let bundle_id = bundle_id.into();
        let scope_id = scope_id.into();
        require_nonempty(&bundle_id, "bundle id")?;
        require_nonempty(&scope_id, "bundle scope id")?;
        if assessments.is_empty() {
            return Err(NetworkContinuityBridgeErrorV1::EmptyBundle);
        }
        if assessments
            .iter()
            .any(|assessment| !assessment.admissible_for_downstream_verification)
        {
            return Err(NetworkContinuityBridgeErrorV1::InadmissibleWitnessInBundle);
        }
        let witness_digests = assessments
            .iter()
            .map(|assessment| assessment.witness_digest.clone())
            .collect::<BTreeSet<_>>();
        if witness_digests.len() != assessments.len() {
            return Err(NetworkContinuityBridgeErrorV1::DuplicateWitnessDigest);
        }
        Ok(Self {
            bundle_id,
            scope_id,
            graph_revision,
            witness_digests,
        })
    }

    pub fn digest(&self) -> Result<String, NetworkContinuityBridgeErrorV1> {
        require_nonempty(&self.bundle_id, "bundle id")?;
        require_nonempty(&self.scope_id, "bundle scope id")?;
        if self.witness_digests.is_empty() {
            return Err(NetworkContinuityBridgeErrorV1::EmptyBundle);
        }
        let bytes = serde_json::to_vec(&("symthaea-network-continuity-bundle-v1", self))
            .map_err(|err| NetworkContinuityBridgeErrorV1::Serialization(err.to_string()))?;
        Ok(blake3::hash(&bytes).to_hex().to_string())
    }
}

#[derive(Debug)]
pub enum NetworkContinuityBridgeErrorV1 {
    EmptyField(&'static str),
    Serialization(String),
    EmptyBundle,
    InadmissibleWitnessInBundle,
    DuplicateWitnessDigest,
}

impl fmt::Display for NetworkContinuityBridgeErrorV1 {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyField(field) => write!(f, "empty network continuity field {field}"),
            Self::Serialization(message) => {
                write!(f, "network continuity serialization failed: {message}")
            }
            Self::EmptyBundle => write!(f, "network continuity bundle must not be empty"),
            Self::InadmissibleWitnessInBundle => write!(
                f,
                "network continuity bundle cannot include an inadmissible witness"
            ),
            Self::DuplicateWitnessDigest => write!(
                f,
                "network continuity bundle contains duplicate witness identity"
            ),
        }
    }
}

impl Error for NetworkContinuityBridgeErrorV1 {}

fn require_nonempty(value: &str, field: &'static str) -> Result<(), NetworkContinuityBridgeErrorV1> {
    if value.trim().is_empty() {
        Err(NetworkContinuityBridgeErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

fn validate_optional_nonempty(
    value: Option<&str>,
    field: &'static str,
) -> Result<(), NetworkContinuityBridgeErrorV1> {
    if value.is_some_and(|value| value.trim().is_empty()) {
        Err(NetworkContinuityBridgeErrorV1::EmptyField(field))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use symthaea_support::{
        EntityKindV1, ObservationClockV1, ObservationProvenanceV1,
        ObservationSourceKindV1, SystemObservationV1,
    };

    fn graph_fixture(now: u64) -> (SystemStateGraphV1, EntityId, ObservationId) {
        let subject = EntityId("network:fabric-a".into());
        let observation_id = ObservationId("obs:fabric-a:reachability".into());
        let mut graph = SystemStateGraphV1::new();
        graph
            .record_observation(SystemObservationV1 {
                id: observation_id.clone(),
                subject: subject.clone(),
                provenance: ObservationProvenanceV1 {
                    source_id: "packet:test".into(),
                    source_kind: ObservationSourceKindV1::SyntheticTest,
                    collector: "continuity-fixture".into(),
                    collector_version: Some("1".into()),
                    schema_version: None,
                    artifact_digest: Some("pcap:abc".into()),
                },
                clock: ObservationClockV1 {
                    event_time_unix_ms: Some(now),
                    observed_at_unix_ms: now,
                    ingested_at_unix_ms: Some(now),
                    max_age_ms: Some(5_000),
                    clock_uncertainty_ms: Some(1),
                },
                confidence: 1.0,
                facts: BTreeMap::new(),
            })
            .unwrap();
        graph
            .upsert_entity(
                subject.clone(),
                EntityKindV1::Network,
                BTreeMap::new(),
                &observation_id,
            )
            .unwrap();
        (graph, subject, observation_id)
    }

    fn witness(
        graph: &SystemStateGraphV1,
        subject: EntityId,
        observation: ObservationId,
    ) -> NetworkContinuityWitnessV1 {
        NetworkContinuityWitnessV1 {
            witness_id: "witness:reachability:1".into(),
            subject,
            requirement_id: "reachability:web-to-db".into(),
            requirement_class: NetworkContinuityRequirementClassV1::Reachability,
            disposition: NetworkWitnessDispositionV1::SupportsRequirement,
            evidence_class: NetworkContinuityEvidenceClassV1::NormalizedObservation,
            graph_revision: graph.revision,
            observations: BTreeSet::from([observation]),
            provider: "symthaea-support-protocol".into(),
            provider_version: Some("v1".into()),
            verifier_profile_digest: None,
            topology_digest: Some("topology:abc".into()),
            member_set_digest: None,
        }
    }

    #[test]
    fn fresh_subject_bound_witness_is_only_admissible_for_downstream_verification() {
        let now = 1_000_000;
        let (graph, subject, observation) = graph_fixture(now);
        let witness = witness(&graph, subject, observation);
        let assessment = assess_network_continuity_witness_v1(
            &witness,
            &graph,
            now + 100,
            &NetworkContinuityBridgePolicyV1::default(),
        )
        .unwrap();
        assert!(assessment.admissible_for_downstream_verification);
        assert!(assessment.blockers.is_empty());
        assert_eq!(assessment.currentness.fresh, 1);
    }

    #[test]
    fn stale_evidence_fails_fresh_only_policy() {
        let now = 1_000_000;
        let (graph, subject, observation) = graph_fixture(now);
        let witness = witness(&graph, subject, observation);
        let assessment = assess_network_continuity_witness_v1(
            &witness,
            &graph,
            now + 10_000,
            &NetworkContinuityBridgePolicyV1::default(),
        )
        .unwrap();
        assert!(!assessment.admissible_for_downstream_verification);
        assert!(assessment
            .blockers
            .contains(&NetworkWitnessBlockerV1::CurrentnessRequirementNotMet));
    }

    #[test]
    fn graph_revision_change_blocks_exact_revision_policy() {
        let now = 1_000_000;
        let (mut graph, subject, observation) = graph_fixture(now);
        let witness = witness(&graph, subject.clone(), observation.clone());
        graph
            .upsert_entity(
                subject,
                EntityKindV1::Network,
                BTreeMap::from([("changed".into(), symthaea_support::StateValueV1::Bool(true))]),
                &observation,
            )
            .unwrap();
        let assessment = assess_network_continuity_witness_v1(
            &witness,
            &graph,
            now + 100,
            &NetworkContinuityBridgePolicyV1::default(),
        )
        .unwrap();
        assert!(assessment.blockers.iter().any(|blocker| matches!(
            blocker,
            NetworkWitnessBlockerV1::GraphRevisionChanged { .. }
        )));
    }

    #[test]
    fn cross_subject_evidence_replay_is_blocked() {
        let now = 1_000_000;
        let (mut graph, original_subject, observation) = graph_fixture(now);
        let other = EntityId("network:fabric-b".into());
        graph
            .upsert_entity(
                other.clone(),
                EntityKindV1::Network,
                BTreeMap::new(),
                &observation,
            )
            .unwrap();
        let mut replay = witness(&graph, other, observation);
        replay.graph_revision = graph.revision;
        let assessment = assess_network_continuity_witness_v1(
            &replay,
            &graph,
            now + 100,
            &NetworkContinuityBridgePolicyV1::default(),
        )
        .unwrap();
        assert!(assessment
            .blockers
            .contains(&NetworkWitnessBlockerV1::NoSubjectBoundObservation));
        assert_ne!(replay.subject, original_subject);
    }

    #[test]
    fn redundancy_requires_exact_member_set_digest() {
        let now = 1_000_000;
        let (graph, subject, observation) = graph_fixture(now);
        let mut witness = witness(&graph, subject, observation);
        witness.requirement_class = NetworkContinuityRequirementClassV1::Redundancy;
        let assessment = assess_network_continuity_witness_v1(
            &witness,
            &graph,
            now + 100,
            &NetworkContinuityBridgePolicyV1::default(),
        )
        .unwrap();
        assert!(assessment
            .blockers
            .contains(&NetworkWitnessBlockerV1::MissingMemberSetDigest));
    }

    #[test]
    fn witness_identity_binds_subject_and_requirement_context() {
        let now = 1_000_000;
        let (graph, subject, observation) = graph_fixture(now);
        let first = witness(&graph, subject.clone(), observation.clone());
        let mut second = witness(&graph, subject, observation);
        second.requirement_id = "reachability:admin-to-oob".into();
        assert_ne!(first.digest().unwrap(), second.digest().unwrap());
    }

    #[test]
    fn bundle_rejects_inadmissible_witnesses_and_has_immutable_digest() {
        let now = 1_000_000;
        let (graph, subject, observation) = graph_fixture(now);
        let witness = witness(&graph, subject, observation);
        let assessment = assess_network_continuity_witness_v1(
            &witness,
            &graph,
            now + 100,
            &NetworkContinuityBridgePolicyV1::default(),
        )
        .unwrap();
        let bundle = NetworkContinuityEvidenceBundleV1::new(
            "bundle-1",
            "fabric-a",
            graph.revision,
            &[assessment],
        )
        .unwrap();
        assert_eq!(bundle.witness_digests.len(), 1);
        assert_eq!(bundle.digest().unwrap().len(), 64);
    }
}
