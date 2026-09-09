// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic capability routing for Symthaea extensions.
//!
//! The registry answers "who claims to provide this capability?". Admission
//! answers "which exact provider/package is currently authorized?". Runtime
//! observation answers "how healthy and well-evidenced is it right now?".
//!
//! The router composes those three facts without performing discovery,
//! cryptography, permission granting, loading, or execution.
//!
//! Routing deliberately separates **authorization** from **quality**:
//!
//! - a current [`ActiveAdmission`] for the exact registered manifest is mandatory;
//! - admission capability and minimum trust are hard gates;
//! - trust above the requested minimum does not improve provider quality;
//! - eligible providers are ranked by readiness, evidence, measured reliability,
//!   latency, and finally stable extension ID.
//!
//! Selection is lexicographic rather than a weighted floating-point score so a
//! routing decision is deterministic, explainable, and suitable for evidence
//! recording.

#![deny(unsafe_code)]

use std::cmp::Ordering;
use symthaea_extension_admission::{ActiveAdmission, Sha256Digest};
pub use symthaea_extension_admission::TrustLevel;
use symthaea_extension_core::{
    CapabilityId, EffectClass, ExtensionId, ExtensionManifest, RuntimeKind,
};
use symthaea_extension_registry::ExtensionRegistry;

/// Runtime readiness observed by the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderState {
    Ready,
    Degraded,
    Unavailable,
}

/// Mutable runtime observation used for routing.
///
/// Admission/trust are intentionally absent. Runtime telemetry is not allowed
/// to mint authorization. `evidence_grade` follows Symthaea's E0..E5
/// convention, encoded as 0..=5. `reliability_bps` is measured 0..=10_000.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderObservation {
    pub extension: ExtensionId,
    pub state: ProviderState,
    pub evidence_grade: u8,
    pub reliability_bps: u16,
    /// Unknown latency is represented as `None`; bounded-latency requests fail
    /// closed rather than treating missing telemetry as zero.
    pub estimated_latency_ms: Option<u64>,
}

impl ProviderObservation {
    pub fn validate(&self) -> Result<(), ObservationProblem> {
        if self.evidence_grade > 5 {
            return Err(ObservationProblem::EvidenceGradeOutOfRange(
                self.evidence_grade,
            ));
        }
        if self.reliability_bps > 10_000 {
            return Err(ObservationProblem::ReliabilityOutOfRange(
                self.reliability_bps,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ObservationProblem {
    EvidenceGradeOutOfRange(u8),
    ReliabilityOutOfRange(u16),
}

/// Hard constraints for one routing decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingConstraints {
    pub minimum_trust: TrustLevel,
    pub minimum_evidence_grade: u8,
    pub minimum_reliability_bps: u16,
    pub maximum_latency_ms: Option<u64>,
    pub maximum_memory_bytes: Option<u64>,
    pub maximum_fuel: Option<u64>,
    /// Highest effect class permitted for this invocation.
    pub maximum_effect: EffectClass,
    /// Empty means any runtime is acceptable.
    pub allowed_runtimes: Vec<RuntimeKind>,
    /// Whether degraded providers may remain eligible.
    pub allow_degraded: bool,
}

impl Default for RoutingConstraints {
    fn default() -> Self {
        Self {
            minimum_trust: TrustLevel::Untrusted,
            minimum_evidence_grade: 0,
            minimum_reliability_bps: 0,
            maximum_latency_ms: None,
            maximum_memory_bytes: None,
            maximum_fuel: None,
            maximum_effect: EffectClass::Pure,
            allowed_runtimes: Vec::new(),
            allow_degraded: false,
        }
    }
}

impl RoutingConstraints {
    pub fn validate(&self) -> Result<(), ConstraintProblem> {
        if self.minimum_evidence_grade > 5 {
            return Err(ConstraintProblem::EvidenceGradeOutOfRange(
                self.minimum_evidence_grade,
            ));
        }
        if self.minimum_reliability_bps > 10_000 {
            return Err(ConstraintProblem::ReliabilityOutOfRange(
                self.minimum_reliability_bps,
            ));
        }
        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConstraintProblem {
    EvidenceGradeOutOfRange(u8),
    ReliabilityOutOfRange(u16),
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingRequest {
    pub capability: CapabilityId,
    pub constraints: RoutingConstraints,
}

/// Why one provider was excluded before quality ranking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectionReason {
    MissingAdmission,
    DuplicateAdmission,
    AdmissionManifestMismatch,
    CapabilityNotAdmitted,
    TrustBelowMinimum,
    MissingObservation,
    DuplicateObservation,
    ObservationInvalid(ObservationProblem),
    Unavailable,
    DegradedNotAllowed,
    EvidenceBelowMinimum,
    ReliabilityBelowMinimum,
    LatencyUnknown,
    LatencyAboveMaximum,
    MemoryAboveMaximum,
    FuelAboveMaximum,
    RuntimeNotAllowed,
    EffectNotAllowed,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CandidateAssessment {
    pub extension: ExtensionId,
    pub rejection_reasons: Vec<RejectionReason>,
}

impl CandidateAssessment {
    pub fn eligible(&self) -> bool {
        self.rejection_reasons.is_empty()
    }
}

/// Auditable result of one deterministic routing decision.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoutingDecision {
    pub capability: CapabilityId,
    pub selected: ExtensionId,
    /// Admission lineage used by this exact routing decision.
    pub selected_admission_generation: u64,
    pub selected_manifest_sha256: Sha256Digest,
    pub selected_payload_sha256: Sha256Digest,
    pub selected_policy_sha256: Sha256Digest,
    /// Selected and rejected candidates in stable extension-ID order.
    pub assessments: Vec<CandidateAssessment>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RoutingError {
    InvalidConstraints(ConstraintProblem),
    NoProvider {
        capability: CapabilityId,
    },
    NoEligibleProvider {
        capability: CapabilityId,
        assessments: Vec<CandidateAssessment>,
    },
}

/// Stateless deterministic router.
#[derive(Debug, Default)]
pub struct ExtensionRouter;

impl ExtensionRouter {
    pub fn route(
        registry: &ExtensionRegistry,
        request: &RoutingRequest,
        admissions: &[ActiveAdmission],
        observations: &[ProviderObservation],
    ) -> Result<RoutingDecision, RoutingError> {
        request
            .constraints
            .validate()
            .map_err(RoutingError::InvalidConstraints)?;

        let providers: Vec<_> = registry.providers_for(&request.capability).collect();
        if providers.is_empty() {
            return Err(RoutingError::NoProvider {
                capability: request.capability.clone(),
            });
        }

        let mut assessments = Vec::with_capacity(providers.len());
        let mut eligible = Vec::new();

        for manifest in providers {
            let (assessment, admission, observation) =
                assess_candidate(manifest, request, admissions, observations);
            if assessment.eligible() {
                eligible.push((
                    manifest,
                    admission.expect("eligible provider has admission"),
                    observation.expect("eligible provider has observation"),
                ));
            }
            assessments.push(assessment);
        }

        assessments.sort_by(|a, b| a.extension.cmp(&b.extension));

        if eligible.is_empty() {
            return Err(RoutingError::NoEligibleProvider {
                capability: request.capability.clone(),
                assessments,
            });
        }

        eligible.sort_by(|(manifest_a, _, obs_a), (manifest_b, _, obs_b)| {
            compare_candidates(manifest_a, obs_a, manifest_b, obs_b)
        });

        let (selected_manifest, selected_admission, _) = eligible[0];
        Ok(RoutingDecision {
            capability: request.capability.clone(),
            selected: selected_manifest.id.clone(),
            selected_admission_generation: selected_admission.generation(),
            selected_manifest_sha256: selected_admission.manifest_sha256(),
            selected_payload_sha256: selected_admission.payload_sha256(),
            selected_policy_sha256: selected_admission.policy_sha256(),
            assessments,
        })
    }
}

fn assess_candidate<'a>(
    manifest: &'a ExtensionManifest,
    request: &RoutingRequest,
    admissions: &'a [ActiveAdmission],
    observations: &'a [ProviderObservation],
) -> (
    CandidateAssessment,
    Option<&'a ActiveAdmission>,
    Option<&'a ProviderObservation>,
) {
    let admission_matches: Vec<_> = admissions
        .iter()
        .filter(|admission| admission.extension() == &manifest.id)
        .collect();
    if admission_matches.is_empty() {
        return (
            CandidateAssessment {
                extension: manifest.id.clone(),
                rejection_reasons: vec![RejectionReason::MissingAdmission],
            },
            None,
            None,
        );
    }
    if admission_matches.len() != 1 {
        return (
            CandidateAssessment {
                extension: manifest.id.clone(),
                rejection_reasons: vec![RejectionReason::DuplicateAdmission],
            },
            None,
            None,
        );
    }
    let admission = admission_matches[0];

    let observation_matches: Vec<_> = observations
        .iter()
        .filter(|observation| observation.extension == manifest.id)
        .collect();
    if observation_matches.is_empty() {
        return (
            CandidateAssessment {
                extension: manifest.id.clone(),
                rejection_reasons: vec![RejectionReason::MissingObservation],
            },
            Some(admission),
            None,
        );
    }
    if observation_matches.len() != 1 {
        return (
            CandidateAssessment {
                extension: manifest.id.clone(),
                rejection_reasons: vec![RejectionReason::DuplicateObservation],
            },
            Some(admission),
            None,
        );
    }
    let observation = observation_matches[0];

    let mut reasons = Vec::new();
    if !admission.matches_manifest(manifest) {
        reasons.push(RejectionReason::AdmissionManifestMismatch);
    }
    if !admission.allows_capability(&request.capability) {
        reasons.push(RejectionReason::CapabilityNotAdmitted);
    }
    if admission.trust() < request.constraints.minimum_trust {
        reasons.push(RejectionReason::TrustBelowMinimum);
    }
    if let Err(problem) = observation.validate() {
        reasons.push(RejectionReason::ObservationInvalid(problem));
    }
    match observation.state {
        ProviderState::Unavailable => reasons.push(RejectionReason::Unavailable),
        ProviderState::Degraded if !request.constraints.allow_degraded => {
            reasons.push(RejectionReason::DegradedNotAllowed)
        }
        ProviderState::Ready | ProviderState::Degraded => {}
    }
    if observation.evidence_grade < request.constraints.minimum_evidence_grade {
        reasons.push(RejectionReason::EvidenceBelowMinimum);
    }
    if observation.reliability_bps < request.constraints.minimum_reliability_bps {
        reasons.push(RejectionReason::ReliabilityBelowMinimum);
    }

    if let Some(limit) = request.constraints.maximum_latency_ms {
        match observation.estimated_latency_ms {
            Some(latency) if latency > limit => {
                reasons.push(RejectionReason::LatencyAboveMaximum)
            }
            Some(_) => {}
            None => reasons.push(RejectionReason::LatencyUnknown),
        }
    }

    if request
        .constraints
        .maximum_memory_bytes
        .is_some_and(|limit| manifest.resources.memory_bytes > limit)
    {
        reasons.push(RejectionReason::MemoryAboveMaximum);
    }
    if request
        .constraints
        .maximum_fuel
        .is_some_and(|limit| manifest.resources.fuel > limit)
    {
        reasons.push(RejectionReason::FuelAboveMaximum);
    }
    if !request.constraints.allowed_runtimes.is_empty()
        && !request
            .constraints
            .allowed_runtimes
            .contains(&manifest.runtime)
    {
        reasons.push(RejectionReason::RuntimeNotAllowed);
    }

    let effect = manifest
        .provides
        .iter()
        .find(|capability| capability.id == request.capability)
        .map(|capability| capability.effect)
        .expect("registry provider index must correspond to manifest capability");
    if effect_rank(effect) > effect_rank(request.constraints.maximum_effect) {
        reasons.push(RejectionReason::EffectNotAllowed);
    }

    (
        CandidateAssessment {
            extension: manifest.id.clone(),
            rejection_reasons: reasons,
        },
        Some(admission),
        Some(observation),
    )
}

/// Compare eligible candidates. `Ordering::Less` means `a` is preferred.
///
/// Trust is intentionally absent here. Once candidates satisfy the requested
/// authorization floor, greater signer/operator trust is not evidence of better
/// output quality.
fn compare_candidates(
    manifest_a: &ExtensionManifest,
    obs_a: &ProviderObservation,
    manifest_b: &ExtensionManifest,
    obs_b: &ProviderObservation,
) -> Ordering {
    state_rank(obs_b.state)
        .cmp(&state_rank(obs_a.state))
        .then_with(|| obs_b.evidence_grade.cmp(&obs_a.evidence_grade))
        .then_with(|| obs_b.reliability_bps.cmp(&obs_a.reliability_bps))
        .then_with(|| compare_latency(obs_a.estimated_latency_ms, obs_b.estimated_latency_ms))
        .then_with(|| manifest_a.id.cmp(&manifest_b.id))
}

fn compare_latency(a: Option<u64>, b: Option<u64>) -> Ordering {
    match (a, b) {
        (Some(a), Some(b)) => a.cmp(&b),
        (Some(_), None) => Ordering::Less,
        (None, Some(_)) => Ordering::Greater,
        (None, None) => Ordering::Equal,
    }
}

fn state_rank(state: ProviderState) -> u8 {
    match state {
        ProviderState::Unavailable => 0,
        ProviderState::Degraded => 1,
        ProviderState::Ready => 2,
    }
}

fn effect_rank(effect: EffectClass) -> u8 {
    match effect {
        EffectClass::Pure => 0,
        EffectClass::ReadOnly => 1,
        EffectClass::SideEffecting => 2,
        EffectClass::SafetyCritical => 3,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_extension_admission::{
        AdmissionContext, AdmissionRecord, PrincipalId, Sha256Digest,
    };
    use symthaea_extension_core::{
        AbiVersion, CapabilityDescriptor, ExtensionKind, PermissionSet, ResourceBudget,
    };

    fn digest(byte: u8) -> Sha256Digest {
        Sha256Digest::new([byte; 32])
    }

    fn manifest(
        id: &str,
        capability: &str,
        effect: EffectClass,
        runtime: RuntimeKind,
        memory_bytes: u64,
        fuel: u64,
    ) -> ExtensionManifest {
        ExtensionManifest {
            id: ExtensionId::new(id),
            name: id.into(),
            version: "1.0.0".into(),
            abi: AbiVersion::V1,
            kind: ExtensionKind::Domain,
            runtime,
            description: String::new(),
            provides: vec![CapabilityDescriptor {
                id: CapabilityId::new(capability),
                description: "test capability".into(),
                effect,
            }],
            requires: vec![],
            permissions: PermissionSet::default(),
            resources: ResourceBudget {
                memory_bytes,
                fuel,
                ..ResourceBudget::default()
            },
        }
    }

    fn admission(
        manifest: &ExtensionManifest,
        trust: TrustLevel,
        granted_capabilities: Vec<CapabilityId>,
        generation: u64,
        seed: u8,
    ) -> ActiveAdmission {
        AdmissionRecord::issue(
            manifest.id.clone(),
            manifest.version.clone(),
            digest(seed),
            digest(seed.wrapping_add(1)),
            digest(seed.wrapping_add(2)),
            PrincipalId::new("local:test-authority").unwrap(),
            Some(PrincipalId::new("did:example:test-publisher").unwrap()),
            trust,
            granted_capabilities,
            PermissionSet::default(),
            generation,
        )
        .unwrap()
        .activate(manifest, AdmissionContext::active(generation))
        .unwrap()
    }

    fn observation(
        id: &str,
        evidence_grade: u8,
        reliability_bps: u16,
        latency_ms: Option<u64>,
    ) -> ProviderObservation {
        ProviderObservation {
            extension: ExtensionId::new(id),
            state: ProviderState::Ready,
            evidence_grade,
            reliability_bps,
            estimated_latency_ms: latency_ms,
        }
    }

    fn request(capability: &str) -> RoutingRequest {
        RoutingRequest {
            capability: CapabilityId::new(capability),
            constraints: RoutingConstraints {
                maximum_effect: EffectClass::SafetyCritical,
                allow_degraded: true,
                ..RoutingConstraints::default()
            },
        }
    }

    #[test]
    fn missing_admission_fails_closed() {
        let capability = "science.orbits.propagate";
        let manifest = manifest(
            "org.example.orbit",
            capability,
            EffectClass::Pure,
            RuntimeKind::Wasm,
            1024,
            10_000,
        );
        let mut registry = ExtensionRegistry::new();
        registry.register(manifest).unwrap();
        let observations = vec![observation("org.example.orbit", 5, 10_000, Some(1))];

        let error = ExtensionRouter::route(&registry, &request(capability), &[], &observations)
            .unwrap_err();
        let RoutingError::NoEligibleProvider { assessments, .. } = error else {
            panic!("expected no eligible provider");
        };
        assert_eq!(
            assessments[0].rejection_reasons,
            vec![RejectionReason::MissingAdmission]
        );
    }

    #[test]
    fn capability_admission_is_narrower_than_manifest_claim() {
        let capability = "science.orbits.propagate";
        let other = CapabilityId::new("science.orbits.inspect");
        let mut manifest = manifest(
            "org.example.orbit",
            capability,
            EffectClass::Pure,
            RuntimeKind::Wasm,
            1024,
            10_000,
        );
        manifest.provides.push(CapabilityDescriptor {
            id: other.clone(),
            description: "inspect an orbit".into(),
            effect: EffectClass::ReadOnly,
        });
        let active = admission(&manifest, TrustLevel::Trusted, vec![other], 1, 10);
        let mut registry = ExtensionRegistry::new();
        registry.register(manifest).unwrap();
        let observations = vec![observation("org.example.orbit", 5, 10_000, Some(1))];

        let error = ExtensionRouter::route(
            &registry,
            &request(capability),
            &[active],
            &observations,
        )
        .unwrap_err();
        let RoutingError::NoEligibleProvider { assessments, .. } = error else {
            panic!("expected no eligible provider");
        };
        assert!(assessments[0]
            .rejection_reasons
            .contains(&RejectionReason::CapabilityNotAdmitted));
    }

    #[test]
    fn exact_manifest_binding_rejects_same_id_version_substitution() {
        let capability = "science.example.compute";
        let admitted_manifest = manifest(
            "org.example.compute",
            capability,
            EffectClass::Pure,
            RuntimeKind::Wasm,
            1024,
            10_000,
        );
        let active = admission(
            &admitted_manifest,
            TrustLevel::Trusted,
            vec![CapabilityId::new(capability)],
            1,
            15,
        );
        let mut substituted = admitted_manifest.clone();
        substituted.description = "substituted manifest".into();

        let mut registry = ExtensionRegistry::new();
        registry.register(substituted).unwrap();
        let observations = vec![observation("org.example.compute", 5, 10_000, Some(1))];

        let error = ExtensionRouter::route(
            &registry,
            &request(capability),
            &[active],
            &observations,
        )
        .unwrap_err();
        let RoutingError::NoEligibleProvider { assessments, .. } = error else {
            panic!("expected no eligible provider");
        };
        assert!(assessments[0]
            .rejection_reasons
            .contains(&RejectionReason::AdmissionManifestMismatch));
    }

    #[test]
    fn trust_is_a_floor_not_a_quality_score() {
        let capability = "science.orbits.propagate";
        let fast = manifest(
            "org.example.fast",
            capability,
            EffectClass::Pure,
            RuntimeKind::Wasm,
            1024,
            10_000,
        );
        let more_trusted = manifest(
            "org.example.more-trusted",
            capability,
            EffectClass::Pure,
            RuntimeKind::Wasm,
            1024,
            10_000,
        );
        let mut registry = ExtensionRegistry::new();
        registry.register(fast.clone()).unwrap();
        registry.register(more_trusted.clone()).unwrap();

        let admissions = vec![
            admission(
                &fast,
                TrustLevel::Community,
                vec![CapabilityId::new(capability)],
                1,
                20,
            ),
            admission(
                &more_trusted,
                TrustLevel::Trusted,
                vec![CapabilityId::new(capability)],
                1,
                30,
            ),
        ];
        let observations = vec![
            observation("org.example.fast", 5, 9_999, Some(1)),
            observation("org.example.more-trusted", 3, 9_000, Some(100)),
        ];
        let mut request = request(capability);
        request.constraints.minimum_trust = TrustLevel::Community;

        let decision =
            ExtensionRouter::route(&registry, &request, &admissions, &observations).unwrap();
        assert_eq!(decision.selected, ExtensionId::new("org.example.fast"));
    }

    #[test]
    fn routing_decision_binds_admission_lineage() {
        let capability = "science.example.compute";
        let manifest = manifest(
            "org.example.compute",
            capability,
            EffectClass::Pure,
            RuntimeKind::Wasm,
            1024,
            10_000,
        );
        let active = admission(
            &manifest,
            TrustLevel::Trusted,
            vec![CapabilityId::new(capability)],
            9,
            40,
        );
        let expected_manifest = active.manifest_sha256();
        let expected_payload = active.payload_sha256();
        let expected_policy = active.policy_sha256();
        let mut registry = ExtensionRegistry::new();
        registry.register(manifest).unwrap();
        let observations = vec![observation("org.example.compute", 5, 10_000, Some(3))];

        let decision = ExtensionRouter::route(
            &registry,
            &request(capability),
            &[active],
            &observations,
        )
        .unwrap();
        assert_eq!(decision.selected_admission_generation, 9);
        assert_eq!(decision.selected_manifest_sha256, expected_manifest);
        assert_eq!(decision.selected_payload_sha256, expected_payload);
        assert_eq!(decision.selected_policy_sha256, expected_policy);
    }

    #[test]
    fn duplicate_admissions_fail_closed() {
        let capability = "science.example.compute";
        let manifest = manifest(
            "org.example.compute",
            capability,
            EffectClass::Pure,
            RuntimeKind::Wasm,
            1024,
            10_000,
        );
        let first = admission(
            &manifest,
            TrustLevel::Trusted,
            vec![CapabilityId::new(capability)],
            1,
            50,
        );
        let second = admission(
            &manifest,
            TrustLevel::Trusted,
            vec![CapabilityId::new(capability)],
            1,
            51,
        );
        let mut registry = ExtensionRegistry::new();
        registry.register(manifest).unwrap();
        let observations = vec![observation("org.example.compute", 5, 10_000, Some(1))];

        let error = ExtensionRouter::route(
            &registry,
            &request(capability),
            &[first, second],
            &observations,
        )
        .unwrap_err();
        let RoutingError::NoEligibleProvider { assessments, .. } = error else {
            panic!("expected no eligible provider");
        };
        assert_eq!(
            assessments[0].rejection_reasons,
            vec![RejectionReason::DuplicateAdmission]
        );
    }

    #[test]
    fn bounded_latency_rejects_unknown_latency() {
        let capability = "science.example.compute";
        let manifest = manifest(
            "org.example.compute",
            capability,
            EffectClass::Pure,
            RuntimeKind::Wasm,
            1024,
            10_000,
        );
        let active = admission(
            &manifest,
            TrustLevel::Trusted,
            vec![CapabilityId::new(capability)],
            1,
            60,
        );
        let mut registry = ExtensionRegistry::new();
        registry.register(manifest).unwrap();
        let observations = vec![observation("org.example.compute", 5, 10_000, None)];
        let mut request = request(capability);
        request.constraints.maximum_latency_ms = Some(10);

        let error = ExtensionRouter::route(&registry, &request, &[active], &observations)
            .unwrap_err();
        let RoutingError::NoEligibleProvider { assessments, .. } = error else {
            panic!("expected no eligible provider");
        };
        assert!(assessments[0]
            .rejection_reasons
            .contains(&RejectionReason::LatencyUnknown));
    }

    #[test]
    fn complete_quality_tie_resolves_by_extension_id() {
        let capability = "science.example.compute";
        let a = manifest(
            "org.example.a",
            capability,
            EffectClass::Pure,
            RuntimeKind::Wasm,
            1024,
            10_000,
        );
        let b = manifest(
            "org.example.b",
            capability,
            EffectClass::Pure,
            RuntimeKind::Wasm,
            1024,
            10_000,
        );
        let mut registry = ExtensionRegistry::new();
        registry.register(b.clone()).unwrap();
        registry.register(a.clone()).unwrap();
        let admissions = vec![
            admission(
                &b,
                TrustLevel::Trusted,
                vec![CapabilityId::new(capability)],
                1,
                70,
            ),
            admission(
                &a,
                TrustLevel::Trusted,
                vec![CapabilityId::new(capability)],
                1,
                80,
            ),
        ];
        let observations = vec![
            observation("org.example.b", 4, 9_000, Some(5)),
            observation("org.example.a", 4, 9_000, Some(5)),
        ];

        let decision = ExtensionRouter::route(
            &registry,
            &request(capability),
            &admissions,
            &observations,
        )
        .unwrap();
        assert_eq!(decision.selected, ExtensionId::new("org.example.a"));
    }
}
