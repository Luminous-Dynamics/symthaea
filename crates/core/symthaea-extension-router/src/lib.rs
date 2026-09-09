// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Deterministic capability routing for Symthaea extensions.
//!
//! The registry answers "who claims to provide this capability?". This crate
//! answers the narrower policy question "which already-admitted provider best
//! satisfies this invocation's hard constraints?" It deliberately does not
//! load plugins, grant permissions, verify signatures, or execute code.
//!
//! Selection is intentionally lexicographic rather than a weighted floating-
//! point score. Hard constraints are applied first. Eligible candidates are
//! then ranked by runtime state, trust, evidence, reliability, latency,
//! resource demand, and finally stable extension ID. This makes routing
//! explainable, deterministic, and safe to record in an evidence trail.

#![deny(unsafe_code)]

use std::cmp::Ordering;
use symthaea_extension_core::{
    CapabilityId, EffectClass, ExtensionId, ExtensionManifest, RuntimeKind,
};
use symthaea_extension_registry::ExtensionRegistry;

/// Runtime readiness of a provider as observed by the host.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProviderState {
    Ready,
    Degraded,
    Unavailable,
}

/// Host-assigned authorization/trust level.
///
/// Trust is intentionally not declared by the extension manifest. The host
/// derives it from installation policy, signer authorization, provenance, and
/// local operator decisions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum TrustLevel {
    Untrusted,
    Community,
    Trusted,
    Privileged,
}

/// Mutable host observation used for routing.
///
/// `evidence_grade` follows Symthaea's existing E0..E5 convention, encoded as
/// 0..=5. `reliability_bps` is a measured 0..=10_000 basis-point rate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderObservation {
    pub extension: ExtensionId,
    /// True only after the host's independent admission/trust checks pass.
    pub admitted: bool,
    pub state: ProviderState,
    pub trust: TrustLevel,
    pub evidence_grade: u8,
    pub reliability_bps: u16,
    pub estimated_latency_ms: u64,
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
    /// Whether a degraded provider may be used when no ready provider wins.
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

/// Why one provider was excluded before ranking.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RejectionReason {
    MissingObservation,
    ObservationInvalid(ObservationProblem),
    NotAdmitted,
    Unavailable,
    DegradedNotAllowed,
    TrustBelowMinimum,
    EvidenceBelowMinimum,
    ReliabilityBelowMinimum,
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
    /// Includes selected and rejected providers in stable extension-ID order.
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
            let (assessment, observation) = assess_candidate(manifest, request, observations);
            if assessment.eligible() {
                // Eligible implies a present, valid observation.
                eligible.push((manifest, observation.expect("eligible provider has observation")));
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

        eligible.sort_by(|(manifest_a, obs_a), (manifest_b, obs_b)| {
            compare_candidates(manifest_a, obs_a, manifest_b, obs_b)
        });

        Ok(RoutingDecision {
            capability: request.capability.clone(),
            selected: eligible[0].0.id.clone(),
            assessments,
        })
    }
}

fn assess_candidate<'a>(
    manifest: &'a ExtensionManifest,
    request: &RoutingRequest,
    observations: &'a [ProviderObservation],
) -> (CandidateAssessment, Option<&'a ProviderObservation>) {
    let mut reasons = Vec::new();
    let observation = observations
        .iter()
        .find(|observation| observation.extension == manifest.id);

    let Some(observation) = observation else {
        return (
            CandidateAssessment {
                extension: manifest.id.clone(),
                rejection_reasons: vec![RejectionReason::MissingObservation],
            },
            None,
        );
    };

    if let Err(problem) = observation.validate() {
        reasons.push(RejectionReason::ObservationInvalid(problem));
    }
    if !observation.admitted {
        reasons.push(RejectionReason::NotAdmitted);
    }
    match observation.state {
        ProviderState::Unavailable => reasons.push(RejectionReason::Unavailable),
        ProviderState::Degraded if !request.constraints.allow_degraded => {
            reasons.push(RejectionReason::DegradedNotAllowed)
        }
        ProviderState::Ready | ProviderState::Degraded => {}
    }
    if observation.trust < request.constraints.minimum_trust {
        reasons.push(RejectionReason::TrustBelowMinimum);
    }
    if observation.evidence_grade < request.constraints.minimum_evidence_grade {
        reasons.push(RejectionReason::EvidenceBelowMinimum);
    }
    if observation.reliability_bps < request.constraints.minimum_reliability_bps {
        reasons.push(RejectionReason::ReliabilityBelowMinimum);
    }
    if request
        .constraints
        .maximum_latency_ms
        .is_some_and(|limit| observation.estimated_latency_ms > limit)
    {
        reasons.push(RejectionReason::LatencyAboveMaximum);
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
        Some(observation),
    )
}

/// Compare eligible candidates. `Ordering::Less` means `a` is preferred.
fn compare_candidates(
    manifest_a: &ExtensionManifest,
    obs_a: &ProviderObservation,
    manifest_b: &ExtensionManifest,
    obs_b: &ProviderObservation,
) -> Ordering {
    state_rank(obs_b.state)
        .cmp(&state_rank(obs_a.state))
        .then_with(|| obs_b.trust.cmp(&obs_a.trust))
        .then_with(|| obs_b.evidence_grade.cmp(&obs_a.evidence_grade))
        .then_with(|| obs_b.reliability_bps.cmp(&obs_a.reliability_bps))
        .then_with(|| obs_a.estimated_latency_ms.cmp(&obs_b.estimated_latency_ms))
        .then_with(|| manifest_a.resources.memory_bytes.cmp(&manifest_b.resources.memory_bytes))
        .then_with(|| manifest_a.resources.fuel.cmp(&manifest_b.resources.fuel))
        .then_with(|| manifest_a.id.cmp(&manifest_b.id))
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
    use symthaea_extension_core::{
        AbiVersion, CapabilityDescriptor, ExtensionKind, PermissionSet, ResourceBudget,
    };

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

    fn observation(
        id: &str,
        trust: TrustLevel,
        evidence_grade: u8,
        reliability_bps: u16,
        latency_ms: u64,
    ) -> ProviderObservation {
        ProviderObservation {
            extension: ExtensionId::new(id),
            admitted: true,
            state: ProviderState::Ready,
            trust,
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
    fn higher_trust_wins_before_latency() {
        let capability = "science.orbits.propagate";
        let mut registry = ExtensionRegistry::new();
        registry
            .register(manifest(
                "org.example.fast",
                capability,
                EffectClass::Pure,
                RuntimeKind::Wasm,
                1024,
                10_000,
            ))
            .unwrap();
        registry
            .register(manifest(
                "org.example.trusted",
                capability,
                EffectClass::Pure,
                RuntimeKind::Wasm,
                1024,
                10_000,
            ))
            .unwrap();

        let observations = vec![
            observation("org.example.fast", TrustLevel::Community, 5, 9_999, 1),
            observation("org.example.trusted", TrustLevel::Trusted, 3, 9_000, 100),
        ];

        let decision = ExtensionRouter::route(&registry, &request(capability), &observations).unwrap();
        assert_eq!(decision.selected, ExtensionId::new("org.example.trusted"));
    }

    #[test]
    fn hard_constraints_reject_before_ranking() {
        let capability = "robotics.motion.command";
        let mut registry = ExtensionRegistry::new();
        registry
            .register(manifest(
                "org.example.actuator",
                capability,
                EffectClass::SafetyCritical,
                RuntimeKind::Wasm,
                128 * 1024 * 1024,
                100_000_000,
            ))
            .unwrap();

        let observations = vec![observation(
            "org.example.actuator",
            TrustLevel::Privileged,
            5,
            10_000,
            1,
        )];
        let mut request = request(capability);
        request.constraints.maximum_effect = EffectClass::ReadOnly;
        request.constraints.maximum_memory_bytes = Some(64 * 1024 * 1024);
        request.constraints.maximum_fuel = Some(50_000_000);

        let err = ExtensionRouter::route(&registry, &request, &observations).unwrap_err();
        let RoutingError::NoEligibleProvider { assessments, .. } = err else {
            panic!("expected no eligible provider");
        };
        let reasons = &assessments[0].rejection_reasons;
        assert!(reasons.contains(&RejectionReason::EffectNotAllowed));
        assert!(reasons.contains(&RejectionReason::MemoryAboveMaximum));
        assert!(reasons.contains(&RejectionReason::FuelAboveMaximum));
    }

    #[test]
    fn unadmitted_provider_never_routes() {
        let capability = "science.example.compute";
        let mut registry = ExtensionRegistry::new();
        registry
            .register(manifest(
                "org.example.untrusted",
                capability,
                EffectClass::Pure,
                RuntimeKind::Wasm,
                1024,
                100,
            ))
            .unwrap();

        let mut obs = observation("org.example.untrusted", TrustLevel::Privileged, 5, 10_000, 1);
        obs.admitted = false;

        let err = ExtensionRouter::route(&registry, &request(capability), &[obs]).unwrap_err();
        let RoutingError::NoEligibleProvider { assessments, .. } = err else {
            panic!("expected no eligible provider");
        };
        assert_eq!(
            assessments[0].rejection_reasons,
            vec![RejectionReason::NotAdmitted]
        );
    }

    #[test]
    fn stable_id_breaks_complete_ties() {
        let capability = "science.example.compute";
        let mut registry = ExtensionRegistry::new();
        for id in ["org.example.beta", "org.example.alpha"] {
            registry
                .register(manifest(
                    id,
                    capability,
                    EffectClass::Pure,
                    RuntimeKind::Wasm,
                    1024,
                    100,
                ))
                .unwrap();
        }
        let observations = vec![
            observation("org.example.beta", TrustLevel::Trusted, 4, 9_900, 10),
            observation("org.example.alpha", TrustLevel::Trusted, 4, 9_900, 10),
        ];

        let decision = ExtensionRouter::route(&registry, &request(capability), &observations).unwrap();
        assert_eq!(decision.selected, ExtensionId::new("org.example.alpha"));
    }

    #[test]
    fn missing_observation_is_explicitly_rejected() {
        let capability = "science.example.compute";
        let mut registry = ExtensionRegistry::new();
        registry
            .register(manifest(
                "org.example.provider",
                capability,
                EffectClass::Pure,
                RuntimeKind::Native,
                1024,
                100,
            ))
            .unwrap();

        let err = ExtensionRouter::route(&registry, &request(capability), &[]).unwrap_err();
        let RoutingError::NoEligibleProvider { assessments, .. } = err else {
            panic!("expected no eligible provider");
        };
        assert_eq!(
            assessments[0].rejection_reasons,
            vec![RejectionReason::MissingObservation]
        );
    }
}
