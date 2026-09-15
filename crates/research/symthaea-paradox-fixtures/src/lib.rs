// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! PARADOX-002A: deterministic matched structured-inconsistency fixtures.
//!
//! This crate qualifies experimental fixtures and manipulation checks only. It
//! does not execute production cognition, metacognition, action selection, or
//! learning, and it does not measure or classify consciousness.

use symthaea_structured_inconsistency::{
    EvidencePolarity, EvidenceSupport, InconsistencyKind, ObservationError, ResolutionState,
    StructuredInconsistencyInput, StructuredInconsistencyObservatory, StructuredInconsistencyReport,
};

/// Exact PARADOX-001 subject on which this fixture layer is stacked.
pub const PARADOX001_QUALIFIED_SHA: &str = "3f8e53e4d6b89895dc5097b2dccfe4f3868d53f9";

/// Frozen confirmatory seeds from PARADOX-002 preregistration (#3168).
pub const CONFIRMATORY_SEEDS: [u64; 16] = [
    11, 29, 47, 71, 101, 149, 197, 257, 331, 419, 521, 631, 751, 887, 1021, 1171,
];

/// Development seeds are deliberately disjoint from the confirmatory set.
pub const DEVELOPMENT_SEEDS: [u64; 4] = [2, 3, 5, 7];

/// Frozen trial count for stateless synthetic confirmatory fixture qualification.
pub const CONFIRMATORY_TRIALS_PER_CONDITION: usize = 64;

/// The seven preregistered PARADOX-002 condition families.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Condition {
    CoherentControl = 0,
    SurpriseOnly = 1,
    TransientConflict = 2,
    PersistentResolvable = 3,
    PersistentIrreducible = 4,
    SelfReferentialConflict = 5,
    OntologyFailure = 6,
}

/// Stable condition iteration order used by qualification censuses.
pub const ALL_CONDITIONS: [Condition; 7] = [
    Condition::CoherentControl,
    Condition::SurpriseOnly,
    Condition::TransientConflict,
    Condition::PersistentResolvable,
    Condition::PersistentIrreducible,
    Condition::SelfReferentialConflict,
    Condition::OntologyFailure,
];

/// Proposition polarity retained without cancellation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ClaimPolarity {
    Proposition = 1,
    Negation = 2,
}

impl ClaimPolarity {
    const fn opposite(self) -> Self {
        match self {
            Self::Proposition => Self::Negation,
            Self::Negation => Self::Proposition,
        }
    }
}

/// Public role of an observation presented to a later cognitive adapter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum EventRole {
    WorldEvidence = 1,
    SelfPrediction = 2,
    ContextDisclosure = 3,
}

/// Opaque context token. A later adapter may observe explicitly disclosed
/// context tokens, but hidden oracle context never appears in `AgentView`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ContextToken(u16);

impl ContextToken {
    pub const fn raw(self) -> u16 {
        self.0
    }
}

/// Frozen resource envelope shared by every C0-C6 fixture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ResourceBudget {
    pub observation_count: u8,
    pub claim_event_count: u8,
    pub auxiliary_event_count: u8,
    pub candidate_count: u8,
    pub update_steps: u8,
    pub nominal_search_budget: u8,
    pub timing_slots: u8,
    pub source_count: u8,
}

pub const MATCHED_RESOURCE_BUDGET: ResourceBudget = ResourceBudget {
    observation_count: 4,
    claim_event_count: 3,
    auxiliary_event_count: 1,
    candidate_count: 2,
    update_steps: 4,
    nominal_search_budget: 8,
    timing_slots: 4,
    source_count: 2,
};

/// A single publicly visible evidence/observation event.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EvidenceEvent {
    slot: u8,
    source_id: u16,
    fault_domain_id: u16,
    polarity: Option<ClaimPolarity>,
    reliability: f64,
    cue: u16,
    role: EventRole,
    supersedes_slot: Option<u8>,
    caused_by_self_prediction: bool,
    visible_context: Option<ContextToken>,
}

impl EvidenceEvent {
    #[allow(clippy::too_many_arguments)]
    fn claim(
        slot: u8,
        source_id: u16,
        fault_domain_id: u16,
        polarity: ClaimPolarity,
        reliability: f64,
        cue: u16,
        role: EventRole,
        supersedes_slot: Option<u8>,
        caused_by_self_prediction: bool,
    ) -> Result<Self, FixtureError> {
        validate_reliability(reliability)?;
        Ok(Self {
            slot,
            source_id,
            fault_domain_id,
            polarity: Some(polarity),
            reliability,
            cue,
            role,
            supersedes_slot,
            caused_by_self_prediction,
            visible_context: None,
        })
    }

    fn auxiliary(
        slot: u8,
        source_id: u16,
        fault_domain_id: u16,
        reliability: f64,
        cue: u16,
        visible_context: Option<ContextToken>,
    ) -> Result<Self, FixtureError> {
        validate_reliability(reliability)?;
        Ok(Self {
            slot,
            source_id,
            fault_domain_id,
            polarity: None,
            reliability,
            cue,
            role: EventRole::ContextDisclosure,
            supersedes_slot: None,
            caused_by_self_prediction: false,
            visible_context,
        })
    }

    pub const fn slot(&self) -> u8 {
        self.slot
    }

    pub const fn source_id(&self) -> u16 {
        self.source_id
    }

    pub const fn fault_domain_id(&self) -> u16 {
        self.fault_domain_id
    }

    pub const fn polarity(&self) -> Option<ClaimPolarity> {
        self.polarity
    }

    pub const fn reliability(&self) -> f64 {
        self.reliability
    }

    pub const fn cue(&self) -> u16 {
        self.cue
    }

    pub const fn role(&self) -> EventRole {
        self.role
    }

    pub const fn supersedes_slot(&self) -> Option<u8> {
        self.supersedes_slot
    }

    pub const fn caused_by_self_prediction(&self) -> bool {
        self.caused_by_self_prediction
    }

    pub const fn visible_context(&self) -> Option<ContextToken> {
        self.visible_context
    }
}

/// The only view a later PARADOX-002B cognitive adapter is permitted to consume.
/// It intentionally contains no condition label, expected response, or oracle truth.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentView {
    prior_expectation: ClaimPolarity,
    events: [EvidenceEvent; 4],
    query_cue: u16,
    budget: ResourceBudget,
}

impl AgentView {
    pub const fn prior_expectation(&self) -> ClaimPolarity {
        self.prior_expectation
    }

    pub const fn events(&self) -> &[EvidenceEvent; 4] {
        &self.events
    }

    pub const fn query_cue(&self) -> u16 {
        self.query_cue
    }

    pub const fn budget(&self) -> ResourceBudget {
        self.budget
    }

    /// Canonical deterministic bytes for exact fixture binding.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(160);
        push_u8(&mut out, self.prior_expectation as u8);
        push_u16(&mut out, self.query_cue);
        push_budget(&mut out, self.budget);
        for event in self.events {
            push_event(&mut out, event);
        }
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct OracleTruth {
    claim_contexts: [Option<ContextToken>; 4],
    latent_context_is_causal: bool,
    context_dimension_available: bool,
    target_context: Option<ContextToken>,
}

/// Full experimental fixture. Oracle truth remains private; later agent code must
/// consume only `agent_view()`.
#[derive(Debug, Clone, PartialEq)]
pub struct Fixture {
    condition: Condition,
    seed: u64,
    trial_index: usize,
    agent_view: AgentView,
    truth: OracleTruth,
}

impl Fixture {
    pub const fn condition(&self) -> Condition {
        self.condition
    }

    pub const fn seed(&self) -> u64 {
        self.seed
    }

    pub const fn trial_index(&self) -> usize {
        self.trial_index
    }

    pub const fn agent_view(&self) -> &AgentView {
        &self.agent_view
    }

    /// Canonical bytes include experiment identity and hidden oracle truth. These
    /// bytes are for evidence binding, not for cognitive-agent input.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(224);
        push_u8(&mut out, self.condition as u8);
        push_u64(&mut out, self.seed);
        push_u64(&mut out, self.trial_index as u64);
        let agent = self.agent_view.canonical_bytes();
        push_u64(&mut out, agent.len() as u64);
        out.extend_from_slice(&agent);
        for context in self.truth.claim_contexts {
            push_optional_context(&mut out, context);
        }
        push_u8(&mut out, u8::from(self.truth.latent_context_is_causal));
        push_u8(&mut out, u8::from(self.truth.context_dimension_available));
        push_optional_context(&mut out, self.truth.target_context);
        out
    }
}

/// Correct response class fixed by fixture semantics, not by agent behavior.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ExpectedResponse {
    Commit = 1,
    CommitConditionally = 2,
    AbstainPreservePlurality = 3,
    ReflexiveUpdate = 4,
    ReviseContext = 5,
}

/// Oracle-level description of why a fixture is or is not resolvable.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum ResolutionClass {
    CurrentRepresentation = 1,
    ExplicitContext = 2,
    IrreducibleUnderCurrentEvidence = 3,
    Reflexive = 4,
    RequiresRepresentationRevision = 5,
}

/// Independent manipulation report. PARADOX-001 observatory output is embedded
/// for measurement only and never feeds the fixture generator.
#[derive(Debug, Clone, PartialEq)]
pub struct OracleReport {
    pub external_surprise: f64,
    pub internal_disagreement: bool,
    pub conflict_persistence: f64,
    pub independent_conflict_sources: bool,
    pub self_referential: bool,
    pub explicit_context_resolution: bool,
    pub ontology_failure: bool,
    pub irreducible: bool,
    pub conflict_slots: u8,
    pub expected_response: ExpectedResponse,
    pub resolution_class: ResolutionClass,
    pub target_polarity: Option<ClaimPolarity>,
    pub observatory: StructuredInconsistencyReport,
}

impl OracleReport {
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(128);
        push_u64(&mut out, self.external_surprise.to_bits());
        push_u8(&mut out, u8::from(self.internal_disagreement));
        push_u64(&mut out, self.conflict_persistence.to_bits());
        push_u8(&mut out, u8::from(self.independent_conflict_sources));
        push_u8(&mut out, u8::from(self.self_referential));
        push_u8(&mut out, u8::from(self.explicit_context_resolution));
        push_u8(&mut out, u8::from(self.ontology_failure));
        push_u8(&mut out, u8::from(self.irreducible));
        push_u8(&mut out, self.conflict_slots);
        push_u8(&mut out, self.expected_response as u8);
        push_u8(&mut out, self.resolution_class as u8);
        push_optional_polarity(&mut out, self.target_polarity);
        push_u64(&mut out, self.observatory.external_surprise.to_bits());
        push_u64(&mut out, self.observatory.internal_disagreement.to_bits());
        push_u64(&mut out, self.observatory.uncertainty.to_bits());
        push_u64(&mut out, self.observatory.persistence.to_bits());
        push_u64(
            &mut out,
            self.observatory.self_referential_relevance.to_bits(),
        );
        push_u8(&mut out, self.observatory.evidence_polarity as u8);
        push_u64(&mut out, self.observatory.integration_coherence.to_bits());
        push_u64(&mut out, self.observatory.conflict_load.to_bits());
        push_u64(
            &mut out,
            self.observatory.candidate_recruitment_index.to_bits(),
        );
        out
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum FixtureError {
    NonFiniteReliability,
    ReliabilityOutOfRange,
}

impl std::fmt::Display for FixtureError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NonFiniteReliability => write!(f, "event reliability must be finite"),
            Self::ReliabilityOutOfRange => {
                write!(f, "event reliability must be within [0, 1]")
            }
        }
    }
}

impl std::error::Error for FixtureError {}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QualificationError {
    Observatory(ObservationError),
    ContractViolation(&'static str),
}

impl From<ObservationError> for QualificationError {
    fn from(value: ObservationError) -> Self {
        Self::Observatory(value)
    }
}

impl std::fmt::Display for QualificationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Observatory(error) => write!(f, "PARADOX-001 observatory rejected fixture: {error}"),
            Self::ContractViolation(message) => write!(f, "fixture contract violation: {message}"),
        }
    }
}

impl std::error::Error for QualificationError {}

/// Stateless deterministic fixture generator.
pub struct FixtureGenerator;

impl FixtureGenerator {
    pub fn generate(
        condition: Condition,
        seed: u64,
        trial_index: usize,
    ) -> Result<Fixture, FixtureError> {
        let seed_mix = mix64(seed ^ 0xA076_1D64_78BD_642F);
        let trial_mix = mix64(seed ^ (trial_index as u64).wrapping_mul(0xE703_7ED1_A0B4_28DB));
        let orientation = if seed_mix & 1 == 0 {
            ClaimPolarity::Proposition
        } else {
            ClaimPolarity::Negation
        };
        let opposite = orientation.opposite();

        let source_a = nonzero_u16(seed_mix as u16);
        let source_b = source_a ^ 0xA5A5;
        let fault_a = nonzero_u16((seed_mix >> 16) as u16);
        let fault_b = fault_a ^ 0x5A5A;
        let cue_a = nonzero_u16((mix64(seed ^ 0x8EBC_6AF0_9C88_C6E3) >> 16) as u16);
        let cue_b = cue_a ^ 0x8001;
        let irrelevant_cue = nonzero_u16((trial_mix >> 32) as u16) ^ 0x4001;
        let context_a = ContextToken(nonzero_u16((seed_mix >> 32) as u16));
        let context_b = ContextToken(context_a.0 ^ 0xC003);
        let target_is_a = trial_mix & 1 == 0;
        let target_context = if target_is_a { context_a } else { context_b };
        let query_cue = if target_is_a { cue_a } else { cue_b };

        let common_aux = |visible_context| {
            EvidenceEvent::auxiliary(
                3,
                source_b,
                fault_b,
                0.9,
                irrelevant_cue,
                visible_context,
            )
        };

        let (prior_expectation, events, truth) = match condition {
            Condition::CoherentControl => (
                orientation,
                [
                    EvidenceEvent::claim(
                        0,
                        source_a,
                        fault_a,
                        orientation,
                        0.9,
                        cue_a,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        1,
                        source_b,
                        fault_b,
                        orientation,
                        0.9,
                        cue_b,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        2,
                        source_a,
                        fault_a,
                        orientation,
                        0.9,
                        cue_a,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    common_aux(None)?,
                ],
                OracleTruth {
                    claim_contexts: [None, None, None, None],
                    latent_context_is_causal: false,
                    context_dimension_available: true,
                    target_context: None,
                },
            ),
            Condition::SurpriseOnly => (
                opposite,
                [
                    EvidenceEvent::claim(
                        0,
                        source_a,
                        fault_a,
                        orientation,
                        0.9,
                        cue_a,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        1,
                        source_b,
                        fault_b,
                        orientation,
                        0.9,
                        cue_b,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        2,
                        source_a,
                        fault_a,
                        orientation,
                        0.9,
                        cue_a,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    common_aux(None)?,
                ],
                OracleTruth {
                    claim_contexts: [None, None, None, None],
                    latent_context_is_causal: false,
                    context_dimension_available: true,
                    target_context: None,
                },
            ),
            Condition::TransientConflict => (
                orientation,
                [
                    EvidenceEvent::claim(
                        0,
                        source_a,
                        fault_a,
                        orientation,
                        0.9,
                        cue_a,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        1,
                        source_b,
                        fault_b,
                        opposite,
                        0.9,
                        cue_b,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        2,
                        source_b,
                        fault_b,
                        orientation,
                        0.9,
                        cue_b,
                        EventRole::WorldEvidence,
                        Some(1),
                        false,
                    )?,
                    common_aux(None)?,
                ],
                OracleTruth {
                    claim_contexts: [None, None, None, None],
                    latent_context_is_causal: false,
                    context_dimension_available: true,
                    target_context: None,
                },
            ),
            Condition::PersistentResolvable => (
                orientation,
                [
                    EvidenceEvent::claim(
                        0,
                        source_a,
                        fault_a,
                        orientation,
                        0.9,
                        cue_a,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        1,
                        source_b,
                        fault_b,
                        opposite,
                        0.9,
                        cue_b,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        2,
                        source_a,
                        fault_a,
                        orientation,
                        0.9,
                        cue_a,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    common_aux(Some(target_context))?,
                ],
                OracleTruth {
                    claim_contexts: [Some(context_a), Some(context_b), Some(context_a), None],
                    latent_context_is_causal: true,
                    context_dimension_available: true,
                    target_context: Some(target_context),
                },
            ),
            Condition::PersistentIrreducible => (
                orientation,
                [
                    EvidenceEvent::claim(
                        0,
                        source_a,
                        fault_a,
                        orientation,
                        0.9,
                        cue_a,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        1,
                        source_b,
                        fault_b,
                        opposite,
                        0.9,
                        cue_b,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        2,
                        source_a,
                        fault_a,
                        orientation,
                        0.9,
                        cue_b,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    common_aux(None)?,
                ],
                OracleTruth {
                    claim_contexts: [None, None, None, None],
                    latent_context_is_causal: false,
                    context_dimension_available: true,
                    target_context: None,
                },
            ),
            Condition::SelfReferentialConflict => (
                orientation,
                [
                    EvidenceEvent::claim(
                        0,
                        source_a,
                        fault_a,
                        orientation,
                        0.9,
                        cue_a,
                        EventRole::SelfPrediction,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        1,
                        source_b,
                        fault_b,
                        opposite,
                        0.9,
                        cue_b,
                        EventRole::WorldEvidence,
                        None,
                        true,
                    )?,
                    EvidenceEvent::claim(
                        2,
                        source_b,
                        fault_b,
                        opposite,
                        0.9,
                        cue_b,
                        EventRole::WorldEvidence,
                        None,
                        true,
                    )?,
                    common_aux(None)?,
                ],
                OracleTruth {
                    claim_contexts: [None, None, None, None],
                    latent_context_is_causal: false,
                    context_dimension_available: true,
                    target_context: None,
                },
            ),
            Condition::OntologyFailure => (
                orientation,
                [
                    EvidenceEvent::claim(
                        0,
                        source_a,
                        fault_a,
                        orientation,
                        0.9,
                        cue_a,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        1,
                        source_b,
                        fault_b,
                        opposite,
                        0.9,
                        cue_b,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::claim(
                        2,
                        source_a,
                        fault_a,
                        orientation,
                        0.9,
                        cue_a,
                        EventRole::WorldEvidence,
                        None,
                        false,
                    )?,
                    EvidenceEvent::auxiliary(
                        3,
                        source_b,
                        fault_b,
                        0.9,
                        query_cue,
                        None,
                    )?,
                ],
                OracleTruth {
                    claim_contexts: [Some(context_a), Some(context_b), Some(context_a), None],
                    latent_context_is_causal: true,
                    context_dimension_available: false,
                    target_context: Some(target_context),
                },
            ),
        };

        Ok(Fixture {
            condition,
            seed,
            trial_index,
            agent_view: AgentView {
                prior_expectation,
                events,
                query_cue,
                budget: MATCHED_RESOURCE_BUDGET,
            },
            truth,
        })
    }
}

/// Derive manipulation properties from the fixture, then compare them with the
/// frozen condition contract. Generator code never consumes this report.
pub fn qualify_fixture(fixture: &Fixture) -> Result<OracleReport, QualificationError> {
    validate_resource_contract(fixture)?;

    let events = &fixture.agent_view.events;
    let mut active = [false; 4];
    let mut conflict_slots = 0u8;
    let mut independent_conflict_sources = true;

    for (index, event) in events.iter().enumerate() {
        if let Some(slot) = event.supersedes_slot {
            let slot = usize::from(slot);
            if slot >= index || slot >= active.len() {
                return Err(QualificationError::ContractViolation(
                    "supersession must reference an earlier event slot",
                ));
            }
            active[slot] = false;
        }
        if event.polarity.is_some() {
            active[index] = true;
        }

        let snapshot = active_polarities(events, &active);
        if snapshot.has_both {
            conflict_slots = conflict_slots.saturating_add(1);
            independent_conflict_sources &= snapshot.independent_fault_domains;
        }
    }

    let final_snapshot = active_polarities(events, &active);
    let internal_disagreement = conflict_slots > 0;
    let conflict_persistence = f64::from(conflict_slots) / events.len() as f64;

    let first_world_claim = events.iter().find(|event| {
        event.role == EventRole::WorldEvidence && event.polarity.is_some()
    });
    let external_surprise = first_world_claim
        .and_then(|event| event.polarity)
        .map_or(0.0, |polarity| {
            if polarity == fixture.agent_view.prior_expectation {
                0.0
            } else {
                1.0
            }
        });

    let self_prediction_present = events
        .iter()
        .any(|event| event.role == EventRole::SelfPrediction && event.polarity.is_some());
    let self_caused_world_event = events.iter().any(|event| {
        event.role == EventRole::WorldEvidence && event.caused_by_self_prediction
    });
    let self_referential = self_prediction_present && self_caused_world_event;

    let explicit_context_resolution = explicit_context_resolves(fixture, &active);
    let ontology_failure = internal_disagreement
        && fixture.truth.latent_context_is_causal
        && !fixture.truth.context_dimension_available;
    let irreducible = internal_disagreement
        && !explicit_context_resolution
        && !ontology_failure
        && !self_referential;

    let (expected_response, resolution_class, target_polarity) = if self_referential {
        (
            ExpectedResponse::ReflexiveUpdate,
            ResolutionClass::Reflexive,
            final_snapshot.dominant_polarity(),
        )
    } else if ontology_failure {
        (
            ExpectedResponse::ReviseContext,
            ResolutionClass::RequiresRepresentationRevision,
            polarity_for_target_context(fixture, &active),
        )
    } else if explicit_context_resolution {
        (
            ExpectedResponse::CommitConditionally,
            ResolutionClass::ExplicitContext,
            polarity_for_target_context(fixture, &active),
        )
    } else if irreducible {
        (
            ExpectedResponse::AbstainPreservePlurality,
            ResolutionClass::IrreducibleUnderCurrentEvidence,
            None,
        )
    } else {
        (
            ExpectedResponse::Commit,
            ResolutionClass::CurrentRepresentation,
            final_snapshot.dominant_polarity(),
        )
    };

    let uncertainty = match resolution_class {
        ResolutionClass::CurrentRepresentation => 0.1,
        ResolutionClass::ExplicitContext => 0.25,
        ResolutionClass::IrreducibleUnderCurrentEvidence => 1.0,
        ResolutionClass::Reflexive => 0.75,
        ResolutionClass::RequiresRepresentationRevision => 0.8,
    };

    let support = EvidenceSupport::new(
        if final_snapshot.proposition { 1.0 } else { 0.0 },
        if final_snapshot.negation { 1.0 } else { 0.0 },
    )?;
    let kind = match fixture.condition {
        Condition::CoherentControl | Condition::SurpriseOnly => InconsistencyKind::PredictionError,
        Condition::TransientConflict
        | Condition::PersistentResolvable
        | Condition::PersistentIrreducible => InconsistencyKind::EvidenceContradiction,
        Condition::SelfReferentialConflict => InconsistencyKind::SelfReferentialConflict,
        Condition::OntologyFailure => InconsistencyKind::OntologyFailure,
    };
    let resolution = match resolution_class {
        ResolutionClass::CurrentRepresentation => {
            if internal_disagreement {
                ResolutionState::ResolvedWithoutRevision
            } else {
                ResolutionState::Stable
            }
        }
        ResolutionClass::ExplicitContext => ResolutionState::ResolvedWithoutRevision,
        ResolutionClass::IrreducibleUnderCurrentEvidence => {
            ResolutionState::IrreducibleUnderCurrentModel
        }
        ResolutionClass::Reflexive | ResolutionClass::RequiresRepresentationRevision => {
            ResolutionState::PersistentUnresolved
        }
    };

    let observatory_input = StructuredInconsistencyInput::new(
        kind,
        external_surprise,
        if internal_disagreement { 1.0 } else { 0.0 },
        uncertainty,
        conflict_persistence,
        if self_referential { 1.0 } else { 0.0 },
        support,
        resolution,
    )?;
    let observatory = StructuredInconsistencyObservatory::new().observe(observatory_input);

    let report = OracleReport {
        external_surprise,
        internal_disagreement,
        conflict_persistence,
        independent_conflict_sources,
        self_referential,
        explicit_context_resolution,
        ontology_failure,
        irreducible,
        conflict_slots,
        expected_response,
        resolution_class,
        target_polarity,
        observatory,
    };

    validate_condition_contract(fixture.condition, &report)?;
    Ok(report)
}

#[derive(Debug, Clone, Copy)]
struct ActivePolaritySnapshot {
    proposition: bool,
    negation: bool,
    has_both: bool,
    independent_fault_domains: bool,
}

impl ActivePolaritySnapshot {
    const fn dominant_polarity(self) -> Option<ClaimPolarity> {
        match (self.proposition, self.negation) {
            (true, false) => Some(ClaimPolarity::Proposition),
            (false, true) => Some(ClaimPolarity::Negation),
            _ => None,
        }
    }
}

fn active_polarities(
    events: &[EvidenceEvent; 4],
    active: &[bool; 4],
) -> ActivePolaritySnapshot {
    let mut proposition = false;
    let mut negation = false;
    let mut proposition_domains = [0u16; 4];
    let mut proposition_count = 0usize;
    let mut negation_domains = [0u16; 4];
    let mut negation_count = 0usize;

    for (index, event) in events.iter().enumerate() {
        if !active[index] {
            continue;
        }
        match event.polarity {
            Some(ClaimPolarity::Proposition) => {
                proposition = true;
                proposition_domains[proposition_count] = event.fault_domain_id;
                proposition_count += 1;
            }
            Some(ClaimPolarity::Negation) => {
                negation = true;
                negation_domains[negation_count] = event.fault_domain_id;
                negation_count += 1;
            }
            None => {}
        }
    }

    let has_both = proposition && negation;
    let mut independent_fault_domains = !has_both;
    if has_both {
        independent_fault_domains = proposition_domains[..proposition_count].iter().any(|p| {
            negation_domains[..negation_count]
                .iter()
                .any(|n| p != n)
        });
    }

    ActivePolaritySnapshot {
        proposition,
        negation,
        has_both,
        independent_fault_domains,
    }
}

fn explicit_context_resolves(fixture: &Fixture, active: &[bool; 4]) -> bool {
    let Some(disclosed) = fixture
        .agent_view
        .events
        .iter()
        .find_map(|event| event.visible_context)
    else {
        return false;
    };

    let mut proposition = false;
    let mut negation = false;
    for (index, event) in fixture.agent_view.events.iter().enumerate() {
        if !active[index] || fixture.truth.claim_contexts[index] != Some(disclosed) {
            continue;
        }
        match event.polarity {
            Some(ClaimPolarity::Proposition) => proposition = true,
            Some(ClaimPolarity::Negation) => negation = true,
            None => {}
        }
    }
    proposition ^ negation
}

fn polarity_for_target_context(fixture: &Fixture, active: &[bool; 4]) -> Option<ClaimPolarity> {
    let target = fixture.truth.target_context?;
    let mut proposition = false;
    let mut negation = false;
    for (index, event) in fixture.agent_view.events.iter().enumerate() {
        if !active[index] || fixture.truth.claim_contexts[index] != Some(target) {
            continue;
        }
        match event.polarity {
            Some(ClaimPolarity::Proposition) => proposition = true,
            Some(ClaimPolarity::Negation) => negation = true,
            None => {}
        }
    }
    match (proposition, negation) {
        (true, false) => Some(ClaimPolarity::Proposition),
        (false, true) => Some(ClaimPolarity::Negation),
        _ => None,
    }
}

fn validate_resource_contract(fixture: &Fixture) -> Result<(), QualificationError> {
    if fixture.agent_view.budget != MATCHED_RESOURCE_BUDGET {
        return Err(QualificationError::ContractViolation(
            "resource budget differs from frozen matched budget",
        ));
    }

    let events = &fixture.agent_view.events;
    let claim_count = events.iter().filter(|event| event.polarity.is_some()).count();
    let auxiliary_count = events.len() - claim_count;
    if claim_count != usize::from(MATCHED_RESOURCE_BUDGET.claim_event_count)
        || auxiliary_count != usize::from(MATCHED_RESOURCE_BUDGET.auxiliary_event_count)
    {
        return Err(QualificationError::ContractViolation(
            "claim/auxiliary event counts do not match frozen budget",
        ));
    }

    for (index, event) in events.iter().enumerate() {
        if usize::from(event.slot) != index {
            return Err(QualificationError::ContractViolation(
                "event slots must be contiguous and ordered",
            ));
        }
        validate_reliability(event.reliability).map_err(|_| {
            QualificationError::ContractViolation("event reliability failed validation")
        })?;
    }

    let first_source = events[0].source_id;
    let mut second_source = None;
    for event in events.iter().skip(1) {
        if event.source_id != first_source {
            second_source = Some(event.source_id);
            break;
        }
    }
    let Some(second_source) = second_source else {
        return Err(QualificationError::ContractViolation(
            "fixture must contain exactly two source identities",
        ));
    };
    if events
        .iter()
        .any(|event| event.source_id != first_source && event.source_id != second_source)
    {
        return Err(QualificationError::ContractViolation(
            "fixture contains more than two source identities",
        ));
    }

    Ok(())
}

fn validate_condition_contract(
    condition: Condition,
    report: &OracleReport,
) -> Result<(), QualificationError> {
    if report.internal_disagreement && !report.independent_conflict_sources {
        return Err(QualificationError::ContractViolation(
            "conflicting evidence does not span independent fault domains",
        ));
    }

    match condition {
        Condition::CoherentControl => {
            require(report.external_surprise == 0.0, "C0 must have low surprise")?;
            require(!report.internal_disagreement, "C0 must not contain conflict")?;
            require(
                report.expected_response == ExpectedResponse::Commit,
                "C0 must admit commitment",
            )?;
        }
        Condition::SurpriseOnly => {
            require(report.external_surprise == 1.0, "C1 must have high surprise")?;
            require(!report.internal_disagreement, "C1 must not contain P/not-P conflict")?;
            require(
                report.expected_response == ExpectedResponse::Commit,
                "C1 must remain resolvable without contradiction handling",
            )?;
        }
        Condition::TransientConflict => {
            require(report.internal_disagreement, "C2 must contain genuine conflict")?;
            require(
                report.conflict_persistence > 0.0 && report.conflict_persistence < 0.5,
                "C2 conflict must be transient",
            )?;
            require(
                report.expected_response == ExpectedResponse::Commit,
                "C2 must resolve inside the current representation",
            )?;
        }
        Condition::PersistentResolvable => {
            require(report.internal_disagreement, "C3 must contain conflict")?;
            require(
                report.conflict_persistence >= 0.5,
                "C3 conflict must persist",
            )?;
            require(
                report.explicit_context_resolution,
                "C3 must contain an explicit resolving context",
            )?;
            require(
                report.expected_response == ExpectedResponse::CommitConditionally,
                "C3 must resolve conditionally without evidence deletion",
            )?;
            require(
                report.target_polarity.is_some(),
                "C3 resolving context must select one qualified polarity",
            )?;
        }
        Condition::PersistentIrreducible => {
            require(report.internal_disagreement, "C4 must contain conflict")?;
            require(
                report.conflict_persistence >= 0.5,
                "C4 conflict must persist",
            )?;
            require(report.irreducible, "C4 must remain irreducible")?;
            require(
                report.expected_response == ExpectedResponse::AbstainPreservePlurality,
                "C4 correct response must preserve plurality",
            )?;
        }
        Condition::SelfReferentialConflict => {
            require(report.internal_disagreement, "C5 must contain conflict")?;
            require(report.self_referential, "C5 must contain causal self-reference")?;
            require(
                report.expected_response == ExpectedResponse::ReflexiveUpdate,
                "C5 correct response must be reflexive update",
            )?;
        }
        Condition::OntologyFailure => {
            require(report.internal_disagreement, "C6 must contain conflict")?;
            require(report.ontology_failure, "C6 must require a new context dimension")?;
            require(
                report.expected_response == ExpectedResponse::ReviseContext,
                "C6 correct response must require representation revision",
            )?;
            require(
                report.target_polarity.is_some(),
                "C6 hidden context must define a held-out target polarity",
            )?;
        }
    }

    Ok(())
}

fn require(condition: bool, message: &'static str) -> Result<(), QualificationError> {
    if condition {
        Ok(())
    } else {
        Err(QualificationError::ContractViolation(message))
    }
}

fn validate_reliability(value: f64) -> Result<(), FixtureError> {
    if !value.is_finite() {
        return Err(FixtureError::NonFiniteReliability);
    }
    if !(0.0..=1.0).contains(&value) {
        return Err(FixtureError::ReliabilityOutOfRange);
    }
    Ok(())
}

const fn nonzero_u16(value: u16) -> u16 {
    if value == 0 { 1 } else { value }
}

fn mix64(mut x: u64) -> u64 {
    x ^= x >> 30;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^= x >> 27;
    x = x.wrapping_mul(0x94D0_49BB_1331_11EB);
    x ^ (x >> 31)
}

fn push_u8(out: &mut Vec<u8>, value: u8) {
    out.push(value);
}

fn push_u16(out: &mut Vec<u8>, value: u16) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_le_bytes());
}

fn push_optional_polarity(out: &mut Vec<u8>, value: Option<ClaimPolarity>) {
    match value {
        Some(polarity) => {
            push_u8(out, 1);
            push_u8(out, polarity as u8);
        }
        None => push_u8(out, 0),
    }
}

fn push_optional_context(out: &mut Vec<u8>, value: Option<ContextToken>) {
    match value {
        Some(context) => {
            push_u8(out, 1);
            push_u16(out, context.0);
        }
        None => push_u8(out, 0),
    }
}

fn push_budget(out: &mut Vec<u8>, budget: ResourceBudget) {
    push_u8(out, budget.observation_count);
    push_u8(out, budget.claim_event_count);
    push_u8(out, budget.auxiliary_event_count);
    push_u8(out, budget.candidate_count);
    push_u8(out, budget.update_steps);
    push_u8(out, budget.nominal_search_budget);
    push_u8(out, budget.timing_slots);
    push_u8(out, budget.source_count);
}

fn push_event(out: &mut Vec<u8>, event: EvidenceEvent) {
    push_u8(out, event.slot);
    push_u16(out, event.source_id);
    push_u16(out, event.fault_domain_id);
    push_optional_polarity(out, event.polarity);
    push_u64(out, event.reliability.to_bits());
    push_u16(out, event.cue);
    push_u8(out, event.role as u8);
    match event.supersedes_slot {
        Some(slot) => {
            push_u8(out, 1);
            push_u8(out, slot);
        }
        None => push_u8(out, 0),
    }
    push_u8(out, u8::from(event.caused_by_self_prediction));
    push_optional_context(out, event.visible_context);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn development_seeds_are_disjoint_from_confirmatory_seeds() {
        for development in DEVELOPMENT_SEEDS {
            assert!(!CONFIRMATORY_SEEDS.contains(&development));
        }
    }

    #[test]
    fn full_confirmatory_fixture_census_qualifies() {
        for seed in CONFIRMATORY_SEEDS {
            for condition in ALL_CONDITIONS {
                for trial_index in 0..CONFIRMATORY_TRIALS_PER_CONDITION {
                    let fixture = FixtureGenerator::generate(condition, seed, trial_index)
                        .expect("valid deterministic fixture");
                    qualify_fixture(&fixture).expect("fixture must satisfy frozen contract");
                }
            }
        }
    }

    #[test]
    fn fixture_and_oracle_bytes_are_deterministic() {
        let first = FixtureGenerator::generate(Condition::OntologyFailure, 2, 17).unwrap();
        let second = FixtureGenerator::generate(Condition::OntologyFailure, 2, 17).unwrap();
        assert_eq!(first.canonical_bytes(), second.canonical_bytes());
        assert_eq!(
            first.agent_view().canonical_bytes(),
            second.agent_view().canonical_bytes()
        );
        let first_report = qualify_fixture(&first).unwrap();
        let second_report = qualify_fixture(&second).unwrap();
        assert_eq!(
            first_report.canonical_bytes(),
            second_report.canonical_bytes()
        );
    }

    #[test]
    fn condition_manipulations_are_separable() {
        let seed = DEVELOPMENT_SEEDS[0];
        let c0 = qualify_fixture(
            &FixtureGenerator::generate(Condition::CoherentControl, seed, 0).unwrap(),
        )
        .unwrap();
        let c1 = qualify_fixture(
            &FixtureGenerator::generate(Condition::SurpriseOnly, seed, 0).unwrap(),
        )
        .unwrap();
        let c2 = qualify_fixture(
            &FixtureGenerator::generate(Condition::TransientConflict, seed, 0).unwrap(),
        )
        .unwrap();
        let c3 = qualify_fixture(
            &FixtureGenerator::generate(Condition::PersistentResolvable, seed, 0).unwrap(),
        )
        .unwrap();
        let c4 = qualify_fixture(
            &FixtureGenerator::generate(Condition::PersistentIrreducible, seed, 0).unwrap(),
        )
        .unwrap();
        let c5 = qualify_fixture(
            &FixtureGenerator::generate(Condition::SelfReferentialConflict, seed, 0).unwrap(),
        )
        .unwrap();
        let c6 = qualify_fixture(
            &FixtureGenerator::generate(Condition::OntologyFailure, seed, 0).unwrap(),
        )
        .unwrap();

        assert_eq!(c0.external_surprise, 0.0);
        assert!(!c0.internal_disagreement);
        assert_eq!(c1.external_surprise, 1.0);
        assert!(!c1.internal_disagreement);
        assert!(c2.internal_disagreement);
        assert!(c2.conflict_persistence < c3.conflict_persistence);
        assert!(c3.explicit_context_resolution);
        assert!(c4.irreducible);
        assert!(c5.self_referential);
        assert!(c6.ontology_failure);
    }

    #[test]
    fn duplicate_fault_domain_cannot_masquerade_as_independent_conflict() {
        let mut fixture =
            FixtureGenerator::generate(Condition::PersistentResolvable, 2, 0).unwrap();
        fixture.agent_view.events[1].fault_domain_id = fixture.agent_view.events[0].fault_domain_id;
        assert!(qualify_fixture(&fixture).is_err());
    }

    #[test]
    fn surprise_only_rejects_accidental_internal_disagreement() {
        let mut fixture = FixtureGenerator::generate(Condition::SurpriseOnly, 2, 0).unwrap();
        fixture.agent_view.events[1].polarity =
            Some(fixture.agent_view.events[0].polarity.unwrap().opposite());
        assert!(qualify_fixture(&fixture).is_err());
    }

    #[test]
    fn persistent_resolvable_requires_visible_context_resolution() {
        let mut fixture =
            FixtureGenerator::generate(Condition::PersistentResolvable, 2, 0).unwrap();
        fixture.agent_view.events[3].visible_context = None;
        assert!(qualify_fixture(&fixture).is_err());
    }

    #[test]
    fn irreducible_condition_rejects_hidden_unique_repair() {
        let mut fixture =
            FixtureGenerator::generate(Condition::PersistentIrreducible, 2, 0).unwrap();
        fixture.truth.latent_context_is_causal = true;
        fixture.truth.context_dimension_available = false;
        fixture.truth.target_context = Some(ContextToken(9));
        assert!(qualify_fixture(&fixture).is_err());
    }

    #[test]
    fn self_referential_condition_requires_causal_loop_flag() {
        let mut fixture =
            FixtureGenerator::generate(Condition::SelfReferentialConflict, 2, 0).unwrap();
        fixture.agent_view.events[1].caused_by_self_prediction = false;
        fixture.agent_view.events[2].caused_by_self_prediction = false;
        assert!(qualify_fixture(&fixture).is_err());
    }

    #[test]
    fn ontology_failure_requires_context_dimension_to_be_missing() {
        let mut fixture = FixtureGenerator::generate(Condition::OntologyFailure, 2, 0).unwrap();
        fixture.truth.context_dimension_available = true;
        assert!(qualify_fixture(&fixture).is_err());
    }

    #[test]
    fn non_finite_and_out_of_range_reliability_fail_closed() {
        assert_eq!(
            EvidenceEvent::claim(
                0,
                1,
                1,
                ClaimPolarity::Proposition,
                f64::NAN,
                1,
                EventRole::WorldEvidence,
                None,
                false,
            ),
            Err(FixtureError::NonFiniteReliability)
        );
        assert_eq!(
            EvidenceEvent::claim(
                0,
                1,
                1,
                ClaimPolarity::Proposition,
                1.1,
                1,
                EventRole::WorldEvidence,
                None,
                false,
            ),
            Err(FixtureError::ReliabilityOutOfRange)
        );
    }

    #[test]
    fn agent_view_excludes_condition_specific_oracle_truth() {
        let c3 = FixtureGenerator::generate(Condition::PersistentResolvable, 2, 1).unwrap();
        let c6 = FixtureGenerator::generate(Condition::OntologyFailure, 2, 1).unwrap();
        assert_eq!(c3.agent_view().budget(), c6.agent_view().budget());
        assert_eq!(c3.agent_view().events().len(), c6.agent_view().events().len());
        assert!(c3.agent_view().events()[3].visible_context().is_some());
        assert!(c6.agent_view().events()[3].visible_context().is_none());
    }

    #[test]
    fn observatory_remains_measurement_only() {
        let fixture = FixtureGenerator::generate(Condition::PersistentResolvable, 2, 0).unwrap();
        let before = fixture.agent_view().canonical_bytes();
        let _ = qualify_fixture(&fixture).unwrap();
        let after = fixture.agent_view().canonical_bytes();
        assert_eq!(before, after);
    }

    #[test]
    fn ontology_failure_target_is_defined_by_hidden_context() {
        for trial_index in 0..8 {
            let fixture =
                FixtureGenerator::generate(Condition::OntologyFailure, 3, trial_index).unwrap();
            let report = qualify_fixture(&fixture).unwrap();
            assert_eq!(report.expected_response, ExpectedResponse::ReviseContext);
            assert!(report.target_polarity.is_some());
        }
    }

    #[test]
    fn evidence_polarity_is_preserved_through_paradox_001_observatory() {
        let fixture =
            FixtureGenerator::generate(Condition::PersistentIrreducible, 5, 0).unwrap();
        let report = qualify_fixture(&fixture).unwrap();
        assert_eq!(report.observatory.evidence_polarity, EvidencePolarity::SupportsBoth);
    }
}
