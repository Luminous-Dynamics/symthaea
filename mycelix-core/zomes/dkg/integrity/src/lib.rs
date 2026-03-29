// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! DKG (Distributed Knowledge Graph) Integrity Zome
//!
//! The "Neural Tissue" of the Mycelix Protocol.
//! PoGQ filters inputs (gradients), DKG stores conclusions (knowledge).
//!
//! This is not a database - it's a Truth Ledger with confidence-weighted claims.

use hdi::prelude::*;

// ============================================================================
// CONSTANTS
// ============================================================================

/// Minimum confidence to accept a new claim (prevents spam)
pub const MIN_INITIAL_CONFIDENCE: f32 = 0.1;

/// Maximum confidence (asymptotic limit)
pub const MAX_CONFIDENCE: f32 = 0.9999;

/// Decay rate per day for unreinforced claims
pub const CONFIDENCE_DECAY_RATE: f32 = 0.01;

/// Reinforcement bonus per attestation (diminishing returns)
pub const REINFORCEMENT_BASE: f32 = 0.1;

/// Minimum reputation to attest (prevents Sybil)
pub const MIN_ATTESTATION_REPUTATION: f32 = 0.3;

/// Maximum subject/predicate/object length
pub const MAX_TRIPLE_COMPONENT_LENGTH: usize = 512;

// ============================================================================
// ENTRY TYPES
// ============================================================================

/// The atomic unit of knowledge: a claim with epistemic metadata.
///
/// Instead of storing bare facts, we store claims with:
/// - Provenance (who claimed it, when)
/// - Confidence (how certain is the network?)
/// - Evidence (link to PoGQ proof that validated the source)
/// - Epistemic classification (E/N/M axes)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct VerifiableTriple {
    // The claim itself (RDF-style triple)
    pub subject: String,      // e.g., "The Beatles"
    pub predicate: String,    // e.g., "HasMember"
    pub object: String,       // e.g., "John Lennon"

    // Epistemic Metadata
    pub confidence: f32,      // 0.0 to 1.0 - strength of this "synapse"
    pub evidence_hash: Option<String>,  // Link to PoGQ proof (if ML-derived)

    // Epistemic Classification (E/N/M axes from Epistemic Charter v2.0)
    pub empirical_level: u8,    // 0-4: How verifiable?
    pub normative_level: u8,    // 0-3: Who has authority?
    pub materiality_level: u8,  // 0-3: How long should it persist?

    // Provenance (as strings for compatibility)
    pub creator: String,
    pub created_at: i64,
    pub last_reinforced_at: i64,

    // Version for conflict resolution
    pub version: u64,
}

/// An attestation to a triple (reinforces confidence)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct Attestation {
    pub triple_hash: String,      // The triple being attested
    pub attester: String,
    pub attester_reputation: f32, // Reputation at time of attestation
    pub agreement: bool,          // true = confirms, false = disputes
    pub evidence_hash: Option<String>,  // Optional additional evidence
    pub attested_at: i64,
}

/// Agent's knowledge reputation (separate from PoGQ trust)
#[hdk_entry_helper]
#[derive(Clone, PartialEq)]
pub struct KnowledgeReputation {
    pub agent: String,
    pub total_claims: u64,
    pub confirmed_claims: u64,      // Claims that gained confidence
    pub disputed_claims: u64,       // Claims that lost confidence
    pub attestations_given: u64,
    pub attestation_accuracy: f32,  // How often attestations aligned with consensus
    pub reputation_score: f32,      // Derived from above
    pub updated_at: i64,
    pub version: u64,
}

// ============================================================================
// ENTRY DEFINITIONS
// ============================================================================

/// Entry types for DKG
#[hdk_entry_types]
#[unit_enum(EntryTypesUnit)]
pub enum EntryTypes {
    VerifiableTriple(VerifiableTriple),
    Attestation(Attestation),
    KnowledgeReputation(KnowledgeReputation),
}

// ============================================================================
// LINK TYPES
// ============================================================================

#[hdk_link_types]
pub enum LinkTypes {
    /// Subject string → Triples about that subject
    SubjectToTriples,
    /// Predicate string → Triples using that predicate
    PredicateToTriples,
    /// Object string → Triples with that object
    ObjectToTriples,
    /// Triple → Attestations for that triple
    TripleToAttestations,
    /// Agent → Triples they created
    AgentToTriples,
    /// Agent → Their knowledge reputation
    AgentToReputation,
    /// Evidence hash → Triples derived from that evidence
    EvidenceToTriples,
}

// ============================================================================
// VALIDATION FUNCTIONS
// ============================================================================

/// Validate a VerifiableTriple entry
pub fn validate_verifiable_triple(triple: &VerifiableTriple) -> ExternResult<ValidateCallbackResult> {
    // Check subject
    if triple.subject.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Subject cannot be empty".to_string()
        ));
    }
    if triple.subject.len() > MAX_TRIPLE_COMPONENT_LENGTH {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Subject exceeds max length of {}", MAX_TRIPLE_COMPONENT_LENGTH)
        ));
    }

    // Check predicate
    if triple.predicate.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Predicate cannot be empty".to_string()
        ));
    }
    if triple.predicate.len() > MAX_TRIPLE_COMPONENT_LENGTH {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Predicate exceeds max length of {}", MAX_TRIPLE_COMPONENT_LENGTH)
        ));
    }

    // Check object
    if triple.object.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Object cannot be empty".to_string()
        ));
    }
    if triple.object.len() > MAX_TRIPLE_COMPONENT_LENGTH {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Object exceeds max length of {}", MAX_TRIPLE_COMPONENT_LENGTH)
        ));
    }

    // Check confidence bounds
    if triple.confidence < 0.0 || triple.confidence > 1.0 || !triple.confidence.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Confidence must be between 0.0 and 1.0 and finite".to_string()
        ));
    }
    if triple.confidence < MIN_INITIAL_CONFIDENCE {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Initial confidence must be at least {}", MIN_INITIAL_CONFIDENCE)
        ));
    }

    // Validate epistemic levels
    if triple.empirical_level > 4 {
        return Ok(ValidateCallbackResult::Invalid(
            "Empirical level must be 0-4".to_string()
        ));
    }
    if triple.normative_level > 3 {
        return Ok(ValidateCallbackResult::Invalid(
            "Normative level must be 0-3".to_string()
        ));
    }
    if triple.materiality_level > 3 {
        return Ok(ValidateCallbackResult::Invalid(
            "Materiality level must be 0-3".to_string()
        ));
    }

    // Validate epistemic classification consistency
    // E4 (reproducible) claims should have higher minimum confidence
    if triple.empirical_level == 4 && triple.confidence < 0.5 {
        return Ok(ValidateCallbackResult::Invalid(
            "Reproducible claims (E4) require minimum confidence of 0.5".to_string()
        ));
    }

    // N3 (axiomatic) claims require cryptographic evidence (E3/E4)
    if triple.normative_level == 3 && triple.empirical_level < 3 {
        return Ok(ValidateCallbackResult::Invalid(
            "Axiomatic claims (N3) require cryptographic or reproducible evidence (E3/E4)".to_string()
        ));
    }

    // M3 (foundational) claims cannot be personal scope (N0)
    if triple.materiality_level == 3 && triple.normative_level == 0 {
        return Ok(ValidateCallbackResult::Invalid(
            "Foundational claims (M3) cannot be personal scope (N0)".to_string()
        ));
    }

    // Check creator is non-empty
    if triple.creator.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Creator cannot be empty".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate an Attestation entry
pub fn validate_attestation(attestation: &Attestation) -> ExternResult<ValidateCallbackResult> {
    // Check triple hash is non-empty
    if attestation.triple_hash.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Triple hash cannot be empty".to_string()
        ));
    }

    // Check attester is non-empty
    if attestation.attester.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Attester cannot be empty".to_string()
        ));
    }

    // Check reputation threshold
    if attestation.attester_reputation < MIN_ATTESTATION_REPUTATION {
        return Ok(ValidateCallbackResult::Invalid(
            format!("Attester reputation {} below minimum {}",
                attestation.attester_reputation, MIN_ATTESTATION_REPUTATION)
        ));
    }

    // Reputation must be valid
    if attestation.attester_reputation < 0.0 || attestation.attester_reputation > 1.0
        || !attestation.attester_reputation.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Attester reputation must be between 0.0 and 1.0 and finite".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

/// Validate a KnowledgeReputation entry
pub fn validate_knowledge_reputation(reputation: &KnowledgeReputation) -> ExternResult<ValidateCallbackResult> {
    // Check agent is non-empty
    if reputation.agent.trim().is_empty() {
        return Ok(ValidateCallbackResult::Invalid(
            "Agent cannot be empty".to_string()
        ));
    }

    // Confirmed + disputed should not exceed total
    if reputation.confirmed_claims + reputation.disputed_claims > reputation.total_claims {
        return Ok(ValidateCallbackResult::Invalid(
            "Confirmed + disputed claims cannot exceed total claims".to_string()
        ));
    }

    // Reputation score must be valid
    if reputation.reputation_score < 0.0 || reputation.reputation_score > 1.0
        || !reputation.reputation_score.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Reputation score must be between 0.0 and 1.0 and finite".to_string()
        ));
    }

    // Attestation accuracy must be valid
    if reputation.attestation_accuracy < 0.0 || reputation.attestation_accuracy > 1.0
        || !reputation.attestation_accuracy.is_finite() {
        return Ok(ValidateCallbackResult::Invalid(
            "Attestation accuracy must be between 0.0 and 1.0 and finite".to_string()
        ));
    }

    Ok(ValidateCallbackResult::Valid)
}

// ============================================================================
// HELPER FUNCTIONS
// ============================================================================

impl VerifiableTriple {
    /// Calculate effective confidence after time decay
    pub fn effective_confidence(&self, now_micros: i64) -> f32 {
        let micros_per_day: i64 = 24 * 60 * 60 * 1_000_000;
        let days_since_reinforcement = (now_micros - self.last_reinforced_at) as f64
            / micros_per_day as f64;

        // Exponential decay
        let decay_factor = (-CONFIDENCE_DECAY_RATE * days_since_reinforcement as f32).exp();

        // M3 (foundational) claims decay slower
        let adjusted_decay = if self.materiality_level == 3 {
            decay_factor.powf(0.5) // Square root = slower decay
        } else if self.materiality_level == 0 {
            decay_factor.powf(2.0) // Squared = faster decay
        } else {
            decay_factor
        };

        self.confidence * adjusted_decay
    }

    /// Calculate reinforcement from an attestation
    pub fn reinforcement_delta(attester_reputation: f32, current_confidence: f32) -> f32 {
        // Diminishing returns: harder to increase already-high confidence
        let headroom = MAX_CONFIDENCE - current_confidence;
        let base_increase = REINFORCEMENT_BASE * attester_reputation;

        // Asymptotic approach to MAX_CONFIDENCE
        headroom * base_increase
    }

    /// Calculate dispute penalty from a negative attestation
    pub fn dispute_delta(attester_reputation: f32, current_confidence: f32) -> f32 {
        // Higher reputation attesters have more impact
        let penalty = REINFORCEMENT_BASE * attester_reputation * 1.5; // Disputes hit harder

        let max_penalty = current_confidence - MIN_INITIAL_CONFIDENCE;
        (current_confidence * penalty).min(max_penalty.max(0.0))
    }

    /// Get the epistemic classification as a string (e.g., "E3-N2-M2")
    pub fn epistemic_classification(&self) -> String {
        format!("E{}-N{}-M{}", self.empirical_level, self.normative_level, self.materiality_level)
    }
}

impl KnowledgeReputation {
    /// Calculate reputation score from metrics
    pub fn calculate_score(&self) -> f32 {
        if self.total_claims == 0 {
            return 0.5; // Neutral starting point
        }

        let confirmation_rate = self.confirmed_claims as f32 / self.total_claims as f32;
        let dispute_rate = self.disputed_claims as f32 / self.total_claims as f32;

        // Weighted combination
        let claim_quality = confirmation_rate * 0.4 - dispute_rate * 0.3;
        let attestation_quality = self.attestation_accuracy * 0.3;

        // Clamp to valid range
        (0.5 + claim_quality + attestation_quality).clamp(0.0, 1.0)
    }
}

// ============================================================================
// EPISTEMIC LEVEL CONSTANTS (for coordinator use)
// ============================================================================

/// Empirical Level Constants
pub mod empirical {
    pub const E0_NULL: u8 = 0;           // Unverifiable belief
    pub const E1_TESTIMONIAL: u8 = 1;     // Personal attestation
    pub const E2_PRIVATE_VERIFY: u8 = 2;  // Audit guild can verify
    pub const E3_CRYPTOGRAPHIC: u8 = 3;   // ZKP or cryptographic proof
    pub const E4_REPRODUCIBLE: u8 = 4;    // Publicly reproducible
}

/// Normative Level Constants
pub mod normative {
    pub const N0_PERSONAL: u8 = 0;    // Self only
    pub const N1_COMMUNAL: u8 = 1;    // Local DAO
    pub const N2_NETWORK: u8 = 2;     // Global consensus
    pub const N3_AXIOMATIC: u8 = 3;   // Constitutional/mathematical
}

/// Materiality Level Constants
pub mod materiality {
    pub const M0_EPHEMERAL: u8 = 0;     // Discard immediately
    pub const M1_TEMPORAL: u8 = 1;      // Prune after state change
    pub const M2_PERSISTENT: u8 = 2;    // Archive after time
    pub const M3_FOUNDATIONAL: u8 = 3;  // Preserve forever
}
