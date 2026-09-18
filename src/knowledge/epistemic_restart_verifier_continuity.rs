// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Read-only continuity checks for EKM-049 verifier provenance.
//!
//! A verifier profile can itself roll back even when the candidate restart state
//! moves forward. This module compares one EKM-049 provenance receipt with a
//! caller-held trusted verifier checkpoint. It never updates that checkpoint.

use super::epistemic_restart_verifier_provenance::{
    RestartVerifierProfileDigestV1, RestartVerifierProvenanceDigestV1,
    RestartVerifierProvenanceReceiptV1,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TrustedRestartVerifierStateV1 {
    verifier_id: String,
    implementation_id: String,
    implementation_version: String,
    trust_snapshot_sequence: u64,
    trust_snapshot_digest: [u8; 32],
    configuration_digest: [u8; 32],
    profile_digest: RestartVerifierProfileDigestV1,
    provenance_digest: RestartVerifierProvenanceDigestV1,
}

impl TrustedRestartVerifierStateV1 {
    pub fn from_receipt(receipt: &RestartVerifierProvenanceReceiptV1) -> Self {
        let profile = receipt.verifier_profile();
        Self {
            verifier_id: profile.verifier_id().to_string(),
            implementation_id: profile.implementation_id().to_string(),
            implementation_version: profile.implementation_version().to_string(),
            trust_snapshot_sequence: profile.trust_snapshot_sequence(),
            trust_snapshot_digest: profile.trust_snapshot_digest(),
            configuration_digest: profile.configuration_digest(),
            profile_digest: receipt.verifier_profile_digest(),
            provenance_digest: receipt.provenance_digest(),
        }
    }

    pub fn trust_snapshot_sequence(&self) -> u64 {
        self.trust_snapshot_sequence
    }

    pub fn profile_digest(&self) -> RestartVerifierProfileDigestV1 {
        self.profile_digest
    }

    pub fn provenance_digest(&self) -> RestartVerifierProvenanceDigestV1 {
        self.provenance_digest
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestartVerifierContinuityDispositionV1 {
    StableTrustReuse,
    TrustSnapshotAdvance,
    Rejected,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RestartVerifierContinuityFailureV1 {
    VerifierIdentityChanged,
    ImplementationIdentityChanged,
    ImplementationVersionChanged,
    ConfigurationChanged,
    TrustSnapshotRollback { trusted: u64, candidate: u64 },
    SameSequenceSnapshotSubstitution,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RestartVerifierContinuityDecisionV1 {
    disposition: RestartVerifierContinuityDispositionV1,
    failures: Vec<RestartVerifierContinuityFailureV1>,
    candidate_trust_snapshot_sequence: u64,
    trusted_trust_snapshot_sequence: u64,
    further_review_eligible: bool,
    trusted_state_mutated: bool,
    quarantine_construction_authorized: bool,
    activation_authorized: bool,
}

impl RestartVerifierContinuityDecisionV1 {
    pub fn disposition(&self) -> RestartVerifierContinuityDispositionV1 {
        self.disposition
    }

    pub fn failures(&self) -> &[RestartVerifierContinuityFailureV1] {
        &self.failures
    }

    pub fn further_review_eligible(&self) -> bool {
        self.further_review_eligible
    }

    pub fn trusted_state_mutated(&self) -> bool {
        self.trusted_state_mutated
    }

    pub fn quarantine_construction_authorized(&self) -> bool {
        self.quarantine_construction_authorized
    }

    pub fn activation_authorized(&self) -> bool {
        self.activation_authorized
    }
}

pub struct RestartVerifierContinuityGateV1;

impl RestartVerifierContinuityGateV1 {
    pub fn evaluate(
        trusted: &TrustedRestartVerifierStateV1,
        candidate: &RestartVerifierProvenanceReceiptV1,
    ) -> RestartVerifierContinuityDecisionV1 {
        let profile = candidate.verifier_profile();
        let mut failures = Vec::new();

        if profile.verifier_id() != trusted.verifier_id {
            failures.push(RestartVerifierContinuityFailureV1::VerifierIdentityChanged);
        }
        if profile.implementation_id() != trusted.implementation_id {
            failures.push(RestartVerifierContinuityFailureV1::ImplementationIdentityChanged);
        }
        if profile.implementation_version() != trusted.implementation_version {
            failures.push(RestartVerifierContinuityFailureV1::ImplementationVersionChanged);
        }
        if profile.configuration_digest() != trusted.configuration_digest {
            failures.push(RestartVerifierContinuityFailureV1::ConfigurationChanged);
        }

        let candidate_sequence = profile.trust_snapshot_sequence();
        if candidate_sequence < trusted.trust_snapshot_sequence {
            failures.push(RestartVerifierContinuityFailureV1::TrustSnapshotRollback {
                trusted: trusted.trust_snapshot_sequence,
                candidate: candidate_sequence,
            });
        } else if candidate_sequence == trusted.trust_snapshot_sequence
            && profile.trust_snapshot_digest() != trusted.trust_snapshot_digest
        {
            failures.push(RestartVerifierContinuityFailureV1::SameSequenceSnapshotSubstitution);
        }

        let disposition = if !failures.is_empty() {
            RestartVerifierContinuityDispositionV1::Rejected
        } else if candidate_sequence == trusted.trust_snapshot_sequence {
            RestartVerifierContinuityDispositionV1::StableTrustReuse
        } else {
            RestartVerifierContinuityDispositionV1::TrustSnapshotAdvance
        };

        RestartVerifierContinuityDecisionV1 {
            disposition,
            failures,
            candidate_trust_snapshot_sequence: candidate_sequence,
            trusted_trust_snapshot_sequence: trusted.trust_snapshot_sequence,
            further_review_eligible: !matches!(
                disposition,
                RestartVerifierContinuityDispositionV1::Rejected
            ),
            trusted_state_mutated: false,
            quarantine_construction_authorized: false,
            activation_authorized: false,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::epistemic_restart_verifier_provenance::{
        RestartVerifierProfileV1, RestartVerifierProvenanceReceiptV1,
    };

    // EKM-050 deliberately tests the pure comparison semantics through a small
    // internal fixture instead of introducing public constructors for provenance receipts.
    fn profile(
        verifier: &str,
        implementation: &str,
        version: &str,
        sequence: u64,
        snapshot: [u8; 32],
        configuration: [u8; 32],
    ) -> RestartVerifierProfileV1 {
        RestartVerifierProfileV1::new(
            verifier,
            implementation,
            version,
            snapshot,
            sequence,
            0,
            100,
            configuration,
        )
        .unwrap()
    }

    #[test]
    fn profile_constructor_supports_continuity_inputs() {
        let current = profile("v", "impl", "1", 4, [4; 32], [9; 32]);
        assert_eq!(current.trust_snapshot_sequence(), 4);
        assert_eq!(current.trust_snapshot_digest(), [4; 32]);
    }

    #[test]
    fn receipt_type_remains_non_authorizing() {
        // Compile-level guard: EKM-050 depends only on the read-only EKM-049 receipt type.
        fn assert_receipt_type(_: Option<&RestartVerifierProvenanceReceiptV1>) {}
        assert_receipt_type(None);
    }
}
