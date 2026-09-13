// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Canonical formal-safety obligations for evidence-first domain awareness.
//!
//! DA-001..DA-037 remain in the original public catalog for API compatibility.
//! DA-038..DA-042 extend the canonical DomainAwareness template without adding
//! variants to the public enum and breaking downstream exhaustive matches.

#[path = "domain_awareness_base.rs"]
mod base;

use serde::{Deserialize, Serialize};

use crate::{EvidenceKind, ProofObligation};

pub use base::DomainAwarenessObligation;

/// Typed compatibility-safe extension for hardware-root and measured-boot
/// reference-integrity obligations.
///
/// These codes are part of the canonical DomainAwareness template, but are kept
/// separate from [`DomainAwarenessObligation`] so adding them is not a breaking
/// change for downstream exhaustive enum matches.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum DomainAwarenessTpmObligation {
    HardwareTrustRootSubjectBindingIsExact,
    FreshAttestationKeyPossessionRequired,
    MeasuredBootReplayMustMatchFreshQuote,
    ReferenceIntegrityApprovalRequired,
    ReferenceIntegrityLineageMustBeCurrentAndAuthorized,
}

impl DomainAwarenessTpmObligation {
    pub const ALL: [Self; 5] = [
        Self::HardwareTrustRootSubjectBindingIsExact,
        Self::FreshAttestationKeyPossessionRequired,
        Self::MeasuredBootReplayMustMatchFreshQuote,
        Self::ReferenceIntegrityApprovalRequired,
        Self::ReferenceIntegrityLineageMustBeCurrentAndAuthorized,
    ];

    pub const fn code(self) -> &'static str {
        match self {
            Self::HardwareTrustRootSubjectBindingIsExact => "DA-038",
            Self::FreshAttestationKeyPossessionRequired => "DA-039",
            Self::MeasuredBootReplayMustMatchFreshQuote => "DA-040",
            Self::ReferenceIntegrityApprovalRequired => "DA-041",
            Self::ReferenceIntegrityLineageMustBeCurrentAndAuthorized => "DA-042",
        }
    }

    pub const fn claim(self) -> &'static str {
        match self {
            Self::HardwareTrustRootSubjectBindingIsExact => {
                "hardware-backed monotonic trust-root evidence is accepted only when the exact reviewed hardware/runtime subject, TPM NV public identity, counter epoch, and rollback floor are bound and independently qualified; substitution or rollback cannot preserve readiness"
            }
            Self::FreshAttestationKeyPossessionRequired => {
                "current platform assurance requires fresh nonce-bound proof of possession of the reviewed attestation key over the exact policy-selected PCR set; stored key or certificate material alone cannot establish current possession"
            }
            Self::MeasuredBootReplayMustMatchFreshQuote => {
                "measured-boot event evidence, including required final events, must independently replay in order to the exact PCR state covered by the fresh quote; a valid quote alone cannot establish event-log consistency"
            }
            Self::ReferenceIntegrityApprovalRequired => {
                "measured-state approval requires the exact current signed and versioned reference-integrity policy; explicit denied or unknown-critical measurements reject approval, while missing required or unresolved evidence cannot be promoted to approved"
            }
            Self::ReferenceIntegrityLineageMustBeCurrentAndAuthorized => {
                "the active reference-integrity manifest must be the exact tip of a contiguous signed lineage with authorized signer and key transitions; history truncation, forged predecessors, revision gaps, or unreviewed signer replacement cannot preserve approval"
            }
        }
    }

    pub const fn expected_evidence(self) -> EvidenceKind {
        EvidenceKind::Test
    }

    pub fn stable_key(self) -> String {
        ProofObligation::new(self.claim(), self.expected_evidence()).stable_key()
    }
}

pub(crate) fn obligations() -> Vec<(&'static str, EvidenceKind)> {
    let mut obligations = base::obligations();
    obligations.extend(
        DomainAwarenessTpmObligation::ALL
            .iter()
            .copied()
            .map(|obligation| (obligation.claim(), obligation.expected_evidence())),
    );
    obligations
}

#[cfg(test)]
mod extension_tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn legacy_public_catalog_is_unchanged_and_extension_is_unique() {
        assert_eq!(DomainAwarenessObligation::ALL.len(), 37);
        assert_eq!(DomainAwarenessTpmObligation::ALL.len(), 5);
        assert_eq!(obligations().len(), 42);

        let base_codes = DomainAwarenessObligation::ALL
            .iter()
            .map(|obligation| obligation.code())
            .collect::<BTreeSet<_>>();
        let extension_codes = DomainAwarenessTpmObligation::ALL
            .iter()
            .map(|obligation| obligation.code())
            .collect::<BTreeSet<_>>();
        assert!(base_codes.is_disjoint(&extension_codes));

        let keys = DomainAwarenessTpmObligation::ALL
            .iter()
            .map(|obligation| obligation.stable_key())
            .collect::<BTreeSet<_>>();
        assert_eq!(keys.len(), DomainAwarenessTpmObligation::ALL.len());
    }

    #[test]
    fn tpm_reference_integrity_codes_and_evidence_are_stable() {
        assert_eq!(
            DomainAwarenessTpmObligation::HardwareTrustRootSubjectBindingIsExact.code(),
            "DA-038"
        );
        assert_eq!(
            DomainAwarenessTpmObligation::FreshAttestationKeyPossessionRequired.code(),
            "DA-039"
        );
        assert_eq!(
            DomainAwarenessTpmObligation::MeasuredBootReplayMustMatchFreshQuote.code(),
            "DA-040"
        );
        assert_eq!(
            DomainAwarenessTpmObligation::ReferenceIntegrityApprovalRequired.code(),
            "DA-041"
        );
        assert_eq!(
            DomainAwarenessTpmObligation::ReferenceIntegrityLineageMustBeCurrentAndAuthorized.code(),
            "DA-042"
        );
        assert!(DomainAwarenessTpmObligation::ALL
            .iter()
            .all(|obligation| obligation.expected_evidence() == EvidenceKind::Test));
    }
}
