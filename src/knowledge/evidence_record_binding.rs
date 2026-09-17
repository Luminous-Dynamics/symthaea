// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Canonical mapping between an EKM evidence-draft identity and ledger fields.
//!
//! EKM-017/018/019 currently reconstruct the same `context` and `method` strings
//! at multiple boundaries. This module freezes one canonical representation and
//! exact matcher before those call sites are migrated, so future refactors have
//! an explicit compatibility target instead of relying on duplicated literals.
//!
//! The binding is an internal ledger encoding, not a cryptographic digest and not
//! a substitute for provenance, authorization, or the full typed draft identity.

use super::claim_evidence::EvidenceRecord;
use super::evidence_mutation_firewall::EvidenceDraftIdentity;

/// Versioned canonical ledger-field binding for an admitted evidence draft.
///
/// The version is explicit so a future representation change can be introduced
/// as a migration instead of silently changing what counts as an exact replay.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EvidenceRecordBindingVersion {
    V1,
}

/// Canonical binding helper. It performs no ledger mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EvidenceRecordBinding {
    version: EvidenceRecordBindingVersion,
}

impl Default for EvidenceRecordBinding {
    fn default() -> Self {
        Self::v1()
    }
}

impl EvidenceRecordBinding {
    pub const fn v1() -> Self {
        Self {
            version: EvidenceRecordBindingVersion::V1,
        }
    }

    pub const fn version(self) -> EvidenceRecordBindingVersion {
        self.version
    }

    /// Deterministic ledger context for a draft identity.
    pub fn context(self, identity: &EvidenceDraftIdentity) -> Option<String> {
        match self.version {
            EvidenceRecordBindingVersion::V1 => {
                Some(format!("inquiry-result: {}", identity.result_summary))
            }
        }
    }

    /// Deterministic ledger method/protocol field for a draft identity.
    pub fn method(self, identity: &EvidenceDraftIdentity) -> Option<String> {
        match self.version {
            EvidenceRecordBindingVersion::V1 => Some(format!(
                "preregistered-decision[{}]: {}",
                identity.decision_rule_label, identity.decision_criterion
            )),
        }
    }

    /// Exact semantic match between a ledger evidence record and a draft.
    pub fn matches(self, record: &EvidenceRecord, identity: &EvidenceDraftIdentity) -> bool {
        record.claim_id == identity.claim_id
            && record.kind == identity.kind
            && record.polarity == identity.polarity
            && record.provenance_id == identity.provenance_id
            && record.observed_at_cycle == identity.observed_at_cycle
            && record.context == self.context(identity)
            && record.method == self.method(identity)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::knowledge::{
        ClaimId, EvidenceId, EvidenceKind, EvidencePolarity, ProvenanceId,
    };

    fn identity() -> EvidenceDraftIdentity {
        EvidenceDraftIdentity {
            claim_id: ClaimId(7),
            kind: EvidenceKind::Measurement,
            polarity: EvidencePolarity::Supports,
            provenance_id: ProvenanceId(11),
            observed_at_cycle: 19,
            result_summary: "bounded result".into(),
            decision_rule_label: "supports".into(),
            decision_criterion: "measure > upper".into(),
        }
    }

    fn matching_record(binding: EvidenceRecordBinding, identity: &EvidenceDraftIdentity) -> EvidenceRecord {
        EvidenceRecord {
            id: EvidenceId(23),
            claim_id: identity.claim_id,
            kind: identity.kind,
            polarity: identity.polarity,
            provenance_id: identity.provenance_id,
            observed_at_cycle: identity.observed_at_cycle,
            context: binding.context(identity),
            method: binding.method(identity),
        }
    }

    #[test]
    fn v1_freezes_current_firewall_encoding() {
        let binding = EvidenceRecordBinding::v1();
        let identity = identity();
        assert_eq!(
            binding.context(&identity),
            Some("inquiry-result: bounded result".into())
        );
        assert_eq!(
            binding.method(&identity),
            Some("preregistered-decision[supports]: measure > upper".into())
        );
    }

    #[test]
    fn exact_record_matches() {
        let binding = EvidenceRecordBinding::v1();
        let identity = identity();
        let record = matching_record(binding, &identity);
        assert!(binding.matches(&record, &identity));
    }

    #[test]
    fn evidence_kind_cannot_be_hidden_by_matching_strings() {
        let binding = EvidenceRecordBinding::v1();
        let identity = identity();
        let mut record = matching_record(binding, &identity);
        record.kind = EvidenceKind::Report;
        assert!(!binding.matches(&record, &identity));
    }

    #[test]
    fn polarity_cannot_be_hidden_by_matching_strings() {
        let binding = EvidenceRecordBinding::v1();
        let identity = identity();
        let mut record = matching_record(binding, &identity);
        record.polarity = EvidencePolarity::Contradicts;
        assert!(!binding.matches(&record, &identity));
    }

    #[test]
    fn provenance_and_time_are_part_of_exact_identity() {
        let binding = EvidenceRecordBinding::v1();
        let identity = identity();
        let mut record = matching_record(binding, &identity);
        record.provenance_id = ProvenanceId(99);
        assert!(!binding.matches(&record, &identity));

        let mut record = matching_record(binding, &identity);
        record.observed_at_cycle += 1;
        assert!(!binding.matches(&record, &identity));
    }
}
