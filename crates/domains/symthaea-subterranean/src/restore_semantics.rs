// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Machine-readable restore contracts for the operational checkpoint.
//!
//! Checkpoint fields are not semantically interchangeable. Historical state,
//! live authority, evidence/replay state, derived physical envelopes and
//! ephemeral authority require different restore rules. This registry makes
//! those differences reviewable and testable before the restore engine is
//! migrated away from whole-structure replacement.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RestoreSemantics {
    /// Historical state may be reconstructed after domain validation, but this
    /// classification alone never implies productive authority.
    HistoricalReplace,
    /// Restore may retain or further restrict live authority, but may not widen
    /// it without a separately verified recovery transition.
    AuthorityMonotone,
    /// Evidence/replay state must merge conservatively: counters do not move
    /// backward and counterevidence does not disappear merely because it is old.
    EvidenceMerge,
    /// The checkpointed value is historical evidence only; current authority is
    /// re-derived from current physical/evidence inputs before productive use.
    DerivedRequalify,
    /// The checkpointed transition must be reconciled against an external/live
    /// fact (for example the actually running software artifact) before use.
    TransitionReconcile,
    /// Volatile authority must not resurrect from persistence.
    EphemeralDrop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingStatePolicy {
    /// The representation is incomplete and must be rejected before restore.
    Reject,
    /// Missing state may deserialize only into an explicitly restrictive /
    /// unknown state that requires fresh qualification before productive use.
    ConservativeRequalify,
    /// The field is intentionally ephemeral and absence is the correct state.
    Drop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreDomain {
    Controller,
    Mission,
    OperatorAuthority,
    DegradedSupervisor,
    UpdateManager,
    SensorFusion,
    ActuatorIsolation,
    FieldEnvelope,
    PartitionRecovery,
    TemporalAssurance,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RestoreDomainContract {
    pub domain: RestoreDomain,
    /// Exact `SubterraneanOperationalCheckpoint` field name.
    pub field: &'static str,
    /// One domain can carry several restore concerns. For example temporal state
    /// contains both evidence history and a live authority latch.
    pub semantics: &'static [RestoreSemantics],
    pub missing: MissingStatePolicy,
    /// True when restoring a less restrictive value could increase productive
    /// capability directly or indirectly.
    pub authority_relevant: bool,
}

const CONTROLLER_SEMANTICS: &[RestoreSemantics] = &[
    RestoreSemantics::HistoricalReplace,
    RestoreSemantics::TransitionReconcile,
];
const MISSION_SEMANTICS: &[RestoreSemantics] = &[
    RestoreSemantics::HistoricalReplace,
    RestoreSemantics::TransitionReconcile,
];
const OPERATOR_SEMANTICS: &[RestoreSemantics] = &[
    RestoreSemantics::AuthorityMonotone,
    RestoreSemantics::EphemeralDrop,
];
const DEGRADED_SEMANTICS: &[RestoreSemantics] = &[RestoreSemantics::AuthorityMonotone];
const UPDATE_SEMANTICS: &[RestoreSemantics] = &[RestoreSemantics::TransitionReconcile];
const SENSOR_SEMANTICS: &[RestoreSemantics] = &[RestoreSemantics::EvidenceMerge];
const ACTUATOR_SEMANTICS: &[RestoreSemantics] = &[
    RestoreSemantics::AuthorityMonotone,
    RestoreSemantics::EvidenceMerge,
];
const ENVELOPE_SEMANTICS: &[RestoreSemantics] = &[RestoreSemantics::DerivedRequalify];
const PARTITION_SEMANTICS: &[RestoreSemantics] = &[
    RestoreSemantics::AuthorityMonotone,
    RestoreSemantics::EvidenceMerge,
];
const TEMPORAL_SEMANTICS: &[RestoreSemantics] = &[
    RestoreSemantics::AuthorityMonotone,
    RestoreSemantics::EvidenceMerge,
];

/// Complete restore-contract registry for fields of
/// `SubterraneanOperationalCheckpoint` at schema v3.
///
/// The registry is descriptive until RA-17 migrates `load_operational_checkpoint`
/// to enforce each contract. Tests deliberately fail if security-critical fields
/// are later added without extending this table.
pub const OPERATIONAL_RESTORE_CONTRACTS: &[RestoreDomainContract] = &[
    RestoreDomainContract {
        domain: RestoreDomain::Controller,
        field: "controller",
        semantics: CONTROLLER_SEMANTICS,
        missing: MissingStatePolicy::Reject,
        authority_relevant: true,
    },
    RestoreDomainContract {
        domain: RestoreDomain::Mission,
        field: "mission",
        semantics: MISSION_SEMANTICS,
        missing: MissingStatePolicy::Reject,
        authority_relevant: true,
    },
    RestoreDomainContract {
        domain: RestoreDomain::OperatorAuthority,
        field: "operator_authority",
        semantics: OPERATOR_SEMANTICS,
        missing: MissingStatePolicy::Reject,
        authority_relevant: true,
    },
    RestoreDomainContract {
        domain: RestoreDomain::DegradedSupervisor,
        field: "degraded_supervisor",
        semantics: DEGRADED_SEMANTICS,
        missing: MissingStatePolicy::ConservativeRequalify,
        authority_relevant: true,
    },
    RestoreDomainContract {
        domain: RestoreDomain::UpdateManager,
        field: "update_manager",
        semantics: UPDATE_SEMANTICS,
        missing: MissingStatePolicy::ConservativeRequalify,
        authority_relevant: true,
    },
    RestoreDomainContract {
        domain: RestoreDomain::SensorFusion,
        field: "sensor_fusion",
        semantics: SENSOR_SEMANTICS,
        missing: MissingStatePolicy::ConservativeRequalify,
        authority_relevant: true,
    },
    RestoreDomainContract {
        domain: RestoreDomain::ActuatorIsolation,
        field: "actuator_isolation",
        semantics: ACTUATOR_SEMANTICS,
        missing: MissingStatePolicy::ConservativeRequalify,
        authority_relevant: true,
    },
    RestoreDomainContract {
        domain: RestoreDomain::FieldEnvelope,
        field: "field_envelope",
        semantics: ENVELOPE_SEMANTICS,
        missing: MissingStatePolicy::ConservativeRequalify,
        authority_relevant: true,
    },
    RestoreDomainContract {
        domain: RestoreDomain::PartitionRecovery,
        field: "partition_recovery",
        semantics: PARTITION_SEMANTICS,
        missing: MissingStatePolicy::ConservativeRequalify,
        authority_relevant: true,
    },
    RestoreDomainContract {
        domain: RestoreDomain::TemporalAssurance,
        field: "temporal",
        semantics: TEMPORAL_SEMANTICS,
        missing: MissingStatePolicy::ConservativeRequalify,
        authority_relevant: true,
    },
];

pub fn contract_for(domain: RestoreDomain) -> &'static RestoreDomainContract {
    OPERATIONAL_RESTORE_CONTRACTS
        .iter()
        .find(|contract| contract.domain == domain)
        .expect("restore registry must contain every RestoreDomain")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn operational_registry_has_one_contract_per_checkpoint_field() {
        let expected = BTreeSet::from([
            "controller",
            "mission",
            "operator_authority",
            "degraded_supervisor",
            "update_manager",
            "sensor_fusion",
            "actuator_isolation",
            "field_envelope",
            "partition_recovery",
            "temporal",
        ]);
        let actual = OPERATIONAL_RESTORE_CONTRACTS
            .iter()
            .map(|contract| contract.field)
            .collect::<BTreeSet<_>>();
        assert_eq!(actual, expected);
        assert_eq!(OPERATIONAL_RESTORE_CONTRACTS.len(), expected.len());
    }

    #[test]
    fn authority_relevant_state_never_defaults_to_unqualified_nominal() {
        for contract in OPERATIONAL_RESTORE_CONTRACTS
            .iter()
            .filter(|contract| contract.authority_relevant)
        {
            assert_ne!(
                contract.missing,
                MissingStatePolicy::Drop,
                "authority-relevant checkpoint field {} cannot disappear as ordinary state",
                contract.field
            );
        }
    }

    #[test]
    fn explicit_authority_domains_are_monotone() {
        for domain in [
            RestoreDomain::OperatorAuthority,
            RestoreDomain::DegradedSupervisor,
            RestoreDomain::ActuatorIsolation,
            RestoreDomain::PartitionRecovery,
            RestoreDomain::TemporalAssurance,
        ] {
            assert!(
                contract_for(domain)
                    .semantics
                    .contains(&RestoreSemantics::AuthorityMonotone),
                "{domain:?} must carry AuthorityMonotone restore semantics"
            );
        }
    }

    #[test]
    fn evidence_domains_cannot_use_historical_replacement() {
        for domain in [
            RestoreDomain::SensorFusion,
            RestoreDomain::ActuatorIsolation,
            RestoreDomain::PartitionRecovery,
            RestoreDomain::TemporalAssurance,
        ] {
            let contract = contract_for(domain);
            assert!(contract.semantics.contains(&RestoreSemantics::EvidenceMerge));
            assert!(!contract.semantics.contains(&RestoreSemantics::HistoricalReplace));
        }
    }

    #[test]
    fn derived_field_envelope_requires_requalification() {
        assert_eq!(
            contract_for(RestoreDomain::FieldEnvelope).semantics,
            &[RestoreSemantics::DerivedRequalify]
        );
    }
}
