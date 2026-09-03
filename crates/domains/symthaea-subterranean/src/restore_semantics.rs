// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Machine-readable restore contracts for the operational checkpoint.
//!
//! Checkpoint fields are not semantically interchangeable. Historical state,
//! live authority, evidence/replay state, derived physical envelopes, embedded
//! safety policy and ephemeral authority require different restore rules.

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum RestoreSemantics {
    HistoricalReplace,
    AuthorityMonotone,
    EvidenceMerge,
    DerivedRequalify,
    TransitionReconcile,
    EphemeralDrop,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissingStatePolicy {
    Reject,
    ConservativeRequalify,
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
    pub field: &'static str,
    pub semantics: &'static [RestoreSemantics],
    pub missing: MissingStatePolicy,
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
    RestoreSemantics::EvidenceMerge,
    RestoreSemantics::TransitionReconcile,
    RestoreSemantics::EphemeralDrop,
];
const DEGRADED_SEMANTICS: &[RestoreSemantics] = &[
    RestoreSemantics::AuthorityMonotone,
    RestoreSemantics::EvidenceMerge,
    RestoreSemantics::TransitionReconcile,
];
const UPDATE_SEMANTICS: &[RestoreSemantics] = &[RestoreSemantics::TransitionReconcile];
const SENSOR_SEMANTICS: &[RestoreSemantics] = &[
    RestoreSemantics::EvidenceMerge,
    RestoreSemantics::DerivedRequalify,
    RestoreSemantics::TransitionReconcile,
];
const ACTUATOR_SEMANTICS: &[RestoreSemantics] = &[
    RestoreSemantics::AuthorityMonotone,
    RestoreSemantics::EvidenceMerge,
    RestoreSemantics::DerivedRequalify,
    RestoreSemantics::TransitionReconcile,
];
const ENVELOPE_SEMANTICS: &[RestoreSemantics] = &[
    RestoreSemantics::DerivedRequalify,
    RestoreSemantics::TransitionReconcile,
];
const PARTITION_SEMANTICS: &[RestoreSemantics] = &[
    RestoreSemantics::AuthorityMonotone,
    RestoreSemantics::EvidenceMerge,
    RestoreSemantics::DerivedRequalify,
    RestoreSemantics::TransitionReconcile,
];
const TEMPORAL_SEMANTICS: &[RestoreSemantics] = &[
    RestoreSemantics::AuthorityMonotone,
    RestoreSemantics::EvidenceMerge,
    RestoreSemantics::DerivedRequalify,
    RestoreSemantics::TransitionReconcile,
];

pub const OPERATIONAL_RESTORE_CONTRACTS: &[RestoreDomainContract] = &[
    RestoreDomainContract { domain: RestoreDomain::Controller, field: "controller", semantics: CONTROLLER_SEMANTICS, missing: MissingStatePolicy::Reject, authority_relevant: true },
    RestoreDomainContract { domain: RestoreDomain::Mission, field: "mission", semantics: MISSION_SEMANTICS, missing: MissingStatePolicy::Reject, authority_relevant: true },
    RestoreDomainContract { domain: RestoreDomain::OperatorAuthority, field: "operator_authority", semantics: OPERATOR_SEMANTICS, missing: MissingStatePolicy::Reject, authority_relevant: true },
    RestoreDomainContract { domain: RestoreDomain::DegradedSupervisor, field: "degraded_supervisor", semantics: DEGRADED_SEMANTICS, missing: MissingStatePolicy::ConservativeRequalify, authority_relevant: true },
    RestoreDomainContract { domain: RestoreDomain::UpdateManager, field: "update_manager", semantics: UPDATE_SEMANTICS, missing: MissingStatePolicy::ConservativeRequalify, authority_relevant: true },
    RestoreDomainContract { domain: RestoreDomain::SensorFusion, field: "sensor_fusion", semantics: SENSOR_SEMANTICS, missing: MissingStatePolicy::ConservativeRequalify, authority_relevant: true },
    RestoreDomainContract { domain: RestoreDomain::ActuatorIsolation, field: "actuator_isolation", semantics: ACTUATOR_SEMANTICS, missing: MissingStatePolicy::ConservativeRequalify, authority_relevant: true },
    RestoreDomainContract { domain: RestoreDomain::FieldEnvelope, field: "field_envelope", semantics: ENVELOPE_SEMANTICS, missing: MissingStatePolicy::ConservativeRequalify, authority_relevant: true },
    RestoreDomainContract { domain: RestoreDomain::PartitionRecovery, field: "partition_recovery", semantics: PARTITION_SEMANTICS, missing: MissingStatePolicy::ConservativeRequalify, authority_relevant: true },
    RestoreDomainContract { domain: RestoreDomain::TemporalAssurance, field: "temporal", semantics: TEMPORAL_SEMANTICS, missing: MissingStatePolicy::ConservativeRequalify, authority_relevant: true },
];

pub fn contract_for(domain: RestoreDomain) -> &'static RestoreDomainContract {
    OPERATIONAL_RESTORE_CONTRACTS.iter().find(|contract| contract.domain == domain).expect("restore registry must contain every RestoreDomain")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeSet;

    #[test]
    fn operational_registry_has_one_contract_per_checkpoint_field() {
        let expected = BTreeSet::from(["controller","mission","operator_authority","degraded_supervisor","update_manager","sensor_fusion","actuator_isolation","field_envelope","partition_recovery","temporal"]);
        let actual = OPERATIONAL_RESTORE_CONTRACTS.iter().map(|contract| contract.field).collect::<BTreeSet<_>>();
        assert_eq!(actual, expected);
        assert_eq!(OPERATIONAL_RESTORE_CONTRACTS.len(), expected.len());
    }

    #[test]
    fn authority_relevant_state_never_defaults_to_unqualified_nominal() {
        for contract in OPERATIONAL_RESTORE_CONTRACTS.iter().filter(|contract| contract.authority_relevant) {
            assert_ne!(contract.missing, MissingStatePolicy::Drop);
        }
    }

    #[test]
    fn explicit_authority_domains_are_monotone() {
        for domain in [RestoreDomain::OperatorAuthority,RestoreDomain::DegradedSupervisor,RestoreDomain::ActuatorIsolation,RestoreDomain::PartitionRecovery,RestoreDomain::TemporalAssurance] {
            assert!(contract_for(domain).semantics.contains(&RestoreSemantics::AuthorityMonotone));
        }
    }

    #[test]
    fn evidence_domains_cannot_use_historical_replacement() {
        for domain in [RestoreDomain::OperatorAuthority,RestoreDomain::DegradedSupervisor,RestoreDomain::SensorFusion,RestoreDomain::ActuatorIsolation,RestoreDomain::PartitionRecovery,RestoreDomain::TemporalAssurance] {
            let contract = contract_for(domain);
            assert!(contract.semantics.contains(&RestoreSemantics::EvidenceMerge));
            assert!(!contract.semantics.contains(&RestoreSemantics::HistoricalReplace));
        }
    }

    #[test]
    fn checkpointed_safety_policy_requires_reconciliation() {
        for domain in [RestoreDomain::OperatorAuthority,RestoreDomain::DegradedSupervisor,RestoreDomain::SensorFusion,RestoreDomain::ActuatorIsolation,RestoreDomain::FieldEnvelope,RestoreDomain::PartitionRecovery,RestoreDomain::TemporalAssurance] {
            assert!(contract_for(domain).semantics.contains(&RestoreSemantics::TransitionReconcile));
        }
    }

    #[test]
    fn current_truth_domains_require_fresh_requalification() {
        for domain in [RestoreDomain::SensorFusion,RestoreDomain::ActuatorIsolation,RestoreDomain::FieldEnvelope,RestoreDomain::PartitionRecovery,RestoreDomain::TemporalAssurance] {
            assert!(contract_for(domain).semantics.contains(&RestoreSemantics::DerivedRequalify));
        }
    }
}
