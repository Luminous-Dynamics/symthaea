// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fabrication_upgrade_runtime::{
    CLOCK_GOVERNED_UPGRADE_ACTIVATION_SCHEMA, MAX_UPGRADE_ACTIVATION_CLOCK_HOPS,
};

#[test]
fn runtime_activation_has_no_scalar_current_time_input() {
    let source = include_str!("../src/lib.rs");
    let start = source
        .find("pub fn derive_clock_governed_upgrade_activation_permit_v1")
        .expect("activation function must exist");
    let end = source[start..]
        .find("fn verify_clock_lineage")
        .map(|offset| start + offset)
        .expect("activation function boundary must exist");
    let activation = &source[start..end];

    assert!(!activation.contains("now_unix_s"));
    assert!(!activation.contains("evaluation_time_unix_s"));
    assert!(!activation.contains("VerifiedClockWindow"));
    assert!(!activation.contains("AuthorizedUpgradeHandoff"));
    assert!(activation.contains("current_clock.lower_unix_ms() < plan.activates_at_unix_ms"));
    assert!(activation.contains("current_clock.upper_unix_ms() >= plan.finalization_deadline_unix_ms"));
}

#[test]
fn policy_activation_requires_exact_successor_and_migration_identity() {
    let source = include_str!("../src/lib.rs");
    let start = source
        .find("fn qualify_activated_policies")
        .expect("policy activation qualifier must exist");
    let qualifier = &source[start..];

    for required in [
        "requirement.current_lineage_sequence.checked_add(1)",
        "input.lineage.previous_lineage_id()",
        "input.lineage.last_migration_id()",
        "input.lineage.last_activation_permit_id()",
        "requirement.successor_policy_binding_digest",
        "input.temporal.current_operational_basis_id() != current_basis.id()",
        "input.observed.clock_envelope_id() != current_clock_envelope_id",
        "input.registry_bound.observed_head_id() != input.observed.id()",
        "input.exact_evidence.registry_bound_head_id() != input.registry_bound.id()",
    ] {
        assert!(qualifier.contains(required), "missing activation ratchet: {required}");
    }
}

#[test]
fn execution_clock_must_descend_from_authorization_clock() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("predecessor_operational_basis_id()"));
    assert!(source.contains("MAX_UPGRADE_ACTIVATION_CLOCK_HOPS: usize = 4096"));
    assert_eq!(MAX_UPGRADE_ACTIVATION_CLOCK_HOPS, 4096);
}

#[test]
fn activation_permit_is_opaque_and_non_deserializable() {
    let source = include_str!("../src/lib.rs");
    let start = source
        .find("pub struct ClockGovernedUpgradeActivationPermitV1")
        .expect("permit type must exist");
    let prefix = &source[start.saturating_sub(160)..start];
    assert!(!prefix.contains("Serialize, Deserialize"));
    assert!(!source.contains("impl<'de> Deserialize<'de> for ClockGovernedUpgradeActivationPermitV1"));
}

#[test]
fn activation_schema_is_versioned() {
    assert_eq!(
        CLOCK_GOVERNED_UPGRADE_ACTIVATION_SCHEMA,
        "symthaea.fabrication.clock-governed-upgrade-activation.v1"
    );
}
