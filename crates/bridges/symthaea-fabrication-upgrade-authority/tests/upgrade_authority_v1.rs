// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fabrication_upgrade_authority::{
    CLOCK_GOVERNED_UPGRADE_HANDOFF_PLAN_SCHEMA, CLOCK_GOVERNED_UPGRADE_HANDOFF_PURPOSE,
};

#[test]
fn live_upgrade_authority_excludes_legacy_scalar_authority_inputs() {
    let source = include_str!("../src/lib.rs");
    let live_start = source
        .find("pub fn prepare_clock_governed_upgrade_handoff_v1")
        .expect("preparation function must exist");
    let live_end = source
        .find("pub fn authorize_clock_governed_upgrade_handoff_v1")
        .expect("authorization function must exist");
    let preparation = &source[live_start..live_end];

    assert!(!preparation.contains("now_unix_s"));
    assert!(!preparation.contains("evaluation_time_unix_s"));
    assert!(!preparation.contains("VerifiedClockWindow"));
    assert!(!preparation.contains("AuthorizedPolicyMigration"));
    assert!(!preparation.contains("AuthorizedUpgradeHandoff"));
    assert!(!preparation.contains("prepared_at_unix_ms"));
}

#[test]
fn portable_plan_has_no_legacy_clock_or_preparation_timestamp() {
    let source = include_str!("../src/lib.rs");
    let plan_start = source
        .find("pub struct ClockGovernedUpgradeHandoffPlanV1")
        .expect("plan must exist");
    let plan_end = source[plan_start..]
        .find("}\n\n/// Borrowed live-authority bundle")
        .map(|offset| plan_start + offset)
        .expect("plan boundary must exist");
    let plan = &source[plan_start..plan_end];

    assert!(!plan.contains("clock_evidence_digest"));
    assert!(!plan.contains("prepared_at_unix_ms"));
    assert!(plan.contains("policy_requirements"));
}

#[test]
fn live_authority_is_opaque_and_not_serde_constructible() {
    let source = include_str!("../src/lib.rs");
    let authority_start = source
        .find("pub struct ClockGovernedUpgradeHandoffV1")
        .expect("authority type must exist");
    let prefix_start = authority_start.saturating_sub(160);
    let prefix = &source[prefix_start..authority_start];
    assert!(!prefix.contains("Serialize, Deserialize"));

    let prepared_start = source
        .find("pub struct PreparedClockGovernedUpgradeHandoffV1")
        .expect("prepared type must exist");
    let prepared_prefix = &source[prepared_start.saturating_sub(160)..prepared_start];
    assert!(!prepared_prefix.contains("Serialize, Deserialize"));
}

#[test]
fn handoff_quorum_has_dedicated_usage_and_purpose() {
    let source = include_str!("../src/lib.rs");
    assert_eq!(
        CLOCK_GOVERNED_UPGRADE_HANDOFF_PURPOSE,
        "clock-governed-upgrade-handoff-v1"
    );
    assert_eq!(
        CLOCK_GOVERNED_UPGRADE_HANDOFF_PLAN_SCHEMA,
        "symthaea.fabrication.clock-governed-upgrade-handoff-plan.v1"
    );
    assert!(source.contains("threshold_policy.key_usage != KeyUsage::UpgradeHandoff"));
}

#[test]
fn policy_chain_requires_all_hardened_currentness_layers() {
    let source = include_str!("../src/lib.rs");
    let derive_start = source
        .find("fn derive_policy_requirements")
        .expect("policy requirement derivation must exist");
    let derive = &source[derive_start..];
    for required in [
        "authority.temporal.lineage_id()",
        "authority.observed.lineage_id()",
        "authority.registry_bound.observed_head_id()",
        "authority.exact_evidence.observed_head_id()",
        "authority.exact_evidence.registry_bound_head_id()",
        "migration_plan.predecessor != *authority.lineage.current_policy()",
    ] {
        assert!(derive.contains(required), "missing authority ratchet: {required}");
    }
}
