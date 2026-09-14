// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fabrication_upgrade_probation_authority::{
    CLOCK_GOVERNED_UPGRADE_PROBATION_CLEARANCE_PURPOSE,
    CLOCK_GOVERNED_UPGRADE_PROBATION_CLEARANCE_SCHEMA,
    MAX_PROBATION_CLOCK_HOPS, ProbationObservationVerificationPolicyV1,
};

#[test]
fn probation_clearance_has_no_scalar_current_time_or_legacy_clearance_input() {
    let source = include_str!("../src/lib.rs");
    let start = source
        .find("pub fn prepare_clock_governed_upgrade_probation_clearance_v1")
        .expect("preparation function must exist");
    let end = source[start..]
        .find("pub fn authorize_clock_governed_upgrade_probation_clearance_v1")
        .map(|offset| start + offset)
        .expect("preparation boundary must exist");
    let preparation = &source[start..end];

    assert!(!preparation.contains("now_unix_s"));
    assert!(!preparation.contains("evaluation_time_unix_s"));
    assert!(!preparation.contains("AuthorizedUpgradeProbationClearance"));
    assert!(!preparation.contains("AuthorizedUpgradeHandoff"));
    assert!(preparation.contains("current_clock.upper_unix_ms() >= handoff.plan().finalization_deadline_unix_ms"));
}

#[test]
fn observations_require_exact_signatures_and_probation_key_usage() {
    let source = include_str!("../src/lib.rs");
    for required in [
        "digest_upgrade_probation_observation(&signed.observation)",
        "provider.verify_observation_signature",
        "KeyUsage::UpgradeProbation",
        "signed.observation.machine_id",
        "signed.observation.region_id",
        "ObserverMachineMismatch",
        "ObserverFailureDomainMismatch",
    ] {
        assert!(source.contains(required), "missing probation authority ratchet: {required}");
    }
}

#[test]
fn probation_observations_must_be_after_activation_and_definitely_complete() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains(
        "signed.observation.started_at_unix_ms < handoff.plan().activates_at_unix_ms"
    ));
    assert!(source.contains(
        "signed.observation.ended_at_unix_ms > current_clock.lower_unix_ms()"
    ));
    assert!(source.contains("predecessor_operational_basis_id()"));
    assert_eq!(MAX_PROBATION_CLOCK_HOPS, 4096);
}

#[test]
fn clearance_governance_is_separate_from_observer_keys() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("threshold_policy.key_usage != KeyUsage::ThresholdCeremony"));
    assert!(source.contains("record.usages.contains(&KeyUsage::UpgradeProbation)"));
    assert_eq!(
        CLOCK_GOVERNED_UPGRADE_PROBATION_CLEARANCE_PURPOSE,
        "clock-governed-upgrade-probation-clearance-v1"
    );
}

#[test]
fn exact_verifier_diversity_defaults_to_two() {
    let policy = ProbationObservationVerificationPolicyV1::default();
    assert_eq!(policy.minimum_distinct_providers, 2);
    assert!(policy.maximum_providers >= policy.minimum_distinct_providers);
}

#[test]
fn live_clearance_is_opaque_and_telemetry_reference_is_not_overclaimed() {
    let source = include_str!("../src/lib.rs");
    let start = source
        .find("pub struct ClockGovernedUpgradeProbationClearanceV1")
        .expect("clearance type must exist");
    let prefix = &source[start.saturating_sub(160)..start];
    assert!(!prefix.contains("Serialize, Deserialize"));
    assert!(source.contains("telemetry_evidence_digest` remains a signed, non-zero reference"));
    assert_eq!(
        CLOCK_GOVERNED_UPGRADE_PROBATION_CLEARANCE_SCHEMA,
        "symthaea.fabrication.clock-governed-upgrade-probation-clearance.v1"
    );
}
