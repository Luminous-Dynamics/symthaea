// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_fabrication_kernel::hardware_reauthorization::HardwareReauthorizationPolicy;
use symthaea_fabrication_upgrade_hardware_authority::ExactHardwareReauthorizationVerificationPolicyV1;

#[test]
fn default_exact_hardware_verification_requires_two_providers() {
    let policy = ExactHardwareReauthorizationVerificationPolicyV1::default();
    assert_eq!(policy.minimum_distinct_providers, 2);
    assert!(policy.maximum_providers >= policy.minimum_distinct_providers);
}

#[test]
fn hardware_policy_still_requires_positive_windows() {
    let policy = HardwareReauthorizationPolicy::default();
    assert!(policy.maximum_authorization_duration_s > 0);
    assert!(policy.maximum_statement_age_s > 0);
}

#[test]
fn source_ratchets_exclude_legacy_scalar_live_authority() {
    let source = include_str!("../src/lib.rs");
    assert!(!source.contains("now_unix_s:"));
    assert!(!source.contains("AuthorizedUpgradeHandoff"));
    assert!(!source.contains("AuthorizedUpgradeProbationClearance"));
    assert!(!source.contains("VerifiedClockWindow"));
    assert!(source.contains("ClockGovernedUpgradeHandoffV1"));
    assert!(source.contains("TelemetryBoundUpgradeProbationClearanceV1"));
    assert!(source.contains("KeyUsage::HardwareReauthorization"));
    assert!(source.contains("require_effective_time_after_envelope_seconds"));
    assert!(source.contains("HARDWARE_SIGNATURE_DOMAIN"));
    assert!(!source.contains(
        "Serialize, Deserialize)]\npub struct ClockGovernedHardwareReauthorizationV1"
    ));
}

#[test]
fn statement_timing_is_interval_safe_and_post_probation() {
    let source = include_str!("../src/lib.rs");
    assert!(source.contains("issued_ms < clearance_clock.upper_unix_ms()"));
    assert!(source.contains("issued_ms > current_clock.lower_unix_ms()"));
    assert!(source.contains("expires_ms <= current_clock.upper_unix_ms()"));
    assert!(source.contains("current_clock\n        .upper_unix_ms()"));
    assert!(source.contains("StatementOutlivesHandoff"));
}
