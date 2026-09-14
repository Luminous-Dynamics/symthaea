// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Evidence-authority regression for issue #1034.
//!
//! The current runtime-qualification resolver is a public conformance helper
//! that accepts caller-supplied booleans. Until functional evidence is bound
//! to the exact same intervention through the verifier-owned #1015 path,
//! this legacy helper must have zero authority to mint
//! `SupportTier::FunctionallySupported`.
//!
//! This test is intentionally expected to fail on the pre-#1034
//! implementation. It changes no production behavior.

use symthaea_psych_bench::benchmarks::butlin::{
    EvidenceOutcome, RuntimeQualification, SupportTier, resolve_outcome,
};

fn fully_passing_fixture() -> RuntimeQualification {
    RuntimeQualification {
        static_design_qualifies: true,
        intervention_applied: true,
        intervention_specificity_passed: true,
        positive_control_effect_observed: true,
        sham_behaved_as_expected: true,
        probe_signal_usable: true,
        identity_and_config_match: true,
    }
}

#[test]
fn legacy_runtime_resolver_cannot_mint_functional_support() {
    let outcome = resolve_outcome(&fully_passing_fixture(), true, true);
    assert_eq!(
        outcome,
        EvidenceOutcome::Supported(SupportTier::CausallySupported),
        "the legacy boolean resolver has no same-intervention downstream provenance; \
         even an attractive functional_effect_observed=true input must be capped at \
         CausallySupported until #1015's verifier-owned path is wired to a real runner"
    );
}

#[test]
fn failed_runtime_qualification_still_fails_closed() {
    let mut qualification = fully_passing_fixture();
    qualification.intervention_specificity_passed = false;
    assert_eq!(
        resolve_outcome(&qualification, true, true),
        EvidenceOutcome::Inconclusive,
        "removing legacy functional authority must not weaken existing runtime qualification gates"
    );
}

#[test]
fn qualified_null_remains_a_negative_finding() {
    assert_eq!(
        resolve_outcome(&fully_passing_fixture(), false, true),
        EvidenceOutcome::NotDemonstrated,
        "the authority ceiling must preserve the existing distinction between a qualified null \
         and an inconclusive measurement"
    );
}
