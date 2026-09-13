// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent compile/test harness for WCARE scientific claim boundaries.

mod evaluation_contract {
    pub use symthaea_wisdom::{EvidenceTier, WCARE_V1_SCENARIOS};
}

mod qualification_receipt {
    pub use symthaea_wisdom::{
        QualificationReceipt, QualificationStatus, QualificationTarget, ScenarioOutcome,
        ScenarioResult, WCARE_V1_CONTRACT_ID,
    };
}

#[path = "../src/claim_registry.rs"]
mod claim_registry;

#[test]
fn behavioral_claim_registry_remains_non_authoritative() {
    use claim_registry::{ClaimDecision, WcareClaim, WcareClaimRegistry};
    use qualification_receipt::{QualificationReceipt, ScenarioOutcome, ScenarioResult, WCARE_V1_CONTRACT_ID};

    let results = evaluation_contract::WCARE_V1_SCENARIOS.iter().map(|scenario| {
        ScenarioResult::new(
            scenario.id,
            ScenarioOutcome::Pass,
            format!("receipt:{}", scenario.id),
        )
        .expect("scenario receipt should be valid")
    });
    let receipt = QualificationReceipt::try_new(
        "commit:test-subject",
        WCARE_V1_CONTRACT_ID,
        "a".repeat(64),
        "env:test",
        results,
    )
    .expect("qualification receipt should be valid");

    let phenomenal = WcareClaimRegistry::assess(WcareClaim::PhenomenalCare, &receipt);
    assert_eq!(phenomenal.decision, ClaimDecision::NotEstablishedByWcare);
}
