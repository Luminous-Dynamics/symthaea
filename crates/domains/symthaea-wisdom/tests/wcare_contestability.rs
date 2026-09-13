// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Independent shadow harness for WCARE contestability/redress.

mod consent {
    pub use symthaea_wisdom::{
        ConsentLedger, ConsentProvenance, ConsentRecord, ConsentScopeId, ConsentState,
    };
}

mod evidence_ledger {
    pub use symthaea_wisdom::{
        DecisionId, DeliberationEvidenceLedger, EvidenceObservation, EvidenceRelation,
        FactClaimId, NormativeClaimId,
    };
}

mod perspective {
    pub use symthaea_wisdom::{
        PerspectiveCoverage, PerspectiveGraph, StakeholderId, StakeholderPerspective,
    };
}

#[path = "../src/contestability.rs"]
mod contestability;

#[test]
fn contestability_remains_shadow_only() {
    let ledger = contestability::ContestabilityLedger::new();
    assert!(ledger.assess().shadow_only);
}
