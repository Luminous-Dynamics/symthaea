// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Integration-level marker for the executable extraction contract.
//!
//! Detailed canonicalization and source-multiplicity regressions live next to
//! private transport types in the library unit tests. This integration target
//! intentionally remains tiny so `cargo test --all-targets` proves the crate is
//! discoverable as an external consumer without exposing raw qmpy transport types.

use symthaea_materials_oqmd_extraction_executor::{
    COMPOSITION_CANONICALIZER_ID, PROPERTY_CONDITION_POLICY_ID, QMPY_SOURCE_COMMIT,
    QMPY_VERSION, STRUCTURE_CANONICALIZER_ID,
};

#[test]
fn frozen_public_contract_ids_are_stable() {
    assert_eq!(QMPY_VERSION, "1.4.0");
    assert_eq!(QMPY_SOURCE_COMMIT.len(), 40);
    assert_eq!(
        COMPOSITION_CANONICALIZER_ID,
        "reduced-integer-stoichiometry-v1"
    );
    assert_eq!(
        STRUCTURE_CANONICALIZER_ID,
        "species-lattice-fractional-sites-v1"
    );
    assert_eq!(
        PROPERTY_CONDITION_POLICY_ID,
        "oqmd-dft-method-condition-v1"
    );
}
