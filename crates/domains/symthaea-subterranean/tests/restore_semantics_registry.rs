// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Ensure the machine-readable restore contract tracks the actual serialized
//! operational-checkpoint field set rather than a second hand-maintained list.

use std::collections::BTreeSet;
use symthaea_core::genesis::GenesisSeed;
use symthaea_subterranean::embodiment::SubterraneanEmbodiment;
use symthaea_subterranean::operational_checkpoint::restore_semantics::OPERATIONAL_RESTORE_CONTRACTS;

#[test]
fn every_serialized_checkpoint_field_has_restore_semantics() {
    let genesis = GenesisSeed::from_phrase("restore semantics schema coverage");
    let embodiment = SubterraneanEmbodiment::new(&genesis);
    let checkpoint = embodiment.operational_checkpoint();
    let encoded = serde_json::to_value(checkpoint).expect("checkpoint must serialize");
    let object = encoded
        .as_object()
        .expect("operational checkpoint must serialize as an object");

    let actual_fields = object.keys().map(String::as_str).collect::<BTreeSet<_>>();
    let registered_fields = OPERATIONAL_RESTORE_CONTRACTS
        .iter()
        .map(|contract| contract.field)
        .collect::<BTreeSet<_>>();

    assert_eq!(
        registered_fields, actual_fields,
        "every serialized operational-checkpoint field must be classified before the schema grows"
    );
    assert_eq!(
        OPERATIONAL_RESTORE_CONTRACTS.len(),
        actual_fields.len(),
        "restore registry must contain exactly one contract per checkpoint field"
    );
}
