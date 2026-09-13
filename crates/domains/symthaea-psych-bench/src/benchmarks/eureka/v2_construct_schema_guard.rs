// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Temporary drift guard between the V2 construct prototype and the canonical
//! V2 public schema.
//!
//! `v2_construct.rs` predates `v2_public_schema.rs` and still owns private
//! duplicate declarations. Rewriting a thousand-line target-blind construct
//! capsule only to move those declarations would be a poor source-risk trade
//! before executable qualification is available. This test therefore freezes
//! the exact remaining duplication and fails if either side drifts.
//!
//! This is not the end state. Before V2 identity is promoted from prototype to
//! frozen instrument, `v2_construct.rs` must consume `v2_public_schema` directly
//! and this textual guard must be deleted.

use super::hidden_world::PublicAction;
use super::v2_public_schema::{
    V2_CONTEXT_DENOMINATOR, V2_COUNT_DENOMINATOR, V2_OBSERVATION_DIM,
    V2_PUBLIC_MODES_PER_FAMILY, V2_REQUIRED_ACTIONS, V2PublicFamily, action_from_index,
};

const CONSTRUCT_SOURCE: &str = include_str!("v2_construct.rs");

#[test]
fn legacy_construct_private_grammar_matches_canonical_schema_pending_refactor() {
    assert_eq!(V2_COUNT_DENOMINATOR, 31);
    assert_eq!(V2_CONTEXT_DENOMINATOR, 7);
    assert_eq!(V2_OBSERVATION_DIM, 4);
    assert_eq!(V2_PUBLIC_MODES_PER_FAMILY, 4);
    assert_eq!(V2_REQUIRED_ACTIONS, 4);

    // State cardinality 32 means generated count channels are exactly 0..=31.
    assert!(CONSTRUCT_SOURCE.contains("const STATE_CARDINALITY: u16 = 32;"));
    assert!(CONSTRUCT_SOURCE.contains("const MODES: u8 = 4;"));
    assert!(CONSTRUCT_SOURCE.contains("const ACTIONS: u8 = 4;"));
    assert!(CONSTRUCT_SOURCE.contains("struct State([i32; 4]);"));

    // The old prototype name `ConservedFlow` is a legacy alias only. The
    // canonical scientific name is PublicFlowV2 because one mode injects mass.
    assert!(CONSTRUCT_SOURCE.contains("ConservedFlow"));
    assert!(CONSTRUCT_SOURCE.contains("PublicRelay"));
    assert_eq!(V2PublicFamily::PublicFlowV2.tag(), 1);
    assert_eq!(V2PublicFamily::PublicRelayV2.tag(), 2);

    // Context bands must remain exactly flow 0..=3 and relay 4..=7.
    assert!(CONSTRUCT_SOURCE.contains("Self::ConservedFlow => i32::from(mode)"));
    assert!(CONSTRUCT_SOURCE.contains("Self::PublicRelay => 4 + i32::from(mode)"));
    for mode in 0..V2_PUBLIC_MODES_PER_FAMILY {
        assert_eq!(
            V2PublicFamily::PublicFlowV2.context(mode).unwrap(),
            i32::from(mode)
        );
        assert_eq!(
            V2PublicFamily::PublicRelayV2.context(mode).unwrap(),
            4 + i32::from(mode)
        );
    }

    // Canonical action vocabulary must stay NoOp, Pulse(0), Pulse(1), Pulse(2)
    // with no aliases or modulo/clamp fallback.
    let expected = [
        PublicAction::NoOp,
        PublicAction::Pulse { slot: 0 },
        PublicAction::Pulse { slot: 1 },
        PublicAction::Pulse { slot: 2 },
    ];
    for (index, expected_action) in expected.into_iter().enumerate() {
        assert_eq!(action_from_index(index).unwrap(), expected_action);
    }
    assert!(CONSTRUCT_SOURCE.contains("0 => PublicAction::NoOp"));
    assert!(CONSTRUCT_SOURCE.contains("1 => PublicAction::Pulse { slot: 0 }"));
    assert!(CONSTRUCT_SOURCE.contains("2 => PublicAction::Pulse { slot: 1 }"));
    assert!(CONSTRUCT_SOURCE.contains("3 => PublicAction::Pulse { slot: 2 }"));
}

#[test]
fn canonical_schema_revision_is_not_owned_by_construct_source() {
    // The construct prototype must not invent a second canonical schema
    // revision while the temporary private aliases still exist.
    assert!(!CONSTRUCT_SOURCE.contains("EUREKA.002.V2.PUBLIC_SCHEMA"));
}
