// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_xenia_continuity_anchor::{
    XENIA_SYMTHAEA_CONTINUITY_NAMESPACE, symthaea_xenia_policy_commitment,
};

const EXPECTED_NAMESPACE: &str = "symthaea.episodic-continuity.xenia-anchor.v1";
const EXPECTED_POLICY_COMMITMENT: [u8; 32] = [
    0xa7, 0xd9, 0x7c, 0xf9, 0x2f, 0xfc, 0x98, 0x61,
    0x4c, 0x3a, 0x63, 0x87, 0x54, 0x37, 0x8a, 0x76,
    0x2e, 0x9a, 0x5f, 0x77, 0x22, 0xe7, 0x19, 0x3c,
    0x6a, 0xe2, 0x7b, 0x5d, 0xfd, 0x6d, 0x1e, 0x39,
];

#[test]
fn xenia_symthaea_v1_namespace_and_policy_commitment_are_stable() {
    assert_eq!(XENIA_SYMTHAEA_CONTINUITY_NAMESPACE, EXPECTED_NAMESPACE);
    assert_eq!(
        symthaea_xenia_policy_commitment().0,
        EXPECTED_POLICY_COMMITMENT
    );
}
