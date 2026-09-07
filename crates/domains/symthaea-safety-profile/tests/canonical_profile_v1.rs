// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_safety_profile::{
    ComponentRequirement, SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1, SafetyConfigurationProfile,
};

#[test]
fn canonical_profile_bytes_match_pinned_cross_tool_v1_vector() {
    let profile = SafetyConfigurationProfile {
        schema_version: SAFETY_CONFIGURATION_PROFILE_SCHEMA_V1.to_owned(),
        profile_id: "test-profile-v1".to_owned(),
        hardware_inventory: ComponentRequirement::Required,
        firmware: ComponentRequirement::NotApplicable,
        software_closure: ComponentRequirement::NotApplicable,
        electrical_topology: ComponentRequirement::NotApplicable,
        thermal_topology: ComponentRequirement::NotApplicable,
        protection_settings: ComponentRequirement::NotApplicable,
        sensor_map: ComponentRequirement::NotApplicable,
        actuator_map: ComponentRequirement::NotApplicable,
        calibration: ComponentRequirement::NotApplicable,
        network_topology: ComponentRequirement::NotApplicable,
    };

    let expected = hex_bytes(
        "73796d74686165613a7361666574792d636f6e66696775726174696f6e2d70726f66696c653a7631000000002873796d74686165612d7361666574792d636f6e66696775726174696f6e2d70726f66696c652d76310000000f746573742d70726f66696c652d76310101020003000400050006000700080009000a00",
    );
    assert_eq!(profile.canonical_bytes().unwrap(), expected);
}

fn hex_bytes(hex: &str) -> Vec<u8> {
    assert_eq!(hex.len() % 2, 0);
    hex.as_bytes()
        .chunks_exact(2)
        .map(|pair| (from_hex(pair[0]) << 4) | from_hex(pair[1]))
        .collect()
}

fn from_hex(byte: u8) -> u8 {
    match byte {
        b'0'..=b'9' => byte - b'0',
        b'a'..=b'f' => byte - b'a' + 10,
        _ => panic!("invalid hex byte"),
    }
}
