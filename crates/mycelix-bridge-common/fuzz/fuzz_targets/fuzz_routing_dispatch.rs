// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
#![no_main]

use arbitrary::Arbitrary;
use libfuzzer_sys::fuzz_target;
use mycelix_bridge_common::routing::{
    resolve_civic_zome, resolve_commons_zome, BridgeDomain, COMMONS_DOMAINS,
    CIVIC_DOMAINS,
};

// ---------------------------------------------------------------------------
// Arbitrary input types
// ---------------------------------------------------------------------------

#[derive(Arbitrary, Debug)]
struct FuzzRoutingInput {
    // Fuzz BridgeDomain::from_str_loose with arbitrary strings
    domain_strings: Vec<Vec<u8>>,

    // Fuzz resolve functions with domain index + arbitrary query_type
    domain_idx: u8,
    query_types: Vec<Vec<u8>>,
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn domain_from_idx(idx: u8) -> BridgeDomain {
    match idx % 12 {
        0 => BridgeDomain::Property,
        1 => BridgeDomain::Housing,
        2 => BridgeDomain::Care,
        3 => BridgeDomain::Mutualaid,
        4 => BridgeDomain::Water,
        5 => BridgeDomain::Food,
        6 => BridgeDomain::Transport,
        7 => BridgeDomain::Support,
        8 => BridgeDomain::Space,
        9 => BridgeDomain::Justice,
        10 => BridgeDomain::Emergency,
        _ => BridgeDomain::Media,
    }
}

// ---------------------------------------------------------------------------
// Fuzz target
// ---------------------------------------------------------------------------

fuzz_target!(|input: FuzzRoutingInput| {
    let domain = domain_from_idx(input.domain_idx);

    // 1. Fuzz BridgeDomain::from_str_loose with arbitrary byte sequences
    for raw_bytes in &input.domain_strings {
        // Convert to String lossily — from_str_loose must handle any UTF-8 string
        let s = String::from_utf8_lossy(raw_bytes);
        let parsed = BridgeDomain::from_str_loose(&s);

        // If parsed successfully, verify round-trip consistency
        if let Some(d) = parsed {
            let canonical = d.as_str();
            let reparsed = BridgeDomain::from_str_loose(canonical);
            assert_eq!(
                reparsed,
                Some(d),
                "round-trip failed: {:?} -> {:?} -> {:?}",
                s,
                canonical,
                reparsed
            );
        }
    }

    // 2. Fuzz resolve_commons_zome and resolve_civic_zome with arbitrary query types
    for raw_qt in &input.query_types {
        let query_type = String::from_utf8_lossy(raw_qt);

        // resolve_commons_zome: must not panic
        let commons_result = resolve_commons_zome(domain, &query_type);

        // Invariant: commons domains must return Some, civic domains must return None
        if domain.is_commons() {
            assert!(
                commons_result.is_some(),
                "commons domain {:?} returned None for query {:?}",
                domain,
                query_type
            );
        }
        if domain.is_civic() {
            assert!(
                commons_result.is_none(),
                "civic domain {:?} returned Some for resolve_commons_zome",
                domain
            );
        }

        // resolve_civic_zome: must not panic
        let civic_result = resolve_civic_zome(domain, &query_type);

        // Invariant: civic domains must return Some, commons domains must return None
        if domain.is_civic() {
            assert!(
                civic_result.is_some(),
                "civic domain {:?} returned None for query {:?}",
                domain,
                query_type
            );
        }
        if domain.is_commons() {
            assert!(
                civic_result.is_none(),
                "commons domain {:?} returned Some for resolve_civic_zome",
                domain
            );
        }

        // If resolved, verify the zome name string is non-empty
        if let Some(zome) = commons_result {
            assert!(!zome.as_str().is_empty(), "commons zome has empty name");
        }
        if let Some(zome) = civic_result {
            assert!(!zome.as_str().is_empty(), "civic zome has empty name");
        }
    }

    // 3. Verify domain classification consistency
    assert!(
        domain.is_commons() || domain.is_civic(),
        "domain {:?} is neither commons nor civic",
        domain
    );
    assert!(
        !(domain.is_commons() && domain.is_civic()),
        "domain {:?} is both commons and civic",
        domain
    );

    // 4. Verify domain constants are consistent
    for d in COMMONS_DOMAINS {
        assert!(d.is_commons(), "{:?} in COMMONS_DOMAINS but is_commons() false", d);
        assert!(!d.is_civic(), "{:?} in COMMONS_DOMAINS but is_civic() true", d);
    }
    for d in CIVIC_DOMAINS {
        assert!(d.is_civic(), "{:?} in CIVIC_DOMAINS but is_civic() false", d);
        assert!(!d.is_commons(), "{:?} in CIVIC_DOMAINS but is_commons() true", d);
    }

    // 5. Fuzz all 12 domains x arbitrary query to stress all resolve paths
    for idx in 0..12u8 {
        let d = domain_from_idx(idx);
        for raw_qt in &input.query_types {
            let qt = String::from_utf8_lossy(raw_qt);
            let _ = resolve_commons_zome(d, &qt);
            let _ = resolve_civic_zome(d, &qt);
        }
    }
});
