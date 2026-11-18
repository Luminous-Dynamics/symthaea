// Copyright (C) 2026 Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Targeted static regression for the swarm portion of SWARM-AUTH-QUAL-001.
//!
//! This registry intentionally covers only the swarm cognition representation
//! and its manager accessor. It is not a repository-wide authority audit.

const AUDITED_SWARM_SURFACES: &[(&str, &str)] = &[
    (
        "src/cognitive_loop/managers/swarm_consciousness.rs",
        include_str!("../src/cognitive_loop/managers/swarm_consciousness.rs"),
    ),
    (
        "src/cognitive_loop/managers/swarm_manager.rs",
        include_str!("../src/cognitive_loop/managers/swarm_manager.rs"),
    ),
];

#[test]
fn registered_swarm_surfaces_do_not_reintroduce_authority_delegation_names() {
    let forbidden = [
        "AuthorityDelegation",
        "inherited_phi",
        "delegation_ratio",
        "swarm_consciousness.delegation",
        "effective_phi",
    ];

    for (path, source) in AUDITED_SWARM_SURFACES {
        for token in forbidden {
            assert!(
                !source.contains(token),
                "{path} reintroduced forbidden swarm authority token {token:?}"
            );
        }
    }
}

#[test]
fn collective_cognitive_influence_has_no_authority_bearing_fields() {
    let source = AUDITED_SWARM_SURFACES
        .iter()
        .find(|(path, _)| path.ends_with("swarm_consciousness.rs"))
        .map(|(_, source)| *source)
        .expect("swarm cognition source must be in the static registry");

    let marker = "pub struct CollectiveCognitiveInfluence";
    let start = source
        .find(marker)
        .expect("typed cognition-only collective influence must exist");
    let body_start = source[start..]
        .find('{')
        .map(|offset| start + offset + 1)
        .expect("collective influence struct must have a body");
    let body_end = source[body_start..]
        .find("\n}")
        .map(|offset| body_start + offset)
        .expect("collective influence struct must close");
    let body = &source[body_start..body_end];

    for forbidden_field in ["authority", "capability", "permit", "lease", "delegation"] {
        assert!(
            !body.contains(forbidden_field),
            "CollectiveCognitiveInfluence must not expose {forbidden_field:?} as data"
        );
    }

    for required_field in [
        "individual_phi",
        "collective_phi",
        "blended_cognitive_phi",
        "collective_influence_weight",
        "in_conscious_collective",
    ] {
        assert!(
            body.contains(required_field),
            "CollectiveCognitiveInfluence lost cognitive field {required_field:?}"
        );
    }
}

#[test]
fn swarm_manager_exposes_a_typed_cognitive_influence_not_an_effective_phi_scalar() {
    let source = AUDITED_SWARM_SURFACES
        .iter()
        .find(|(path, _)| path.ends_with("swarm_manager.rs"))
        .map(|(_, source)| *source)
        .expect("swarm manager source must be in the static registry");

    let signature = "pub fn collective_cognitive_influence(";
    let start = source
        .find(signature)
        .expect("typed collective cognitive influence accessor must exist");
    let snippet = &source[start..source[start..].find('{').map(|n| start + n).unwrap_or(source.len())];

    assert!(
        snippet.contains("&super::swarm_consciousness::CollectiveCognitiveInfluence"),
        "swarm manager accessor must return the cognition-only type by reference"
    );
    assert!(
        !snippet.contains("f64"),
        "swarm manager must not expose collective Phi as an untyped authority-adjacent scalar"
    );
}
