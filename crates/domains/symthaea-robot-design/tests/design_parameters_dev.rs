// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Development-only executable specification for ROB-DESIGN-001B0.
//!
//! This integration harness compiles the production-intended module directly
//! by path while the parent ROB-DESIGN-001A source remains qualification-gated.
//! It must be replaced by the normal library export path before 001B0 can be
//! considered a mergeable/qualified product tranche.

#[path = "../src/design_parameters.rs"]
mod design_parameters;

use design_parameters::{
    DesignLengthDomainV1, DesignLengthParameterV1, DesignLengthUm, DesignParameterId,
    DesignParameterSetV1, DesignSearchDomainV1,
};

fn id(value: &str) -> DesignParameterId {
    DesignParameterId::new(value).unwrap()
}

fn parameter(name: &str, micrometres: u64) -> DesignLengthParameterV1 {
    DesignLengthParameterV1 {
        id: id(name),
        value: DesignLengthUm::from_micrometres(micrometres),
    }
}

#[test]
fn c0_reference_domain_accepts_exact_baseline_and_candidate() {
    let domain = DesignSearchDomainV1::new(vec![
        DesignLengthDomainV1::Explicit {
            id: id("section-width"),
            values: vec![
                DesignLengthUm::from_micrometres(16_000),
                DesignLengthUm::from_micrometres(18_000),
                DesignLengthUm::from_micrometres(20_000),
                DesignLengthUm::from_micrometres(22_000),
                DesignLengthUm::from_micrometres(24_000),
            ],
        },
        DesignLengthDomainV1::Explicit {
            id: id("section-height"),
            values: vec![
                DesignLengthUm::from_micrometres(4_800),
                DesignLengthUm::from_micrometres(5_400),
                DesignLengthUm::from_micrometres(6_000),
                DesignLengthUm::from_micrometres(6_600),
                DesignLengthUm::from_micrometres(7_200),
            ],
        },
    ])
    .unwrap();

    let baseline = DesignParameterSetV1::new(vec![
        parameter("section-width", 20_000),
        parameter("section-height", 6_000),
    ])
    .unwrap();
    let candidate = DesignParameterSetV1::new(vec![
        parameter("section-width", 16_000),
        parameter("section-height", 7_200),
    ])
    .unwrap();

    assert!(domain.admits(&baseline).unwrap());
    assert!(domain.admits(&candidate).unwrap());
    assert_ne!(baseline.id().unwrap(), candidate.id().unwrap());
}

#[test]
fn domain_change_does_not_reidentify_an_existing_design_parameter_set() {
    let set = DesignParameterSetV1::new(vec![parameter("section-width", 20_000)]).unwrap();
    let before = set.id().unwrap();

    let narrow = DesignSearchDomainV1::new(vec![DesignLengthDomainV1::Explicit {
        id: id("section-width"),
        values: vec![DesignLengthUm::from_micrometres(20_000)],
    }])
    .unwrap();
    let broad = DesignSearchDomainV1::new(vec![DesignLengthDomainV1::Explicit {
        id: id("section-width"),
        values: vec![
            DesignLengthUm::from_micrometres(16_000),
            DesignLengthUm::from_micrometres(18_000),
            DesignLengthUm::from_micrometres(20_000),
            DesignLengthUm::from_micrometres(22_000),
            DesignLengthUm::from_micrometres(24_000),
        ],
    }])
    .unwrap();

    assert!(narrow.admits(&set).unwrap());
    assert!(broad.admits(&set).unwrap());
    assert_ne!(narrow.id().unwrap(), broad.id().unwrap());
    assert_eq!(before, set.id().unwrap());
}
