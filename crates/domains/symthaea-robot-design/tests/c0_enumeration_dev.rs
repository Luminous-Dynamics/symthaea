// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Development integration harness for C0D exhaustive enumeration/selection.

#[path = "../src/design_parameters.rs"]
mod design_parameters;
#[path = "../src/c0_normalized.rs"]
mod c0_normalized;
#[path = "../src/c0_enumeration.rs"]
mod c0_enumeration;

use c0_enumeration::{
    C0_HEIGHT_PARAMETER_ID, C0_WIDTH_PARAMETER_ID, C0SelectionPolicyV1,
    c0_pareto_frontier, enumerate_c0_domain, select_c0_candidate,
};
use c0_normalized::{C0MatchedInvariantProfileId, PositiveRationalV1};
use design_parameters::{
    DesignLengthDomainV1, DesignLengthParameterV1, DesignLengthUm, DesignParameterId,
    DesignParameterSetV1, DesignSearchDomainV1,
};

fn id(value: &str) -> DesignParameterId {
    DesignParameterId::new(value).unwrap()
}

fn domain() -> DesignSearchDomainV1 {
    DesignSearchDomainV1::new(vec![
        DesignLengthDomainV1::Explicit {
            id: id(C0_WIDTH_PARAMETER_ID),
            values: [16_000, 18_000, 20_000, 22_000, 24_000]
                .into_iter()
                .map(DesignLengthUm::from_micrometres)
                .collect(),
        },
        DesignLengthDomainV1::Explicit {
            id: id(C0_HEIGHT_PARAMETER_ID),
            values: [4_800, 5_400, 6_000, 6_600, 7_200]
                .into_iter()
                .map(DesignLengthUm::from_micrometres)
                .collect(),
        },
    ])
    .unwrap()
}

fn baseline() -> DesignParameterSetV1 {
    DesignParameterSetV1::new(vec![
        DesignLengthParameterV1 {
            id: id(C0_WIDTH_PARAMETER_ID),
            value: DesignLengthUm::from_micrometres(20_000),
        },
        DesignLengthParameterV1 {
            id: id(C0_HEIGHT_PARAMETER_ID),
            value: DesignLengthUm::from_micrometres(6_000),
        },
    ])
    .unwrap()
}

#[test]
fn frozen_reference_grid_enumerates_and_selects_deterministically() {
    let candidates = enumerate_c0_domain(
        &domain(),
        &baseline(),
        C0MatchedInvariantProfileId::from_bytes([0xA5; 32]),
    )
    .unwrap();
    assert_eq!(candidates.len(), 25);
    assert_eq!(c0_pareto_frontier(&candidates).unwrap().len(), 9);

    let selected = select_c0_candidate(
        &candidates,
        C0SelectionPolicyV1 {
            max_deflection_ratio: PositiveRationalV1::new(19, 20).unwrap(),
            max_stress_ratio: PositiveRationalV1::new(1, 1).unwrap(),
        },
    )
    .unwrap();

    assert_eq!(selected.width.micrometres(), 16_000);
    assert_eq!(selected.height.micrometres(), 7_200);
    assert_eq!(selected.normalized.mass_ratio, PositiveRationalV1::new(24, 25).unwrap());
    assert_eq!(
        selected.normalized.deflection_ratio,
        PositiveRationalV1::new(625, 864).unwrap()
    );
    assert_eq!(
        selected.normalized.stress_ratio,
        PositiveRationalV1::new(125, 144).unwrap()
    );
}
