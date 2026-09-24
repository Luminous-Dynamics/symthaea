// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Development harness for resource-bounded C0 enumeration.

#[path = "../src/design_parameters.rs"]
mod design_parameters;
#[path = "../src/c0_normalized.rs"]
mod c0_normalized;
#[path = "../src/c0_enumeration.rs"]
mod c0_enumeration;
#[path = "../src/c0_enumeration_guard.rs"]
mod c0_enumeration_guard;

use c0_enumeration::{C0_HEIGHT_PARAMETER_ID, C0_WIDTH_PARAMETER_ID};
use c0_enumeration_guard::{
    C0EnumerationGuardError, DEFAULT_MAX_C0_ENUMERATED_CANDIDATES,
    enumerate_c0_domain_bounded,
};
use c0_normalized::C0MatchedInvariantProfileId;
use design_parameters::{
    DesignLengthDomainV1, DesignLengthParameterV1, DesignLengthUm, DesignParameterId,
    DesignParameterSetV1, DesignSearchDomainV1,
};

fn id(value: &str) -> DesignParameterId {
    DesignParameterId::new(value).unwrap()
}

#[test]
fn million_candidate_cartesian_domain_fails_before_enumeration() {
    let domain = DesignSearchDomainV1::new(vec![
        DesignLengthDomainV1::SteppedInclusive {
            id: id(C0_WIDTH_PARAMETER_ID),
            lower: DesignLengthUm::from_micrometres(1),
            upper: DesignLengthUm::from_micrometres(1_000),
            step: DesignLengthUm::from_micrometres(1),
        },
        DesignLengthDomainV1::SteppedInclusive {
            id: id(C0_HEIGHT_PARAMETER_ID),
            lower: DesignLengthUm::from_micrometres(1),
            upper: DesignLengthUm::from_micrometres(1_000),
            step: DesignLengthUm::from_micrometres(1),
        },
    ])
    .unwrap();
    let baseline = DesignParameterSetV1::new(vec![
        DesignLengthParameterV1 {
            id: id(C0_WIDTH_PARAMETER_ID),
            value: DesignLengthUm::from_micrometres(1),
        },
        DesignLengthParameterV1 {
            id: id(C0_HEIGHT_PARAMETER_ID),
            value: DesignLengthUm::from_micrometres(1),
        },
    ])
    .unwrap();

    assert!(matches!(
        enumerate_c0_domain_bounded(
            &domain,
            &baseline,
            C0MatchedInvariantProfileId::from_bytes([1; 32]),
            DEFAULT_MAX_C0_ENUMERATED_CANDIDATES
        ),
        Err(C0EnumerationGuardError::CandidateBudgetExceeded {
            count: 1_000_000,
            maximum: DEFAULT_MAX_C0_ENUMERATED_CANDIDATES
        })
    ));
}
