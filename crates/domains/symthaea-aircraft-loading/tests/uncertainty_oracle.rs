//! Independent corner-enumeration oracle for the bounded aircraft-loading uncertainty API.

use symthaea_aircraft_loading::{
    AircraftProfile, AuditInputError, CgEnvelopeBand, ItemLoadUncertainty, LoadedItem,
    LoadedItemKind, LoadingPhase, LoadingScenario, LoadingStation, LoadingUncertaintyBounds,
    analyze_loading_uncertainty,
};

fn profile() -> AircraftProfile {
    AircraftProfile {
        aircraft_id: "SYNTHETIC-UNCERTAINTY-AIRCRAFT".to_owned(),
        configuration_revision: "uncertainty-config-001".to_owned(),
        source_reference: "fixture://uncertainty/profile".to_owned(),
        source_sha256: "a".repeat(64),
        approval_reference: "fixture://uncertainty/approval-label".to_owned(),
        empty_mass_kg: 1_000.0,
        empty_moment_kg_m: 2_500.0,
        maximum_takeoff_mass_kg: 10_000.0,
        stations: vec![
            LoadingStation {
                station_id: "FWD".to_owned(),
                arm_m: 2.0,
                maximum_station_load_kg: 500.0,
            },
            LoadingStation {
                station_id: "AFT".to_owned(),
                arm_m: 5.0,
                maximum_station_load_kg: 500.0,
            },
        ],
        cg_envelope: vec![CgEnvelopeBand {
            minimum_mass_kg: 1_000.0,
            maximum_mass_kg: 10_000.0,
            forward_cg_m: 0.0,
            aft_cg_m: 10.0,
        }],
    }
}

fn scenario() -> LoadingScenario {
    LoadingScenario {
        scenario_id: "synthetic-uncertainty-001".to_owned(),
        phase: LoadingPhase::Takeoff,
        aircraft_id: "SYNTHETIC-UNCERTAINTY-AIRCRAFT".to_owned(),
        configuration_revision: "uncertainty-config-001".to_owned(),
        items: vec![
            LoadedItem {
                item_id: "fwd-item".to_owned(),
                kind: LoadedItemKind::Cargo,
                station_id: "FWD".to_owned(),
                mass_kg: 100.0,
                securement_evidence_reference: Some("fixture://securement/fwd-item".to_owned()),
            },
            LoadedItem {
                item_id: "aft-item".to_owned(),
                kind: LoadedItemKind::Cargo,
                station_id: "AFT".to_owned(),
                mass_kg: 80.0,
                securement_evidence_reference: Some("fixture://securement/aft-item".to_owned()),
            },
        ],
    }
}

fn bounds() -> LoadingUncertaintyBounds {
    LoadingUncertaintyBounds {
        minimum_empty_mass_kg: 990.0,
        maximum_empty_mass_kg: 1_010.0,
        minimum_empty_moment_kg_m: 2_400.0,
        maximum_empty_moment_kg_m: 2_600.0,
        items: vec![
            ItemLoadUncertainty {
                item_id: "fwd-item".to_owned(),
                minimum_mass_kg: 90.0,
                maximum_mass_kg: 110.0,
                minimum_station_arm_m: 1.8,
                maximum_station_arm_m: 2.2,
            },
            ItemLoadUncertainty {
                item_id: "aft-item".to_owned(),
                minimum_mass_kg: 70.0,
                maximum_mass_kg: 90.0,
                minimum_station_arm_m: 4.8,
                maximum_station_arm_m: 5.2,
            },
        ],
    }
}

#[test]
fn interval_envelope_contains_all_64_independent_endpoint_combinations() {
    let aircraft = profile();
    let loading = scenario();
    let uncertainty = bounds();
    let envelope = analyze_loading_uncertainty(&aircraft, &loading, &uncertainty)
        .expect("synthetic bounded uncertainty input should be valid");

    assert!(envelope.minimum_total_mass_kg <= 1_150.0);
    assert!(envelope.maximum_total_mass_kg >= 1_210.0);
    assert!(envelope.minimum_total_moment_kg_m <= 2_898.0);
    assert!(envelope.maximum_total_moment_kg_m >= 3_310.0);
    assert_eq!(envelope.scenario_id, loading.scenario_id);
    assert_eq!(envelope.configuration_revision, aircraft.configuration_revision);

    for empty_mass in [990.0, 1_010.0] {
        for empty_moment in [2_400.0, 2_600.0] {
            for fwd_mass in [90.0, 110.0] {
                for fwd_arm in [1.8, 2.2] {
                    for aft_mass in [70.0, 90.0] {
                        for aft_arm in [4.8, 5.2] {
                            let total_mass = empty_mass + fwd_mass + aft_mass;
                            let total_moment =
                                empty_moment + fwd_mass * fwd_arm + aft_mass * aft_arm;
                            let cg = total_moment / total_mass;
                            assert!(envelope.minimum_total_mass_kg <= total_mass);
                            assert!(envelope.maximum_total_mass_kg >= total_mass);
                            assert!(envelope.minimum_total_moment_kg_m <= total_moment);
                            assert!(envelope.maximum_total_moment_kg_m >= total_moment);
                            assert!(envelope.minimum_center_of_gravity_m <= cg);
                            assert!(envelope.maximum_center_of_gravity_m >= cg);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn uncertainty_aggregation_is_deterministic_under_item_bound_reordering() {
    let aircraft = profile();
    let loading = scenario();
    let mut first = bounds();
    let mut second = bounds();
    second.items.reverse();
    assert_eq!(
        analyze_loading_uncertainty(&aircraft, &loading, &first).unwrap(),
        analyze_loading_uncertainty(&aircraft, &loading, &second).unwrap()
    );
    first.items.clear();
    assert!(matches!(
        analyze_loading_uncertainty(&aircraft, &loading, &first),
        Err(AuditInputError::InvalidUncertainty(_))
    ));
}

#[test]
fn uncertainty_bounds_must_contain_nominal_mass_and_arm() {
    let aircraft = profile();
    let loading = scenario();
    let mut invalid_mass = bounds();
    invalid_mass.items[0].maximum_mass_kg = 99.0;
    assert!(matches!(
        analyze_loading_uncertainty(&aircraft, &loading, &invalid_mass),
        Err(AuditInputError::InvalidUncertainty(_))
    ));

    let mut invalid_arm = bounds();
    invalid_arm.items[1].minimum_station_arm_m = 5.1;
    assert!(matches!(
        analyze_loading_uncertainty(&aircraft, &loading, &invalid_arm),
        Err(AuditInputError::InvalidUncertainty(_))
    ));
}

#[test]
fn empty_aircraft_bounds_must_contain_nominal_profile_values() {
    let aircraft = profile();
    let loading = scenario();
    let mut invalid = bounds();
    invalid.minimum_empty_mass_kg = 1_001.0;
    assert!(matches!(
        analyze_loading_uncertainty(&aircraft, &loading, &invalid),
        Err(AuditInputError::InvalidUncertainty(_))
    ));
}
