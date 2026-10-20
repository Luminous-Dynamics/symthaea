//! Independent known-answer arithmetic oracle for the aircraft loading API.
//!
//! This test uses bounded, integer-valued masses and arms so the expected mass and moment sums
//! are independently calculated in simple integer arithmetic before the CG division. It does not
//! reuse the production accumulator or station-aggregation logic.

use symthaea_aircraft_loading::{
    AircraftProfile, AuditDisposition, CgEnvelopeBand, LoadedItem, LoadedItemKind, LoadingPhase,
    LoadingScenario, LoadingStation, audit_loading,
};

fn item(id: &str, station_id: &str, mass_kg: f64) -> LoadedItem {
    LoadedItem {
        item_id: id.to_owned(),
        kind: LoadedItemKind::Cargo,
        station_id: station_id.to_owned(),
        mass_kg,
        securement_evidence_reference: Some(format!("fixture://securement/{id}")),
    }
}

fn profile(arm_fwd: f64, arm_aft: f64) -> AircraftProfile {
    AircraftProfile {
        aircraft_id: "SYNTHETIC-ORACLE-AIRCRAFT".to_owned(),
        configuration_revision: "oracle-config-001".to_owned(),
        source_reference: "fixture://synthetic-aircraft-profile".to_owned(),
        source_sha256: "a".repeat(64),
        approval_reference: "fixture://synthetic-approval-label".to_owned(),
        empty_mass_kg: 1_000.0,
        empty_moment_kg_m: 2_500.0,
        maximum_takeoff_mass_kg: 10_000.0,
        stations: vec![
            LoadingStation {
                station_id: "FWD".to_owned(),
                arm_m: arm_fwd,
                maximum_station_load_kg: 100.0,
            },
            LoadingStation {
                station_id: "AFT".to_owned(),
                arm_m: arm_aft,
                maximum_station_load_kg: 100.0,
            },
        ],
        cg_envelope: vec![CgEnvelopeBand {
            minimum_mass_kg: 1.000,
            maximum_mass_kg: 10_000.0,
            forward_cg_m: 0.0,
            aft_cg_m: 10.0,
        }],
    }
}

fn scenario(items: Vec<LoadedItem>) -> LoadingScenario {
    LoadingScenario {
        scenario_id: "synthetic-loading-oracle-001".to_owned(),
        phase: LoadingPhase::Takeoff,
        aircraft_id: "SYNTHETIC-ORACLE-AIRCRAFT".to_owned(),
        configuration_revision: "oracle-config-001".to_owned(),
        items,
    }
}

#[test]
fn exhaustive_small_integer_grid_matches_independent_mass_moment_oracle() {
    let mut checked = 0_usize;

    for arm_fwd in 1_i32..=3 {
        for arm_aft in 4_i32..=6 {
            let aircraft = profile(f64::from(arm_fwd), f64::from(arm_aft));
            for mass_fwd in 1_i32..=10 {
                for mass_aft in 1_i32..=10 {
                    let loaded = scenario(vec![
                        item("fwd-item", "FWD", f64::from(mass_fwd)),
                        item("aft-item", "AFT", f64::from(mass_aft)),
                    ]);

                    // Independent closed-form integer arithmetic. The fixture's empty mass and
                    // moment are explicit constants, and all item masses/arms are small integers.
                    let expected_mass_kg = 1_000_i64 + i64::from(mass_fwd) + i64::from(mass_aft);
                    let expected_moment_kg_m = 2_500_i64
                        + i64::from(mass_fwd) * i64::from(arm_fwd)
                        + i64::from(mass_aft) * i64::from(arm_aft);
                    let expected_cg_m = expected_moment_kg_m as f64 / expected_mass_kg as f64;

                    let report =
                        audit_loading(&aircraft, &loaded).expect("oracle fixture is valid");

                    assert_eq!(report.total_mass_kg, expected_mass_kg as f64);
                    assert_eq!(report.total_moment_kg_m, expected_moment_kg_m as f64);
                    assert!(
                        (report.center_of_gravity_m - expected_cg_m).abs() < 1e-12,
                        "CG mismatch at arms ({arm_fwd},{arm_aft}) and masses ({mass_fwd},{mass_aft})"
                    );
                    assert_eq!(
                        report.disposition,
                        AuditDisposition::WithinDeclaredEnvelopeForHumanReview
                    );
                    checked += 1;
                }
            }
        }
    }

    assert_eq!(checked, 900);
}

#[test]
fn exact_mass_station_and_cg_boundaries_are_inclusive() {
    let mut aircraft = profile(0.0, 4.0);
    aircraft.maximum_takeoff_mass_kg = 2_000.0;
    aircraft.stations[0].maximum_station_load_kg = 1_000.0;
    aircraft.cg_envelope = vec![CgEnvelopeBand {
        minimum_mass_kg: 2_000.0,
        maximum_mass_kg: 2_000.0,
        forward_cg_m: 1.25,
        aft_cg_m: 1.25,
    }];

    let report = audit_loading(
        &aircraft,
        &scenario(vec![item("boundary-item", "FWD", 1_000.0)]),
    )
    .expect("inclusive exact-boundary fixture is valid");

    assert_eq!(report.total_mass_kg, 2_000.0);
    assert_eq!(report.total_moment_kg_m, 2_500.0);
    assert_eq!(report.center_of_gravity_m, 1.25);
    assert_eq!(
        report.disposition,
        AuditDisposition::WithinDeclaredEnvelopeForHumanReview
    );
}
