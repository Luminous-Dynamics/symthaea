//! GROUND-TRUTH TESTS for the Phase-1 invariant validators (CULINARY_PLAN Phase 1b).
//! Each asserts the plan's published-bound example: the physically-impossible spec
//! is rejected, the physically-sound one passes. These are the differentiated core
//! — a flat recipe generator cannot make these calls.

use symthaea_culinary::nutrition::NutrientProfile;
use symthaea_culinary::spec::{
    Candy, Coagulation, CulinarySpec, DairyAcidification, DoughClass, Emulsion, Fat, FryingFat,
    Hydration, Pasteurization, Pathogen, Protein, SugarStage,
};
use symthaea_culinary::thermal::ThermalTrajectory;
use symthaea_culinary::validate::{
    CulinaryViolation, validate, validate_candy, validate_coagulation, validate_dairy,
    validate_emulsion, validate_frying, validate_hydration, validate_nutrition,
    validate_pasteurization,
};

// --- Nutrition: FDA Daily Value limits (sodium 2,300 mg, added sugar 50 g) ---

#[test]
fn a_very_salty_very_sweet_profile_is_flagged_on_both_axes() {
    let n = NutrientProfile {
        sodium_mg: 4000.0,
        added_sugar_g: 90.0,
        ..Default::default()
    };
    assert_eq!(validate_nutrition(&n).len(), 2);
}

#[test]
fn a_modest_profile_within_daily_values_passes() {
    let n = NutrientProfile {
        protein_g: 25.0,
        carb_g: 40.0,
        fat_g: 15.0,
        sodium_mg: 500.0,
        added_sugar_g: 8.0,
        ..Default::default()
    };
    assert!(validate_nutrition(&n).is_empty());
    // Atwater: 25*4 + 40*4 + 15*9 = 100 + 160 + 135 = 395 kcal.
    assert!((n.energy_kcal() - 395.0).abs() < 1e-9);
}

// --- Emulsion: φ ≤ random close packing (0.7405) -----------------------------

#[test]
fn mayonnaise_above_close_packing_is_rejected() {
    // 80 % oil: droplets cannot pack — breaks.
    let r = validate_emulsion(&Emulsion {
        dispersed_phase_fraction: 0.80,
    });
    assert!(matches!(r, Err(CulinaryViolation::EmulsionBreaks { .. })));
}

#[test]
fn mayonnaise_below_close_packing_passes() {
    // 70 % oil: a real, stable mayonnaise.
    assert!(
        validate_emulsion(&Emulsion {
            dispersed_phase_fraction: 0.70,
        })
        .is_ok()
    );
}

// --- Coagulation: egg set points; custard curdle point -----------------------

#[test]
fn custard_held_too_hot_is_flagged() {
    // A stirred custard taken to 90 °C curdles.
    let c = Coagulation {
        protein: Protein::Custard,
        trajectory: ThermalTrajectory::new(vec![(0.0, 20.0), (8.0, 90.0)]),
    };
    assert!(matches!(
        validate_coagulation(&c),
        Err(CulinaryViolation::Overcoagulated { .. })
    ));
}

#[test]
fn custard_cooked_gently_passes() {
    // Crème anglaise brought to 80 °C: set but below the 82 °C curdle point.
    let c = Coagulation {
        protein: Protein::Custard,
        trajectory: ThermalTrajectory::new(vec![(0.0, 20.0), (10.0, 80.0)]),
    };
    assert!(validate_coagulation(&c).is_ok());
}

// --- Pasteurization: Salmonella 7-log in poultry (D₆₀=0.396, z=5.56) ----------

#[test]
fn sous_vide_chicken_at_55c_for_10min_fails() {
    // Below reference temperature: nowhere near a 7-log kill in 10 minutes.
    let p = Pasteurization {
        pathogen: Pathogen::SalmonellaPoultry,
        trajectory: ThermalTrajectory::hold(55.0, 10.0),
        required_log_reduction: None,
    };
    assert!(matches!(
        validate_pasteurization(&p),
        Err(CulinaryViolation::InsufficientPasteurization { .. })
    ));
}

#[test]
fn sous_vide_chicken_at_60c_sufficient_hold_passes() {
    // At reference 60 °C, 7-log needs 7 × 0.396 = 2.77 min; a 5-min hold clears it.
    let p = Pasteurization {
        pathogen: Pathogen::SalmonellaPoultry,
        trajectory: ThermalTrajectory::hold(60.0, 5.0),
        required_log_reduction: None,
    };
    assert!(validate_pasteurization(&p).is_ok());
}

// --- Hydration: baker's-percentage windows -----------------------------------

#[test]
fn under_hydrated_bread_is_flagged() {
    // 40 % hydration is far below the 60–85 % bread window.
    let h = Hydration {
        dough: DoughClass::Bread,
        flour_g: 1000.0,
        water_g: 400.0,
    };
    assert!(matches!(
        validate_hydration(&h),
        Err(CulinaryViolation::HydrationOutOfRange { .. })
    ));
}

#[test]
fn well_hydrated_bread_passes() {
    // 70 % hydration: a standard rustic loaf.
    let h = Hydration {
        dough: DoughClass::Bread,
        flour_g: 1000.0,
        water_g: 700.0,
    };
    assert!(validate_hydration(&h).is_ok());
}

// --- Candy: sugar-syrup stage windows -----------------------------------------

#[test]
fn syrup_at_108c_undercooked_for_soft_ball() {
    // Below the 112-116 °C soft-ball window.
    let c = Candy {
        stage: SugarStage::SoftBall,
        trajectory: ThermalTrajectory::hold(108.0, 2.0),
    };
    assert!(matches!(
        validate_candy(&c),
        Err(CulinaryViolation::CandyUndercooked { .. })
    ));
}

#[test]
fn syrup_at_114c_passes_soft_ball() {
    let c = Candy {
        stage: SugarStage::SoftBall,
        trajectory: ThermalTrajectory::hold(114.0, 2.0),
    };
    assert!(validate_candy(&c).is_ok());
}

// --- Dairy: casein isoelectric point (pH 4.6) ---------------------------------

#[test]
fn fresh_milk_wont_curdle_into_paneer() {
    // pH 6.5 fresh milk, intent to curdle it: physically won't happen.
    let d = DairyAcidification {
        ph: 6.5,
        should_curdle: true,
    };
    assert!(matches!(
        validate_dairy(&d),
        Err(CulinaryViolation::DairyWontCurdle { .. })
    ));
}

#[test]
fn acidified_milk_curdles_into_paneer() {
    // pH 4.2, intent to curdle it: matches reality.
    let d = DairyAcidification {
        ph: 4.2,
        should_curdle: true,
    };
    assert!(validate_dairy(&d).is_ok());
}

// --- Frying: cooking-fat smoke points ------------------------------------------

#[test]
fn extra_virgin_olive_oil_smokes_at_searing_heat() {
    let fr = FryingFat {
        fat: Fat::ExtraVirginOliveOil,
        trajectory: ThermalTrajectory::hold(220.0, 3.0),
    };
    assert!(matches!(
        validate_frying(&fr),
        Err(CulinaryViolation::FatSmoking { .. })
    ));
}

#[test]
fn canola_oil_handles_searing_heat() {
    let fr = FryingFat {
        fat: Fat::CanolaOil,
        trajectory: ThermalTrajectory::hold(190.0, 3.0),
    };
    assert!(validate_frying(&fr).is_ok());
}

// --- Whole-spec: multiple invariants at once ---------------------------------

#[test]
fn full_spec_collects_every_violation() {
    // A deliberately impossible "dish": broken emulsion + curdled custard +
    // unsafe pasteurization + bad hydration + undercooked candy + a dairy
    // intent mismatch + a smoking fat + sodium AND added sugar both over their
    // FDA Daily Values (nutrition contributes 2 of the 9 total). All nine
    // must be reported.
    let spec = CulinarySpec::new("kitchen nightmare")
        .with_emulsion(Emulsion {
            dispersed_phase_fraction: 0.9,
        })
        .with_coagulation(Coagulation {
            protein: Protein::Custard,
            trajectory: ThermalTrajectory::hold(95.0, 5.0),
        })
        .with_pasteurization(Pasteurization {
            pathogen: Pathogen::SalmonellaPoultry,
            trajectory: ThermalTrajectory::hold(50.0, 5.0),
            required_log_reduction: None,
        })
        .with_hydration(Hydration {
            dough: DoughClass::Bread,
            flour_g: 1000.0,
            water_g: 300.0,
        })
        .with_candy(Candy {
            stage: SugarStage::HardCrack,
            trajectory: ThermalTrajectory::hold(108.0, 2.0),
        })
        .with_dairy(DairyAcidification {
            ph: 6.5,
            should_curdle: true,
        })
        .with_frying(FryingFat {
            fat: Fat::ExtraVirginOliveOil,
            trajectory: ThermalTrajectory::hold(220.0, 3.0),
        })
        .with_nutrition(NutrientProfile {
            sodium_mg: 4000.0,
            added_sugar_g: 90.0,
            ..Default::default()
        });
    let violations = validate(&spec);
    assert_eq!(violations.len(), 9, "got: {violations:#?}");
}

#[test]
fn a_sound_spec_is_accepted() {
    // A real hollandaise-ish emulsion + gently-cooked yolk + safe hold + good
    // dough + a real soft-ball caramel + intentionally-acidified paneer-style
    // dairy + a high-smoke-point frying oil.
    let spec = CulinarySpec::new("sunday brunch")
        .with_emulsion(Emulsion {
            dispersed_phase_fraction: 0.68,
        })
        .with_coagulation(Coagulation {
            protein: Protein::EggYolk,
            trajectory: ThermalTrajectory::hold(70.0, 3.0),
        })
        .with_pasteurization(Pasteurization {
            pathogen: Pathogen::SalmonellaPoultry,
            trajectory: ThermalTrajectory::hold(62.0, 4.0),
            required_log_reduction: None,
        })
        .with_hydration(Hydration {
            dough: DoughClass::Bread,
            flour_g: 1000.0,
            water_g: 720.0,
        })
        .with_candy(Candy {
            stage: SugarStage::SoftBall,
            trajectory: ThermalTrajectory::hold(114.0, 2.0),
        })
        .with_dairy(DairyAcidification {
            ph: 4.2,
            should_curdle: true,
        })
        .with_frying(FryingFat {
            fat: Fat::CanolaOil,
            trajectory: ThermalTrajectory::hold(190.0, 3.0),
        })
        .with_nutrition(NutrientProfile {
            protein_g: 20.0,
            carb_g: 30.0,
            fat_g: 25.0,
            sodium_mg: 450.0,
            added_sugar_g: 5.0,
            ..Default::default()
        });
    assert!(
        validate(&spec).is_empty(),
        "unexpected: {:#?}",
        validate(&spec)
    );
}
