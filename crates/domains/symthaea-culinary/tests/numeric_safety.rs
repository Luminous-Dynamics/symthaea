//! Adversarial numeric-input regression tests.
//!
//! NaN comparisons are false in IEEE-754 arithmetic. Validators must reject
//! invalid input explicitly rather than accidentally treating it as safe.

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

fn assert_invalid_numeric(result: Result<(), CulinaryViolation>) {
    assert!(
        matches!(result, Err(CulinaryViolation::InvalidNumericInput { .. })),
        "expected InvalidNumericInput, got {result:?}"
    );
}

#[test]
fn emulsion_rejects_nan_instead_of_accepting_it() {
    assert_invalid_numeric(validate_emulsion(&Emulsion {
        dispersed_phase_fraction: f64::NAN,
    }));
}

#[test]
fn hydration_rejects_negative_mass_even_when_ratio_looks_valid() {
    assert_invalid_numeric(validate_hydration(&Hydration {
        dough: DoughClass::Bread,
        flour_g: -100.0,
        water_g: -70.0,
    }));
}

#[test]
fn dairy_rejects_non_finite_and_out_of_range_ph() {
    assert_invalid_numeric(validate_dairy(&DairyAcidification {
        ph: f64::NAN,
        should_curdle: false,
    }));
    assert_invalid_numeric(validate_dairy(&DairyAcidification {
        ph: 15.0,
        should_curdle: false,
    }));
}

#[test]
fn nutrition_rejects_nan_and_negative_nutrient_quantities() {
    let nan_profile = NutrientProfile {
        sodium_mg: f64::NAN,
        ..Default::default()
    };
    assert!(
        validate_nutrition(&nan_profile)
            .iter()
            .any(|v| matches!(v, CulinaryViolation::InvalidNumericInput { field: "nutrition.sodium_mg", .. }))
    );

    let negative_profile = NutrientProfile {
        protein_g: -1.0,
        ..Default::default()
    };
    assert!(
        validate_nutrition(&negative_profile)
            .iter()
            .any(|v| matches!(v, CulinaryViolation::InvalidNumericInput { field: "nutrition.protein_g", .. }))
    );
}

#[test]
fn pasteurization_rejects_nan_target_and_non_finite_trajectory() {
    assert_invalid_numeric(validate_pasteurization(&Pasteurization {
        pathogen: Pathogen::SalmonellaPoultry,
        trajectory: ThermalTrajectory::hold(60.0, 5.0),
        required_log_reduction: Some(f64::NAN),
    }));

    assert_invalid_numeric(validate_pasteurization(&Pasteurization {
        pathogen: Pathogen::SalmonellaPoultry,
        trajectory: ThermalTrajectory::new(vec![(0.0, 60.0), (f64::NAN, 60.0)]),
        required_log_reduction: None,
    }));
}

#[test]
fn process_validators_reject_empty_trajectories() {
    let empty = ThermalTrajectory::new(vec![]);

    assert_invalid_numeric(validate_coagulation(&Coagulation {
        protein: Protein::EggWhite,
        trajectory: empty.clone(),
    }));
    assert_invalid_numeric(validate_candy(&Candy {
        stage: SugarStage::SoftBall,
        trajectory: empty.clone(),
    }));
    assert_invalid_numeric(validate_frying(&FryingFat {
        fat: Fat::CanolaOil,
        trajectory: empty,
    }));
}

#[test]
fn process_validators_reject_below_absolute_zero() {
    let impossible = ThermalTrajectory::hold(-300.0, 5.0);
    assert_invalid_numeric(validate_candy(&Candy {
        stage: SugarStage::SoftBall,
        trajectory: impossible,
    }));
}

#[test]
fn pasteurization_rejects_overflowed_lethality_calculation() {
    // A finite but absurd input can overflow the exponential lethality model.
    // The validator must reject the result rather than interpret infinity as safe.
    assert_invalid_numeric(validate_pasteurization(&Pasteurization {
        pathogen: Pathogen::SalmonellaPoultry,
        trajectory: ThermalTrajectory::hold(f64::MAX, 5.0),
        required_log_reduction: None,
    }));
}

#[test]
fn process_validators_reject_reversed_time() {
    let reversed = ThermalTrajectory::new(vec![(5.0, 60.0), (0.0, 60.0)]);
    assert_invalid_numeric(validate_candy(&Candy {
        stage: SugarStage::SoftBall,
        trajectory: reversed,
    }));
}

#[test]
fn whole_spec_keeps_invalid_numeric_values_from_passing() {
    let spec = CulinarySpec::new("adversarial")
        .with_emulsion(Emulsion {
            dispersed_phase_fraction: f64::INFINITY,
        });
    let violations = validate(&spec);
    assert!(
        !violations.is_empty(),
        "non-finite values must never result in an empty violation set"
    );
    assert!(
        violations
            .iter()
            .any(|v| matches!(v, CulinaryViolation::InvalidNumericInput { .. }))
    );
}
