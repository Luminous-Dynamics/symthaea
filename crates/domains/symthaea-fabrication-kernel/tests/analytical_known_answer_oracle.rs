//! Independent known-answer cases for the bounded simple-beam analytical model.
//!
//! Expected numbers below are hand-derived from elementary beam/section equations.
//! They are intentionally literal reference values and never call production
//! helpers to compute expected outputs.
//!
//! This qualifies equation implementations for the stated idealized profile only.
//! It is not physical validation, general FEA, or structural safety evidence.

use symthaea_fabrication_kernel::analytical::{
    AnalyticalBackend, AnalyticalInputError, CrossSection, MaterialProperties,
};
use symthaea_fabrication_kernel::units::Meters;

fn assert_close(label: &str, actual: f64, expected: f64, relative_tolerance: f64) {
    assert!(
        actual.is_finite(),
        "{label}: actual value must be finite, got {actual}"
    );
    let allowed = expected.abs().max(1.0e-15) * relative_tolerance + 1.0e-30;
    assert!(
        (actual - expected).abs() <= allowed,
        "{label}: expected {expected:.16e}, got {actual:.16e}, allowed error {allowed:.3e}"
    );
}

fn steel_beam(width: f64, height: f64, length: f64) -> AnalyticalBackend {
    AnalyticalBackend::new(
        MaterialProperties::steel(),
        CrossSection::Rectangle { width, height },
        length,
    )
}

#[test]
fn rectangle_section_properties_match_hand_calculated_reference() {
    // b=0.02 m, h=0.04 m:
    // A=b*h=8e-4 m²; I=b*h³/12=1.066666...e-7 m⁴; Z=I/(h/2).
    let section = CrossSection::Rectangle {
        width: 0.02,
        height: 0.04,
    };
    assert_close("rectangle area [m²]", section.area(), 8.0e-4, 1.0e-12);
    assert_close(
        "rectangle I [m⁴]",
        section.moment_of_inertia(),
        1.0666666666666667e-7,
        1.0e-12,
    );
    assert_close(
        "rectangle Z [m³]",
        section.section_modulus(),
        5.333333333333333e-6,
        1.0e-12,
    );
}

#[test]
fn solid_circle_section_properties_match_hand_calculated_reference() {
    // r=0.01 m: A=πr², I=πr⁴/4, Z=I/r.
    let section = CrossSection::Circle { radius: 0.01 };
    assert_close("circle area [m²]", section.area(), 3.1415926535897932e-4, 1.0e-12);
    assert_close(
        "circle I [m⁴]",
        section.moment_of_inertia(),
        7.853981633974483e-9,
        1.0e-12,
    );
    assert_close(
        "circle Z [m³]",
        section.section_modulus(),
        7.853981633974483e-7,
        1.0e-12,
    );
}

#[test]
fn hollow_circle_section_properties_match_hand_calculated_reference() {
    // ro=0.02 m, ri=0.01 m:
    // A=π(ro²-ri²)=3π*1e-4; I=π(ro⁴-ri⁴)/4=3.75π*1e-8.
    let section = CrossSection::HollowCircle {
        outer_radius: 0.02,
        inner_radius: 0.01,
    };
    assert_close(
        "hollow-circle area [m²]",
        section.area(),
        9.42477796076938e-4,
        1.0e-12,
    );
    assert_close(
        "hollow-circle I [m⁴]",
        section.moment_of_inertia(),
        1.1780972450961724e-7,
        1.0e-12,
    );
    assert_close(
        "hollow-circle Z [m³]",
        section.section_modulus(),
        5.890486225480862e-6,
        1.0e-12,
    );
}

#[test]
fn symmetric_i_beam_section_properties_match_hand_calculated_reference() {
    // bf=0.1, tf=0.01, hw=0.08, tw=0.01 m; total depth=0.1 m.
    // A=2*bf*tf+hw*tw=0.0028 m².
    // Ix=bf*D³/12-(bf-tw)*hw³/12=4.493333...e-6 m⁴.
    let section = CrossSection::IBeam {
        flange_width: 0.1,
        flange_height: 0.01,
        web_height: 0.08,
        web_thickness: 0.01,
    };
    assert_close("I-beam area [m²]", section.area(), 2.8e-3, 1.0e-12);
    assert_close(
        "I-beam I [m⁴]",
        section.moment_of_inertia(),
        4.493333333333333e-6,
        1.0e-12,
    );
    assert_close(
        "I-beam Z [m³]",
        section.section_modulus(),
        8.986666666666666e-5,
        1.0e-12,
    );
}

#[test]
fn axial_stress_matches_force_over_area_oracle() {
    // 1000 N / (0.02 m * 0.04 m) = 1.25 MPa.
    let beam = steel_beam(0.02, 0.04, 1.2);
    assert_close("axial stress [Pa]", beam.axial_stress(1000.0), 1.25e6, 1.0e-12);
}

#[test]
fn centered_point_load_matches_known_bending_stress_and_deflection() {
    // Simply supported, center point load:
    // Mmax=P*L/4=180 N m; Z=5.333333...e-6 m³; sigma=M/Z=33.75 MPa.
    // delta=P*L³/(48EI)=0.0010125 m for E=200 GPa, I=1.066666...e-7 m⁴.
    let beam = steel_beam(0.02, 0.04, 1.2);
    assert_close(
        "center-load bending stress [Pa]",
        beam.bending_stress(600.0),
        33.75e6,
        1.0e-12,
    );
    assert_close(
        "center-load maximum deflection [m]",
        beam.max_deflection(600.0),
        1.0125e-3,
        1.0e-12,
    );
}

#[test]
fn combined_axial_bending_and_average_shear_match_known_answer() {
    // A=8e-4 m²; axial stress=1.25 MPa; bending=33.75 MPa;
    // average shear proxy=(600/2)/A=0.375 MPa; total normal stress=35 MPa.
    let beam = steel_beam(0.02, 0.04, 1.2);
    let result = beam.analyze(1000.0, 600.0);
    let expected_von_mises = ((35.0e6_f64).powi(2)
        + 3.0 * (375_000.0_f64).powi(2))
    .sqrt();

    assert_close("combined axial stress [Pa]", result.axial_stress, 1.25e6, 1.0e-12);
    assert_close("combined bending stress [Pa]", result.bending_stress, 33.75e6, 1.0e-12);
    assert_close("average shear proxy [Pa]", result.shear_stress, 375_000.0, 1.0e-12);
    assert_close("von-Mises-style proxy [Pa]", result.von_mises_stress, expected_von_mises, 1.0e-12);
}

#[test]
fn elastic_modulus_changes_deflection_but_not_elastic_stress() {
    // For fixed geometry and load in this linear-elastic model, stress is
    // independent of E while deflection scales as 1/E.
    let baseline = steel_beam(0.02, 0.04, 1.2);
    let mut doubled_stiffness_material = MaterialProperties::steel();
    doubled_stiffness_material.elastic_modulus *= 2.0;
    let doubled_stiffness = AnalyticalBackend::new(
        doubled_stiffness_material,
        CrossSection::Rectangle {
            width: 0.02,
            height: 0.04,
        },
        1.2,
    );

    assert_close(
        "stress invariant under E change [Pa]",
        doubled_stiffness.bending_stress(600.0),
        baseline.bending_stress(600.0),
        1.0e-12,
    );
    assert_close(
        "deflection halves when E doubles [m]",
        doubled_stiffness.max_deflection(600.0),
        baseline.max_deflection(600.0) / 2.0,
        1.0e-12,
    );
}

#[test]
fn load_and_span_scaling_follow_center_load_beam_equations() {
    let baseline = steel_beam(0.02, 0.04, 1.2);
    let doubled_load = steel_beam(0.02, 0.04, 1.2);
    let doubled_span = steel_beam(0.02, 0.04, 2.4);

    assert_close(
        "stress doubles when load doubles [Pa]",
        doubled_load.bending_stress(1200.0),
        baseline.bending_stress(600.0) * 2.0,
        1.0e-12,
    );
    assert_close(
        "stress doubles when span doubles [Pa]",
        doubled_span.bending_stress(600.0),
        baseline.bending_stress(600.0) * 2.0,
        1.0e-12,
    );
    assert_close(
        "deflection doubles with load [m]",
        doubled_load.max_deflection(1200.0),
        baseline.max_deflection(600.0) * 2.0,
        1.0e-12,
    );
    assert_close(
        "deflection increases eightfold with span [m]",
        doubled_span.max_deflection(600.0),
        baseline.max_deflection(600.0) * 8.0,
        1.0e-12,
    );
}

#[test]
fn zero_load_has_zero_stress_and_deflection_but_unbounded_yield_ratio() {
    // The implementation represents zero computed stress as an infinite
    // yield/stress ratio. That ratio is not an independent safety qualification.
    let beam = steel_beam(0.02, 0.04, 1.2);
    let result = beam.analyze(0.0, 0.0);
    assert_eq!(result.axial_stress, 0.0);
    assert_eq!(result.bending_stress, 0.0);
    assert_eq!(result.shear_stress, 0.0);
    assert_eq!(result.max_deflection, 0.0);
    assert_eq!(result.von_mises_stress, 0.0);
    assert!(result.safety_factor.is_infinite());
}

#[test]
fn checked_constructor_rejects_singular_geometry() {
    let error = AnalyticalBackend::new_checked(
        MaterialProperties::steel(),
        CrossSection::Rectangle {
            width: 0.0,
            height: 0.04,
        },
        Meters::positive(1.2).expect("positive span fixture"),
    )
    .err()
    .expect("zero-width section must be rejected");
    assert!(matches!(
        error,
        AnalyticalInputError::InvalidCrossSection("width")
    ));

    let invalid_circle = CrossSection::HollowCircle {
        outer_radius: 0.01,
        inner_radius: 0.01,
    };
    assert!(matches!(
        invalid_circle.validate(),
        Err(AnalyticalInputError::InvalidCrossSection(_))
    ));
}
