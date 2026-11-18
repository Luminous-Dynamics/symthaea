//! First-order, SI-unit stem-section and beam-test reference calculations.
//!
//! These equations are deliberately limited to small-deflection,
//! linear-elastic Euler-Bernoulli beam assumptions. Real crop stems can be
//! tapered, anisotropic, viscoelastic, hydrated, locally damaged and root-
//! coupled. Treat these functions as transparent baselines for interpreting
//! controlled tests, never as a causal classifier or a substitute for measured
//! force-displacement curves.

use std::f64::consts::PI;

/// Invalid or out-of-domain input to a reference mechanics calculation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MechanicsError {
    NonFinite(&'static str),
    NonPositive(&'static str),
    InvalidInnerAxes,
    InvalidBeamProfile(&'static str),
}

/// Concentric hollow elliptical cross-section aligned to its principal axes.
///
/// All semi-axes are in metres. Both inner semi-axes may be zero for a solid
/// ellipse. Otherwise both must be strictly positive and smaller than the
/// corresponding outer semi-axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EllipticalHollowSection {
    pub outer_semi_axis_x_m: f64,
    pub outer_semi_axis_y_m: f64,
    pub inner_semi_axis_x_m: f64,
    pub inner_semi_axis_y_m: f64,
}

/// Cross-section properties about the two aligned principal axes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SectionProperties {
    /// Second moment of area about the x axis, in m^4.
    pub second_moment_area_about_x_m4: f64,
    /// Second moment of area about the y axis, in m^4.
    pub second_moment_area_about_y_m4: f64,
    /// Extreme-fibre section modulus for bending about x, in m^3.
    pub section_modulus_about_x_m3: f64,
    /// Extreme-fibre section modulus for bending about y, in m^3.
    pub section_modulus_about_y_m3: f64,
}

fn positive_finite(value: f64, name: &'static str) -> Result<f64, MechanicsError> {
    if !value.is_finite() {
        return Err(MechanicsError::NonFinite(name));
    }
    if value <= 0.0 {
        return Err(MechanicsError::NonPositive(name));
    }
    Ok(value)
}

fn finite(value: f64, name: &'static str) -> Result<f64, MechanicsError> {
    if value.is_finite() {
        Ok(value)
    } else {
        Err(MechanicsError::NonFinite(name))
    }
}

impl EllipticalHollowSection {
    /// Construct a concentric elliptical section after validating its axes.
    pub fn new(
        outer_semi_axis_x_m: f64,
        outer_semi_axis_y_m: f64,
        inner_semi_axis_x_m: f64,
        inner_semi_axis_y_m: f64,
    ) -> Result<Self, MechanicsError> {
        positive_finite(outer_semi_axis_x_m, "outer_semi_axis_x_m")?;
        positive_finite(outer_semi_axis_y_m, "outer_semi_axis_y_m")?;
        finite(inner_semi_axis_x_m, "inner_semi_axis_x_m")?;
        finite(inner_semi_axis_y_m, "inner_semi_axis_y_m")?;

        if inner_semi_axis_x_m < 0.0 || inner_semi_axis_y_m < 0.0 {
            return Err(MechanicsError::InvalidInnerAxes);
        }

        let solid = inner_semi_axis_x_m == 0.0 && inner_semi_axis_y_m == 0.0;
        let hollow = inner_semi_axis_x_m > 0.0
            && inner_semi_axis_y_m > 0.0
            && inner_semi_axis_x_m < outer_semi_axis_x_m
            && inner_semi_axis_y_m < outer_semi_axis_y_m;

        if !solid && !hollow {
            return Err(MechanicsError::InvalidInnerAxes);
        }

        Ok(Self {
            outer_semi_axis_x_m,
            outer_semi_axis_y_m,
            inner_semi_axis_x_m,
            inner_semi_axis_y_m,
        })
    }

    /// Compute principal second moments of area and extreme-fibre section
    /// moduli for a concentric, aligned elliptical annulus.
    pub fn properties(self) -> SectionProperties {
        let outer_x = self.outer_semi_axis_x_m;
        let outer_y = self.outer_semi_axis_y_m;
        let inner_x = self.inner_semi_axis_x_m;
        let inner_y = self.inner_semi_axis_y_m;

        let ix = PI / 4.0 * (outer_x * outer_y.powi(3) - inner_x * inner_y.powi(3));
        let iy = PI / 4.0 * (outer_x.powi(3) * outer_y - inner_x.powi(3) * inner_y);

        SectionProperties {
            second_moment_area_about_x_m4: ix,
            second_moment_area_about_y_m4: iy,
            section_modulus_about_x_m3: ix / outer_y,
            section_modulus_about_y_m3: iy / outer_x,
        }
    }
}

/// Infer flexural rigidity EI (N·m²) from the linear force/displacement slope
/// of a simply supported, centre-loaded three-point bend test:
/// EI = k L³ / 48.
///
/// The input slope must be measured in the linear, nondestructive region.
pub fn three_point_bend_flexural_rigidity(
    force_displacement_slope_n_per_m: f64,
    support_span_m: f64,
) -> Result<f64, MechanicsError> {
    let slope = positive_finite(
        force_displacement_slope_n_per_m,
        "force_displacement_slope_n_per_m",
    )?;
    let span = positive_finite(support_span_m, "support_span_m")?;
    Ok(slope * span.powi(3) / 48.0)
}

/// Peak mid-span bending moment (N·m) for a simply supported beam with a
/// single centre point load: M_max = F L / 4.
pub fn three_point_bend_peak_moment(
    force_n: f64,
    support_span_m: f64,
) -> Result<f64, MechanicsError> {
    let force = positive_finite(force_n, "force_n")?;
    let span = positive_finite(support_span_m, "support_span_m")?;
    Ok(force * span / 4.0)
}

/// Infer EI (N·m²) for an ideal cantilever with a transverse tip force:
/// delta = F L³ / (3 EI).
///
/// Use only for a declared cantilever fixture and a small-deflection regime.
pub fn cantilever_tip_load_flexural_rigidity(
    force_n: f64,
    cantilever_length_m: f64,
    tip_deflection_m: f64,
) -> Result<f64, MechanicsError> {
    let force = positive_finite(force_n, "force_n")?;
    let length = positive_finite(cantilever_length_m, "cantilever_length_m")?;
    let deflection = positive_finite(tip_deflection_m, "tip_deflection_m")?;
    Ok(force * length.powi(3) / (3.0 * deflection))
}

/// Base bending moment (N·m) for an ideal cantilever with a transverse tip
/// force: M_base = F L.
pub fn cantilever_base_moment(
    force_n: f64,
    cantilever_length_m: f64,
) -> Result<f64, MechanicsError> {
    let force = positive_finite(force_n, "force_n")?;
    let length = positive_finite(cantilever_length_m, "cantilever_length_m")?;
    Ok(force * length)
}

/// Signed curvature (1/m) under a local, linear-elastic beam approximation:
/// kappa = M / EI. The sign follows the supplied moment convention.
pub fn curvature_from_moment(
    bending_moment_nm: f64,
    flexural_rigidity_nm2: f64,
) -> Result<f64, MechanicsError> {
    finite(bending_moment_nm, "bending_moment_nm")?;
    let rigidity = positive_finite(flexural_rigidity_nm2, "flexural_rigidity_nm2")?;
    Ok(bending_moment_nm / rigidity)
}

/// Absolute outer-fibre bending stress (Pa) for a declared bending axis:
/// sigma_max = |M| / Z. Supply the matching principal-axis section modulus.
pub fn outer_fibre_bending_stress_pa(
    bending_moment_nm: f64,
    section_modulus_m3: f64,
) -> Result<f64, MechanicsError> {
    finite(bending_moment_nm, "bending_moment_nm")?;
    let modulus = positive_finite(section_modulus_m3, "section_modulus_m3")?;
    Ok(bending_moment_nm.abs() / modulus)
}


/// One piecewise-constant flexural-rigidity interval along a stem axis.
/// Positions are measured from the root in metres; EI is in N·m².
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BeamSegment {
    pub start_m: f64,
    pub end_m: f64,
    pub flexural_rigidity_nm2: f64,
}

/// Piecewise Euler-Bernoulli cantilever profile. Segments must cover [0, L]
/// exactly once, in increasing order, without gaps or overlaps.
#[derive(Debug, Clone, PartialEq)]
pub struct BeamProfile {
    length_m: f64,
    segments: Vec<BeamSegment>,
}

/// Linearized root support. An absent stiffness means rigid in that degree
/// of freedom; finite positive values represent compliant root/soil support.
/// Translational stiffness is N/m and rotational stiffness is N·m/rad.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct RootSupport {
    pub translational_stiffness_n_per_m: Option<f64>,
    pub rotational_stiffness_nm_per_rad: Option<f64>,
}

impl RootSupport {
    /// Ideal clamp: no root translation or rotation.
    pub const RIGID: Self = Self {
        translational_stiffness_n_per_m: None,
        rotational_stiffness_nm_per_rad: None,
    };

    fn validate(self) -> Result<Self, MechanicsError> {
        if let Some(value) = self.translational_stiffness_n_per_m {
            positive_finite(value, "translational_root_stiffness_n_per_m")?;
        }
        if let Some(value) = self.rotational_stiffness_nm_per_rad {
            positive_finite(value, "rotational_root_stiffness_nm_per_rad")?;
        }
        Ok(self)
    }

    fn tip_displacement_under_resultant(self, force_n: f64, moment_nm: f64, length_m: f64) -> f64 {
        let translation = self
            .translational_stiffness_n_per_m
            .map_or(0.0, |stiffness| force_n / stiffness);
        let rotation = self
            .rotational_stiffness_nm_per_rad
            .map_or(0.0, |stiffness| moment_nm / stiffness * length_m);
        translation + rotation
    }
}

impl BeamProfile {
    /// Create a stem profile. Adjacent boundaries may differ only by a
    /// scale-aware floating-point tolerance; the supplied intervals must
    /// otherwise cover the complete length without overlap.
    pub fn new(length_m: f64, segments: Vec<BeamSegment>) -> Result<Self, MechanicsError> {
        positive_finite(length_m, "beam_length_m")?;
        if segments.is_empty() {
            return Err(MechanicsError::InvalidBeamProfile("at least one segment is required"));
        }

        let tolerance = 1.0e-12 * length_m.max(1.0);
        if segments[0].start_m.abs() > tolerance {
            return Err(MechanicsError::InvalidBeamProfile("first segment must start at zero"));
        }

        let mut previous_end = 0.0;
        for segment in &segments {
            finite(segment.start_m, "segment_start_m")?;
            finite(segment.end_m, "segment_end_m")?;
            positive_finite(segment.flexural_rigidity_nm2, "segment_flexural_rigidity_nm2")?;

            if segment.end_m <= segment.start_m {
                return Err(MechanicsError::InvalidBeamProfile(
                    "segment end must exceed segment start",
                ));
            }
            if (segment.start_m - previous_end).abs() > tolerance {
                return Err(MechanicsError::InvalidBeamProfile(
                    "segments must be contiguous and non-overlapping",
                ));
            }
            previous_end = segment.end_m;
        }

        if (previous_end - length_m).abs() > tolerance {
            return Err(MechanicsError::InvalidBeamProfile("segments must end at beam length"));
        }

        Ok(Self { length_m, segments })
    }

    pub fn length_m(&self) -> f64 {
        self.length_m
    }

    pub fn segments(&self) -> &[BeamSegment] {
        &self.segments
    }

    /// Tip deflection under a transverse point force applied at the free tip.
    ///
    /// For piecewise constant EI, the beam contribution is evaluated exactly:
    /// F * sum(((L-a)^3 - (L-b)^3) / (3 EI_segment)).
    /// Root translation and rotation are added using the linear spring model.
    pub fn tip_deflection_under_tip_load(
        &self,
        force_n: f64,
        root_support: RootSupport,
    ) -> Result<f64, MechanicsError> {
        finite(force_n, "force_n")?;
        let root = root_support.validate()?;
        let mut beam_deflection = 0.0;
        for segment in &self.segments {
            let outer = (self.length_m - segment.start_m).powi(3);
            let inner = (self.length_m - segment.end_m).powi(3);
            beam_deflection += (outer - inner) / (3.0 * segment.flexural_rigidity_nm2);
        }
        let applied_beam_deflection = force_n * beam_deflection;
        let root_deflection = root.tip_displacement_under_resultant(
            force_n,
            force_n * self.length_m,
            self.length_m,
        );
        Ok(applied_beam_deflection + root_deflection)
    }

    /// Tip deflection under a uniform transverse load q (N/m) over the full
    /// stem length. This is a quasi-static small-deflection reference model.
    pub fn tip_deflection_under_uniform_load(
        &self,
        load_n_per_m: f64,
        root_support: RootSupport,
    ) -> Result<f64, MechanicsError> {
        finite(load_n_per_m, "load_n_per_m")?;
        let root = root_support.validate()?;
        let mut beam_deflection = 0.0;
        for segment in &self.segments {
            let outer = (self.length_m - segment.start_m).powi(4);
            let inner = (self.length_m - segment.end_m).powi(4);
            beam_deflection += (outer - inner) / (8.0 * segment.flexural_rigidity_nm2);
        }
        let total_force = load_n_per_m * self.length_m;
        let root_moment = load_n_per_m * self.length_m.powi(2) / 2.0;
        let root_deflection = root.tip_displacement_under_resultant(
            total_force,
            root_moment,
            self.length_m,
        );
        Ok(load_n_per_m * beam_deflection + root_deflection)
    }
}

/// Convert Young's modulus (Pa) and a matching principal second moment of area
/// (m⁴) into flexural rigidity EI (N·m²).
pub fn flexural_rigidity_from_modulus(
    young_modulus_pa: f64,
    second_moment_area_m4: f64,
) -> Result<f64, MechanicsError> {
    let modulus = positive_finite(young_modulus_pa, "young_modulus_pa")?;
    let inertia = positive_finite(second_moment_area_m4, "second_moment_area_m4")?;
    Ok(modulus * inertia)
}

#[cfg(test)]
mod tests {
    use super::*;

    const EPS: f64 = 1.0e-10;


    #[test]
    fn modulus_and_second_moment_form_flexural_rigidity() {
        assert!((flexural_rigidity_from_modulus(2.0e9, 1.0e-9).unwrap() - 2.0).abs() < EPS);
    }

    #[test]
    fn uniform_profile_reproduces_constant_ei_cantilever_tip_load_solution() {
        let profile = BeamProfile::new(0.5, vec![BeamSegment {
            start_m: 0.0, end_m: 0.5, flexural_rigidity_nm2: 4.0,
        }]).unwrap();
        let actual = profile.tip_deflection_under_tip_load(2.0, RootSupport::RIGID).unwrap();
        let expected = 2.0 * 0.5_f64.powi(3) / (3.0 * 4.0);
        assert!((actual - expected).abs() < EPS);
    }

    #[test]
    fn equal_stiffness_partition_is_invariant_for_tip_load_and_uniform_load() {
        let one = BeamProfile::new(0.5, vec![BeamSegment {
            start_m: 0.0, end_m: 0.5, flexural_rigidity_nm2: 4.0,
        }]).unwrap();
        let split = BeamProfile::new(0.5, vec![
            BeamSegment { start_m: 0.0, end_m: 0.2, flexural_rigidity_nm2: 4.0 },
            BeamSegment { start_m: 0.2, end_m: 0.5, flexural_rigidity_nm2: 4.0 },
        ]).unwrap();
        assert!((one.tip_deflection_under_tip_load(2.0, RootSupport::RIGID).unwrap()
            - split.tip_deflection_under_tip_load(2.0, RootSupport::RIGID).unwrap()).abs() < EPS);
        let one_uniform =
            one.tip_deflection_under_uniform_load(3.0, RootSupport::RIGID).unwrap();
        let split_uniform =
            split.tip_deflection_under_uniform_load(3.0, RootSupport::RIGID).unwrap();
        assert!((one_uniform - split_uniform).abs() < EPS);
    }

    #[test]
    fn uniform_load_profile_reproduces_closed_form_cantilever_solution() {
        let profile = BeamProfile::new(0.5, vec![BeamSegment {
            start_m: 0.0, end_m: 0.5, flexural_rigidity_nm2: 4.0,
        }]).unwrap();
        let actual = profile.tip_deflection_under_uniform_load(3.0, RootSupport::RIGID).unwrap();
        let expected = 3.0 * 0.5_f64.powi(4) / (8.0 * 4.0);
        assert!((actual - expected).abs() < EPS);
    }

    #[test]
    fn a_stiffer_rootward_segment_reduces_tip_load_deflection() {
        let stiff_at_root = BeamProfile::new(1.0, vec![
            BeamSegment { start_m: 0.0, end_m: 0.5, flexural_rigidity_nm2: 10.0 },
            BeamSegment { start_m: 0.5, end_m: 1.0, flexural_rigidity_nm2: 1.0 },
        ]).unwrap();
        let stiff_at_tip = BeamProfile::new(1.0, vec![
            BeamSegment { start_m: 0.0, end_m: 0.5, flexural_rigidity_nm2: 1.0 },
            BeamSegment { start_m: 0.5, end_m: 1.0, flexural_rigidity_nm2: 10.0 },
        ]).unwrap();
        let rootward = stiff_at_root
            .tip_deflection_under_tip_load(1.0, RootSupport::RIGID)
            .unwrap();
        let tipward = stiff_at_tip.tip_deflection_under_tip_load(1.0, RootSupport::RIGID).unwrap();
        assert!(rootward < tipward);
    }

    #[test]
    fn root_spring_compliance_adds_translational_and_rotational_displacement() {
        let profile = BeamProfile::new(0.5, vec![BeamSegment {
            start_m: 0.0, end_m: 0.5, flexural_rigidity_nm2: 4.0,
        }]).unwrap();
        let rigid = profile.tip_deflection_under_tip_load(2.0, RootSupport::RIGID).unwrap();
        let compliant = profile.tip_deflection_under_tip_load(2.0, RootSupport {
            translational_stiffness_n_per_m: Some(1_000.0),
            rotational_stiffness_nm_per_rad: Some(20.0),
        }).unwrap();
        assert!((compliant - rigid - 0.027).abs() < EPS);
    }

    #[test]
    fn root_spring_compliance_for_uniform_load_uses_resultant_and_base_moment() {
        let profile = BeamProfile::new(0.5, vec![BeamSegment {
            start_m: 0.0, end_m: 0.5, flexural_rigidity_nm2: 4.0,
        }]).unwrap();
        let rigid = profile.tip_deflection_under_uniform_load(3.0, RootSupport::RIGID).unwrap();
        let compliant = profile.tip_deflection_under_uniform_load(3.0, RootSupport {
            translational_stiffness_n_per_m: Some(1_000.0),
            rotational_stiffness_nm_per_rad: Some(20.0),
        }).unwrap();
        assert!((compliant - rigid - 0.010875).abs() < EPS);
    }

    #[test]
    fn beam_profile_rejects_gaps_and_overlaps() {
        let gap = BeamProfile::new(1.0, vec![
            BeamSegment { start_m: 0.0, end_m: 0.4, flexural_rigidity_nm2: 1.0 },
            BeamSegment { start_m: 0.5, end_m: 1.0, flexural_rigidity_nm2: 1.0 },
        ]);
        assert_eq!(
            gap,
            Err(MechanicsError::InvalidBeamProfile(
                "segments must be contiguous and non-overlapping"
            ))
        );
        let overlap = BeamProfile::new(1.0, vec![
            BeamSegment { start_m: 0.0, end_m: 0.6, flexural_rigidity_nm2: 1.0 },
            BeamSegment { start_m: 0.5, end_m: 1.0, flexural_rigidity_nm2: 1.0 },
        ]);
        assert_eq!(
            overlap,
            Err(MechanicsError::InvalidBeamProfile(
                "segments must be contiguous and non-overlapping"
            ))
        );
    }

    #[test]
    fn beam_profile_rejects_missing_coverage_and_nonpositive_ei() {
        assert_eq!(
            BeamProfile::new(1.0, vec![BeamSegment {
                start_m: 0.0, end_m: 0.9, flexural_rigidity_nm2: 1.0,
            }]),
            Err(MechanicsError::InvalidBeamProfile("segments must end at beam length"))
        );
        assert_eq!(
            BeamProfile::new(1.0, vec![BeamSegment {
                start_m: 0.0, end_m: 1.0, flexural_rigidity_nm2: 0.0,
            }]),
            Err(MechanicsError::NonPositive("segment_flexural_rigidity_nm2"))
        );
    }

    #[test]
    fn invalid_root_spring_stiffness_is_rejected() {
        let profile = BeamProfile::new(1.0, vec![BeamSegment {
            start_m: 0.0, end_m: 1.0, flexural_rigidity_nm2: 1.0,
        }]).unwrap();
        assert_eq!(
            profile.tip_deflection_under_tip_load(1.0, RootSupport {
                translational_stiffness_n_per_m: Some(0.0),
                rotational_stiffness_nm_per_rad: None,
            }),
            Err(MechanicsError::NonPositive("translational_root_stiffness_n_per_m"))
        );
    }

    #[test]
    fn solid_ellipse_has_expected_principal_second_moments() {
        let section = EllipticalHollowSection::new(0.01, 0.02, 0.0, 0.0).unwrap();
        let p = section.properties();
        let expected_ix = PI / 4.0 * 0.01 * 0.02_f64.powi(3);
        let expected_iy = PI / 4.0 * 0.01_f64.powi(3) * 0.02;
        assert!((p.second_moment_area_about_x_m4 - expected_ix).abs() < EPS);
        assert!((p.second_moment_area_about_y_m4 - expected_iy).abs() < EPS);
    }

    #[test]
    fn hollow_ellipse_has_less_inertia_than_solid_outer_ellipse() {
        let solid = EllipticalHollowSection::new(0.01, 0.02, 0.0, 0.0)
            .unwrap()
            .properties();
        let hollow = EllipticalHollowSection::new(0.01, 0.02, 0.005, 0.01)
            .unwrap()
            .properties();
        assert!(hollow.second_moment_area_about_x_m4 < solid.second_moment_area_about_x_m4);
        assert!(hollow.second_moment_area_about_y_m4 < solid.second_moment_area_about_y_m4);
    }

    #[test]
    fn ellipse_principal_axes_are_anisotropic_when_aspect_ratio_is_not_one() {
        let p = EllipticalHollowSection::new(0.01, 0.02, 0.0, 0.0)
            .unwrap()
            .properties();
        assert!(p.second_moment_area_about_x_m4 < p.second_moment_area_about_y_m4);
    }

    #[test]
    fn section_rejects_invalid_inner_axes() {
        assert_eq!(
            EllipticalHollowSection::new(0.01, 0.02, 0.01, 0.005),
            Err(MechanicsError::InvalidInnerAxes)
        );
        assert_eq!(
            EllipticalHollowSection::new(0.01, 0.02, 0.0, 0.005),
            Err(MechanicsError::InvalidInnerAxes)
        );
        assert_eq!(
            EllipticalHollowSection::new(0.01, 0.02, -0.001, 0.0),
            Err(MechanicsError::InvalidInnerAxes)
        );
    }

    #[test]
    fn three_point_bend_ei_uses_span_cubed_over_48() {
        let ei = three_point_bend_flexural_rigidity(1_000.0, 0.1).unwrap();
        assert!((ei - (1_000.0 * 0.1_f64.powi(3) / 48.0)).abs() < EPS);
    }

    #[test]
    fn three_point_bend_peak_moment_uses_force_times_span_over_four() {
        assert!((three_point_bend_peak_moment(8.0, 0.3).unwrap() - 0.6).abs() < EPS);
    }

    #[test]
    fn cantilever_tip_load_ei_matches_closed_form() {
        let ei = cantilever_tip_load_flexural_rigidity(2.0, 0.5, 0.01).unwrap();
        assert!((ei - (2.0 * 0.5_f64.powi(3) / (3.0 * 0.01))).abs() < EPS);
    }

    #[test]
    fn cantilever_base_moment_is_force_times_length() {
        assert!((cantilever_base_moment(2.0, 0.5).unwrap() - 1.0).abs() < EPS);
    }

    #[test]
    fn curvature_preserves_moment_sign() {
        assert!((curvature_from_moment(-2.0, 4.0).unwrap() + 0.5).abs() < EPS);
    }

    #[test]
    fn outer_fibre_stress_uses_absolute_moment_and_section_modulus() {
        assert!((outer_fibre_bending_stress_pa(-1.0, 1.0e-5).unwrap() - 100_000.0).abs() < EPS);
    }

    #[test]
    fn beam_calculations_reject_non_physical_test_inputs() {
        assert_eq!(
            three_point_bend_flexural_rigidity(0.0, 0.1),
            Err(MechanicsError::NonPositive("force_displacement_slope_n_per_m"))
        );
        assert_eq!(
            three_point_bend_peak_moment(10.0, f64::NAN),
            Err(MechanicsError::NonFinite("support_span_m"))
        );
        assert_eq!(
            cantilever_tip_load_flexural_rigidity(1.0, 0.5, 0.0),
            Err(MechanicsError::NonPositive("tip_deflection_m"))
        );
        assert_eq!(
            outer_fibre_bending_stress_pa(1.0, 0.0),
            Err(MechanicsError::NonPositive("section_modulus_m3"))
        );
    }
}
