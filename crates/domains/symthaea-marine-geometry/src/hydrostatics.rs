//! G2c hydrostatic properties of the bounded waterplane-capped polyhedral mesh.
//!
//! The calculations describe the supplied mesh only. They do not establish
//! that the mesh represents an as-built vessel or that any stability criteria
//! are satisfied.

use serde::{Deserialize, Serialize};

use crate::loft::{verify_loft_artifact_integrity, LoftedHullArtifact};
use crate::{admit, Disposition};

const WATERPLANE_TOLERANCE_M: f64 = 1.0e-9;
const MINIMUM_POSITIVE_AREA_M2: f64 = 1.0e-14;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct HydrostaticViolation {
    pub code: String,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HydrostaticProperties {
    pub result_schema_version: u32,
    pub candidate_id: String,
    pub geometry_digest_blake3: String,
    pub loft_manifest_digest_blake3: String,
    pub waterplane_face_range_first: usize,
    pub waterplane_face_count: usize,
    pub water_density_kg_m3: f64,
    pub displacement_volume_m3: f64,
    pub displacement_mass_kg: f64,
    pub displacement_tonnes: f64,
    /// Centre of buoyancy in the declared x-forward, y-starboard, z-down frame.
    pub centre_of_buoyancy_m: [f64; 3],
    pub longitudinal_centre_of_buoyancy_m: f64,
    pub transverse_centre_of_buoyancy_m: f64,
    /// Positive z is below the declared design waterplane.
    pub vertical_centre_of_buoyancy_below_waterplane_m: f64,
    pub waterplane_area_m2: f64,
    pub longitudinal_centre_of_flotation_m: f64,
    pub transverse_centre_of_flotation_m: f64,
    /// Integral of y^2 dA about the longitudinal axis through the waterplane centroid.
    pub transverse_waterplane_second_moment_m4: f64,
    /// Integral of x^2 dA about the transverse axis through the waterplane centroid.
    pub longitudinal_waterplane_second_moment_m4: f64,
    /// Integral of (x-LCF)(y-TCF) dA.
    pub waterplane_product_of_inertia_m4: f64,
    pub transverse_metacentric_radius_bm_t_m: f64,
    pub longitudinal_metacentric_radius_bm_l_m: f64,
    pub tonnes_per_centimetre_immersion: f64,
}

#[derive(Debug, Clone, Copy, Default)]
struct CompensatedSum {
    sum: f64,
    correction: f64,
}

impl CompensatedSum {
    fn add(&mut self, value: f64) {
        let adjusted = value - self.correction;
        let next = self.sum + adjusted;
        self.correction = (next - self.sum) - adjusted;
        self.sum = next;
    }

    fn value(self) -> f64 {
        self.sum
    }
}

fn violation(
    violations: &mut Vec<HydrostaticViolation>,
    code: &str,
    detail: impl Into<String>,
) {
    violations.push(HydrostaticViolation {
        code: code.to_owned(),
        detail: detail.into(),
    });
}

fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn subtract(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn triangle_area_xy(a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    0.5 * ((b[0] - a[0]) * (c[1] - a[1])
        - (b[1] - a[1]) * (c[0] - a[0]))
        .abs()
}

fn volume_and_buoyancy_centre(
    vertices: &[[f64; 3]],
    faces: &[[u32; 3]],
) -> Result<(f64, [f64; 3]), HydrostaticViolation> {
    if vertices.is_empty() || faces.is_empty() {
        return Err(HydrostaticViolation {
            code: "empty_mesh".into(),
            detail: "volume integration requires vertices and faces".into(),
        });
    }

    let mut minimum = [f64::INFINITY; 3];
    let mut maximum = [f64::NEG_INFINITY; 3];
    for vertex in vertices {
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(vertex[axis]);
            maximum[axis] = maximum[axis].max(vertex[axis]);
        }
    }
    let reference = [
        (minimum[0] + maximum[0]) * 0.5,
        (minimum[1] + maximum[1]) * 0.5,
        (minimum[2] + maximum[2]) * 0.5,
    ];

    let mut volume = CompensatedSum::default();
    let mut first_moments = [
        CompensatedSum::default(),
        CompensatedSum::default(),
        CompensatedSum::default(),
    ];

    for (face_index, face) in faces.iter().enumerate() {
        let Some(a) = vertices.get(face[0] as usize).copied() else {
            return Err(HydrostaticViolation {
                code: "face_index".into(),
                detail: format!("face {face_index} has an invalid first vertex index"),
            });
        };
        let Some(b) = vertices.get(face[1] as usize).copied() else {
            return Err(HydrostaticViolation {
                code: "face_index".into(),
                detail: format!("face {face_index} has an invalid second vertex index"),
            });
        };
        let Some(c) = vertices.get(face[2] as usize).copied() else {
            return Err(HydrostaticViolation {
                code: "face_index".into(),
                detail: format!("face {face_index} has an invalid third vertex index"),
            });
        };

        let a_rel = subtract(a, reference);
        let b_rel = subtract(b, reference);
        let c_rel = subtract(c, reference);
        let tetra_volume = dot(a_rel, cross(b_rel, c_rel)) / 6.0;
        if !tetra_volume.is_finite() {
            return Err(HydrostaticViolation {
                code: "non_finite_tetra_volume".into(),
                detail: format!("face {face_index} produced a non-finite tetrahedral volume"),
            });
        }
        volume.add(tetra_volume);
        for axis in 0..3 {
            first_moments[axis].add(
                tetra_volume * (a_rel[axis] + b_rel[axis] + c_rel[axis]) / 4.0,
            );
        }
    }

    let displacement_volume = volume.value();
    if !displacement_volume.is_finite() || displacement_volume <= 0.0 {
        return Err(HydrostaticViolation {
            code: "positive_displacement_volume".into(),
            detail: format!(
                "signed displacement volume must be finite and positive, observed {displacement_volume}"
            ),
        });
    }

    let centre = [
        reference[0] + first_moments[0].value() / displacement_volume,
        reference[1] + first_moments[1].value() / displacement_volume,
        reference[2] + first_moments[2].value() / displacement_volume,
    ];
    if centre.iter().any(|coordinate| !coordinate.is_finite()) {
        return Err(HydrostaticViolation {
            code: "finite_buoyancy_centre".into(),
            detail: "centre of buoyancy is not finite".into(),
        });
    }

    Ok((displacement_volume, centre))
}

struct WaterplaneIntegrals {
    area: f64,
    centroid_x: f64,
    centroid_y: f64,
    transverse_second_moment: f64,
    longitudinal_second_moment: f64,
    product_of_inertia: f64,
}

fn waterplane_integrals(
    artifact: &LoftedHullArtifact,
) -> Result<WaterplaneIntegrals, Vec<HydrostaticViolation>> {
    let vertices = &artifact.candidate.geometry.vertices;
    let faces = &artifact.candidate.geometry.faces;
    let range = artifact.waterplane_face_range;
    let end = range.first_face.checked_add(range.face_count);
    let Some(end) = end.filter(|end| *end <= faces.len()) else {
        return Err(vec![HydrostaticViolation {
            code: "waterplane_face_range".into(),
            detail: "waterplane face range is outside the candidate mesh".into(),
        }]);
    };

    // Shift waterplane coordinates near the local origin before calculating
    // second moments. This avoids catastrophic cancellation for large global
    // coordinate offsets.
    let mut minimum_x = f64::INFINITY;
    let mut maximum_x = f64::NEG_INFINITY;
    let mut minimum_y = f64::INFINITY;
    let mut maximum_y = f64::NEG_INFINITY;
    for face_index in range.first_face..end {
        let face = faces[face_index];
        for vertex_index in face {
            let point = vertices[vertex_index as usize];
            if point.iter().any(|coordinate| !coordinate.is_finite())
                || point[2].abs() > WATERPLANE_TOLERANCE_M
            {
                return Err(vec![HydrostaticViolation {
                    code: "waterplane_not_at_z0".into(),
                    detail: format!("waterplane face {face_index} has a non-finite or non-z=0 vertex"),
                }]);
            }
            minimum_x = minimum_x.min(point[0]);
            maximum_x = maximum_x.max(point[0]);
            minimum_y = minimum_y.min(point[1]);
            maximum_y = maximum_y.max(point[1]);
        }
    }
    if !minimum_x.is_finite()
        || !maximum_x.is_finite()
        || !minimum_y.is_finite()
        || !maximum_y.is_finite()
    {
        return Err(vec![HydrostaticViolation {
            code: "waterplane_bounds".into(),
            detail: "waterplane bounds are not finite".into(),
        }]);
    }
    let reference_x = (minimum_x + maximum_x) * 0.5;
    let reference_y = (minimum_y + maximum_y) * 0.5;

    let mut area = CompensatedSum::default();
    let mut first_x = CompensatedSum::default();
    let mut first_y = CompensatedSum::default();
    let mut raw_x2 = CompensatedSum::default();
    let mut raw_y2 = CompensatedSum::default();
    let mut raw_xy = CompensatedSum::default();

    for face_index in range.first_face..end {
        let face = faces[face_index];
        let a = vertices[face[0] as usize];
        let b = vertices[face[1] as usize];
        let c = vertices[face[2] as usize];

        let triangle_area = triangle_area_xy(a, b, c);
        if !triangle_area.is_finite() || triangle_area <= MINIMUM_POSITIVE_AREA_M2 {
            return Err(vec![HydrostaticViolation {
                code: "waterplane_degenerate_triangle".into(),
                detail: format!("waterplane face {face_index} has zero or invalid area"),
            }]);
        }
        area.add(triangle_area);

        let xs = [a[0] - reference_x, b[0] - reference_x, c[0] - reference_x];
        let ys = [a[1] - reference_y, b[1] - reference_y, c[1] - reference_y];
        first_x.add(triangle_area * (xs[0] + xs[1] + xs[2]) / 3.0);
        first_y.add(triangle_area * (ys[0] + ys[1] + ys[2]) / 3.0);

        let x_square_sum = xs.iter().map(|x| x * x).sum::<f64>();
        let x_pair_sum = xs[0] * xs[1] + xs[1] * xs[2] + xs[2] * xs[0];
        let y_square_sum = ys.iter().map(|y| y * y).sum::<f64>();
        let y_pair_sum = ys[0] * ys[1] + ys[1] * ys[2] + ys[2] * ys[0];

        raw_x2.add(triangle_area * (x_square_sum + x_pair_sum) / 6.0);
        raw_y2.add(triangle_area * (y_square_sum + y_pair_sum) / 6.0);

        let diagonal_xy = xs[0] * ys[0] + xs[1] * ys[1] + xs[2] * ys[2];
        let cross_xy = xs[0] * ys[1]
            + xs[1] * ys[0]
            + xs[1] * ys[2]
            + xs[2] * ys[1]
            + xs[2] * ys[0]
            + xs[0] * ys[2];
        raw_xy.add(triangle_area * (2.0 * diagonal_xy + cross_xy) / 12.0);
    }

    let total_area = area.value();
    if !total_area.is_finite() || total_area <= MINIMUM_POSITIVE_AREA_M2 {
        return Err(vec![HydrostaticViolation {
            code: "positive_waterplane_area".into(),
            detail: format!("waterplane area must be positive, observed {total_area}"),
        }]);
    }

    let centroid_x_relative = first_x.value() / total_area;
    let centroid_y_relative = first_y.value() / total_area;
    let centroid_x = reference_x + centroid_x_relative;
    let centroid_y = reference_y + centroid_y_relative;
    let transverse_second_moment =
        raw_y2.value() - total_area * centroid_y_relative * centroid_y_relative;
    let longitudinal_second_moment =
        raw_x2.value() - total_area * centroid_x_relative * centroid_x_relative;
    let product_of_inertia =
        raw_xy.value() - total_area * centroid_x_relative * centroid_y_relative;

    if !centroid_x.is_finite()
        || !centroid_y.is_finite()
        || !transverse_second_moment.is_finite()
        || !longitudinal_second_moment.is_finite()
        || transverse_second_moment <= 0.0
        || longitudinal_second_moment <= 0.0
        || !product_of_inertia.is_finite()
    {
        return Err(vec![HydrostaticViolation {
            code: "waterplane_moments",
            detail: "waterplane centroid or central moments are non-finite or non-positive".into(),
        }]);
    }

    Ok(WaterplaneIntegrals {
        area: total_area,
        centroid_x,
        centroid_y,
        transverse_second_moment,
        longitudinal_second_moment,
        product_of_inertia,
    })
}

/// Calculate displacement and waterplane properties for the exact lofted mesh.
///
/// This verifies geometry and loft-manifest digests, checks a fresh G0/G1
/// admission result, then integrates the oriented polyhedral surface. It does
/// not compute KG, GM or stability pass/fail.
pub fn calculate_hydrostatic_properties(
    artifact: &LoftedHullArtifact,
    water_density_kg_m3: f64,
) -> Result<HydrostaticProperties, Vec<HydrostaticViolation>> {
    if !water_density_kg_m3.is_finite() || water_density_kg_m3 <= 0.0 {
        return Err(vec![HydrostaticViolation {
            code: "water_density",
            detail: "water density must be finite and strictly positive in kg/m^3".into(),
        }]);
    }
    if artifact.cap_semantics != "closed-displacement-volume-at-design-waterplane-z0" {
        return Err(vec![HydrostaticViolation {
            code: "cap_semantics",
            detail: "hydrostatics requires the declared z=0 design-waterplane cap".into(),
        }]);
    }

    verify_loft_artifact_integrity(artifact).map_err(|errors| {
        errors
            .into_iter()
            .map(|item| HydrostaticViolation {
                code: item.code,
                detail: item.detail,
            })
            .collect::<Vec<_>>()
    })?;

    let fresh_admission = admit(&artifact.candidate);
    if fresh_admission.disposition != Disposition::AdmissibleForProfile
        || fresh_admission != artifact.admission_report
    {
        return Err(vec![HydrostaticViolation {
            code: "fresh_geometry_admission",
            detail: "fresh G0/G1 admission must match the stored report and be admissible-for-profile".into(),
        }]);
    }

    let geometry = &artifact.candidate.geometry;
    if geometry.units != "m"
        || geometry.coordinate_frame.handedness != "right-handed"
        || !geometry.coordinate_frame.axis_convention.contains("x-forward")
        || !geometry.coordinate_frame.axis_convention.contains("y-starboard")
        || !geometry.coordinate_frame.axis_convention.contains("z-down")
    {
        return Err(vec![HydrostaticViolation {
            code: "coordinate_frame",
            detail: "hydrostatics requires metres in a right-handed x-forward/y-starboard/z-down frame".into(),
        }]);
    }

    let (displacement_volume_m3, centre_of_buoyancy_m) =
        volume_and_buoyancy_centre(&geometry.vertices, &geometry.faces)
            .map_err(|error| vec![error])?;
    let waterplane = waterplane_integrals(artifact)?;

    let displacement_mass_kg = water_density_kg_m3 * displacement_volume_m3;
    let displacement_tonnes = displacement_mass_kg / 1_000.0;
    let transverse_metacentric_radius_bm_t_m =
        waterplane.transverse_second_moment / displacement_volume_m3;
    let longitudinal_metacentric_radius_bm_l_m =
        waterplane.longitudinal_second_moment / displacement_volume_m3;
    let tonnes_per_centimetre_immersion =
        water_density_kg_m3 * waterplane.area * 0.01 / 1_000.0;

    let values = [
        displacement_mass_kg,
        displacement_tonnes,
        transverse_metacentric_radius_bm_t_m,
        longitudinal_metacentric_radius_bm_l_m,
        tonnes_per_centimetre_immersion,
    ];
    if values.iter().any(|value| !value.is_finite()) {
        return Err(vec![HydrostaticViolation {
            code: "finite_hydrostatic_properties",
            detail: "one or more derived hydrostatic properties are non-finite".into(),
        }]);
    }

    Ok(HydrostaticProperties {
        result_schema_version: 1,
        candidate_id: artifact.candidate.candidate_id.clone(),
        geometry_digest_blake3: artifact.candidate.geometry_digest_blake3.clone(),
        loft_manifest_digest_blake3: artifact.loft_manifest_digest_blake3.clone(),
        waterplane_face_range_first: artifact.waterplane_face_range.first_face,
        waterplane_face_count: artifact.waterplane_face_range.face_count,
        water_density_kg_m3,
        displacement_volume_m3,
        displacement_mass_kg,
        displacement_tonnes,
        centre_of_buoyancy_m,
        longitudinal_centre_of_buoyancy_m: centre_of_buoyancy_m[0],
        transverse_centre_of_buoyancy_m: centre_of_buoyancy_m[1],
        vertical_centre_of_buoyancy_below_waterplane_m: centre_of_buoyancy_m[2],
        waterplane_area_m2: waterplane.area,
        longitudinal_centre_of_flotation_m: waterplane.centroid_x,
        transverse_centre_of_flotation_m: waterplane.centroid_y,
        transverse_waterplane_second_moment_m4: waterplane.transverse_second_moment,
        longitudinal_waterplane_second_moment_m4: waterplane.longitudinal_second_moment,
        waterplane_product_of_inertia_m4: waterplane.product_of_inertia,
        transverse_metacentric_radius_bm_t_m,
        longitudinal_metacentric_radius_bm_l_m,
        tonnes_per_centimetre_immersion,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::loft::loft_waterplane_capped_hull;
    use crate::section::{
        generate_section_artifact, SectionControlPoint, SectionHullSpec, SectionStation,
    };
    use crate::{AdmissionProfile, CoordinateFrame};

    fn prismatic_loft() -> LoftedHullArtifact {
        let control_points = vec![
            SectionControlPoint {
                breadth_fraction: 0.0,
                depth_fraction: 1.0,
            },
            SectionControlPoint {
                breadth_fraction: 1.0,
                depth_fraction: 0.0,
            },
        ];
        let spec = SectionHullSpec {
            schema_version: 1,
            candidate_id: "hydro-prism-known-answer-001".into(),
            length_m: 100.0,
            beam_m: 20.0,
            draft_m: 6.0,
            coordinate_frame: CoordinateFrame {
                id: "ship-local".into(),
                axis_convention: "x-forward,y-starboard,z-down".into(),
                handedness: "right-handed".into(),
            },
            samples_per_segment: 1,
            stations: vec![
                SectionStation {
                    x_over_length: 0.0,
                    half_breadth_fraction: 1.0,
                    draft_fraction: 1.0,
                    control_points: control_points.clone(),
                },
                SectionStation {
                    x_over_length: 1.0,
                    half_breadth_fraction: 1.0,
                    draft_fraction: 1.0,
                    control_points,
                },
            ],
        };
        let sections = generate_section_artifact(&spec).unwrap();
        let profile = AdmissionProfile {
            id: "hydrostatic-prism-v1".into(),
            require_closed_surface: true,
            minimum_face_area_m2: 1.0e-12,
            minimum_enclosed_volume_m3: 1.0e-12,
            additional_required_checks: vec![],
        };
        loft_waterplane_capped_hull(&sections, profile).unwrap()
    }

    #[test]
    fn triangular_prism_hydrostatics_match_known_answers() {
        let result = calculate_hydrostatic_properties(&prismatic_loft(), 1_025.0).unwrap();
        assert!((result.displacement_volume_m3 - 6_000.0).abs() < 1.0e-9);
        assert!((result.displacement_tonnes - 6_150.0).abs() < 1.0e-9);
        assert!((result.centre_of_buoyancy_m[0] - 50.0).abs() < 1.0e-9);
        assert!(result.centre_of_buoyancy_m[1].abs() < 1.0e-9);
        assert!((result.centre_of_buoyancy_m[2] - 2.0).abs() < 1.0e-9);
        assert!((result.waterplane_area_m2 - 2_000.0).abs() < 1.0e-9);
        assert!((result.longitudinal_centre_of_flotation_m - 50.0).abs() < 1.0e-9);
        assert!(result.transverse_centre_of_flotation_m.abs() < 1.0e-9);
        assert!(
            (result.transverse_waterplane_second_moment_m4 - 66_666.666_666_666_67).abs()
                < 1.0e-6
        );
        assert!(
            (result.longitudinal_waterplane_second_moment_m4 - 1_666_666.666_666_666_7).abs()
                < 1.0e-5
        );
        assert!((result.transverse_metacentric_radius_bm_t_m - 11.111_111_111_111_11).abs() < 1.0e-9);
        assert!((result.longitudinal_metacentric_radius_bm_l_m - 277.777_777_777_777_77).abs() < 1.0e-9);
        assert!((result.tonnes_per_centimetre_immersion - 20.5).abs() < 1.0e-9);
    }

    #[test]
    fn hydrostatics_are_translation_invariant_in_the_waterplane() {
        let mut artifact = prismatic_loft();
        for point in &mut artifact.candidate.geometry.vertices {
            point[0] += 1_000_000_000.0;
            point[1] -= 200_000_000.0;
        }
        artifact.candidate.geometry_digest_blake3 =
            crate::compute_geometry_digest(&artifact.candidate.geometry).unwrap();
        artifact.mesh_digest_blake3 = artifact.candidate.geometry_digest_blake3.clone();
        artifact.admission_report = admit(&artifact.candidate);
        artifact.loft_manifest_digest_blake3 =
            crate::loft::compute_loft_manifest_digest(&artifact).unwrap();

        let result = calculate_hydrostatic_properties(&artifact, 1_025.0).unwrap();
        assert!((result.displacement_volume_m3 - 6_000.0).abs() < 1.0e-8);
        assert!((result.waterplane_area_m2 - 2_000.0).abs() < 1.0e-8);
        assert!((result.centre_of_buoyancy_m[0] - 1_000_000_050.0).abs() < 1.0e-6);
        assert!((result.centre_of_buoyancy_m[1] + 200_000_000.0).abs() < 1.0e-6);
        assert!(
            (result.transverse_waterplane_second_moment_m4 - 66_666.666_666_666_67).abs()
                < 1.0e-5
        );
        assert!(
            (result.longitudinal_waterplane_second_moment_m4 - 1_666_666.666_666_666_7).abs()
                < 1.0e-4
        );
    }

    #[test]
    fn non_positive_density_is_rejected() {
        let error = calculate_hydrostatic_properties(&prismatic_loft(), 0.0).unwrap_err();
        assert!(error.iter().any(|item| item.code == "water_density"));
    }

    #[test]
    fn tampered_face_range_is_rejected_by_manifest_digest() {
        let mut artifact = prismatic_loft();
        artifact.waterplane_face_range.first_face += 1;
        let error = calculate_hydrostatic_properties(&artifact, 1_025.0).unwrap_err();
        assert!(error.iter().any(|item| item.code == "loft_manifest_digest"));
    }

    #[test]
    fn tampered_waterplane_height_is_rejected() {
        let mut artifact = prismatic_loft();
        let first_face = artifact.waterplane_face_range.first_face;
        let vertex_index =
            artifact.candidate.geometry.faces[first_face][0] as usize;
        artifact.candidate.geometry.vertices[vertex_index][2] = 0.05;
        let error = calculate_hydrostatic_properties(&artifact, 1_025.0).unwrap_err();
        assert!(error.iter().any(|item| item.code == "mesh_digest"));
    }

    #[test]
    fn invalid_geometry_admission_is_rejected() {
        let mut artifact = prismatic_loft();
        artifact.candidate.geometry.faces.pop();
        let error = calculate_hydrostatic_properties(&artifact, 1_025.0).unwrap_err();
        assert!(error.iter().any(|item| item.code == "mesh_digest"));
    }
}
