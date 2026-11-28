// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Analytic metamorphic oracles for spatial coordinate-frame changes.
//!
//! These are test-only reference calculations. They do not introduce a
//! production transform API or claim that a candidate representation is
//! equivariant. The tests distinguish transformed coordinates, invariant
//! world geometry, and observer-relative predicates that must be recomputed.

use std::f64::consts::{FRAC_PI_2, PI};

type Vec3 = [f64; 3];
type Mat3 = [[f64; 3]; 3];

const EPS: f64 = 1.0e-10;

fn add(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn sub(a: Vec3, b: Vec3) -> Vec3 {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn scale(v: Vec3, factor: f64) -> Vec3 {
    [v[0] * factor, v[1] * factor, v[2] * factor]
}

fn dot(a: Vec3, b: Vec3) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn norm(v: Vec3) -> f64 {
    dot(v, v).sqrt()
}

fn distance(a: Vec3, b: Vec3) -> f64 {
    norm(sub(a, b))
}

fn cross(a: Vec3, b: Vec3) -> Vec3 {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn identity_matrix() -> Mat3 {
    [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
}

fn matrix_vector(matrix: Mat3, vector: Vec3) -> Vec3 {
    [
        dot(matrix[0], vector),
        dot(matrix[1], vector),
        dot(matrix[2], vector),
    ]
}

fn matrix_multiply(a: Mat3, b: Mat3) -> Mat3 {
    let mut result = [[0.0; 3]; 3];
    for (row_index, result_row) in result.iter_mut().enumerate() {
        for (col_index, value) in result_row.iter_mut().enumerate() {
            *value = (0..3)
                .map(|inner| a[row_index][inner] * b[inner][col_index])
                .sum();
        }
    }
    result
}

fn transpose(matrix: Mat3) -> Mat3 {
    [
        [matrix[0][0], matrix[1][0], matrix[2][0]],
        [matrix[0][1], matrix[1][1], matrix[2][1]],
        [matrix[0][2], matrix[1][2], matrix[2][2]],
    ]
}

fn determinant(matrix: Mat3) -> f64 {
    let [a, b, c] = matrix;
    a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
}

fn matrix_is_close(a: Mat3, b: Mat3, tolerance: f64) -> bool {
    a.iter().zip(b.iter()).all(|(a_row, b_row)| {
        a_row
            .iter()
            .zip(b_row.iter())
            .all(|(a_value, b_value)| (a_value - b_value).abs() <= tolerance)
    })
}

fn rotation_axis_angle(axis: Vec3, angle: f64) -> Mat3 {
    let axis_norm = norm(axis);
    assert!(axis_norm.is_finite() && axis_norm > 0.0);
    let [x, y, z] = scale(axis, 1.0 / axis_norm);
    let cosine = angle.cos();
    let sine = angle.sin();
    let one_minus_cosine = 1.0 - cosine;

    [
        [
            cosine + x * x * one_minus_cosine,
            x * y * one_minus_cosine - z * sine,
            x * z * one_minus_cosine + y * sine,
        ],
        [
            y * x * one_minus_cosine + z * sine,
            cosine + y * y * one_minus_cosine,
            y * z * one_minus_cosine - x * sine,
        ],
        [
            z * x * one_minus_cosine - y * sine,
            z * y * one_minus_cosine + x * sine,
            cosine + z * z * one_minus_cosine,
        ],
    ]
}

#[derive(Clone, Copy, Debug)]
struct Rigid3 {
    /// Rotation from the source frame into the target frame.
    rotation: Mat3,
    /// Translation expressed in the target frame.
    translation: Vec3,
}

impl Rigid3 {
    fn identity() -> Self {
        Self {
            rotation: identity_matrix(),
            translation: [0.0; 3],
        }
    }

    fn from_axis_angle(axis: Vec3, angle: f64, translation: Vec3) -> Self {
        Self {
            rotation: rotation_axis_angle(axis, angle),
            translation,
        }
    }

    fn transform_point(self, point_in_source: Vec3) -> Vec3 {
        add(matrix_vector(self.rotation, point_in_source), self.translation)
    }

    fn transform_vector(self, vector_in_source: Vec3) -> Vec3 {
        matrix_vector(self.rotation, vector_in_source)
    }

    /// Return self ∘ rhs: apply rhs first, then self.
    fn compose(self, rhs: Self) -> Self {
        Self {
            rotation: matrix_multiply(self.rotation, rhs.rotation),
            translation: add(
                matrix_vector(self.rotation, rhs.translation),
                self.translation,
            ),
        }
    }

    fn inverse(self) -> Self {
        let inverse_rotation = transpose(self.rotation);
        Self {
            rotation: inverse_rotation,
            translation: scale(matrix_vector(inverse_rotation, self.translation), -1.0),
        }
    }

    fn is_proper_rotation(self, tolerance: f64) -> bool {
        let gram = matrix_multiply(transpose(self.rotation), self.rotation);
        matrix_is_close(gram, identity_matrix(), tolerance)
            && (determinant(self.rotation) - 1.0).abs() <= tolerance
    }
}

fn quaternion_wxyz_to_rotation([w, x, y, z]: [f64; 4]) -> Mat3 {
    let length = (w * w + x * x + y * y + z * z).sqrt();
    assert!(length.is_finite() && length > 0.0);
    let w = w / length;
    let x = x / length;
    let y = y / length;
    let z = z / length;

    [
        [
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - w * z),
            2.0 * (x * z + w * y),
        ],
        [
            2.0 * (x * y + w * z),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - w * x),
        ],
        [
            2.0 * (x * z - w * y),
            2.0 * (y * z + w * x),
            1.0 - 2.0 * (x * x + y * y),
        ],
    ]
}

fn quaternion_xyzw_to_rotation([x, y, z, w]: [f64; 4]) -> Mat3 {
    quaternion_wxyz_to_rotation([w, x, y, z])
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FrameId {
    namespace: &'static str,
    local_id: &'static str,
    generation: u64,
    unit: &'static str,
}

impl FrameId {
    fn is_compatible_with(self, other: Self) -> bool {
        self == other
    }
}

fn assert_vec_close(actual: Vec3, expected: Vec3, tolerance: f64) {
    assert!(
        distance(actual, expected) <= tolerance,
        "actual={actual:?}, expected={expected:?}, tolerance={tolerance}"
    );
}

#[test]
fn planar_translation_and_arbitrary_rotation_preserve_xy_geometry() {
    let transforms = [
        Rigid3::from_axis_angle([0.0, 0.0, 1.0], 0.0, [2.5, -1.25, 0.0]),
        Rigid3::from_axis_angle([0.0, 0.0, 1.0], 0.37, [-4.0, 2.0, 0.0]),
        Rigid3::from_axis_angle([0.0, 0.0, 1.0], -1.21, [0.5, 3.0, 0.0]),
        Rigid3::from_axis_angle([0.0, 0.0, 1.0], FRAC_PI_2, [0.0, 0.0, 0.0]),
    ];
    let a = [0.5, -1.0, 0.0];
    let b = [2.0, 0.25, 0.0];
    let c = [-0.75, 3.0, 0.0];
    let original_ab = distance(a, b);
    let original_ac = distance(a, c);

    for transform in transforms {
        let ta = transform.transform_point(a);
        let tb = transform.transform_point(b);
        let tc = transform.transform_point(c);
        assert!((distance(ta, tb) - original_ab).abs() < EPS);
        assert!((distance(ta, tc) - original_ac).abs() < EPS);
        assert!(ta[2].abs() < EPS && tb[2].abs() < EPS && tc[2].abs() < EPS);
    }
}

#[test]
fn arbitrary_se3_actions_preserve_distances_angles_and_incidence() {
    let points = [
        [-1.25, 0.5, 2.0],
        [2.0, -0.75, 0.25],
        [0.5, 1.5, -2.25],
        [3.25, 2.0, 1.75],
    ];
    let transforms = [
        Rigid3::from_axis_angle([1.0, 2.0, 3.0], 0.13, [2.0, -1.0, 0.5]),
        Rigid3::from_axis_angle([-2.0, 1.0, 0.5], -0.91, [-3.0, 4.0, 1.0]),
        Rigid3::from_axis_angle([0.5, -1.0, 2.0], 2.17, [0.25, 0.5, -0.75]),
    ];

    let vector_a = sub(points[1], points[0]);
    let vector_b = sub(points[2], points[0]);
    let original_cosine = dot(vector_a, vector_b) / (norm(vector_a) * norm(vector_b));
    let original_ab = distance(points[0], points[1]);
    let original_cd = distance(points[2], points[3]);

    for transform in transforms {
        assert!(transform.is_proper_rotation(EPS));
        let transformed = points.map(|point| transform.transform_point(point));
        assert!((distance(transformed[0], transformed[1]) - original_ab).abs() < EPS);
        assert!((distance(transformed[2], transformed[3]) - original_cd).abs() < EPS);

        let transformed_a = sub(transformed[1], transformed[0]);
        let transformed_b = sub(transformed[2], transformed[0]);
        let transformed_cosine =
            dot(transformed_a, transformed_b) / (norm(transformed_a) * norm(transformed_b));
        assert!((transformed_cosine - original_cosine).abs() < EPS);

        // Collinearity is an incidence property and must survive a common SE(3) action.
        let line_start = [0.0, 0.0, 0.0];
        let line_mid = [1.0, 2.0, 3.0];
        let line_end = [2.0, 4.0, 6.0];
        let ta = transform.transform_point(line_start);
        let tb = transform.transform_point(line_mid);
        let tc = transform.transform_point(line_end);
        assert!(norm(cross(sub(tb, ta), sub(tc, ta))) < 1.0e-9);
    }
}

#[test]
fn transform_composition_is_ordered_and_inverse_round_trip_closes() {
    let first = Rigid3::from_axis_angle([1.0, 0.0, 0.0], 0.73, [1.0, 2.0, -1.0]);
    let second = Rigid3::from_axis_angle([0.0, 1.0, 0.0], -0.61, [-2.0, 0.5, 3.0]);
    let point = [0.25, -1.5, 2.75];

    let sequential = second.transform_point(first.transform_point(point));
    let composed = second.compose(first).transform_point(point);
    assert_vec_close(composed, sequential, EPS);

    let reversed = first.compose(second).transform_point(point);
    assert!(
        distance(composed, reversed) > 1.0e-3,
        "non-commuting rotations must expose reversed composition order"
    );

    let returned = first.inverse().transform_point(first.transform_point(point));
    assert_vec_close(returned, point, EPS);
    let inverse_matrix_product = matrix_multiply(first.rotation, first.inverse().rotation);
    assert!(matrix_is_close(inverse_matrix_product, identity_matrix(), EPS));
}

#[test]
fn non_origin_pivot_rotation_preserves_local_geometry() {
    let pivot = [3.0, -2.0, 1.25];
    let rotation = Rigid3::from_axis_angle([1.0, 2.0, -0.5], 0.84, [0.0; 3]);
    let translate_to_origin = Rigid3 {
        rotation: identity_matrix(),
        translation: scale(pivot, -1.0),
    };
    let translate_back = Rigid3 {
        rotation: identity_matrix(),
        translation: pivot,
    };
    let about_pivot = translate_back.compose(rotation).compose(translate_to_origin);

    let point = [4.0, -1.0, 2.0];
    let pivot_point = about_pivot.transform_point(pivot);
    assert_vec_close(pivot_point, pivot, EPS);
    assert!(
        distance(about_pivot.transform_point(point), point) > 1.0e-3,
        "a non-pivot point must move under a non-identity pivot rotation"
    );

    let wrong_origin_rotation = rotation.transform_point(point);
    assert!(
        distance(wrong_origin_rotation, about_pivot.transform_point(point)) > 1.0e-3,
        "rotation about the origin must not be confused with rotation about the declared pivot"
    );
}

#[test]
fn wxyz_and_xyzw_adapters_agree_only_when_decoded_by_the_declared_order() {
    let wxyz = [0.81, 0.23, -0.31, 0.42];
    let xyzw = [wxyz[1], wxyz[2], wxyz[3], wxyz[0]];
    let correct_wxyz = quaternion_wxyz_to_rotation(wxyz);
    let correct_xyzw = quaternion_xyzw_to_rotation(xyzw);
    assert!(matrix_is_close(correct_wxyz, correct_xyzw, EPS));

    let wrong_order = quaternion_wxyz_to_rotation(xyzw);
    assert!(
        !matrix_is_close(correct_wxyz, wrong_order, 1.0e-4),
        "an asymmetric non-identity quaternion must expose WXYZ/XYZW confusion"
    );
}

#[test]
fn transform_direction_is_not_symmetric_and_valid_inverse_is_explicit() {
    // T_world_from_camera maps a camera-frame point into world coordinates.
    let t_world_from_camera =
        Rigid3::from_axis_angle([0.0, 0.0, 1.0], 0.6, [4.0, -2.0, 1.0]);
    let point_camera = [0.5, 1.0, 3.0];
    let point_world = t_world_from_camera.transform_point(point_camera);

    let recovered_camera = t_world_from_camera.inverse().transform_point(point_world);
    assert_vec_close(recovered_camera, point_camera, EPS);

    let wrong_direction = t_world_from_camera.inverse().transform_point(point_camera);
    assert!(
        distance(wrong_direction, point_world) > 1.0e-2,
        "using T_camera_from_world in place of T_world_from_camera must fail"
    );
}

#[test]
fn observer_reframe_changes_camera_relative_left_right_but_not_world_geometry() {
    let camera_world_a = Rigid3::identity();
    let camera_world_b =
        Rigid3::from_axis_angle([0.0, 0.0, 1.0], PI, [0.0, 0.0, 0.0]);
    let fixed_world_point = [1.0, 0.0, 5.0];
    let another_world_point = [2.0, 0.0, 5.0];

    // Camera coordinates are computed from T_camera_from_world, not by mutating
    // or rotating the canonical world points.
    let point_in_camera_a = camera_world_a.inverse().transform_point(fixed_world_point);
    let point_in_camera_b = camera_world_b.inverse().transform_point(fixed_world_point);
    assert!(point_in_camera_a[0] > 0.0);
    assert!(point_in_camera_b[0] < 0.0);

    let world_distance_before = distance(fixed_world_point, another_world_point);
    let world_distance_after = distance(fixed_world_point, another_world_point);
    assert!((world_distance_before - world_distance_after).abs() < EPS);
    assert_vec_close(fixed_world_point, [1.0, 0.0, 5.0], EPS);
}

#[test]
fn reflection_is_not_a_proper_rotation_and_must_not_pass_so3_validation() {
    let proper = Rigid3::from_axis_angle([1.0, 2.0, 3.0], 0.91, [0.0; 3]);
    assert!(proper.is_proper_rotation(EPS));
    assert!((determinant(proper.rotation) - 1.0).abs() < EPS);

    let reflection = [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];
    let reflected = Rigid3 {
        rotation: reflection,
        translation: [0.0; 3],
    };
    assert!((determinant(reflection) + 1.0).abs() < EPS);
    assert!(!reflected.is_proper_rotation(EPS));
}

#[test]
fn explicit_meter_to_millimeter_conversion_preserves_physical_distance() {
    let point_a_m = [0.25, -1.5, 2.0];
    let point_b_m = [1.75, 0.5, -0.25];
    let distance_m = distance(point_a_m, point_b_m);

    let point_a_mm = scale(point_a_m, 1_000.0);
    let point_b_mm = scale(point_b_m, 1_000.0);
    let distance_mm = distance(point_a_mm, point_b_mm);
    assert!((distance_mm / 1_000.0 - distance_m).abs() < EPS);

    // Explicitly converting back recovers the original metric coordinates.
    assert_vec_close(scale(point_a_mm, 0.001), point_a_m, EPS);
    assert_vec_close(scale(point_b_mm, 0.001), point_b_m, EPS);
}

#[test]
fn frame_namespace_generation_and_unit_are_part_of_compatibility() {
    let world_v1 = FrameId {
        namespace: "robot-a",
        local_id: "world",
        generation: 1,
        unit: "m",
    };
    let same_world_v1 = world_v1;
    let world_v2 = FrameId {
        generation: 2,
        ..world_v1
    };
    let world_mm = FrameId {
        unit: "mm",
        ..world_v1
    };
    let other_namespace = FrameId {
        namespace: "robot-b",
        ..world_v1
    };

    assert!(world_v1.is_compatible_with(same_world_v1));
    assert!(!world_v1.is_compatible_with(world_v2));
    assert!(!world_v1.is_compatible_with(world_mm));
    assert!(!world_v1.is_compatible_with(other_namespace));
}

#[test]
fn frame_graph_cycle_closes_only_for_consistent_edges() {
    // T_b_from_a, T_c_from_b, and T_a_from_c compose to identity.
    let t_b_from_a = Rigid3::from_axis_angle([1.0, 0.0, 0.0], 0.3, [1.0, 0.0, 0.5]);
    let t_c_from_b = Rigid3::from_axis_angle([0.0, 1.0, 0.0], -0.5, [0.0, 2.0, -1.0]);
    let t_a_from_c = t_c_from_b.compose(t_b_from_a).inverse();

    let cycle = t_a_from_c.compose(t_c_from_b).compose(t_b_from_a);
    assert_vec_close(cycle.transform_point([0.7, -1.2, 2.5]), [0.7, -1.2, 2.5], EPS);
    assert!(matrix_is_close(cycle.rotation, identity_matrix(), EPS));

    let inconsistent_edge = Rigid3 {
        translation: add(t_a_from_c.translation, [0.05, 0.0, 0.0]),
        ..t_a_from_c
    };
    let inconsistent_cycle =
        inconsistent_edge.compose(t_c_from_b).compose(t_b_from_a);
    assert!(
        norm(inconsistent_cycle.translation) > 1.0e-2,
        "a modified non-identity edge must expose cycle-closure error"
    );
}
