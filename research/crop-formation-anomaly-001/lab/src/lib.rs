//! Small, dependency-free reference algorithms for CROP-INTEL-001.
//!
//! The crate measures descriptive geometry and runs explicitly declared
//! decoders. It does not infer authorship, origin, intent, or semantics.

pub mod circle_layout;
pub mod crop_flow;
pub mod decoder;
pub mod event_split;
pub mod geometry;
pub mod grid;
pub mod mechanics;
pub mod plant_pose_3d;
pub mod statistics;

pub use circle_layout::{
    audit_circle_measurement_sensitivity, class_circle_layout, count_centred_motifs,
    describe_circle_layout, parse_circle_csv, permute_circle_radii,
    run_radius_permutation_ensemble, validate_circles, CircleClassing, CircleContact,
    CircleGrammarFamily, CircleGrammarOutcome, CircleGrammarSpec, CircleLayout,
    CircleLayoutError, CircleRecord, CircleSensitivityConfig, CircleSensitivitySummary,
};
pub use crop_flow::{CropFlowDescriptor, CropFlowError, CropFlowSample, describe_crop_flow, describe_crop_flow_about_center};
pub use event_split::{BenchmarkSample, EventFold, EventFoldPlan, EventSplitError, grouped_event_kfold, validate_event_fold_plan};
pub use decoder::{
    BitOrder, DecodeError, DecodeOutcome, DecoderSpec, PayloadEncoding, decode,
    encode_ascii7, run_tournament,
};
pub use geometry::{
    Bounds2, GeometryDescriptor, GeometryError, Point2, SymmetryScore, describe,
};
pub use grid::{
    BinaryGrid, GridDecodeOutcome, GridDecoderManifest, GridDecoderSpec, GridError, GridTransform,
    ManifestError, NullGridResult,
    TraversalOrder, margin_preserving_shuffle, run_grid_tournament,
    standard_d4_ascii7_manifest,
};
pub use mechanics::{
    cantilever_base_moment, cantilever_tip_load_flexural_rigidity, curvature_from_moment,
    outer_fibre_bending_stress_pa, three_point_bend_flexural_rigidity,
    three_point_bend_peak_moment, flexural_rigidity_from_modulus,
    BeamProfile, BeamSegment, EllipticalHollowSection, MechanicsError, RootSupport,
    SectionProperties,
};
pub use plant_pose_3d::{Point3, StemPose3DDescriptor, StemPose3DError, StemPoseSample, Vector3, describe_stem_pose_3d, describe_stem_pose_3d_about_center};
pub use statistics::{
    FamilyWiseMaxResult, binary_entropy, familywise_max_statistic_p_value,
    upper_tail_max_p_value, wilson_interval,
};

#[cfg(test)]
mod integrated_tests {
    use super::*;

    const CCAT_ZEALS_KNOLL: &str =
        include_str!("../../datasets/ccat-zeals-knoll-2026/circles_drag_60.csv");

    #[test]
    fn geometry_family_replays_all_radius_permutation_nulls_with_familywise_correction() {
        let circles = parse_circle_csv(CCAT_ZEALS_KNOLL).expect("source geometry fixture");
        let family = CircleGrammarFamily::new(
            "ccat-familywise-radius-position-v1",
            1,
            vec![
                CircleGrammarSpec {
                    spec_id: "k4-t40".into(),
                    radius_class_count: 4,
                    tangent_gap_tolerance_px: 40.0,
                },
                CircleGrammarSpec {
                    spec_id: "k5-t45".into(),
                    radius_class_count: 5,
                    tangent_gap_tolerance_px: 45.0,
                },
                CircleGrammarSpec {
                    spec_id: "k6-t50".into(),
                    radius_class_count: 6,
                    tangent_gap_tolerance_px: 50.0,
                },
            ],
        )
        .expect("frozen family");
        let observed = family.evaluate(&circles).expect("observed family");
        let seeds = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111];
        let null_replicates = run_radius_permutation_ensemble(&circles, &family, &seeds)
            .expect("all null replicates rerun complete family");

        let observed_scores: Vec<f64> = observed
            .iter()
            .map(|outcome| outcome.first_last_class_edge_count as f64)
            .collect();
        let null_score_matrix: Vec<Vec<f64>> = null_replicates
            .iter()
            .map(|replicate| {
                replicate
                    .iter()
                    .map(|outcome| outcome.first_last_class_edge_count as f64)
                    .collect()
            })
            .collect();
        let result = familywise_max_statistic_p_value(
            &observed_scores,
            &null_score_matrix,
        )
        .expect("complete family-wise null matrix");

        assert_eq!(result.family_size, family.specifications().len());
        assert_eq!(result.null_replicate_count, seeds.len());
        assert_eq!(result.observed_maximum, observed_scores.iter().copied().fold(0.0, f64::max));
        assert!(result.adjusted_p_value >= 1.0 / (seeds.len() + 1) as f64);
        assert!(result.adjusted_p_value <= 1.0);
    }
}
