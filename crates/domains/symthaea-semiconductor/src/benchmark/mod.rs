//! Versioned benchmark protocols and independent scoring primitives.

pub mod binary_metrics;
pub mod pair_manifest;
pub mod split_manifest;

pub use binary_metrics::{
    BinaryClassCounts, BinaryMetricReportV1, BinaryMetricsError, BinaryScoreSampleV1,
    BinaryScoreSubjectV1, score_binary_subject_v1,
};
pub use pair_manifest::{
    ContrastivePairManifestV1, PairManifestError, PairValidationReportV1, validate_pair_manifest_v1,
};
pub use split_manifest::{
    SplitManifestError, SplitManifestV1, SplitValidationReportV1, validate_split_manifest_v1,
};
