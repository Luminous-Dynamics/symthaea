//! Versioned benchmark protocols and independent scoring primitives.

pub mod binary_metrics;

pub use binary_metrics::{
    BinaryClassCounts, BinaryMetricReportV1, BinaryScoreSampleV1, BinaryScoreSubjectV1,
    BinaryMetricsError, score_binary_subject_v1,
};
