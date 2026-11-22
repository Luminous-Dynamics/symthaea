//! Versioned benchmark protocols and independent scoring primitives.

pub mod binary_metrics;
pub mod dtco_manifest;
pub mod mf_acquisition;
pub mod pair_manifest;
pub mod secom_logistic;
pub mod secom_missingness;
pub mod secom_observation_drift;
pub mod secom_raw;
pub mod secom_tabular_baselines;
pub mod split_manifest;
pub mod stratified_repeat;

pub use binary_metrics::{
    BinaryClassCounts, BinaryMetricReportV1, BinaryMetricsError, BinaryScoreSampleV1,
    BinaryScoreSubjectV1, score_binary_subject_v1,
};
pub use pair_manifest::{
    ContrastivePairManifestV1, PairManifestError, PairValidationReportV1, validate_pair_manifest_v1,
};
pub use stratified_repeat::{
    StratifiedRepeatManifestV1, StratifiedRepeatReportV1, StratifiedSampleV1,
    StratifiedSplitAssignmentV1, StratifiedSplitError, StratifiedSplitSubjectV1,
    generate_repeated_stratified_splits_v1,
};

pub use split_manifest::{
    SplitManifestError, SplitManifestV1, SplitValidationReportV1, validate_split_manifest_v1,
};

pub use secom_logistic::{SecomLogisticError, SecomLogisticReportV1, evaluate_secom_logistic_v1};

pub use secom_missingness::{
    SecomMissingnessError, SecomMissingnessReportV1, evaluate_secom_missingness_v1,
};
pub use secom_observation_drift::{
    SecomObservationDriftError, SecomObservationDriftReportV1, evaluate_secom_observation_drift_v1,
};

pub use secom_tabular_baselines::{
    SecomTabularBaselineError, SecomTabularBaselinesReportV1, evaluate_secom_tabular_baselines_v1,
};

pub use secom_raw::{
    SecomRawDataError, SecomRawExpectationsV1, SecomRawIntegrityReportV1, audit_secom_raw_v1,
};

pub use dtco_manifest::{
    DtcoLineageError, DtcoLineageManifestV1, DtcoLineageReportV1, validate_dtco_lineage_v1,
};

pub use mf_acquisition::{
    AcquisitionCandidateV1, AcquisitionError, AcquisitionReportV1, AcquisitionSubjectV1,
    select_high_fidelity_candidates_v1,
};
