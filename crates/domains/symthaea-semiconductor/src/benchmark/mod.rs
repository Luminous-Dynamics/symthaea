//! Versioned benchmark protocols and independent scoring primitives.

pub mod binary_metrics;
pub mod dtco_manifest;
pub mod mf_acquisition;
pub mod observation_semantics;
pub mod pair_manifest;
pub mod secom_logistic;
pub mod secom_missingness;
pub mod secom_observation_drift;
pub mod secom_raw;
pub mod secom_tabular_baselines;
pub mod split_manifest;

pub use binary_metrics::{
    BinaryClassCounts, BinaryMetricReportV1, BinaryMetricsError, BinaryScoreSampleV1,
    BinaryScoreSubjectV1, score_binary_subject_v1,
};
pub use observation_semantics::{
    MeasurementUncertaintyV1, ObservationStateCountsV1, SensorAffineTransformUncertaintyV1,
    SensorCorrespondenceExecutionReportV1, SensorCorrespondenceExecutionV1,
    SensorCorrespondenceManifestV1, SensorCorrespondenceReportV1,
    SensorCorrespondenceStatusV1, SensorCorrespondenceV1, SensorObservationError,
    SensorObservationFrameV1, SensorObservationReportV1, SensorObservationStateV1,
    SensorObservationV1, SensorUnitTransformV1, adapt_secom_row_to_observation_frame_v1,
    execute_sensor_correspondence_v1, validate_sensor_correspondence_manifest_v1,
    validate_sensor_observation_frame_v1,
};

pub use pair_manifest::{
    ContrastivePairManifestV1, PairManifestError, PairValidationReportV1, validate_pair_manifest_v1,
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
