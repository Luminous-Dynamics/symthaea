//! Versioned benchmark protocols and independent scoring primitives.

pub mod binary_metrics;
pub mod dtco_manifest;
pub mod metrology_audit;
pub mod metrology_selection_receipt_v2;
pub mod metrology_semantics;
pub mod mf_acquisition;
pub mod pair_manifest;
pub mod process_event_graph;
pub mod secom_fold_preprocessor;
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
pub use metrology_selection_receipt_v2::{
    CANONICALIZATION_VERSION_V1, SELECTION_POLICY_CONFIG_V2_SCHEMA, SELECTION_RECEIPT_V2_SCHEMA,
    SEED_DERIVATION_VERSION_V1, SELECTOR_ALGORITHM_ID_V1, SELECTOR_ALGORITHM_VERSION_V1,
    EvsiScoreKindV2, MetrologyScoreProvenanceV2, MetrologySelectionPolicyConfigV2,
    MetrologySelectionReceiptV2, SelectionReceiptAttestationV2, SelectionReceiptV2Error,
    UncertaintyScoreKindV2, build_selection_receipt_v2, verify_selection_receipt_v2,
};

pub use process_event_graph::{
    ClockBasisV1, InformationCutoffSnapshotV1, MetrologyLabelStatusV1, MetrologyLabelV1,
    PredictionInformationCutoffV1, ProcessEventEdgeV1, ProcessEventGraphError, ProcessEventGraphV1,
    ProcessEventKindV1, ProcessEventV1, ProcessGraphValidationReportV1,
    build_information_cutoff_snapshot_v1, validate_process_event_graph_v1,
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

pub use secom_fold_preprocessor::{
    FittedSecomPreprocessorV1, PreparedSecomFoldV1, QuantileOutputV1, SecomFoldPreprocessorError,
    SecomFoldPreprocessorReportV1, SecomPreprocessingRowV1, SecomPreprocessorConfigV1,
    SecomTransformedSampleV1, fit_secom_preprocessor_v1, prepare_secom_stratified_fold_v1,
    transform_secom_rows_v1,
};

pub use secom_raw::{
    SecomRawDataError, SecomRawExpectationsV1, SecomRawIntegrityReportV1, audit_secom_raw_v1,
};

pub use dtco_manifest::{
    DtcoLineageError, DtcoLineageManifestV1, DtcoLineageReportV1, validate_dtco_lineage_v1,
};

pub use metrology_semantics::{
    BinaryMeasurandSemanticsV1, MetrologySemanticsError, PositiveConditionV1,
    ValidatedMetrologyClassificationV1, classify_validated_metrology_label_v1,
    seal_binary_measurand_semantics_v1, validate_binary_measurand_semantics_v1,
};

pub use metrology_audit::{
    MetrologyAuditCandidateV1, MetrologyAuditError, MetrologyAuditOutcomeV1,
    MetrologyAuditPolicyV1, MetrologyAuditPopulationEstimateV1, MetrologyAuditSelectionReportV1,
    MetrologyAuditSelectionRowV1, MetrologyAuditStratumEstimateV1, MetrologySelectionMethodV1,
    estimate_metrology_population_v1, select_metrology_audit_v1,
};

pub use mf_acquisition::{
    AcquisitionCandidateV1, AcquisitionError, AcquisitionReportV1, AcquisitionSubjectV1,
    select_high_fidelity_candidates_v1,
};
