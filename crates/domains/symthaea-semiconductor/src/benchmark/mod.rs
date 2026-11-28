//! Versioned benchmark protocols and independent scoring primitives.

pub mod binary_metrics;
pub mod capacity_aware_feedback;
pub mod conformal_interval_builder;
pub mod delayed_conformal_aci;
pub mod calibration_requalification;
pub mod dtco_manifest;
pub mod label_consumption_guard;
pub mod metrology_audit;
pub mod metrology_qualification_gate;
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
pub mod vm_snapshot_dependency_impact;
pub mod vm_snapshot_evidence;

pub use binary_metrics::{
    BinaryClassCounts, BinaryMetricReportV1, BinaryMetricsError, BinaryScoreSampleV1,
    BinaryScoreSubjectV1, score_binary_subject_v1,
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

pub use metrology_qualification_gate::{
    CALIBRATION_CERTIFICATE_SCHEMA_V1, CalibrationCertificateV1, MEASUREMENT_EVIDENCE_SCHEMA_V1,
    MeasurementEvidenceV1, MeasurementQualificationErrorV1, MeasurementQualificationPolicyV1,
    MeasurementQualificationReasonV1, MeasurementQualificationReportV1,
    MeasurementQualificationStatusV1, MeasurementUncertaintyComponentV1,
    QUALIFICATION_POLICY_SCHEMA_V1, QUALIFICATION_REPORT_SCHEMA_V1, TraceabilityDeclarationV1,
    UncertaintyComponentKindV1, qualify_measurement_evidence_v1,
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

pub use calibration_requalification::{
    CALIBRATION_CHANGE_EVENT_SCHEMA_V1, CALIBRATION_REQUALIFICATION_INPUT_SCHEMA_V1,
    CALIBRATION_REQUALIFICATION_REPORT_SCHEMA_V1, CalibrationChangeEventV1,
    CalibrationChangeKindV1, CalibrationChangeScopeV1, CalibrationChangeTargetV1,
    CalibrationEventFrontierV1, CalibrationEventInventoryStateV1, CalibrationEventOrderingStateV1,
    CalibrationEventQualificationV1, CalibrationOperatingRangeV1,
    CalibrationRequalificationErrorV1, CalibrationRequalificationFrontierReportV1,
    CalibrationRequalificationInputV1, CalibrationRequalificationReasonV1,
    CalibrationRequalificationReportV1, MeasurementTimeBoundsV1,
    requalify_calibration_measurement_v1, seal_calibration_event_inventory_v1,
};

pub use label_consumption_guard::{
    LABEL_CONSUMPTION_GUARD_SCHEMA_V1, LABEL_CONSUMPTION_REPORT_SCHEMA_V1,
    LabelConsumptionEligibilityReportV1, LabelConsumptionEligibilityV1,
    LabelConsumptionGuardErrorV1, LabelConsumptionReasonV1, LabelEvidenceFrontierV1,
    assess_label_consumption_eligibility_v1,
};

pub use vm_snapshot_evidence::{
    VM_SNAPSHOT_INPUT_SCHEMA_V1, VM_SNAPSHOT_MANIFEST_SCHEMA_V1,
    VM_SNAPSHOT_REQUALIFICATION_SCHEMA_V1, VmSnapshotCurrentDispositionV1,
    VmSnapshotEvidenceErrorV1, VmSnapshotInputV1, VmSnapshotLabelCurrentStatusV1,
    VmSnapshotLabelEvidenceV1, VmSnapshotLabelManifestEntryV1, VmSnapshotPurposeV1,
    VmSnapshotRequalificationReportV1, VmTrainingSnapshotManifestV1,
    build_vm_training_snapshot_manifest_v1, requalify_vm_training_snapshot_v1,
};

pub use vm_snapshot_dependency_impact::{
    VM_DEPENDENCY_EDGE_SCHEMA_V1, VM_DEPENDENCY_GRAPH_SCHEMA_V1, VM_IMPACT_CLOSURE_INPUT_SCHEMA_V1,
    VM_IMPACT_CLOSURE_REPORT_SCHEMA_V1, VmDependencyDimensionV1, VmImpactClosureErrorV1,
    VmImpactClosureInputV1, VmImpactClosureReportV1, VmImpactClosureStatusV1,
    VmModelDerivationKindV1, VmModelSnapshotNodeV1, VmSnapshotDependencyEdgeV1,
    VmSnapshotDependencyGraphV1, VmSnapshotImpactEntryV1, VmSnapshotImpactStateV1,
    compute_vm_snapshot_impact_closure_v1, seal_vm_snapshot_dependency_edge_v1,
    seal_vm_snapshot_dependency_graph_v1,
};

pub use delayed_conformal_aci::{
    DELAYED_ACI_DELIVERY_SCHEMA_V1, DELAYED_ACI_ENGINE_SCHEMA_V1,
    DELAYED_ACI_POLICY_SCHEMA_V1, DELAYED_ACI_PREDICTION_RECEIPT_SCHEMA_V1,
    DELAYED_ACI_PREDICTION_SCHEMA_V1, DELAYED_ACI_UPDATE_RECEIPT_SCHEMA_V1,
    DelayToMemoryDiagnosticV1, DelayedAciDispositionV1, DelayedAciEngineV1,
    DelayedAciErrorV1, DelayedAciGuaranteeStateV1, DelayedAciLabelDeliveryV1,
    DelayedAciPolicyV1, DelayedAciPredictionInputV1, DelayedAciPredictionReceiptV1,
    DelayedAciReasonV1, DelayedAciUpdateReceiptV1, ResidualMemoryBasisV1,
    create_delayed_aci_engine_v1, delayed_aci_alpha_for_round_v1,
    deliver_delayed_aci_label_v1, estimate_delay_to_memory_v1,
    register_delayed_aci_prediction_v1,
};

pub use capacity_aware_feedback::{
    CA_OCRC_ENGINE_SCHEMA_V1, CA_OCRC_FEEDBACK_SCHEMA_V1, CA_OCRC_POLICY_SCHEMA_V1,
    CA_OCRC_ROUND_SCHEMA_V1, CA_OCRC_UPDATE_RECEIPT_SCHEMA_V1,
    CaOcrcEngineV1, CaOcrcErrorV1, CaOcrcFeedbackReceiptV1, CaOcrcFeedbackV1,
    CaOcrcGuaranteeStateV1, CaOcrcOptimizedProxyPolicyV1, CaOcrcPolicyV1,
    CaOcrcPredictionKindV1, CaOcrcReasonV1, CaOcrcRiskGapBoundV1,
    CaOcrcRoundInputV1, CaOcrcTrackedRoundV1, CaOcrcTrackingDecisionV1,
    CaOcrcTrackingDispositionV1, ca_ocrc_proxy_survival_v1,
    ca_ocrc_risk_gap_bound_v1, create_ca_ocrc_engine_v1,
    optimize_ca_ocrc_proxy_policy_v1, receive_ca_ocrc_feedback_v1,
    start_ca_ocrc_round_v1,
};

pub use conformal_interval_builder::{
    CONFORMAL_CALIBRATION_SAMPLE_SCHEMA_V1, CONFORMAL_INTERVAL_INPUT_SCHEMA_V1,
    CONFORMAL_SCORE_POLICY_SCHEMA_V1, CONFORMAL_SET_RECEIPT_SCHEMA_V1,
    BoundedConformalScoreContractV1, BoundedConformalScoreKindV1,
    BoundedPredictionSetInputV1, BoundedPredictionSetKindV1, BoundedPredictionSetReceiptV1,
    ConformalCalibrationSampleV1, ConformalCalibrationScoreV1, ConformalIntervalInputV1,
    ConformalScoreKindV1, ConformalScorePolicyV1, ConformalSetBuildReasonV1,
    ConformalSetBuildStatusV1, ConformalSetBuilderErrorV1, ConformalSetReceiptV1,
    bounded_conformal_score_contract_sha256_v1, build_bounded_score_prediction_set_v1,
    build_conformal_interval_v1, verify_bounded_score_prediction_set_v1,
};
