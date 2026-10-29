use symthaea_operations_research::aviation_public_data_contract::{
    canonical_manifest_digest, validate_event_cluster_splits, validate_manifest,
    validate_scoring_ready_manifest, validate_as_of_feature, AcquisitionStatus, BenchmarkUse,
    DataContractError, DatasetPartition, DatasetSource, DatasetUseScope, EventClusterAssignment,
    FeatureAvailability, FeatureValueKind, LicenseStatus, PublicDataManifest, SplitLabel,
    SplitWindow, TemporalSplitContract,
};

const HASH_A: &str = "sha256:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const HASH_B: &str = "sha256:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";

fn source(
    source_id: &str,
    status: AcquisitionStatus,
    coverage_start: i64,
    coverage_end: i64,
    downloaded_at: Option<i64>,
) -> DatasetSource {
    DatasetSource {
        source_id: source_id.to_owned(),
        publisher: "Example Publisher".to_owned(),
        dataset_name: format!("synthetic dataset {source_id}"),
        release_id: "v0.1-test".to_owned(),
        schema_or_methodology_version: "schema-1".to_owned(),
        canonical_url: format!("https://example.test/{source_id}"),
        coverage_start_epoch: coverage_start,
        coverage_end_epoch: coverage_end,
        catalogued_at_epoch: 2_100,
        dataset_downloaded_at_epoch: downloaded_at,
        acquisition_status: status,
        license_status: LicenseStatus::Unknown,
        use_scope: DatasetUseScope::Unknown,
    }
}

fn partition(source_id: &str, id: &str, start: i64, end: i64, digest: &str) -> DatasetPartition {
    DatasetPartition {
        source_id: source_id.to_owned(),
        partition_id: id.to_owned(),
        start_epoch: start,
        end_epoch: end,
        downloaded_at_epoch: 2_300,
        byte_len: 128,
        content_digest: digest.to_owned(),
    }
}

fn manifest() -> PublicDataManifest {
    PublicDataManifest {
        manifest_id: "public-data-fixture-v1".to_owned(),
        manifest_version: "manifest-schema-1".to_owned(),
        created_at_epoch: 3_000,
        sources: vec![
            source("flight-events", AcquisitionStatus::Retrieved, 1_000, 2_000, Some(2_200)),
            source("solar", AcquisitionStatus::Catalogued, 1_000, 2_000, None),
        ],
        partitions: vec![
            partition("flight-events", "flights-001", 1_000, 1_500, HASH_A),
            partition("flight-events", "flights-002", 1_500, 2_000, HASH_B),
        ],
    }
}

fn split_contract() -> TemporalSplitContract {
    TemporalSplitContract {
        contract_id: "time-split-v1".to_owned(),
        train: SplitWindow { start_epoch: 100, end_epoch: 200 },
        calibration: SplitWindow { start_epoch: 220, end_epoch: 250 },
        validation: SplitWindow { start_epoch: 270, end_epoch: 300 },
        test: SplitWindow { start_epoch: 350, end_epoch: 400 },
    }
}

fn feature(kind: FeatureValueKind, information: i64, available: i64, value_time: i64) -> FeatureAvailability {
    FeatureAvailability {
        feature_id: "grid-price".to_owned(),
        source_id: "tariff-source-v1".to_owned(),
        source_digest: HASH_A.to_owned(),
        value_kind: kind,
        information_time_epoch: information,
        available_at_epoch: available,
        value_time_epoch: value_time,
    }
}

#[test]
fn valid_manifest_has_stable_canonical_digest() {
    let original = manifest();
    validate_manifest(&original).unwrap();
    let digest = canonical_manifest_digest(&original).unwrap();
    assert!(digest.starts_with("sha256:"));
    assert_eq!(canonical_manifest_digest(&original).unwrap(), digest);

    let mut reordered = original.clone();
    reordered.sources.reverse();
    reordered.partitions.reverse();
    assert_eq!(canonical_manifest_digest(&reordered).unwrap(), digest);
}

#[test]
fn canonical_manifest_digest_matches_independent_known_answer_vector() {
    assert_eq!(
        canonical_manifest_digest(&manifest()).unwrap(),
        "sha256:83951fc7c040759120fb6509b0ddaecf2c637f700f3051d2f9e7348af1f4414e"
    );
}

#[test]
fn manifest_digest_changes_when_partition_content_changes() {
    let original = manifest();
    let mut changed = original.clone();
    changed.partitions[0].content_digest = HASH_B.to_owned();
    assert_ne!(
        canonical_manifest_digest(&original).unwrap(),
        canonical_manifest_digest(&changed).unwrap()
    );
}

#[test]
fn retrieved_source_requires_download_time_and_at_least_one_partition() {
    let mut case = manifest();
    case.partitions.clear();
    assert_eq!(
        validate_manifest(&case).unwrap_err(),
        DataContractError::MissingRetrievedPartition {
            source_id: "flight-events".to_owned()
        }
    );

    case.sources[0].dataset_downloaded_at_epoch = None;
    case.partitions.push(partition("flight-events", "flights-001", 1_000, 2_000, HASH_A));
    assert_eq!(
        validate_manifest(&case).unwrap_err(),
        DataContractError::InvalidTimestamp("retrieved dataset time")
    );
}

#[test]
fn unknown_source_and_unretrieved_source_partition_are_rejected() {
    let mut case = manifest();
    case.partitions[0].source_id = "absent-source".to_owned();
    assert_eq!(
        validate_manifest(&case).unwrap_err(),
        DataContractError::UnknownPartitionSource {
            source_id: "absent-source".to_owned()
        }
    );

    let mut case = manifest();
    case.partitions[0].source_id = "solar".to_owned();
    assert_eq!(
        validate_manifest(&case).unwrap_err(),
        DataContractError::PartitionSourceNotRetrieved {
            source_id: "solar".to_owned()
        }
    );
}

#[test]
fn overlapping_partitions_are_rejected() {
    let mut case = manifest();
    case.partitions[1].start_epoch = 1_400;
    assert_eq!(
        validate_manifest(&case).unwrap_err(),
        DataContractError::OverlappingSourcePartitions {
            source_id: "flight-events".to_owned()
        }
    );
}

#[test]
fn partition_coverage_and_digest_are_enforced() {
    let mut case = manifest();
    case.partitions[0].start_epoch = 900;
    assert_eq!(
        validate_manifest(&case).unwrap_err(),
        DataContractError::PartitionOutOfSourceCoverage {
            partition_id: "flights-001".to_owned()
        }
    );

    let mut case = manifest();
    case.partitions[0].content_digest = "sha256:BAD".to_owned();
    assert_eq!(
        validate_manifest(&case).unwrap_err(),
        DataContractError::InvalidDigest { field: "partition content" }
    );
}

#[test]
fn catalogued_source_cannot_claim_downloaded_partitions() {
    let mut case = manifest();
    case.sources[1].dataset_downloaded_at_epoch = Some(2_200);
    case.partitions.push(partition("solar", "solar-001", 1_000, 2_000, HASH_A));
    assert_eq!(
        validate_manifest(&case).unwrap_err(),
        DataContractError::PartitionSourceNotRetrieved { source_id: "solar".to_owned() }
    );
}

#[test]
fn scoring_is_blocked_until_sources_are_retrieved_terms_reviewed_and_scope_allows_use() {
    let mut case = manifest();
    case.sources[0].license_status = LicenseStatus::VerifiedOpen;
    case.sources[0].use_scope = DatasetUseScope::CommercialAllowed;
    assert_eq!(
        validate_scoring_ready_manifest(&case, BenchmarkUse::CommercialDevelopment).unwrap_err(),
        DataContractError::SourceNotScoringReady {
            source_id: "solar".to_owned()
        }
    );

    case.sources[1].acquisition_status = AcquisitionStatus::Retrieved;
    case.sources[1].dataset_downloaded_at_epoch = Some(2_200);
    case.partitions.push(partition("solar", "solar-001", 1_000, 2_000, HASH_A));
    assert_eq!(
        validate_scoring_ready_manifest(&case, BenchmarkUse::CommercialDevelopment).unwrap_err(),
        DataContractError::LicenseNotCleared {
            source_id: "solar".to_owned()
        }
    );

    case.sources[1].license_status = LicenseStatus::TermsReviewed;
    case.sources[1].use_scope = DatasetUseScope::NonCommercialOnly;
    assert_eq!(
        validate_scoring_ready_manifest(&case, BenchmarkUse::CommercialDevelopment).unwrap_err(),
        DataContractError::UseScopeNotAuthorized {
            source_id: "solar".to_owned(),
            requested_use: BenchmarkUse::CommercialDevelopment,
        }
    );
    validate_scoring_ready_manifest(&case, BenchmarkUse::NonCommercialResearch).unwrap();

    case.sources[1].use_scope = DatasetUseScope::CommercialAllowed;
    validate_scoring_ready_manifest(&case, BenchmarkUse::CommercialDevelopment).unwrap();
}

#[test]
fn temporal_split_contract_accepts_ordered_gaps_and_half_open_endpoints() {
    let contract = split_contract();
    contract.validate().unwrap();
    assert_eq!(contract.label_at(100).unwrap(), Some(SplitLabel::Train));
    assert_eq!(contract.label_at(199).unwrap(), Some(SplitLabel::Train));
    assert_eq!(contract.label_at(200).unwrap(), None);
    assert_eq!(contract.label_at(220).unwrap(), Some(SplitLabel::Calibration));
    assert_eq!(contract.label_at(399).unwrap(), Some(SplitLabel::Test));
    assert_eq!(contract.label_at(400).unwrap(), None);
}

#[test]
fn overlapping_or_out_of_order_temporal_splits_are_rejected() {
    let mut case = split_contract();
    case.calibration.start_epoch = 190;
    assert_eq!(
        case.validate().unwrap_err(),
        DataContractError::OverlappingSplitWindows
    );

    let mut case = split_contract();
    case.test.end_epoch = case.test.start_epoch;
    assert_eq!(
        case.validate().unwrap_err(),
        DataContractError::InvalidSplitWindow("test")
    );
}

#[test]
fn one_event_cluster_cannot_cross_temporal_splits() {
    validate_event_cluster_splits(&[
        EventClusterAssignment {
            event_cluster_id: "disruption-day-1".to_owned(),
            split: SplitLabel::Train,
        },
        EventClusterAssignment {
            event_cluster_id: "disruption-day-1".to_owned(),
            split: SplitLabel::Train,
        },
        EventClusterAssignment {
            event_cluster_id: "disruption-day-2".to_owned(),
            split: SplitLabel::Test,
        },
    ]).unwrap();

    assert_eq!(
        validate_event_cluster_splits(&[
            EventClusterAssignment {
                event_cluster_id: "same-irregular-operation".to_owned(),
                split: SplitLabel::Train,
            },
            EventClusterAssignment {
                event_cluster_id: "same-irregular-operation".to_owned(),
                split: SplitLabel::Test,
            },
        ]).unwrap_err(),
        DataContractError::EventClusterLeakage {
            event_cluster_id: "same-irregular-operation".to_owned(),
            first_split: SplitLabel::Train,
            conflicting_split: SplitLabel::Test,
        }
    );
}

#[test]
fn observed_feature_must_be_available_and_refer_to_nonfuture_time() {
    validate_as_of_feature(&feature(FeatureValueKind::Observed, 80, 90, 95), 100).unwrap();

    assert_eq!(
        validate_as_of_feature(&feature(FeatureValueKind::Observed, 80, 110, 95), 100).unwrap_err(),
        DataContractError::FutureInformationLeakage {
            feature_id: "grid-price".to_owned(),
            decision_at_epoch: 100,
            available_at_epoch: 110,
        }
    );

    assert_eq!(
        validate_as_of_feature(&feature(FeatureValueKind::Observed, 80, 90, 101), 100).unwrap_err(),
        DataContractError::FutureObservedValue {
            feature_id: "grid-price".to_owned(),
            decision_at_epoch: 100,
            value_time_epoch: 101,
        }
    );
}

#[test]
fn valid_forecast_for_future_interval_is_allowed_only_if_available_in_time() {
    validate_as_of_feature(&feature(FeatureValueKind::Forecast, 70, 90, 200), 100).unwrap();
    assert_eq!(
        validate_as_of_feature(&feature(FeatureValueKind::Forecast, 101, 105, 200), 100).unwrap_err(),
        DataContractError::FutureInformationLeakage {
            feature_id: "grid-price".to_owned(),
            decision_at_epoch: 100,
            available_at_epoch: 105,
        }
    );
}

#[test]
fn information_timestamp_cannot_follow_availability_timestamp() {
    assert_eq!(
        validate_as_of_feature(&feature(FeatureValueKind::Forecast, 95, 90, 200), 100).unwrap_err(),
        DataContractError::InvalidFeatureChronology { feature_id: "grid-price".to_owned() }
    );
}
