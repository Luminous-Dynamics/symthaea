use super::*;

const SUBJECT: &str = "0123456789abcdef0123456789abcdef01234567";

fn digest(byte: u8) -> QualificationDigest {
    QualificationDigest([byte; 32])
}

fn profile(lane: QualificationLane) -> QualificationProfile {
    QualificationProfile::new(
        1,
        lane,
        ".github/workflows/qualify.yml",
        digest(7),
        vec!["workflow_dispatch".into()],
        vec![
            QualificationJobRequirement::required("oracle"),
            QualificationJobRequirement::required("rust"),
        ],
    )
    .unwrap()
}

fn job(id: u64, name: &str, disposition: JobDisposition) -> ObservedQualificationJob {
    ObservedQualificationJob {
        job_id: id,
        name: name.into(),
        disposition,
        step_summary: digest(id as u8),
        provider_status: Some("completed".into()),
        provider_conclusion: Some("success".into()),
    }
}

fn observation(jobs: Vec<ObservedQualificationJob>) -> QualificationRunObservation {
    QualificationRunObservation {
        repository: "Luminous-Dynamics/symthaea".into(),
        workflow_run_id: 42,
        workflow_id: 9,
        workflow_path: ".github/workflows/qualify.yml".into(),
        workflow_definition: digest(7),
        run_attempt: 1,
        event: "workflow_dispatch".into(),
        exact_head_sha: SUBJECT.into(),
        jobs,
        materializer_revision: 1,
        head_ref: Some("feat/example".into()),
        base_ref: Some("main".into()),
        provider_status: Some("completed".into()),
        provider_conclusion: Some("success".into()),
    }
}

fn pass_observation() -> QualificationRunObservation {
    observation(vec![
        job(1, "oracle", JobDisposition::Passed),
        job(2, "rust", JobDisposition::Passed),
    ])
}

#[test]
fn exact_full_head_success_mints_receipt() {
    let profile = profile(QualificationLane::FullExactHead);
    let observation = pass_observation();
    let evaluation = profile.evaluate(SUBJECT, &observation).unwrap();

    assert_eq!(evaluation.classification, QualificationClassification::Pass);
    assert!(evaluation.mechanical_integrity.satisfied);

    let receipt =
        QualifiedHeadReceipt::try_new(&profile, SUBJECT, &observation, &evaluation).unwrap();
    assert_eq!(receipt.expected_subject_sha, SUBJECT);
    assert_eq!(receipt.observed_head_sha, SUBJECT);
    assert_ne!(receipt.identity(), QualificationDigest([0; 32]));
}

#[test]
fn queued_required_job_is_not_executed_and_cannot_mint() {
    let profile = profile(QualificationLane::FullExactHead);
    let observation = observation(vec![
        job(1, "oracle", JobDisposition::Passed),
        job(2, "rust", JobDisposition::NotExecuted),
    ]);
    let evaluation = profile.evaluate(SUBJECT, &observation).unwrap();

    assert_eq!(
        evaluation.classification,
        QualificationClassification::NotExecuted
    );
    assert!(QualifiedHeadReceipt::try_new(&profile, SUBJECT, &observation, &evaluation).is_err());
}

#[test]
fn cancelled_required_job_is_infrastructure_indeterminate() {
    let profile = profile(QualificationLane::FullExactHead);
    let observation = observation(vec![
        job(1, "oracle", JobDisposition::Passed),
        job(2, "rust", JobDisposition::Cancelled),
    ]);
    assert_eq!(
        profile.evaluate(SUBJECT, &observation).unwrap().classification,
        QualificationClassification::InfrastructureIndeterminate
    );
}

#[test]
fn subject_and_qualifier_failures_remain_distinct() {
    let profile = profile(QualificationLane::FullExactHead);

    let subject = observation(vec![
        job(1, "oracle", JobDisposition::Passed),
        job(2, "rust", JobDisposition::FailSubject),
    ]);
    assert_eq!(
        profile.evaluate(SUBJECT, &subject).unwrap().classification,
        QualificationClassification::FailSubject
    );

    let qualifier = observation(vec![
        job(1, "oracle", JobDisposition::FailQualifier),
        job(2, "rust", JobDisposition::Passed),
    ]);
    assert_eq!(
        profile.evaluate(SUBJECT, &qualifier).unwrap().classification,
        QualificationClassification::FailQualifier
    );
}

#[test]
fn missing_and_duplicate_required_jobs_fail_closed() {
    let profile = profile(QualificationLane::FullExactHead);

    let missing = observation(vec![job(1, "oracle", JobDisposition::Passed)]);
    assert_eq!(
        profile.evaluate(SUBJECT, &missing).unwrap().classification,
        QualificationClassification::IncompleteJobSet
    );

    let duplicate = observation(vec![
        job(1, "oracle", JobDisposition::Passed),
        job(2, "oracle", JobDisposition::Passed),
        job(3, "rust", JobDisposition::Passed),
    ]);
    assert_eq!(
        profile.evaluate(SUBJECT, &duplicate).unwrap().classification,
        QualificationClassification::FailQualifier
    );
}

#[test]
fn stale_subject_is_not_reinterpreted_as_pass() {
    let profile = profile(QualificationLane::FullExactHead);
    let mut observation = pass_observation();
    observation.exact_head_sha = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa".into();

    assert_eq!(
        profile.evaluate(SUBJECT, &observation).unwrap().classification,
        QualificationClassification::StaleSubject
    );
}

#[test]
fn green_source_sanity_lane_is_wrong_lane_for_authority() {
    let profile = profile(QualificationLane::SourceSanity);
    let observation = pass_observation();
    let evaluation = profile.evaluate(SUBJECT, &observation).unwrap();

    assert_eq!(
        evaluation.classification,
        QualificationClassification::WrongLane
    );
    assert!(QualifiedHeadReceipt::try_new(&profile, SUBJECT, &observation, &evaluation).is_err());
}

#[test]
fn wrong_workflow_definition_or_event_fails_qualifier() {
    let profile = profile(QualificationLane::FullExactHead);

    let mut wrong_definition = pass_observation();
    wrong_definition.workflow_definition = digest(8);
    assert_eq!(
        profile
            .evaluate(SUBJECT, &wrong_definition)
            .unwrap()
            .classification,
        QualificationClassification::FailQualifier
    );

    let mut wrong_event = pass_observation();
    wrong_event.event = "push".into();
    assert_eq!(
        profile.evaluate(SUBJECT, &wrong_event).unwrap().classification,
        QualificationClassification::FailQualifier
    );
}

#[test]
fn job_order_does_not_change_normalized_identity() {
    let a = pass_observation();
    let b = observation(vec![
        job(2, "rust", JobDisposition::Passed),
        job(1, "oracle", JobDisposition::Passed),
    ]);

    assert_eq!(a.identity().unwrap(), b.identity().unwrap());
    assert_eq!(a.job_set_identity(), b.job_set_identity());
}

#[test]
fn changed_disposition_and_attempt_change_identity() {
    let a = pass_observation();

    let mut changed_disposition = pass_observation();
    changed_disposition.jobs[1].disposition = JobDisposition::FailSubject;
    assert_ne!(
        a.identity().unwrap(),
        changed_disposition.identity().unwrap()
    );

    let mut changed_attempt = pass_observation();
    changed_attempt.run_attempt = 2;
    assert_ne!(a.identity().unwrap(), changed_attempt.identity().unwrap());
}

#[test]
fn descriptive_provider_text_and_refs_do_not_change_authority_identity() {
    let a = pass_observation();
    let mut b = a.clone();

    b.head_ref = Some("renamed".into());
    b.base_ref = None;
    b.provider_status = Some("different wording".into());
    b.provider_conclusion = None;
    b.jobs[0].provider_status = Some("provider changed wording".into());
    b.jobs[0].provider_conclusion = None;

    assert_eq!(a.identity().unwrap(), b.identity().unwrap());
}

#[test]
fn changed_step_summary_changes_identity() {
    let a = pass_observation();
    let mut b = a.clone();
    b.jobs[0].step_summary = digest(99);
    assert_ne!(a.identity().unwrap(), b.identity().unwrap());
}

#[test]
fn conditional_skip_must_be_explicitly_preregistered() {
    let profile = QualificationProfile::new(
        1,
        QualificationLane::FullExactHead,
        ".github/workflows/qualify.yml",
        digest(7),
        vec!["workflow_dispatch".into()],
        vec![
            QualificationJobRequirement::required("oracle"),
            QualificationJobRequirement::conditionally_skippable("optional"),
        ],
    )
    .unwrap();

    let observation = observation(vec![
        job(1, "oracle", JobDisposition::Passed),
        job(2, "optional", JobDisposition::Skipped),
    ]);
    assert_eq!(
        profile.evaluate(SUBJECT, &observation).unwrap().classification,
        QualificationClassification::Pass
    );

    let strict = profile(QualificationLane::FullExactHead);
    let observation = observation(vec![
        job(1, "oracle", JobDisposition::Passed),
        job(2, "rust", JobDisposition::Skipped),
    ]);
    assert_eq!(
        strict.evaluate(SUBJECT, &observation).unwrap().classification,
        QualificationClassification::IncompleteJobSet
    );
}

#[test]
fn profile_rejects_duplicate_job_names_and_events() {
    let duplicate_jobs = QualificationProfile::new(
        1,
        QualificationLane::FullExactHead,
        ".github/workflows/qualify.yml",
        digest(7),
        vec!["workflow_dispatch".into()],
        vec![
            QualificationJobRequirement::required("rust"),
            QualificationJobRequirement::required("rust"),
        ],
    );
    assert_eq!(
        duplicate_jobs,
        Err(QualificationError::DuplicateRequiredJob)
    );

    let duplicate_events = QualificationProfile::new(
        1,
        QualificationLane::FullExactHead,
        ".github/workflows/qualify.yml",
        digest(7),
        vec!["pull_request".into(), "pull_request".into()],
        vec![QualificationJobRequirement::required("rust")],
    );
    assert_eq!(
        duplicate_events,
        Err(QualificationError::DuplicateAllowedEvent)
    );
}

#[test]
fn profile_identity_normalizes_declared_set_order() {
    let a = QualificationProfile::new(
        1,
        QualificationLane::FullExactHead,
        ".github/workflows/qualify.yml",
        digest(7),
        vec!["workflow_dispatch".into(), "pull_request".into()],
        vec![
            QualificationJobRequirement::required("rust"),
            QualificationJobRequirement::required("oracle"),
        ],
    )
    .unwrap();
    let b = QualificationProfile::new(
        1,
        QualificationLane::FullExactHead,
        ".github/workflows/qualify.yml",
        digest(7),
        vec!["pull_request".into(), "workflow_dispatch".into()],
        vec![
            QualificationJobRequirement::required("oracle"),
            QualificationJobRequirement::required("rust"),
        ],
    )
    .unwrap();

    assert_eq!(a.identity(), b.identity());
}

#[test]
fn malformed_sha_is_rejected_before_evaluation() {
    let profile = profile(QualificationLane::FullExactHead);
    let observation = pass_observation();

    assert!(matches!(
        profile.evaluate("ABC", &observation),
        Err(QualificationError::InvalidSha)
    ));
}

#[test]
fn unexpected_job_fails_qualifier() {
    let profile = profile(QualificationLane::FullExactHead);
    let observation = observation(vec![
        job(1, "oracle", JobDisposition::Passed),
        job(2, "rust", JobDisposition::Passed),
        job(3, "surprise", JobDisposition::Passed),
    ]);

    let evaluation = profile.evaluate(SUBJECT, &observation).unwrap();
    assert_eq!(
        evaluation.classification,
        QualificationClassification::FailQualifier
    );
    assert!(!evaluation.mechanical_integrity.satisfied);
    assert!(QualifiedHeadReceipt::try_new(&profile, SUBJECT, &observation, &evaluation).is_err());
}
