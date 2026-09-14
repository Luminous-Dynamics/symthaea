use symthaea_interoception::{
    GateStatus, QualificationGateEvidence, QualificationGateReceipt, QualificationReceipt,
    INTEROCEPTIVE_MODEL_SEMANTICS_VERSION, QUALIFICATION_RECEIPT_SCHEMA_VERSION,
    REQUIRED_QUALIFICATION_GATES,
};

const SOURCE: &str = "0123456789abcdef0123456789abcdef01234567";
const OTHER_SOURCE: &str = "89abcdef0123456789abcdef0123456789abcdef";

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn evidence_for(gate: &str, source_commit: &str) -> QualificationGateEvidence {
    match gate {
        "local_fmt" => QualificationGateEvidence::local_command(
            source_commit,
            "cargo fmt --all --check",
            digest('a'),
            digest('b'),
        ),
        "local_test" => QualificationGateEvidence::local_command(
            source_commit,
            "cargo test -p symthaea-interoception",
            digest('a'),
            digest('c'),
        ),
        "local_clippy" => QualificationGateEvidence::local_command(
            source_commit,
            "cargo clippy -p symthaea-interoception --all-targets -- -D warnings",
            digest('a'),
            digest('d'),
        ),
        "workspace_ci" => {
            QualificationGateEvidence::github_actions(source_commit, "CI", 12345, 1)
        }
        "showroom_integrity" => QualificationGateEvidence::github_actions(
            source_commit,
            "Showroom Integrity",
            12346,
            1,
        ),
        other => panic!("unexpected qualification gate fixture: {other}"),
    }
}

fn receipt_with(status: GateStatus) -> QualificationReceipt {
    QualificationReceipt {
        schema_version: QUALIFICATION_RECEIPT_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        source_commit: SOURCE.into(),
        gates: REQUIRED_QUALIFICATION_GATES
            .iter()
            .map(|gate| {
                if status == GateStatus::Pending {
                    QualificationGateReceipt::pending(*gate)
                } else {
                    QualificationGateReceipt::with_evidence(
                        *gate,
                        status,
                        evidence_for(gate, SOURCE),
                    )
                }
            })
            .collect(),
    }
}

#[test]
fn all_required_passed_is_qualified() {
    let receipt = receipt_with(GateStatus::Passed);
    receipt.validate().expect("valid qualification receipt");
    assert!(receipt.is_qualified());
    assert!(receipt.blocking_required_gates().is_empty());
}

#[test]
fn skipped_required_gate_never_counts_as_passed() {
    let mut receipt = receipt_with(GateStatus::Passed);
    receipt.gates[0].status = GateStatus::Skipped;

    assert!(receipt.validate().is_ok());
    assert!(!receipt.is_qualified());
    assert_eq!(receipt.blocking_required_gates().len(), 1);
}

#[test]
fn optional_skipped_benchmark_does_not_block_required_gates() {
    let mut receipt = receipt_with(GateStatus::Passed);
    receipt.gates.push(QualificationGateReceipt::with_evidence(
        "benchmark_suite",
        GateStatus::Skipped,
        QualificationGateEvidence::github_actions(SOURCE, "Symthaea Benchmark Suite", 12347, 1),
    ));

    assert!(receipt.is_qualified());
}

#[test]
fn missing_or_duplicate_required_gate_invalidates_receipt() {
    let mut missing = receipt_with(GateStatus::Passed);
    missing.gates.pop();
    assert!(missing.validate().is_err());
    assert!(!missing.is_qualified());

    let mut duplicate = receipt_with(GateStatus::Passed);
    duplicate.gates.push(duplicate.gates[0].clone());
    assert!(duplicate.validate().is_err());
    assert!(!duplicate.is_qualified());
}

#[test]
fn semantics_mismatch_invalidates_receipt() {
    let mut receipt = receipt_with(GateStatus::Passed);
    receipt.model_semantics_version = INTEROCEPTIVE_MODEL_SEMANTICS_VERSION + 1;

    assert!(receipt.validate().is_err());
    assert!(!receipt.is_qualified());
}

#[test]
fn gate_evidence_from_another_source_head_is_rejected() {
    let mut receipt = receipt_with(GateStatus::Passed);
    receipt.gates[0].evidence = Some(evidence_for("local_fmt", OTHER_SOURCE));

    let errors = receipt
        .validate()
        .expect_err("cross-head gate evidence must invalidate qualification");
    assert!(errors
        .iter()
        .any(|error| error.contains("subject commit does not match")));
    assert!(!receipt.is_qualified());
}

#[test]
fn required_gate_rejects_wrong_evidence_kind() {
    let mut receipt = receipt_with(GateStatus::Passed);
    receipt.gates[0].evidence = Some(QualificationGateEvidence::github_actions(
        SOURCE,
        "CI",
        12345,
        1,
    ));

    let errors = receipt
        .validate()
        .expect_err("local_fmt must not accept GitHub Actions evidence");
    assert!(errors
        .iter()
        .any(|error| error.contains("incompatible evidence kind")));
}

#[test]
fn malformed_local_evidence_digest_is_rejected() {
    let mut receipt = receipt_with(GateStatus::Passed);
    receipt.gates[1].evidence = Some(QualificationGateEvidence::local_command(
        SOURCE,
        "cargo test -p symthaea-interoception",
        "not-a-digest",
        digest('a'),
    ));

    let errors = receipt
        .validate()
        .expect_err("malformed local evidence digest must fail");
    assert!(errors
        .iter()
        .any(|error| error.contains("environment_sha256")));
}

#[test]
fn pending_gate_may_omit_evidence_but_never_qualifies() {
    let receipt = receipt_with(GateStatus::Pending);
    receipt.validate().expect("pending receipt remains structurally valid");
    assert!(receipt.gates.iter().all(|gate| gate.evidence.is_none()));
    assert!(!receipt.is_qualified());
}

#[test]
fn qualification_receipt_survives_json_round_trip() {
    let receipt = receipt_with(GateStatus::Passed);
    let encoded = serde_json::to_vec(&receipt).expect("serialize qualification receipt");
    let decoded: QualificationReceipt =
        serde_json::from_slice(&encoded).expect("deserialize qualification receipt");

    assert_eq!(decoded, receipt);
    assert!(decoded.is_qualified());
}
