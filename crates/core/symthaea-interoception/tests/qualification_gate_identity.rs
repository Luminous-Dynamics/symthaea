use symthaea_interoception::{
    GateStatus, QualificationGateEvidence, QualificationGateReceipt, QualificationReceipt,
    INTEROCEPTIVE_MODEL_SEMANTICS_VERSION, QUALIFICATION_RECEIPT_SCHEMA_VERSION,
};

const SOURCE: &str = "0123456789abcdef0123456789abcdef01234567";

fn digest(ch: char) -> String {
    std::iter::repeat_n(ch, 64).collect()
}

fn base_receipt(gate: QualificationGateReceipt) -> QualificationReceipt {
    let local_test = QualificationGateReceipt::with_evidence(
        "local_test",
        GateStatus::Passed,
        QualificationGateEvidence::local_command(
            SOURCE,
            "cargo test -p symthaea-interoception",
            digest('a'),
            digest('b'),
        ),
    );
    let local_clippy = QualificationGateReceipt::with_evidence(
        "local_clippy",
        GateStatus::Passed,
        QualificationGateEvidence::local_command(
            SOURCE,
            "cargo clippy -p symthaea-interoception --all-targets -- -D warnings",
            digest('a'),
            digest('c'),
        ),
    );
    let workspace_ci = QualificationGateReceipt::with_evidence(
        "workspace_ci",
        GateStatus::Passed,
        QualificationGateEvidence::github_actions(SOURCE, "CI", 1001, 1),
    );
    let showroom = QualificationGateReceipt::with_evidence(
        "showroom_integrity",
        GateStatus::Passed,
        QualificationGateEvidence::github_actions(SOURCE, "Showroom Integrity", 1002, 1),
    );

    QualificationReceipt {
        schema_version: QUALIFICATION_RECEIPT_SCHEMA_VERSION,
        model_semantics_version: INTEROCEPTIVE_MODEL_SEMANTICS_VERSION,
        source_commit: SOURCE.into(),
        gates: vec![gate, local_test, local_clippy, workspace_ci, showroom],
    }
}

#[test]
fn local_fmt_cannot_be_satisfied_by_an_unrelated_local_command() {
    let receipt = base_receipt(QualificationGateReceipt::with_evidence(
        "local_fmt",
        GateStatus::Passed,
        QualificationGateEvidence::local_command(
            SOURCE,
            "cargo check -p symthaea-interoception",
            digest('a'),
            digest('d'),
        ),
    ));

    let errors = receipt
        .validate()
        .expect_err("a substituted command identity must fail qualification");
    assert!(errors
        .iter()
        .any(|error| error.contains("incompatible evidence kind or identity")));
    assert!(!receipt.is_qualified());
}

#[test]
fn workspace_ci_cannot_be_satisfied_by_an_unrelated_workflow() {
    let mut receipt = base_receipt(QualificationGateReceipt::with_evidence(
        "local_fmt",
        GateStatus::Passed,
        QualificationGateEvidence::local_command(
            SOURCE,
            "cargo fmt --all --check",
            digest('a'),
            digest('d'),
        ),
    ));
    receipt.gates[3].evidence = Some(QualificationGateEvidence::github_actions(
        SOURCE,
        "Unrelated Workflow",
        1001,
        1,
    ));

    let errors = receipt
        .validate()
        .expect_err("a substituted workflow identity must fail qualification");
    assert!(errors
        .iter()
        .any(|error| error.contains("incompatible evidence kind or identity")));
    assert!(!receipt.is_qualified());
}
