use symthaea_interoception::{
    GateStatus, QualificationGateReceipt, QualificationReceipt,
    QUALIFICATION_RECEIPT_SCHEMA_VERSION, REQUIRED_QUALIFICATION_GATES,
};

fn receipt_with(status: GateStatus) -> QualificationReceipt {
    QualificationReceipt {
        schema_version: QUALIFICATION_RECEIPT_SCHEMA_VERSION,
        source_commit: "0123456789abcdef0123456789abcdef01234567".into(),
        gates: REQUIRED_QUALIFICATION_GATES
            .iter()
            .map(|gate| QualificationGateReceipt::new(*gate, status, format!("evidence:{gate}")))
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
    receipt.gates.push(QualificationGateReceipt::new(
        "benchmark_suite",
        GateStatus::Skipped,
        "github-actions:skipped",
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
fn qualification_receipt_survives_json_round_trip() {
    let receipt = receipt_with(GateStatus::Pending);
    let encoded = serde_json::to_vec(&receipt).expect("serialize qualification receipt");
    let decoded: QualificationReceipt =
        serde_json::from_slice(&encoded).expect("deserialize qualification receipt");

    assert_eq!(decoded, receipt);
    assert!(!decoded.is_qualified());
}
