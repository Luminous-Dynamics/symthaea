use symthaea_quantum_comp::{
    BetaReadinessStatus, VerificationStage, alpha9_to_alpha10_migration, current_beta_readiness,
    current_validation_snapshot, current_verification_matrix, known_schema_labels,
};

#[test]
fn alpha10_schema_labels_are_consistent() {
    assert!(
        known_schema_labels()
            .iter()
            .all(|label| label.ends_with("alpha10"))
    );
}

#[test]
fn verification_matrix_has_external_boundary() {
    let matrix = current_verification_matrix();
    assert!(
        matrix
            .rows
            .iter()
            .any(|row| row.stage == VerificationStage::External)
    );
    assert!(matrix.to_text().contains("verification_matrix_caveat"));
}

#[test]
fn migration_guide_is_alpha10_targeted() {
    let guide = alpha9_to_alpha10_migration();
    assert_eq!(guide.to_version, "0.1.0-alpha.10");
    assert!(guide.to_markdown().contains("schema-labels"));
}

#[test]
fn beta_readiness_is_not_overstated() {
    let report = current_beta_readiness();
    assert_ne!(report.status, BetaReadinessStatus::Ready);
    assert!(report.to_text().contains("beta_readiness_status"));
}

#[test]
fn validation_snapshot_contains_manifest_and_matrix() {
    let snapshot = current_validation_snapshot();
    let markdown = snapshot.to_markdown();
    assert!(markdown.contains("Alpha Release Manifest"));
    assert!(markdown.contains("Verification Matrix"));
}
