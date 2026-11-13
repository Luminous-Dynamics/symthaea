use mel_music_edit_adapter_harness::{
    validate_music_edit_capability_matrix_v1, validate_music_edit_dataset_manifest_v1,
    MusicEditCapabilityMatrixV1, MusicEditDatasetManifestV1,
};

const DATASET_MANIFEST: &str =
    include_str!("../fixtures/synthetic-music-edit-dataset-manifest-v1.json");
const CAPABILITY_MATRIX: &str =
    include_str!("../fixtures/synthetic-music-edit-capability-matrix-v1.json");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest: MusicEditDatasetManifestV1 = serde_json::from_str(DATASET_MANIFEST)?;
    let matrix: MusicEditCapabilityMatrixV1 = serde_json::from_str(CAPABILITY_MATRIX)?;
    let manifest_issues = validate_music_edit_dataset_manifest_v1(&manifest);
    let matrix_issues = validate_music_edit_capability_matrix_v1(&matrix);
    if !manifest_issues.is_empty() || !matrix_issues.is_empty() {
        return Err(format!(
            "music-edit adapter fixture invalid: manifest={manifest_issues:?}; matrix={matrix_issues:?}"
        )
        .into());
    }
    println!(
        "{{\"schema_version\":\"mel-music-edit-adapter-fixture-check-v1\",\"dataset_rows\":{},\"operation_count\":{},\"all_operations_qualified\":{},\"model_run_performed\":false,\"benchmark_score_emitted\":false}}",
        manifest.rows.len(),
        matrix.capabilities.len(),
        matrix.capabilities.iter().all(|entry| {
            entry.qualification == mel_music_edit_adapter_harness::OperationQualificationV1::Qualified
        })
    );
    Ok(())
}
