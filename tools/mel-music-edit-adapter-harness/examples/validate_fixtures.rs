use mel_music_edit_adapter_harness::{
    compare_music_edit_scopes_v1, validate_music_edit_capability_matrix_v1,
    validate_music_edit_dataset_manifest_v1, MusicEditArtifactMapV1,
    MusicEditCapabilityMatrixV1, MusicEditComparisonScopeV1, MusicEditDatasetManifestV1,
};

const DATASET_MANIFEST: &str =
    include_str!("../fixtures/synthetic-music-edit-dataset-manifest-v1.json");
const CAPABILITY_MATRIX: &str =
    include_str!("../fixtures/synthetic-music-edit-capability-matrix-v1.json");
const COMPARISON_SCOPE: &str =
    include_str!("../fixtures/synthetic-music-edit-comparison-scope-v1.json");
const ARTIFACT_MAP: &str =
    include_str!("../fixtures/synthetic-music-edit-artifact-map-v1.json");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let manifest: MusicEditDatasetManifestV1 = serde_json::from_str(DATASET_MANIFEST)?;
    let matrix: MusicEditCapabilityMatrixV1 = serde_json::from_str(CAPABILITY_MATRIX)?;
    let scope: MusicEditComparisonScopeV1 = serde_json::from_str(COMPARISON_SCOPE)?;
    let artifact_map: MusicEditArtifactMapV1 = serde_json::from_str(ARTIFACT_MAP)?;

    let manifest_issues = validate_music_edit_dataset_manifest_v1(&manifest);
    let matrix_issues = validate_music_edit_capability_matrix_v1(&matrix);
    let scope_comparison = compare_music_edit_scopes_v1(&scope, &scope);
    let manifest_ids: std::collections::BTreeSet<&str> =
        manifest.rows.iter().map(|row| row.sample_id.as_str()).collect();
    let map_ids: std::collections::BTreeSet<&str> =
        artifact_map.rows.iter().map(|row| row.sample_id.as_str()).collect();
    let map_valid = artifact_map.schema_version
        == mel_music_edit_adapter_harness::MUSIC_EDIT_ARTIFACT_MAP_SCHEMA_V1
        && artifact_map.rows.len() == manifest.rows.len()
        && map_ids == manifest_ids;

    if !manifest_issues.is_empty() || !matrix_issues.is_empty()
        || !scope_comparison.directly_comparable || !map_valid
    {
        return Err(std::io::Error::other(format!(
            "music-edit adapter fixture invalid: manifest={manifest_issues:?}; matrix={matrix_issues:?}; artifact_map_valid={map_valid}"
        )).into());
    }
    println!(
        "{{\"schema_version\":\"mel-music-edit-adapter-fixture-check-v1\",\"dataset_rows\":{},\"artifact_map_rows\":{},\"operation_count\":{},\"all_operations_qualified\":{},\"model_run_performed\":false,\"benchmark_score_emitted\":false}}",
        manifest.rows.len(),
        artifact_map.rows.len(),
        matrix.capabilities.len(),
        matrix.capabilities.iter().all(|entry| {
            entry.qualification == mel_music_edit_adapter_harness::OperationQualificationV1::Qualified
        })
    );
    Ok(())
}
