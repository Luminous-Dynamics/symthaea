use mel_music_edit_adapter_harness::{
    materialize_music_edit_dataset_v1, verify_music_edit_materialization_receipt_v1,
    MusicEditArtifactMapV1, MusicEditDatasetManifestV1, MusicEditMaterializationLimitsV1,
};
use std::{env, fs, io, path::Path};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.len() != 4 {
        return Err(io::Error::other(
            "usage: cargo run --example materialize -- <artifact-root> <dataset-manifest.json> <artifact-map.json> <receipt-output.json>"
        ).into());
    }

    let root = Path::new(&args[0]);
    let manifest: MusicEditDatasetManifestV1 =
        serde_json::from_slice(&fs::read(&args[1])?)?;
    let artifact_map: MusicEditArtifactMapV1 =
        serde_json::from_slice(&fs::read(&args[2])?)?;

    let receipt = materialize_music_edit_dataset_v1(
        root,
        &manifest,
        &artifact_map,
        MusicEditMaterializationLimitsV1::default(),
    )
    .map_err(|issues| io::Error::other(format!("materialization rejected: {issues:?}")))?;

    let replay_issues =
        verify_music_edit_materialization_receipt_v1(&receipt, &manifest, &artifact_map);
    if !replay_issues.is_empty() {
        return Err(io::Error::other(format!("receipt replay rejected: {replay_issues:?}")).into());
    }

    fs::write(&args[3], serde_json::to_vec_pretty(&receipt)?)?;
    println!(
        "{{\"schema_version\":\"mel-music-edit-materialization-command-v1\",\"sample_count\":{},\"artifact_count\":{},\"total_bytes\":{},\"receipt_sha256\":\"{}\",\"benchmark_score_emitted\":false}}",
        receipt.sample_count, receipt.artifact_count, receipt.total_bytes, receipt.receipt_sha256
    );
    Ok(())
}
