use mel_music_edit_adapter_harness::{
    evaluate_music_edit_validity_v1, verify_music_edit_validity_evaluation_receipt_v1,
    MusicEditValidityEnvelopeV1, MusicEditValidityObservationV1, MusicEditValidityDecisionV1,
};

const ENVELOPE: &str =
    include_str!("../fixtures/synthetic-music-edit-validity-envelope-v1.json");
const OBSERVATIONS: &str =
    include_str!("../fixtures/synthetic-music-edit-validity-observation-bundle-v1.json");

#[derive(serde::Deserialize)]
struct ObservationBundle {
    schema_version: String,
    candidate_output_sha256: String,
    source_lineage_sha256: String,
    observations: Vec<MusicEditValidityObservationV1>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let envelope: MusicEditValidityEnvelopeV1 = serde_json::from_str(ENVELOPE)?;
    let mut bundle: ObservationBundle = serde_json::from_str(OBSERVATIONS)?;
    if bundle.schema_version != "mel-music-edit-validity-observation-bundle-v1" {
        return Err(std::io::Error::other("wrong observation bundle schema").into());
    }
    let envelope_sha256 =
        mel_music_edit_adapter_harness::music_edit_validity_envelope_digest_v1(&envelope)?;
    for observation in &mut bundle.observations {
        observation.envelope_sha256 = envelope_sha256.clone();
    }

    let receipt = evaluate_music_edit_validity_v1(
        &envelope,
        &bundle.candidate_output_sha256,
        &bundle.source_lineage_sha256,
        &bundle.observations,
    ).map_err(|issues| std::io::Error::other(format!("evaluation failed closed: {issues:?}")))?;

    let replay_issues = verify_music_edit_validity_evaluation_receipt_v1(
        &receipt,
        &envelope,
        &bundle.candidate_output_sha256,
        &bundle.source_lineage_sha256,
        &bundle.observations,
    );
    if !replay_issues.is_empty() {
        return Err(std::io::Error::other(format!("evaluation receipt replay failed: {replay_issues:?}")).into());
    }
    if receipt.decision != MusicEditValidityDecisionV1::Acceptable {
        return Err(std::io::Error::other(format!(
            "synthetic fixture expected Acceptable but got {:?}",
            receipt.decision
        )).into());
    }
    println!(
        "{{\"schema_version\":\"mel-music-edit-validity-evaluator-check-v1\",\"outcome_count\":{},\"decision\":\"{:?}\",\"receipt_sha256\":\"{}\",\"real_audio_used\":false,\"human_study_performed\":false}}",
        receipt.outcomes.len(),
        receipt.decision,
        receipt.receipt_sha256
    );
    Ok(())
}
