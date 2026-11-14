use mel_music_edit_adapter_harness::{
    music_edit_validity_envelope_digest_v1, validate_music_edit_validity_envelope_v1,
    MusicEditValidityEnvelopeV1,
};

const FIXTURE: &str =
    include_str!("../fixtures/synthetic-music-edit-validity-envelope-v1.json");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let envelope: MusicEditValidityEnvelopeV1 = serde_json::from_str(FIXTURE)?;
    let issues = validate_music_edit_validity_envelope_v1(&envelope);
    if !issues.is_empty() {
        return Err(std::io::Error::other(format!(
            "validity envelope rejected: {issues:?}"
        )).into());
    }
    let digest = music_edit_validity_envelope_digest_v1(&envelope)?;
    println!(
        "{{\"schema_version\":\"mel-music-edit-validity-envelope-check-v1\",\"reference_count\":{},\"valid_alternative_count\":{},\"envelope_sha256\":\"{}\",\"audio_scored\":false,\"human_study_performed\":false}}",
        envelope.references.len(),
        envelope.references.iter().filter(|reference| {
            reference.role == mel_music_edit_adapter_harness::MusicEditReferenceRoleV1::ValidAlternative
                && reference.adequacy == mel_music_edit_adapter_harness::MusicEditReferenceAdequacyV1::Validated
        }).count(),
        digest
    );
    Ok(())
}
