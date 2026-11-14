use mel_music_edit_adapter_harness::{
    music_reference_review_task_digest_v1, MusicReferenceReviewTaskV1,
    MusicReviewPanelPolicyV1,
};

const TASK: &str = include_str!("../fixtures/synthetic-music-reference-review-task-v1.json");
const POLICY: &str = include_str!("../fixtures/synthetic-music-review-panel-policy-v1.json");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let task: MusicReferenceReviewTaskV1 = serde_json::from_str(TASK)?;
    let policy: MusicReviewPanelPolicyV1 = serde_json::from_str(POLICY)?;
    let task_sha256 = music_reference_review_task_digest_v1(&task)?;
    if policy.minimum_reviewers < 2
        || policy.minimum_independence_groups < 2
        || !policy.require_unanimous_adequacy
    {
        return Err(std::io::Error::other("unsafe synthetic panel policy").into());
    }
    println!(
        "{{\"schema_version\":\"mel-music-review-static-fixture-check-v1\",\"task_sha256\":\"{}\",\"minimum_reviewers\":{},\"signature_verification_performed\":false,\"panel_qualified\":false}}",
        task_sha256, policy.minimum_reviewers
    );
    Ok(())
}
