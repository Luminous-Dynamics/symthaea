use mel_music_rights_policy_harness::{
    evaluate_music_rights_use_v1, MusicComponentRightsBundleV1, MusicRightsUseRequestV1,
    RightsDecisionStatusV1,
};

const RIGHTS: &str = include_str!("../fixtures/synthetic-sm ebench-rights-structure-v1.json");
const REQUEST: &str = include_str!("../fixtures/synthetic-commercial-evaluation-request-v1.json");

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let rights: MusicComponentRightsBundleV1 = serde_json::from_str(RIGHTS)?;
    let request: MusicRightsUseRequestV1 = serde_json::from_str(REQUEST)?;
    let decision = evaluate_music_rights_use_v1(&rights, &request)?;
    if decision.status != RightsDecisionStatusV1::Blocked {
        return Err("unreviewed metadata-only fixture must remain blocked".into());
    }
    println!(
        "{{\"schema_version\":\"mel-music-rights-fixture-check-v1\",\"components\":{},\"decision\":\"blocked\",\"reason_count\":{},\"legal_clearance_inferred\":false}}",
        rights.components.len(),
        decision.reasons.len()
    );
    Ok(())
}
