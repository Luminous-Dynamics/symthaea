use marine_mission_evidence_join::{
    assess_case, Currentness, EvidenceKind, MissionCase, MissionEvidence,
    OwnerOutcome,
};

const PROFILE: &str = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
const CONFIG: &str = "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb";
const MISSION: &str = "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc";
const CONDITIONS: &str = "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd";

fn row(owner_id: &str) -> MissionEvidence {
    MissionEvidence {
        owner_id: owner_id.to_owned(),
        report_ref: Some(format!("synthetic-report:{owner_id}")),
        result_digest: Some("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee".to_owned()),
        method_revision_ref: Some(format!("synthetic-method:{owner_id}:v1")),
        profile_id: Some("liveaboard-offshore-v1".to_owned()),
        profile_manifest_digest: Some(PROFILE.to_owned()),
        configuration_digest: Some(CONFIG.to_owned()),
        mission_contract_digest: Some(MISSION.to_owned()),
        operating_case_id: Some("offshore-30d-sea-state-4-v1".to_owned()),
        conditions_digest: Some(CONDITIONS.to_owned()),
        target_duration_seconds: Some(30 * 24 * 60 * 60),
        evidence_kind: Some(EvidenceKind::Synthetic),
        currentness: Some(Currentness::Current),
        currentness_ref: Some(format!("synthetic-currentness:{owner_id}")),
        outcome: Some(OwnerOutcome::SupportsDeclaredScope),
    }
}

fn main() {
    let case = MissionCase {
        profile_id: "liveaboard-offshore-v1".to_owned(),
        profile_manifest_digest: PROFILE.to_owned(),
        configuration_digest: CONFIG.to_owned(),
        mission_contract_digest: MISSION.to_owned(),
        operating_case_id: "offshore-30d-sea-state-4-v1".to_owned(),
        conditions_digest: CONDITIONS.to_owned(),
        target_duration_seconds: 30 * 24 * 60 * 60,
        required_owner_ids: vec![
            "energy_budget".to_owned(),
            "resource_autonomy".to_owned(),
            "habitability_motion".to_owned(),
        ],
        evidence: vec![
            row("energy_budget"),
            row("resource_autonomy"),
            row("habitability_motion"),
        ],
    };
    println!("{:#?}", assess_case(&case));
}
