use symthaea_research_split::{
    AssignedUnit, GroupRef, GroupSeparationPolicy, PartitionRole, ResearchSplitManifest, SplitUnit,
    TemporalSeparationPolicy,
};

fn assigned(id: &str, time: i64, spatial: &str, role: PartitionRole) -> AssignedUnit {
    AssignedUnit::new(
        SplitUnit::new(
            id,
            time,
            format!("digest:{id}"),
            vec![GroupRef::new("spatial-block", spatial).unwrap()],
        )
        .unwrap(),
        role,
    )
}

fn manifest() -> ResearchSplitManifest {
    ResearchSplitManifest::new(
        "serde-split",
        vec![
            assigned("train", 1_000, "a", PartitionRole::Training),
            assigned("eval", 2_000, "b", PartitionRole::Evaluation),
        ],
        GroupSeparationPolicy::EvaluationDisjoint {
            dimensions: vec!["spatial-block".into()],
        },
        TemporalSeparationPolicy::ForwardEvaluation { embargo_ms: 500 },
        vec![],
    )
    .unwrap()
}

#[test]
fn deserialization_rejects_stale_digest_after_semantic_mutation() {
    let original = manifest();
    let mut json = serde_json::to_value(&original).unwrap();
    json["assignments"][1]["unit"]["observed_at_unix_ms"] = serde_json::json!(2_100);

    let decoded = serde_json::from_value::<ResearchSplitManifest>(json);
    assert!(decoded.is_err());
}

#[test]
fn deserialization_rejects_digest_recomputed_over_structurally_invalid_split() {
    // The public serialized representation cannot use a forged digest to bypass semantic
    // validation. This attack mutates evaluation onto the training block; regardless of digest
    // contents, deserialization must fail before the manifest can be used.
    let original = manifest();
    let mut json = serde_json::to_value(&original).unwrap();
    json["assignments"][1]["unit"]["groups"][0]["value"] = serde_json::json!("a");
    json["manifest_digest"] = serde_json::json!("attacker-recomputed-or-forged-digest");

    let decoded = serde_json::from_value::<ResearchSplitManifest>(json);
    assert!(decoded.is_err());
}
