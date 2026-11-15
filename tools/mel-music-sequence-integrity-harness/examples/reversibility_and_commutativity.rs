use mel_music_sequence_integrity_harness::{
    assess_operation_pair_v1, invert_symbolic_edit_patch_v1, symbolic_state_sha256_v1,
    verify_symbolic_undo_redo_v1, DeclaredOperationRelationV1, MusicalEventDeltaV1,
    MusicalEventV1, OperationConflictResolutionV1, SymbolicEditPatchV1,
    SYMBOLIC_PATCH_SCHEMA_V1,
};
use serde::Serialize;

fn event(id: &str, pitch: i32) -> MusicalEventV1 {
    MusicalEventV1 {
        event_id: id.to_owned(),
        part_id: "piano".to_owned(),
        onset_ticks: 0,
        duration_ticks: 480,
        pitch_millist: pitch,
        velocity: 88,
        program: 0,
    }
}
fn patch(id: &str, base: &str, change: MusicalEventDeltaV1) -> SymbolicEditPatchV1 {
    SymbolicEditPatchV1 {
        schema_version: SYMBOLIC_PATCH_SCHEMA_V1.to_owned(),
        patch_id: id.to_owned(),
        session_base_state_sha256: base.to_owned(),
        changes: vec![change],
    }
}
#[derive(Serialize)]
struct Report {
    undo_restores_initial_sha256: String,
    redo_restores_edited_sha256: String,
    commuting_pair: mel_music_sequence_integrity_harness::OperationPairAssessmentV1,
    conflict_pair: mel_music_sequence_integrity_harness::OperationPairAssessmentV1,
    inverse_patch_id: String,
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let initial = vec![event("a", 60_000), event("b", 64_000)];
    let base = symbolic_state_sha256_v1(&initial)?;
    let left = patch("left-a", &base, MusicalEventDeltaV1 {
        event_id: "a".to_owned(),
        before: Some(event("a", 60_000)),
        after: Some(event("a", 62_000)),
    });
    let right_disjoint = patch("right-b", &base, MusicalEventDeltaV1 {
        event_id: "b".to_owned(),
        before: Some(event("b", 64_000)),
        after: Some(event("b", 65_000)),
    });
    let commuting = assess_operation_pair_v1(
        &initial, &left, &right_disjoint,
        DeclaredOperationRelationV1::Commuting, None,
    )?;

    let right_conflicting = patch("right-a", &base, MusicalEventDeltaV1 {
        event_id: "a".to_owned(),
        before: Some(event("a", 60_000)),
        after: Some(event("a", 63_000)),
    });
    let conflict = assess_operation_pair_v1(
        &initial, &left, &right_conflicting,
        DeclaredOperationRelationV1::NonCommuting,
        Some(OperationConflictResolutionV1::ManualMergeRequired),
    )?;
    let (undo_sha, redo_sha) = verify_symbolic_undo_redo_v1(&initial, &left)?;
    let inverse = invert_symbolic_edit_patch_v1(&left)?;

    println!("{}", serde_json::to_string_pretty(&Report {
        undo_restores_initial_sha256: undo_sha,
        redo_restores_edited_sha256: redo_sha,
        commuting_pair: commuting,
        conflict_pair: conflict,
        inverse_patch_id: inverse.patch_id,
    })?);
    Ok(())
}
