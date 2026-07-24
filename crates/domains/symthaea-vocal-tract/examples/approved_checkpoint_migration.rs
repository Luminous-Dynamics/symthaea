// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_vocal_tract::{
    CheckpointKey, CheckpointKeyId, CheckpointMigrationApprovalKey, CheckpointSealContext,
    CheckpointStore, GestureFrame, OpenCheckpointExpectations, PhysicalSpeechSynthesizer,
    approve_checkpoint_migration_plan, open_approved_checkpoint_migration_plan,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root = std::env::temp_dir().join(format!(
        "symthaea-approved-migration-{}",
        std::process::id(),
    ));
    let old = CheckpointKey::from_parts(CheckpointKeyId([1; 16]), [0x31; 32])?;
    let next = CheckpointKey::from_parts(CheckpointKeyId([2; 16]), [0x52; 32])?;
    let mut store = CheckpointStore::new(&root, old);
    let checkpoint = PhysicalSpeechSynthesizer::default()
        .render_gesture_chunk(&vec![GestureFrame::default(); 4], None, false)?
        .next_checkpoint
        .ok_or("physical renderer did not return a checkpoint")?;
    let utterance_id = *b"migration-plan01";
    let first = store.save(
        "old-00000000.checkpoint",
        &checkpoint,
        CheckpointSealContext {
            utterance_id,
            sequence: 0,
            previous_envelope_digest: [0; 32],
        },
    )?;
    store.save(
        "old-00000001.checkpoint",
        &checkpoint,
        CheckpointSealContext {
            utterance_id,
            sequence: 1,
            previous_envelope_digest: first.envelope_digest,
        },
    )?;
    store.keyring_mut().rotate(next)?;

    let plan = store.plan_reencrypt_chain(
        &[
            "old-00000000.checkpoint".to_owned(),
            "old-00000001.checkpoint".to_owned(),
        ],
        "new",
        utterance_id,
        0,
        [0; 32],
        [0; 32],
    )?;
    let approval_key = CheckpointMigrationApprovalKey::new([0x93; 32])?;
    let approved = approve_checkpoint_migration_plan(&plan, &approval_key)?;
    let reviewed_plan = open_approved_checkpoint_migration_plan(&approved, &approval_key)?;
    let receipt = store.execute_reencrypt_plan(&reviewed_plan)?;

    let migrated_first = store.load_verified(
        "new-00000000.checkpoint",
        OpenCheckpointExpectations {
            utterance_id,
            sequence: 0,
            previous_envelope_digest: [0; 32],
        },
    )?;
    let migrated_second = store.load_verified(
        "new-00000001.checkpoint",
        OpenCheckpointExpectations {
            utterance_id,
            sequence: 1,
            previous_envelope_digest: migrated_first.envelope_digest,
        },
    )?;
    assert_eq!(receipt.target_chain_head, migrated_second.envelope_digest);

    std::fs::remove_dir_all(root)?;
    Ok(())
}
