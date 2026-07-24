// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

use symthaea_vocal_tract::{
    CheckpointKey, CheckpointSealContext, CheckpointStore, DurableRollbackProtector, GestureFrame,
    OpenCheckpointExpectations, PhysicalSpeechSynthesizer, RollbackStateKey,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let root =
        std::env::temp_dir().join(format!("symthaea-durable-rollback-{}", std::process::id(),));
    let checkpoint_root = root.join("checkpoints");
    let rollback_root = root.join("rollback-state");
    let store = CheckpointStore::new(
        &checkpoint_root,
        CheckpointKey::from_parts(
            symthaea_vocal_tract::CheckpointKeyId([0x51; 16]),
            [0x63; 32],
        )?,
    );
    let checkpoint = PhysicalSpeechSynthesizer::default()
        .render_gesture_chunk(&vec![GestureFrame::default(); 4], None, false)?
        .next_checkpoint
        .ok_or("physical renderer did not return a checkpoint")?;
    let utterance_id = *b"durable-state-01";

    let first = {
        let protector =
            DurableRollbackProtector::new(&rollback_root, RollbackStateKey::new([0x84; 32])?);
        store.save_with_rollback_protection(
            "state-000.checkpoint",
            &checkpoint,
            CheckpointSealContext {
                utterance_id,
                sequence: 0,
                previous_envelope_digest: [0; 32],
            },
            &protector,
        )?
    };

    // Recreate the protector to demonstrate restart persistence.
    let protector =
        DurableRollbackProtector::new(&rollback_root, RollbackStateKey::new([0x84; 32])?);
    store.load_with_rollback_protection(
        "state-000.checkpoint",
        OpenCheckpointExpectations {
            utterance_id,
            sequence: 0,
            previous_envelope_digest: [0; 32],
        },
        &protector,
    )?;
    store.save_with_rollback_protection(
        "state-001.checkpoint",
        &checkpoint,
        CheckpointSealContext {
            utterance_id,
            sequence: 1,
            previous_envelope_digest: first.envelope_digest,
        },
        &protector,
    )?;

    std::fs::remove_dir_all(root)?;
    Ok(())
}
