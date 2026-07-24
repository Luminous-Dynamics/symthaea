// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::os::unix::net::UnixListener;
    use std::thread;

    use symthaea_vocal_tract::{
        CheckpointAgentToken, CheckpointKey, CheckpointKeyAgentReplayGuard, CheckpointKeyId,
        CheckpointKeyring, CheckpointSealContext, CheckpointStore, GestureFrame,
        OpenCheckpointExpectations, PhysicalSpeechSynthesizer, UnixCheckpointKeyAgent,
        serve_checkpoint_key_agent_connection,
    };

    let root = std::env::temp_dir().join(format!(
        "symthaea-external-key-agent-{}",
        std::process::id(),
    ));
    std::fs::create_dir_all(&root)?;
    let socket = root.join("checkpoint-key-agent.sock");
    let checkpoint_root = root.join("checkpoints");
    let listener = UnixListener::bind(&socket)?;
    let token_bytes = [0xa7; 32];
    let keyring = CheckpointKeyring::new(CheckpointKey::from_parts(
        CheckpointKeyId([0x41; 16]),
        [0x73; 32],
    )?);
    let server = thread::spawn(move || {
        let token = CheckpointAgentToken::new(token_bytes).unwrap();
        let replay_guard = CheckpointKeyAgentReplayGuard::with_default_window();
        // One save performs ActiveKeyId + ActiveEncryptionKey; one load
        // performs DecryptionKey.
        for _ in 0..3 {
            let (mut stream, _) = listener.accept().unwrap();
            serve_checkpoint_key_agent_connection(&mut stream, &keyring, &token, &replay_guard)
                .unwrap();
        }
    });

    let provider = UnixCheckpointKeyAgent::new(&socket, CheckpointAgentToken::new(token_bytes)?);
    let store = CheckpointStore::with_provider(&checkpoint_root, provider);
    let checkpoint = PhysicalSpeechSynthesizer::default()
        .render_gesture_chunk(&vec![GestureFrame::default(); 4], None, false)?
        .next_checkpoint
        .ok_or("physical renderer did not return a checkpoint")?;
    let context = CheckpointSealContext {
        utterance_id: *b"external-agent01",
        sequence: 0,
        previous_envelope_digest: [0; 32],
    };
    store.save("state-000.checkpoint", &checkpoint, context)?;
    store.load(
        "state-000.checkpoint",
        OpenCheckpointExpectations {
            utterance_id: context.utterance_id,
            sequence: context.sequence,
            previous_envelope_digest: context.previous_envelope_digest,
        },
    )?;
    server.join().map_err(|_| "key-agent server panicked")?;
    std::fs::remove_dir_all(root)?;
    Ok(())
}

#[cfg(not(unix))]
fn main() {
    eprintln!("the external checkpoint key-agent example requires Unix-domain sockets");
}
