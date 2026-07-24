// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Operate a restart-durable Series 17 power-loss execution journal.
//!
//! Commands:
//!   claim CAMPAIGN OPS_PLAN LEASE OPS_KEY JOURNAL_DIR SESSION32 EVENT32 NOW
//!   status CAMPAIGN OPS_PLAN LEASE OPS_KEY JOURNAL_DIR NOW
//!   advance CAMPAIGN OPS_PLAN LEASE OPS_KEY JOURNAL_DIR EXPECTED32 STATE EVENT32 SESSION32 NOW
//!
//! Digests and bindings are lowercase or uppercase hexadecimal without a `0x`
//! prefix. `OPS_KEY` is a private 48-byte file containing key-id || secret-key.

#[cfg(unix)]
use std::fs::OpenOptions;
#[cfg(unix)]
use std::io::Read;
#[cfg(unix)]
use std::path::{Path, PathBuf};

#[cfg(unix)]
use symthaea_vocal_tract::{
    CheckpointPowerLossExecutionJournal, CheckpointPowerLossExecutionState,
    CheckpointPowerLossJournalStore, CheckpointPowerLossOperationsAuthority,
    CheckpointPowerLossOperationsKey, CheckpointPowerLossOperationsKeyId,
    checkpoint_power_loss_resume_decision, decode_checkpoint_power_loss_campaign,
    decode_checkpoint_power_loss_operations_plan,
};
#[cfg(unix)]
use zeroize::Zeroize;

#[cfg(unix)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let command = required(&mut args, "command")?;
    let campaign_path = PathBuf::from(required(&mut args, "campaign artifact")?);
    let operations_path = PathBuf::from(required(&mut args, "operations plan")?);
    let lease_path = PathBuf::from(required(&mut args, "sealed lease")?);
    let key_path = PathBuf::from(required(&mut args, "operations key")?);
    let journal_root = PathBuf::from(required(&mut args, "journal directory")?);

    let campaign = decode_checkpoint_power_loss_campaign(&std::fs::read(campaign_path)?)?;
    let operations =
        decode_checkpoint_power_loss_operations_plan(&campaign, &std::fs::read(operations_path)?)?;
    let sealed_lease = std::fs::read(lease_path)?;
    let (key_id, mut key_bytes) = read_private_key_file(&key_path)?;
    let authority =
        CheckpointPowerLossOperationsAuthority::new(CheckpointPowerLossOperationsKey::new(
            CheckpointPowerLossOperationsKeyId::new(key_id)?,
            key_bytes,
        )?);
    key_bytes.zeroize();
    if authority.key_id() != operations.operations_authority_key_id {
        return Err("operations key does not match the operations plan".into());
    }
    let store = CheckpointPowerLossJournalStore::new(journal_root, authority);

    let (journal, now) = match command.as_str() {
        "claim" => {
            let session = parse_hex::<32>(&required(&mut args, "operator session binding")?)?;
            let event = parse_hex::<32>(&required(&mut args, "claim evidence digest")?)?;
            let now = required(&mut args, "unix seconds")?.parse::<u64>()?;
            ensure_no_trailing(&mut args)?;
            (
                store.create(&campaign, &operations, &sealed_lease, session, event, now)?,
                now,
            )
        }
        "status" => {
            let now = required(&mut args, "unix seconds")?.parse::<u64>()?;
            ensure_no_trailing(&mut args)?;
            (store.load(&campaign, &operations, &sealed_lease)?, now)
        }
        "advance" => {
            let expected = parse_hex::<32>(&required(&mut args, "expected journal digest")?)?;
            let state = parse_state(&required(&mut args, "next state")?)?;
            let event = parse_hex::<32>(&required(&mut args, "event evidence digest")?)?;
            let session = parse_hex::<32>(&required(&mut args, "operator session binding")?)?;
            let now = required(&mut args, "unix seconds")?.parse::<u64>()?;
            ensure_no_trailing(&mut args)?;
            (
                store.append(
                    &campaign,
                    &operations,
                    &sealed_lease,
                    expected,
                    state,
                    event,
                    session,
                    now,
                )?,
                now,
            )
        }
        _ => return Err("command must be claim, status, or advance".into()),
    };

    print_status(&store, &campaign, &operations, &sealed_lease, &journal, now)?;
    Ok(())
}

#[cfg(not(unix))]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    Err("the Series 17 journal operator currently requires Unix".into())
}

#[cfg(unix)]
fn print_status(
    store: &CheckpointPowerLossJournalStore,
    campaign: &symthaea_vocal_tract::CheckpointPowerLossCampaignPlan,
    operations: &symthaea_vocal_tract::CheckpointPowerLossOperationsPlan,
    sealed_lease: &[u8],
    journal: &CheckpointPowerLossExecutionJournal,
    now: u64,
) -> Result<(), Box<dyn std::error::Error>> {
    let lease = store
        .authority()
        .open_lease(campaign, operations, sealed_lease)?;
    let digest = journal.digest(campaign, operations, &lease)?;
    let decision =
        checkpoint_power_loss_resume_decision(campaign, operations, &lease, journal, now)?;
    println!("trial_id={}", hex(&lease.trial_id));
    println!("lease_id={}", hex(&lease.lease_id));
    println!("attempt={}", lease.attempt);
    println!("lab_id={}", hex(&lease.lab_id.0));
    println!("journal_digest={}", hex(&digest));
    println!("entries={}", journal.entries.len());
    println!("current_state={:?}", journal.current_state());
    println!("resume_decision={decision:?}");
    Ok(())
}

#[cfg(unix)]
fn parse_state(
    value: &str,
) -> Result<CheckpointPowerLossExecutionState, Box<dyn std::error::Error>> {
    Ok(match value {
        "prepared" => CheckpointPowerLossExecutionState::Prepared,
        "armed" => CheckpointPowerLossExecutionState::Armed,
        "power-event-observed" => CheckpointPowerLossExecutionState::PowerEventObserved,
        "recovery-started" => CheckpointPowerLossExecutionState::RecoveryStarted,
        "recovery-classified" => CheckpointPowerLossExecutionState::RecoveryClassified,
        "evidence-sealed" => CheckpointPowerLossExecutionState::EvidenceSealed,
        "completed" => CheckpointPowerLossExecutionState::Completed,
        "aborted" => CheckpointPowerLossExecutionState::Aborted,
        "quarantined" => CheckpointPowerLossExecutionState::Quarantined,
        _ => return Err("unknown journal state".into()),
    })
}

#[cfg(unix)]
fn required(
    args: &mut impl Iterator<Item = String>,
    name: &str,
) -> Result<String, Box<dyn std::error::Error>> {
    args.next().ok_or_else(|| format!("missing {name}").into())
}

#[cfg(unix)]
fn ensure_no_trailing(
    args: &mut impl Iterator<Item = String>,
) -> Result<(), Box<dyn std::error::Error>> {
    if args.next().is_some() {
        return Err("unexpected trailing arguments".into());
    }
    Ok(())
}

#[cfg(unix)]
fn parse_hex<const N: usize>(value: &str) -> Result<[u8; N], Box<dyn std::error::Error>> {
    if value.len() != N * 2 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        return Err(format!("expected exactly {} hexadecimal characters", N * 2).into());
    }
    let mut output = [0u8; N];
    for (index, slot) in output.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&value[index * 2..index * 2 + 2], 16)?;
    }
    if output.iter().all(|byte| *byte == 0) {
        return Err("zero bindings are not accepted".into());
    }
    Ok(output)
}

#[cfg(unix)]
fn read_private_key_file(path: &Path) -> Result<([u8; 16], [u8; 32]), Box<dyn std::error::Error>> {
    use std::os::unix::fs::{MetadataExt, OpenOptionsExt, PermissionsExt};

    let mut options = OpenOptions::new();
    options
        .read(true)
        .custom_flags(libc::O_CLOEXEC | libc::O_NOFOLLOW);
    let mut file = options.open(path)?;
    let metadata = file.metadata()?;
    if !metadata.is_file()
        || metadata.uid() != symthaea_vocal_tract::effective_uid()
        || metadata.permissions().mode() & 0o077 != 0
        || metadata.len() != 48
    {
        return Err("unsafe Series 17 key file".into());
    }
    let mut encoded = [0u8; 48];
    file.read_exact(&mut encoded)?;
    let mut trailing = [0u8; 1];
    if file.read(&mut trailing)? != 0 {
        encoded.zeroize();
        return Err("Series 17 key file contains trailing bytes".into());
    }
    let mut key_id = [0u8; 16];
    key_id.copy_from_slice(&encoded[..16]);
    let mut key = [0u8; 32];
    key.copy_from_slice(&encoded[16..]);
    encoded.zeroize();
    Ok((key_id, key))
}

#[cfg(unix)]
fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}
