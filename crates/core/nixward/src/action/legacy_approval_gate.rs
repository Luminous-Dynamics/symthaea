// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Transitional replay guard for the legacy plain-text watchdog approval channel.
//!
//! This module exists only to make the current daemon/TUI transport fail closed
//! while the request-bound approval protocol in #5231/#5242 is integrated.
//! Plain text such as `Approved` is not execution authority and must never be
//! transferable from one pending action to another.

use std::io;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

static LEGACY_VERDICT_TEMP_COUNTER: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyApprovalVerdictV1 {
    Approved,
    Vetoed,
    Other,
}

impl LegacyApprovalVerdictV1 {
    pub fn parse(text: &str) -> Self {
        match text.trim().to_ascii_lowercase().as_str() {
            "approved" | "a" => Self::Approved,
            "vetoed" | "v" => Self::Vetoed,
            _ => Self::Other,
        }
    }

    fn persisted_text(self) -> io::Result<&'static str> {
        match self {
            Self::Approved => Ok("Approved"),
            Self::Vetoed => Ok("Vetoed"),
            Self::Other => Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "cannot persist an unrecognized legacy approval verdict",
            )),
        }
    }
}

/// Result of evaluating one legacy verdict against the daemon's current
/// in-memory pending action.
///
/// `AwaitingFreshApproval` deliberately carries an invalidation requirement.
/// If `invalidate_existing_verdict` is true, the caller must successfully
/// remove/invalidate the ambient verdict artifact before installing or keeping
/// the new pending action. Failure to invalidate must fail closed for mutation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyApprovalGateDecisionV1 {
    ApprovedCurrent,
    VetoedCurrent,
    AwaitingFreshApproval {
        invalidate_existing_verdict: bool,
    },
}

pub fn evaluate_legacy_approval_v1(
    pending_action: Option<&str>,
    current_action: &str,
    verdict_text: Option<&str>,
) -> LegacyApprovalGateDecisionV1 {
    let pending_matches = pending_action == Some(current_action);

    if !pending_matches {
        return LegacyApprovalGateDecisionV1::AwaitingFreshApproval {
            invalidate_existing_verdict: verdict_text.is_some(),
        };
    }

    match verdict_text.map(LegacyApprovalVerdictV1::parse) {
        Some(LegacyApprovalVerdictV1::Approved) => {
            LegacyApprovalGateDecisionV1::ApprovedCurrent
        }
        Some(LegacyApprovalVerdictV1::Vetoed) => LegacyApprovalGateDecisionV1::VetoedCurrent,
        Some(LegacyApprovalVerdictV1::Other) => {
            LegacyApprovalGateDecisionV1::AwaitingFreshApproval {
                invalidate_existing_verdict: true,
            }
        }
        None => LegacyApprovalGateDecisionV1::AwaitingFreshApproval {
            invalidate_existing_verdict: false,
        },
    }
}

/// Remove the ambient legacy verdict artifact.
///
/// Absence is already the desired state. Any other removal error is returned so
/// mutation callers can fail closed instead of silently continuing with a stale
/// positive approval still readable on disk.
pub fn invalidate_legacy_verdict_file_v1(path: &Path) -> io::Result<()> {
    match std::fs::remove_file(path) {
        Ok(()) => Ok(()),
        Err(err) if err.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(err) => Err(err),
    }
}

/// Publish one legacy local decision using create-new + write + rename rather
/// than a direct overwrite of the authority-shaped path.
///
/// The caller must surface any returned error and must not claim the decision was
/// published when this function fails. This is still transitional local-file
/// IPC; it does not authenticate the approver and is not equivalent to #5242 or
/// Xenia authority.
pub fn publish_legacy_verdict_file_v1(
    path: &Path,
    verdict: LegacyApprovalVerdictV1,
) -> io::Result<()> {
    let text = verdict.persisted_text()?;
    let temp_path = legacy_temp_path(path)?;

    let result = (|| -> io::Result<()> {
        let mut options = std::fs::OpenOptions::new();
        options.write(true).create_new(true);

        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }

        let mut file = options.open(&temp_path)?;
        file.write_all(text.as_bytes())?;
        file.sync_all()?;
        drop(file);

        // On the NixOS/Unix target this atomically replaces the directory entry
        // itself rather than following an existing destination symlink.
        std::fs::rename(&temp_path, path)?;
        Ok(())
    })();

    if result.is_err() {
        let _ = std::fs::remove_file(&temp_path);
    }

    result
}

fn legacy_temp_path(path: &Path) -> io::Result<PathBuf> {
    let parent = path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "legacy approval path must have a parent directory",
        )
    })?;
    let file_name = path.file_name().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "legacy approval path must have a file name",
        )
    })?;
    let counter = LEGACY_VERDICT_TEMP_COUNTER.fetch_add(1, Ordering::Relaxed);
    let temp_name = format!(
        ".{}.{}.{}.tmp",
        file_name.to_string_lossy(),
        std::process::id(),
        counter
    );
    Ok(parent.join(temp_name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stale_approved_without_pending_action_requires_invalidation() {
        assert_eq!(
            evaluate_legacy_approval_v1(None, "action-b", Some("Approved")),
            LegacyApprovalGateDecisionV1::AwaitingFreshApproval {
                invalidate_existing_verdict: true,
            }
        );
    }

    #[test]
    fn stale_approved_for_different_pending_action_requires_invalidation() {
        assert_eq!(
            evaluate_legacy_approval_v1(Some("action-a"), "action-b", Some("Approved")),
            LegacyApprovalGateDecisionV1::AwaitingFreshApproval {
                invalidate_existing_verdict: true,
            }
        );
    }

    #[test]
    fn exact_matching_approved_is_accepted_by_legacy_guard() {
        assert_eq!(
            evaluate_legacy_approval_v1(Some("action-b"), "action-b", Some("A")),
            LegacyApprovalGateDecisionV1::ApprovedCurrent
        );
    }

    #[test]
    fn exact_matching_veto_is_preserved() {
        assert_eq!(
            evaluate_legacy_approval_v1(Some("action-b"), "action-b", Some("Vetoed")),
            LegacyApprovalGateDecisionV1::VetoedCurrent
        );
    }

    #[test]
    fn unrecognized_matching_verdict_is_invalidated_and_regated() {
        assert_eq!(
            evaluate_legacy_approval_v1(Some("action-b"), "action-b", Some("maybe")),
            LegacyApprovalGateDecisionV1::AwaitingFreshApproval {
                invalidate_existing_verdict: true,
            }
        );
    }

    #[test]
    fn two_cycle_replay_sequence_stays_blocked_after_required_invalidation() {
        let first = evaluate_legacy_approval_v1(Some("action-a"), "action-b", Some("Approved"));
        assert_eq!(
            first,
            LegacyApprovalGateDecisionV1::AwaitingFreshApproval {
                invalidate_existing_verdict: true,
            }
        );

        // The caller invalidates the old artifact before installing action-b as
        // pending. On the next cycle there is no ambient approval to replay.
        let second = evaluate_legacy_approval_v1(Some("action-b"), "action-b", None);
        assert_eq!(
            second,
            LegacyApprovalGateDecisionV1::AwaitingFreshApproval {
                invalidate_existing_verdict: false,
            }
        );
    }

    #[test]
    fn invalidation_removes_existing_file_and_tolerates_absence() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("watchdog_verdict.txt");
        std::fs::write(&path, "Approved").unwrap();

        invalidate_legacy_verdict_file_v1(&path).unwrap();
        assert!(!path.exists());

        invalidate_legacy_verdict_file_v1(&path).unwrap();
        assert!(!path.exists());
    }

    #[test]
    fn publisher_is_fallible_and_writes_exact_known_verdicts() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("watchdog_verdict.txt");

        publish_legacy_verdict_file_v1(&path, LegacyApprovalVerdictV1::Approved).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "Approved");

        publish_legacy_verdict_file_v1(&path, LegacyApprovalVerdictV1::Vetoed).unwrap();
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "Vetoed");

        let err = publish_legacy_verdict_file_v1(&path, LegacyApprovalVerdictV1::Other)
            .unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[cfg(unix)]
    #[test]
    fn publisher_replaces_destination_symlink_instead_of_following_it() {
        use std::os::unix::fs::symlink;

        let dir = tempfile::tempdir().unwrap();
        let victim = dir.path().join("victim.txt");
        let path = dir.path().join("watchdog_verdict.txt");
        std::fs::write(&victim, "do-not-touch").unwrap();
        symlink(&victim, &path).unwrap();

        publish_legacy_verdict_file_v1(&path, LegacyApprovalVerdictV1::Approved).unwrap();

        assert_eq!(std::fs::read_to_string(&victim).unwrap(), "do-not-touch");
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "Approved");
        assert!(!std::fs::symlink_metadata(&path).unwrap().file_type().is_symlink());
    }
}
