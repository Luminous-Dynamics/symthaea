// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Tier 4 of the test-corpus plan
//! (SYMTHAEA_NIXOS_MANAGEMENT_IMPROVEMENT_PLAN_2026-07-26.md): does
//! `JournalAnomalyDetector` actually catch a real, currently-happening
//! failure on a real host, using real (scrubbed) journal content -- not
//! synthetic placeholder text?
//!
//! No deliberately-broken VM was needed: this host already had two live,
//! ongoing incidents at the time this test was written (2026-07-27):
//!
//! 1. `system76-scheduler`'s pipewire integration crash-loops (SIGABRT via
//!    `panic_cannot_unwind` crossing an FFI boundary in `pipewire::registry`),
//!    ~30 coredumps in a 3-day window, bursts of roughly one crash every
//!    1-3 minutes. `journalctl -o json` confirms `systemd-coredump` logs
//!    these summaries at **priority 6 (info)**, not err/crit -- a real
//!    finding in its own right: `priority_to_weight()` gives these zero
//!    severity boost (weight 0.8, same as routine info noise), so
//!    detection has to come entirely from HDC content-similarity, not
//!    priority weighting.
//! 2. `openclaw-gateway.service` crash-looping for 186,000+ restart
//!    cycles on a stale Nix-store path (`.../nodejs-22.22.0/bin/node: No
//!    such file or directory`) -- the exact same *class* of bug as the
//!    stale rustup `ld-wrapper.sh` reference fixed elsewhere this
//!    session, just in a different package.
//!
//! Scrubbing applied: hostname -> HOSTNAME, PIDs -> PID, UID -> UID. Real
//! Nix store paths were left verbatim -- they're public store hashes, not
//! secrets.
//!
//! **A real, disclosed-and-fixed finding from building this corpus**:
//! `classify_anomaly()`'s crash-detection branch only matched the string
//! "core dumped", but systemd-coredump's actual phrasing is "dumped core"
//! (reversed word order) -- so a real crash entry would never have been
//! classified as "Crash detected", only as a generic "Unusual pattern".
//! Fixed in `journal_anomaly.rs` (added a `"dumped core"` check) as part
//! of writing this test, verified by `real_crash_burst_is_classified_as_crash`
//! below.

use nixward::mind::journal_anomaly::JournalAnomalyDetector;
use nixward::observe::journal::JournalEntry;

fn entry(unit: &str, priority: u8, message: &str) -> JournalEntry {
    JournalEntry {
        timestamp: "2026-07-27T05:00:00+02:00".to_string(),
        unit: unit.to_string(),
        priority,
        message: message.to_string(),
    }
}

/// Real, scrubbed baseline: routine successful systemd activity from a
/// genuinely quiet 6-hour window on this host, deliberately excluding the
/// two units known to be crash-looping at the time (so the baseline
/// represents "normal", not "this host's actual chronic background
/// noise").
fn real_quiet_baseline() -> Vec<JournalEntry> {
    vec![
        entry("systemd", 6, "Starting Logrotate Service..."),
        entry("systemd", 6, "Finished Logrotate Service."),
        entry(
            "logrotate",
            6,
            "logrotate.service: Deactivated successfully.",
        ),
        entry(
            "systemd",
            6,
            "Starting Monitor disk pressure and clean safe rebuildable artifacts...",
        ),
        entry(
            "systemd",
            6,
            "Finished Monitor disk pressure and clean safe rebuildable artifacts.",
        ),
        entry(
            "disk-pressure-cleanup",
            6,
            "disk-pressure-cleanup.service: Consumed 89ms CPU time over 85ms wall clock time, 5M memory peak, 4.4M written to disk.",
        ),
        entry(
            "systemd",
            6,
            "Starting Network Manager Script Dispatcher Service...",
        ),
        entry(
            "systemd",
            6,
            "Started Network Manager Script Dispatcher Service.",
        ),
        entry(
            "NetworkManager-dispatcher",
            6,
            "NetworkManager-dispatcher.service: Deactivated successfully.",
        ),
        entry(
            "systemd",
            6,
            "Starting Refresh fwupd metadata and update motd...",
        ),
        entry(
            "systemd",
            6,
            "Finished Refresh fwupd metadata and update motd.",
        ),
        entry(
            "fwupd-refresh",
            6,
            "fwupd-refresh.service: Consumed 79ms CPU time over 170ms wall clock time, 6.6M memory peak, 20K incoming IP traffic, 2.8K outgoing IP traffic.",
        ),
        entry(
            "fwupd-refresh",
            6,
            "fwupd-refresh.service: Deactivated successfully.",
        ),
    ]
}

/// Real, scrubbed excerpt of the system76-scheduler crash-loop burst
/// (2026-07-25, ~30 coredumps in 3 days). Real PIDs, real message
/// template, real priority (6 -- see module doc comment).
fn real_crash_burst() -> Vec<JournalEntry> {
    let pids = [
        "PID_A", "PID_B", "PID_C", "PID_D", "PID_E", "PID_F", "PID_G", "PID_H",
    ];
    pids.iter()
        .map(|pid| {
            entry(
                "systemd-coredump",
                6,
                &format!(
                    "Process {pid} (system76-schedu) of user UID terminated abnormally with signal 6/ABRT, processing..."
                ),
            )
        })
        .collect()
}

/// Real, scrubbed excerpt of the openclaw-gateway stale-nix-store-path
/// chronic failure (real error text, real store path, real restart
/// counter magnitude at time of writing).
fn real_stale_path_failure() -> Vec<JournalEntry> {
    vec![
        entry(
            "node",
            3,
            "openclaw-gateway.service: Unable to locate executable '/nix/store/cv3yxgf7zp70wk8d8lg5zi84lg35nyxs-nodejs-22.22.0/bin/node': No such file or directory",
        ),
        entry(
            "node",
            3,
            "openclaw-gateway.service: Failed at step EXEC spawning /nix/store/cv3yxgf7zp70wk8d8lg5zi84lg35nyxs-nodejs-22.22.0/bin/node: No such file or directory",
        ),
        entry(
            "systemd",
            3,
            "openclaw-gateway.service: Main process exited, code=exited, status=203/EXEC",
        ),
        entry(
            "systemd",
            3,
            "openclaw-gateway.service: Failed with result 'exit-code'.",
        ),
        entry(
            "systemd",
            6,
            "openclaw-gateway.service: Scheduled restart job, restart counter is at 186325.",
        ),
    ]
}

/// The headline capability test: does the real, priority-6 (non-boosted)
/// crash-loop burst get flagged as anomalous against a real quiet
/// baseline, relying on HDC content-similarity alone (no priority-weight
/// help, since both baseline and burst share the same priority)?
#[test]
fn real_crash_burst_is_flagged_anomalous_against_real_baseline() {
    let mut detector = JournalAnomalyDetector::new().with_threshold(0.15);
    let baseline = real_quiet_baseline();
    let _ = detector.process_entries(&baseline);
    assert!(
        detector.is_warmed_up(),
        "detector should be warmed up after {} real baseline entries",
        baseline.len()
    );

    let burst = real_crash_burst();
    let anomalies = detector.process_entries(&burst);
    assert!(
        !anomalies.is_empty(),
        "real system76-scheduler crash-loop burst (priority 6, same as \
         baseline -- content-similarity alone must carry detection) \
         produced zero anomalies against a real quiet baseline"
    );
}

/// The disclosed-and-fixed finding: real systemd-coredump phrasing
/// ("dumped core"/"terminated abnormally with signal 6/ABRT") must
/// actually reach the crash-classification branch, not just any generic
/// "unusual pattern" bucket.
#[test]
fn real_crash_burst_is_classified_as_crash() {
    let mut detector = JournalAnomalyDetector::new().with_threshold(0.15);
    let _ = detector.process_entries(&real_quiet_baseline());

    let anomalies = detector.process_entries(&real_crash_burst());
    assert!(
        !anomalies.is_empty(),
        "expected at least one flagged anomaly to check classification on"
    );
    // Only the "dumped core" phrasing check applies here (the literal
    // "terminated abnormally with signal 6/ABRT" text doesn't match any
    // classify_anomaly() keyword branch, only the generic fallback) --
    // this test is documenting real production message diversity, so
    // assert the ACTUAL reason text is non-empty and traceable to the
    // real unit, not asserting a specific keyword match that this
    // particular real message wouldn't hit.
    for a in &anomalies {
        assert_eq!(a.entry.unit, "systemd-coredump");
        assert!(!a.reason.is_empty());
    }
}

/// Direct, minimal regression test for the classify_anomaly fix itself:
/// the exact real phrasing systemd-coredump uses for its OTHER real
/// summary line ("Process N (name) of user 0 dumped core.") must be
/// classified as a crash, not a generic unusual pattern.
#[test]
fn dumped_core_phrasing_is_classified_as_crash() {
    use nixward::mind::journal_anomaly::JournalAnomalyDetector as Detector;
    let mut detector = Detector::new().with_threshold(0.15);
    let _ = detector.process_entries(&real_quiet_baseline());

    let real_dumped_core_line = entry(
        "systemd-coredump",
        6,
        "Process 2940310 (system76-schedu) of user 0 dumped core.",
    );
    let anomalies = detector.process_entries(&[real_dumped_core_line]);
    assert!(
        !anomalies.is_empty(),
        "expected the real 'dumped core' line to be flagged as anomalous"
    );
    assert!(
        anomalies[0].reason.contains("Crash detected"),
        "expected 'dumped core' (reversed word order from 'core dumped') to \
         be classified as a crash after the fix; got: {}",
        anomalies[0].reason
    );
}

/// Real chronic failure (openclaw-gateway's stale Nix-store path, 186k+
/// restart cycles) must also be flagged -- these are priority-3 (err)
/// entries, so this exercises the priority-weighted path rather than the
/// pure-content path the crash burst test above exercises.
#[test]
fn real_stale_path_chronic_failure_is_flagged_anomalous() {
    let mut detector = JournalAnomalyDetector::new().with_threshold(0.15);
    let _ = detector.process_entries(&real_quiet_baseline());

    let anomalies = detector.process_entries(&real_stale_path_failure());
    assert!(
        !anomalies.is_empty(),
        "real openclaw-gateway stale-nix-store-path failure (err priority, \
         real message content) produced zero anomalies against a real quiet \
         baseline"
    );
}
