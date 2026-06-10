//! Synthetic labeled corpus generator.
//!
//! **Purpose**: exercise the full ingest → encode → cluster → purity pipeline
//! *today*, before a real labeled Evtx corpus is staged. This is a pipeline
//! test, not the Phase 1 kill-criterion test.
//!
//! **The real kill criterion** (purity ≥ 0.50 on a labeled Evtx corpus) still
//! requires real data. See `memory/project_msp_wedge.md`. Do not claim Phase 1
//! has passed just because synthetic fixtures cluster cleanly — they are
//! designed to cluster cleanly. The point of a synthetic test is to catch
//! pipeline bugs that would mask real-data failures, not to validate the
//! thesis.
//!
//! **Classes**: 5 incident categories inspired by the DFIR.training label set,
//! with class-characteristic (provider, event_id, field) distributions that a
//! working role-filler HDC encoder should separate cleanly:
//!
//!   - `benign_login`        — Security 4624, typical workstation sign-in
//!   - `lateral_movement`    — Security 4624 with LogonType=3 + remote host
//!   - `service_restart`     — Service Control Manager 7036
//!   - `ransomware`          — Sysmon 11 (FileCreate) + suspicious extensions
//!   - `network_outage`      — NETLOGON / DNS 1014 + unreachable-host fields

use crate::event::{LogEvent, Severity, Source};
use chrono::{Duration, Utc};
use std::collections::BTreeMap;

/// Deterministic PRNG so fixture runs are reproducible across sessions.
/// Same xorshift64* as the encoder — different seed constant.
fn xorshift(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x.wrapping_mul(0x2545F4914F6CDD1D)
}

/// Generate a labeled synthetic corpus with `n_per_class` events in each of
/// the 5 classes. Events are interleaved (not grouped) so any clusterer that
/// cheats on temporal order gets no advantage.
pub fn generate_synthetic_corpus(n_per_class: usize, seed: u64) -> Vec<LogEvent> {
    generate_noisy_corpus(n_per_class, seed, 0.0)
}

/// Generate a labeled synthetic corpus with class-boundary obfuscation.
///
/// `noise_level` ∈ [0.0, 1.0]:
///   - 0.0 → clean classes (identical to `generate_synthetic_corpus`)
///   - 0.5 → half the events per class have provider/event_id swapped with
///           a randomly chosen other class (class-channel leakage)
///   - 1.0 → every event is scrambled across classes — purity should
///           approach chance (1/n_classes = 0.20 for 5 classes)
///
/// This is the ablation primitive. Real IT-ops data is not as clean as the
/// 0.0 fixtures: the same Event ID 4624 appears in both benign and lateral-
/// movement flows, service restarts overlap with ransomware-triggered
/// service-control activity, etc. The noise sweep measures how much of the
/// encoder's separation comes from *easy* channel differences vs. genuine
/// multi-field composition.
pub fn generate_noisy_corpus(n_per_class: usize, seed: u64, noise_level: f32) -> Vec<LogEvent> {
    assert!((0.0..=1.0).contains(&noise_level));
    let mut state = seed.max(1);
    let classes: &[&dyn Fn(&mut u64, usize) -> LogEvent] = &[
        &gen_benign_login,
        &gen_lateral_movement,
        &gen_service_restart,
        &gen_ransomware,
        &gen_network_outage,
    ];

    let n_classes = classes.len();
    let mut out = Vec::with_capacity(n_per_class * n_classes);
    for i in 0..n_per_class {
        for (cls_idx, cls) in classes.iter().enumerate() {
            let mut ev = cls(&mut state, i);

            // Obfuscation: with probability `noise_level`, swap this event's
            // provider + event_id with a randomly chosen other class's
            // "discriminator" fields. Label stays the same — we are
            // contaminating the features, not the ground truth.
            let roll = (xorshift(&mut state) as f32 / u64::MAX as f32).abs();
            if roll < noise_level {
                let other = (xorshift(&mut state) as usize) % n_classes;
                if other != cls_idx {
                    // Generate a throwaway event of the OTHER class and steal
                    // its discriminator fields.
                    let donor = classes[other](&mut state, i);
                    ev.provider = donor.provider;
                    ev.event_id = donor.event_id;
                    ev.component = donor.component;
                    // Also steal a random subset of the donor's fields to
                    // make the noise more than cosmetic.
                    for (k, v) in donor.fields.into_iter().take(2) {
                        ev.fields.insert(k, v);
                    }
                }
            }
            out.push(ev);
        }
    }
    out
}

fn rand_user(state: &mut u64) -> String {
    let users = ["alice", "bob", "carol", "dave", "eve"];
    users[(xorshift(state) as usize) % users.len()].to_string()
}

fn rand_host(state: &mut u64) -> String {
    let hosts = ["WKSTN-01", "WKSTN-02", "SRV-DC-01", "LAPTOP-07"];
    hosts[(xorshift(state) as usize) % hosts.len()].to_string()
}

fn rand_ip(state: &mut u64) -> String {
    format!(
        "10.0.{}.{}",
        (xorshift(state) as u8),
        (xorshift(state) as u8)
    )
}

fn base_event(provider: &str, event_id: u32, severity: Severity, host: &str) -> LogEvent {
    LogEvent {
        timestamp: Utc::now() - Duration::seconds(event_id as i64),
        source: Source::WindowsEvent,
        severity,
        component: "Security".into(),
        provider: provider.into(),
        event_id,
        message: String::new(),
        fields: BTreeMap::new(),
        host: Some(host.into()),
        label: None,
    }
}

fn gen_benign_login(state: &mut u64, _i: usize) -> LogEvent {
    let host = rand_host(state);
    let mut ev = base_event(
        "Microsoft-Windows-Security-Auditing",
        4624,
        Severity::Info,
        &host,
    );
    ev.component = "Security".into();
    ev.fields.insert("LogonType".into(), "2".into()); // interactive
    ev.fields.insert("TargetUserName".into(), rand_user(state));
    ev.fields.insert("IpAddress".into(), "127.0.0.1".into());
    ev.fields
        .insert("AuthenticationPackageName".into(), "NTLM".into());
    ev.label = Some("benign_login".into());
    ev
}

fn gen_lateral_movement(state: &mut u64, _i: usize) -> LogEvent {
    let host = rand_host(state);
    let mut ev = base_event(
        "Microsoft-Windows-Security-Auditing",
        4624,
        Severity::Warning,
        &host,
    );
    ev.component = "Security".into();
    ev.fields.insert("LogonType".into(), "3".into()); // network
    ev.fields.insert("TargetUserName".into(), rand_user(state));
    ev.fields.insert("IpAddress".into(), rand_ip(state));
    ev.fields
        .insert("AuthenticationPackageName".into(), "Kerberos".into());
    ev.fields
        .insert("ImpersonationLevel".into(), "Impersonation".into());
    ev.label = Some("lateral_movement".into());
    ev
}

fn gen_service_restart(state: &mut u64, _i: usize) -> LogEvent {
    let host = rand_host(state);
    let mut ev = base_event("Service Control Manager", 7036, Severity::Info, &host);
    ev.component = "System".into();
    let services = [
        "Windows Update",
        "Print Spooler",
        "BITS",
        "Windows Defender",
    ];
    ev.fields.insert(
        "ServiceName".into(),
        services[(xorshift(state) as usize) % services.len()].to_string(),
    );
    ev.fields.insert("StateChange".into(), "running".into());
    ev.label = Some("service_restart".into());
    ev
}

fn gen_ransomware(state: &mut u64, _i: usize) -> LogEvent {
    let host = rand_host(state);
    let mut ev = base_event("Microsoft-Windows-Sysmon", 11, Severity::Critical, &host);
    ev.component = "Sysmon".into();
    let exts = [".locked", ".encrypted", ".crypt", ".enc"];
    let ext = exts[(xorshift(state) as usize) % exts.len()];
    ev.fields.insert(
        "TargetFilename".into(),
        format!("C:\\Users\\{}\\Documents\\doc{ext}", rand_user(state)),
    );
    ev.fields
        .insert("Image".into(), "C:\\Windows\\Temp\\evil.exe".into());
    ev.fields
        .insert("ProcessId".into(), format!("{}", xorshift(state) & 0xFFFF));
    ev.label = Some("ransomware".into());
    ev
}

fn gen_network_outage(state: &mut u64, _i: usize) -> LogEvent {
    let host = rand_host(state);
    let mut ev = base_event("NETLOGON", 1014, Severity::Error, &host);
    ev.component = "System".into();
    ev.fields
        .insert("DomainController".into(), "DC-01.corp.local".into());
    ev.fields.insert("ErrorCode".into(), "0x5".into());
    ev.fields
        .insert("FailureReason".into(), "unreachable".into());
    ev.fields.insert("TargetAddress".into(), rand_ip(state));
    ev.label = Some("network_outage".into());
    ev
}

/// Full-event contamination ablation (v2). Stricter than `generate_noisy_corpus`.
///
/// At `noise_level`, each event is replaced with probability `noise_level`
/// by a freshly-generated event of a random *other* class, keeping only the
/// original ground-truth label. This is the honest adversarial test: can the
/// encoder recover class signal when the features and the label have been
/// deliberately decoupled?
///
/// Expected curve:
///   - noise=0.0 → purity 1.000 (clean)
///   - noise=0.5 → ~half the events are fully mislabeled at the feature
///     level; purity should drop noticeably
///   - noise=1.0 → every event's features point at a random other class;
///     purity should approach chance (0.20 for 5 classes)
///
/// If purity stays high under v2, the encoder is somehow finding signal the
/// synthetic data doesn't actually contain, which would mean a bug in the
/// experiment, not a property of the encoder.
pub fn generate_noisy_corpus_v2(n_per_class: usize, seed: u64, noise_level: f32) -> Vec<LogEvent> {
    assert!((0.0..=1.0).contains(&noise_level));
    let mut state = seed.max(1);
    let classes: &[&dyn Fn(&mut u64, usize) -> LogEvent] = &[
        &gen_benign_login,
        &gen_lateral_movement,
        &gen_service_restart,
        &gen_ransomware,
        &gen_network_outage,
    ];
    let class_names = [
        "benign_login",
        "lateral_movement",
        "service_restart",
        "ransomware",
        "network_outage",
    ];
    let n_classes = classes.len();

    let mut out = Vec::with_capacity(n_per_class * n_classes);
    for i in 0..n_per_class {
        for (cls_idx, cls) in classes.iter().enumerate() {
            let mut ev = cls(&mut state, i);
            let roll = (xorshift(&mut state) as f32 / u64::MAX as f32).abs();
            if roll < noise_level {
                // Pick a random OTHER class and generate a full event from it.
                // The ground-truth label sticks with the original class.
                let mut other = (xorshift(&mut state) as usize) % n_classes;
                if other == cls_idx {
                    other = (other + 1) % n_classes;
                }
                let donor = classes[other](&mut state, i);
                ev = LogEvent {
                    timestamp: donor.timestamp,
                    source: donor.source,
                    severity: donor.severity,
                    component: donor.component,
                    provider: donor.provider,
                    event_id: donor.event_id,
                    message: donor.message,
                    fields: donor.fields,
                    host: donor.host,
                    label: Some(class_names[cls_idx].into()),
                };
            }
            out.push(ev);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generates_expected_count() {
        let corpus = generate_synthetic_corpus(10, 42);
        assert_eq!(corpus.len(), 50);
    }

    #[test]
    fn all_events_labeled() {
        let corpus = generate_synthetic_corpus(5, 42);
        for ev in &corpus {
            assert!(ev.label.is_some());
        }
    }

    #[test]
    fn exactly_five_classes() {
        let corpus = generate_synthetic_corpus(10, 42);
        let classes: std::collections::HashSet<_> =
            corpus.iter().map(|e| e.label.as_ref().unwrap()).collect();
        assert_eq!(classes.len(), 5);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = generate_synthetic_corpus(5, 42);
        let b = generate_synthetic_corpus(5, 42);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.label, y.label);
            assert_eq!(x.event_id, y.event_id);
            assert_eq!(x.fields, y.fields);
        }
    }
}
