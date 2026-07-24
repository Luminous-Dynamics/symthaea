// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! In-situ integration test for memetic immune screening
//! (MEMETICS_ANTIMEMETICS_PLAN Phase 2/3): verifies the `MemeticImmuneSystem`
//! actually screens incoming mesh content *inside the running cognitive loop* —
//! not just that the wiring compiles or that the fast-crate logic is unit-tested.
//!
//! Closes the honest caveat carried through Phases 2–3: prior verification was
//! compile + unit level only. This drives a real `CognitiveLoopService`, injects
//! a `SwarmEvent::ContentAnnounced` through the same `swarm_event_tx` channel the
//! mesh receive path uses, and observes the immune telemetry move.

#![cfg(feature = "social-fabric")]

use super::super::managers::swarm_manager::SwarmEvent;
use super::super::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea_core::hdc::BinaryHV;

fn announced(peer: &str, hash_byte: u8, truncated: [u8; 32], created_at: u64) -> SwarmEvent {
    SwarmEvent::ContentAnnounced {
        peer_id: peer.to_string(),
        content_hash: [hash_byte; 32],
        truncated_hdv: truncated,
        domain: "test".to_string(),
        created_at,
    }
}

/// The memetic immune system actually runs on incoming content in the live loop:
/// injecting a peer content announcement makes `seen` advance.
///
/// Verified 2026-07-10: 2 passed in the lib unittest suite (13.15s).
#[test]
fn memetic_immune_screens_incoming_content_live() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let tx = service.swarm_event_sender();

    // Warm up so psi settles (screening runs regardless, but this exercises a
    // realistic state rather than genesis).
    for _ in 0..20 {
        let _ = service.cycle("the garden grows in silence");
    }

    let before = service.memetic_telemetry().seen;

    tx.send(announced("peer-benign", 1, [0x5A; 32], 1_000))
        .expect("swarm_event_tx receiver lives inside the loop");
    // The receive drain + screening happen during a cycle.
    let _ = service.cycle("a memory of light and water");

    let after = service.memetic_telemetry().seen;
    assert!(
        after > before,
        "memetic immune system must screen incoming content in the live loop \
         (seen {before} -> {after}) — it is not actually firing"
    );
}

/// A vaccinated pathogen meme is REJECTED in the live loop (the Phase 1 firewall,
/// observed in-situ). Pathogen rejection is posture-independent, so this is
/// robust regardless of the loop's psi state.
///
/// Verified 2026-07-10 alongside the sibling test.
#[test]
fn memetic_firewall_rejects_vaccinated_pathogen_live() {
    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let tx = service.swarm_event_sender();
    for _ in 0..20 {
        let _ = service.cycle("we are learning to trust each other");
    }

    // The loop reconstructs the incoming embedding via `from_truncated_bytes`,
    // so vaccinate against the *reconstruction* of the exact bytes we will send.
    let pathogen_bytes = [0xC3u8; 32];
    let pathogen_embedding = BinaryHV::from_truncated_bytes(&pathogen_bytes);
    service.vaccinate_meme(pathogen_embedding);

    let before = service.memetic_telemetry().rejected;
    let threats_before = service.threat_memory.pattern_count();

    tx.send(announced("peer-pathogen", 2, pathogen_bytes, 2_000))
        .expect("swarm_event_tx receiver lives inside the loop");
    let _ = service.cycle("input after pathogen");

    let after = service.memetic_telemetry();
    assert!(
        after.rejected > before,
        "a vaccinated pathogen meme must be rejected in the live loop \
         (rejected {before} -> {}); seen={}, immune_memory={}",
        after.rejected,
        after.seen,
        after.immune_memory
    );

    // Bridge A: the rejected pathogen is recorded as a first-class immune
    // memory (as an EpistemicThreat) — ThreatMemory grows.
    assert!(
        service.threat_memory.pattern_count() > threats_before,
        "Bridge A: a rejected pathogen must be stored in ThreatMemory \
         (pattern_count {threats_before} -> {})",
        service.threat_memory.pattern_count()
    );

    // Warded Node Phase 2 (transparency): the rejection is also visible in
    // the guardian-facing audit log, and its meme_id correlates with the
    // content's created_at (2_000, as sent above) — a guardian can see WHAT
    // was blocked, not just that something was.
    let log = service.memetic_filtered_log(10);
    assert_eq!(service.memetic_filtered_log_len(), 1);
    assert_eq!(log.len(), 1, "audit log must contain the rejection");
    assert_eq!(
        log[0].meme_id, 2_000,
        "audit log entry must correlate with the rejected content's id"
    );
}

/// Warded Node design Phase 1 (`WARDED_NODE_DESIGN_2026-07-11.md`): a
/// `WardConfig` with `posture_floor = Red` suppresses ALL uptake in the live
/// loop — even perfectly benign content carrying zero pathogen match — proving
/// the floor overrides whatever posture the loop would otherwise derive (here,
/// Green after a normal warmup). This is the property a guardian relies on: a
/// warded node cannot be talked into a looser posture by its own state.
#[test]
fn ward_config_red_floor_suppresses_benign_uptake_live() {
    use symthaea_memetics::{GuardianPosture, WardConfig};

    // Control: an otherwise-identical unwarded node admits benign content.
    let mut baseline = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let base_tx = baseline.swarm_event_sender();
    for _ in 0..20 {
        let _ = baseline.cycle("we are learning to trust each other");
    }
    base_tx
        .send(announced("peer-benign-control", 3, [0x11; 32], 3_000))
        .expect("swarm_event_tx receiver lives inside the loop");
    let _ = baseline.cycle("control input");
    assert!(
        baseline.memetic_telemetry().accepted > 0,
        "control (no ward floor) should admit benign content, telemetry={:?}",
        baseline.memetic_telemetry()
    );

    // Warded node: identical setup, but a Red posture floor is set.
    let mut warded = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let ward_tx = warded.swarm_event_sender();
    warded.set_ward_config(WardConfig {
        posture_floor: Some(GuardianPosture::Red),
        ..Default::default()
    });
    assert_eq!(
        warded.ward_config().posture_floor,
        Some(GuardianPosture::Red)
    );
    for _ in 0..20 {
        let _ = warded.cycle("we are learning to trust each other");
    }

    ward_tx
        .send(announced("peer-benign-warded", 3, [0x11; 32], 3_000))
        .expect("swarm_event_tx receiver lives inside the loop");
    let _ = warded.cycle("warded input");

    let after = warded.memetic_telemetry();
    assert_eq!(
        after.accepted, 0,
        "a Red posture floor must admit NOTHING, even benign non-pathogen \
         content; telemetry={after:?}"
    );
    assert!(
        after.rejected > 0,
        "the same content the control admitted must be rejected under the \
         Red floor; telemetry={after:?}"
    );
}

/// Warded Node design Phase 3 (Layer B, allowlist-first receive): an
/// `AllowlistOnly` node rejects content from a total stranger — no local
/// trust data at all — even though the content is perfectly benign, and the
/// denial is visible in the guardian-facing audit log (closing the Phase 2/3
/// transparency gap). This is the fail-closed property `AllowlistMode` is
/// for; it holds with or without `mesh-trust` compiled in (a stranger's
/// score is `0.0` either way), so this test only needs `social-fabric`.
#[test]
fn ward_config_allowlist_only_rejects_untrusted_stranger_live() {
    use symthaea_memetics::{AllowlistMode, WardConfig};

    let mut warded = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let tx = warded.swarm_event_sender();
    warded.set_ward_config(WardConfig {
        posture_floor: None,
        allowlist_mode: AllowlistMode::AllowlistOnly { min_trust: 0.5 },
    });
    for _ in 0..20 {
        let _ = warded.cycle("we are learning to trust each other");
    }

    tx.send(announced("peer-stranger", 5, [0x22; 32], 5_000))
        .expect("swarm_event_tx receiver lives inside the loop");
    let _ = warded.cycle("stranger input");

    let after = warded.memetic_telemetry();
    assert_eq!(
        after.accepted, 0,
        "AllowlistOnly must reject a total stranger's benign content; telemetry={after:?}"
    );
    assert_eq!(
        after.seen, 0,
        "the allowlist gate rejects BEFORE screen() runs, so its seen/\
         rejected counters must be untouched; telemetry={after:?}"
    );

    // But the guardian-facing audit log DOES see it.
    let log = warded.memetic_filtered_log(10);
    assert_eq!(warded.memetic_filtered_log_len(), 1);
    assert_eq!(log.len(), 1, "the gate denial must land in the audit log");
    assert_eq!(log[0].meme_id, 5_000);
    assert_eq!(log[0].reason, "allowlist gate: peer trust below threshold");
}

/// Warded Node design Phase 5a (ruleset import): a guardian's `Ruleset`,
/// bulk-imported via `vaccinate_ruleset`, actually protects a running node —
/// content matching one of its entries is rejected exactly like a directly
/// `vaccinate_meme`'d pathogen, proving the bulk path isn't just bookkeeping.
#[test]
fn ward_ruleset_import_protects_the_live_loop() {
    use symthaea_memetics::Ruleset;

    let mut service = CognitiveLoopService::new(CognitiveLoopConfig::default()).unwrap();
    let tx = service.swarm_event_sender();
    for _ in 0..20 {
        let _ = service.cycle("we are learning to trust each other");
    }

    // The loop reconstructs the incoming embedding via from_truncated_bytes,
    // so build the ruleset entry against that same reconstruction.
    let pathogen_bytes = [0x7Fu8; 32];
    let pathogen_embedding = BinaryHV::from_truncated_bytes(&pathogen_bytes);
    let ruleset = Ruleset::new("family-safety-baseline", "2026.07.11", "test-fixture")
        .with_entry(pathogen_embedding, "known-bad pattern imported in bulk");

    let applied = service.vaccinate_ruleset(&ruleset);
    assert_eq!(applied, 1);

    let before = service.memetic_telemetry().rejected;
    tx.send(announced("peer-ruleset-pathogen", 6, pathogen_bytes, 6_000))
        .expect("swarm_event_tx receiver lives inside the loop");
    let _ = service.cycle("input after ruleset import");

    let after = service.memetic_telemetry();
    assert!(
        after.rejected > before,
        "content matching a bulk-imported ruleset entry must be rejected \
         in the live loop; telemetry={after:?}"
    );
}
