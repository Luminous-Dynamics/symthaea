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
}
