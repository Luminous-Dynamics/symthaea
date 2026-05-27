// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Radio Dispatcher — Multi-Band Tiered Radio Architecture
//!
//! **Requires**: `feature = "mesh"`
//!
//! Provides bandwidth-aware routing for Symthaea's mesh consciousness network
//! across heterogeneous radio tiers (Wi-Fi/BLE → LoRa/UHF → HF/NVIS).
//!
//! ## Five Capabilities
//!
//! 1. **Radio-Aware Bandwidth Profiles** — Per-tier AIMD parameters (MTU, duty cycle, latency)
//! 2. **Semantic Delta Compression** — XOR + RLE on BinaryHV diffs (2,048B → ~50-100B typical)
//! 3. **Payload Triage** — Urgency × size × tier routing via `PayloadClassifier`
//! 4. **Network Degradation → Safety** — Connectivity loss escalates `SafetyAgent` levels
//! 5. **Spectrum Manager** — SDR-as-sensory-modality `CognitiveSubsystem` (interval 53)
//!
//! ## Scientific Basis
//!
//! - Shannon (1948): Channel capacity theorem constrains information flow per tier
//! - Jacobson (1988): AIMD congestion control adapted for heterogeneous radio links
//! - Clark & Chalmers (1998): Extended mind thesis — radio network as cognitive extension
//! - Friston (2010): Active Inference over electromagnetic spectrum (prediction error = jamming)
//!
//! ## Module Structure
//!
//! - `tier` — Radio tier abstractions, spectrum manager
//! - `transport` — Compression, FEC, routing, encryption
//! - `hardware` — RadioHardware trait, mock/null implementations, regulatory database
//! - `consciousness_routing` — Consciousness-aware mesh routing, store-and-forward
//! - `advanced` — RTL-SDR hardware, CCSDS space packets

mod advanced;
mod consciousness_routing;
mod hardware;
mod tier;
mod transport;

// ── Re-exports from tier ────────────────────────────────────────────────────
pub use tier::{
    CompressionStrategy, NetworkHealth, PayloadClass, PayloadClassifier, RadioTier,
    RegulatoryConstraints, RoutingDecision, SpectrumManager, SpectrumObservation,
    SpectrumTelemetry, TierProfile,
};

// ── Re-exports from transport ───────────────────────────────────────────────
pub use transport::{
    CompressedDelta, DiscoveryBeacon, FecEncoder, MeshEncryption, PeerSession, RouteEntry,
    RouteTable, TierCompressedPayload,
};

// ── Re-exports from hardware ────────────────────────────────────────────────
pub use hardware::{
    BandAllocation, LicenseType, MockRadioHardware, NullRadioHardware, RadioError, RadioHardware,
    RegulatoryDatabase, RegulatoryRegion,
};

// ── Re-exports from consciousness_routing ───────────────────────────────────
pub use consciousness_routing::{
    ConsciousRoutingDecision, ConsciousnessAwareRouter, ConsolidatedPattern, ConsolidatedWisdom,
    OfflineExperience, OfflineExperienceKind, PeerConsciousnessState, StoreAndForward,
    ThreatObservation,
};

// ── Re-exports from advanced ────────────────────────────────────────────────
pub use advanced::{
    CcsdsPacket, DetectedSignal, RtlSdrHardware, SignalClass, apid, bands, light_time_seconds,
    min_sharing_cadence_for_latency,
};

#[cfg(test)]
mod tests;
