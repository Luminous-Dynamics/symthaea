// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Mycelix Integration Module
//!
//! Integrates Symthaea's consciousness-based cognition with Mycelix's
//! epistemic infrastructure (UESS, SCEI, MATL).
//!
//! ## Architecture
//!
//! ```text
//! Symthaea (Cognition)           Mycelix (Storage + Trust)
//! ┌─────────────────────┐       ┌─────────────────────────┐
//! │ ConsciousnessGraph  │──────▶│ UESS (Epistemic Store)  │
//! │ Φ (Integration)     │       │ E/N/M Classification    │
//! │ HDC (Semantics)     │       │ Content Addressing      │
//! └─────────────────────┘       └─────────────────────────┘
//!          │                             │
//!          ▼                             ▼
//! ┌─────────────────────┐       ┌─────────────────────────┐
//! │ GIS (Ignorance)     │◀─────▶│ SCEI (Self-Correction)  │
//! │ Uncertainty         │       │ Gap Detection           │
//! │ Dark Spot DHT       │       │ Calibration             │
//! └─────────────────────┘       └─────────────────────────┘
//!          │
//!          ▼
//! ┌─────────────────────────────────────────────────────────┐
//! │              KosmicSong (Unified Identity)              │
//! │  ┌──────────┐  ┌───────────┐  ┌────────────────────┐   │
//! │  │ Φ Layer  │  │  Harmonic │  │  Epistemic Layer   │   │
//! │  │          │  │   Layer   │  │  (GIS + Rashomon)  │   │
//! │  └──────────┘  └───────────┘  └────────────────────┘   │
//! └─────────────────────────────────────────────────────────┘
//! ```
//!
//! ## GIS v4.0: Kosmic Song Architecture
//!
//! The KosmicSong is the unified identity synthesizing:
//! - **Consciousness (Φ)**: Integrated information level
//! - **Eight Harmonies**: Value alignment and epistemic lenses
//! - **Graceful Ignorance**: Epistemic humility and uncertainty
//!
//! ## Key Concepts
//!
//! - **Φ → E-Level Mapping**: Consciousness integration (Φ) determines empirical level
//! - **Workspace → N-Level**: Internal scope determines normative access
//! - **Importance → M-Level**: Priority determines materiality/durability
//! - **Harmonies → H-Level**: Which value dimensions are affected
//!
//! ## Example
//!
//! ```rust,ignore
//! use symthaea::mycelix::{KosmicSong, HarmonicProfile};
//!
//! // Create a KosmicSong from awakening
//! let mut song = KosmicSong::from_awakening(0.42, topology);
//!
//! // Synthesize all layers
//! let coherence = song.synthesize();
//! println!("Coherence: {:.2}", coherence.score);
//!
//! // Respond with full epistemic context
//! let response = song.respond("What should we do about climate change?");
//! println!("Perspectives considered: {}", response.perspectives_considered);
//! ```

pub mod collective_identity;
pub mod epistemic_mesh;
pub mod gis;
#[cfg(feature = "mycelix")]
pub mod governance_verify;
pub mod kosmic_song;
pub mod mapper;
pub mod network;
pub mod types;

pub use gis::GracefulIgnoranceSystem;
#[cfg(feature = "mycelix")]
pub use governance_verify::{headless_test, ScenarioConfig, ScenarioReport};
pub use kosmic_song::{HarmonicProfile, KosmicResponse, KosmicSong, KosmicSongBuilder};
pub use mapper::*;
pub use network::{
    MycelixNetworkBridge, NetworkClient, NetworkKVector, SpectralKComputer, SpectralKResult,
};
pub use types::*;
