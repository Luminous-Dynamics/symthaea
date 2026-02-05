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
//! - **Seven Harmonies**: Value alignment and epistemic lenses
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

pub mod types;
pub mod mapper;
pub mod gis;
pub mod kosmic_song;
pub mod network;

pub use types::*;
pub use mapper::*;
pub use gis::GracefulIgnoranceSystem;
pub use kosmic_song::{KosmicSong, KosmicSongBuilder, HarmonicProfile, KosmicResponse};
pub use network::{NetworkClient, NetworkKVector, SpectralKResult, SpectralKComputer, MycelixNetworkBridge};
