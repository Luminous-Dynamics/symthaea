// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! # Symthaea Soma
//!
//! Mobile-embodied consciousness engine wrapping the Spore kernel.
//!
//! Spore is the pure consciousness kernel (~500KB WASM) — HDC, CfC, IIT, neuromod,
//! harmonies, substrate independence, and epistemic honesty. Soma extends it with
//! phone-body embodiment: sensors, haptics, sleep/wake metabolism, BLE mesh,
//! device pairing, holon desktop sync, and screen vision.
//!
//! ## Architecture
//!
//! ```text
//! SomaEngine wraps SporeEngine + adds:
//!   SensorBridge    (accel/gyro/light → neuromod nudges)
//!   HapticManager   (consciousness-gated vibration events)
//!   Metabolism      (Sleep/Drowsy/Alert/Focused state machine)
//!   BleMesh         (BLE peer discovery & consciousness sharing)
//!   HolonBridge     (desktop ↔ phone sync)
//!   PairingManager  (Ed25519 trust establishment)
//!   ScreenVisionBridge  (screen framebuffer → visual perception)
//!   TouchBody       (touch events → proprioceptive signals)
//! ```
//!
//! ## Usage
//!
//! ```rust
//! use symthaea_soma::{SomaEngine, SomaConfig};
//!
//! let config = SomaConfig::default();
//! let mut engine = SomaEngine::new(config);
//! let result = engine.cycle("hello world");
//! println!("Consciousness: {}", result.consciousness_level);
//! ```

#![cfg_attr(
    not(any(feature = "native-ffi", feature = "litert", feature = "prism-search")),
    deny(unsafe_code)
)]

// Re-export Spore kernel types for downstream consumers
pub use symthaea_spore::broca;
pub use symthaea_spore::config;
pub use symthaea_spore::engine::SporeEngine;
pub use symthaea_spore::engine::{CycleResult, EpistemicStatus};
pub use symthaea_spore::persistence;

// Mobile embodiment modules (moved from spore)
pub mod ble_mesh;
pub mod haptic;
pub mod holon_bridge;
pub mod metabolism;
pub mod sensor_bridge;

#[cfg(feature = "pairing")]
pub mod pairing;

// Screen embodiment (Phase 3)
#[cfg(feature = "screen-vision")]
pub mod screen_vision;

#[cfg(feature = "screen-vision")]
pub mod touch_body;

// Decentralized positioning (GPS-independent)
#[cfg(feature = "positioning")]
pub mod positioning_bridge;

// Full Broca language center (replaces BrocaLite)
#[cfg(feature = "broca-full")]
pub mod broca_soma;

// On-device LLM via LiteRT-LM (Gemma 4 E2B)
#[cfg(feature = "litert")]
pub mod litert_bridge;

// Tool-use framework for LLM function calling
#[cfg(feature = "litert")]
pub mod tool_use;

// Broca + LiteRT fusion engine
#[cfg(feature = "fusion")]
pub mod fusion;

// Native FFI for Android/iOS
#[cfg(feature = "native-ffi")]
pub mod native_ffi;

// SomaEngine — the mobile consciousness engine
pub mod engine;

pub use engine::{SomaConfig, SomaEngine, SomaEngineHandle};
