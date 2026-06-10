// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//! Phone Screen Embodiment — ADB-based perception-action loop for Symthaea.
//!
//! Implements the `EmbodimentBridge` trait for controlling an Android phone
//! via ADB. The perception side uses the vision manifold (P1-P8) to see
//! the screen; the action side uses ADB shell commands to tap, swipe, type.
//!
//! # Safety Architecture
//!
//! Actions are Phi-gated following the NRC 4-tier model:
//! - Red (Phi < 0.15): All actions blocked
//! - Orange (Phi < 0.35): Screenshot only (read-only)
//! - Yellow (Phi < 0.6): Navigation (back, home)
//! - Green (Phi >= 0.6): Full control (tap, swipe, type)
//!
//! An additional **confirmation mode** requires explicit user approval
//! before executing any mutating action.
//!
//! # Example
//!
//! ```ignore
//! let mut phone = PhoneBridge::new("41201FDJG000UM", 1008, 2244);
//! phone.capture_and_observe(0.033)?;
//! let proposed = phone.propose_action(phi);
//! println!("Proposed: {:?}", proposed);
//! // User approves → phone.execute_action(proposed)?;
//! ```

pub mod actions;
pub mod adb;
pub mod bridge;
pub mod task;

/// scrcpy-server lifecycle, binary framing, HEVC decode.
/// Gated behind the `scrcpy` feature; requires `nix develop` with ffmpeg
/// for the first build (HEVC decoder via `ffmpeg-next`/pkg-config).
#[cfg(feature = "scrcpy")]
pub mod scrcpy;

/// Streaming-capture wrapper around `PhoneBridge` — owns both a
/// `PhoneBridge` and a `ScrcpyCaptureStream`. See module-level doc for
/// the rationale behind keeping this as a separate type rather than
/// embedding the stream into `PhoneBridge` itself.
#[cfg(feature = "scrcpy")]
pub mod streaming_bridge;

pub use actions::PhoneAction;
pub use bridge::PhoneBridge;
pub use task::{StepAction, StepTarget, Task, TaskStep};
