// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Visual grammar primitives — the locked semantic language of the projection system.
//!
//! ## Immutable Grammar Rules
//!
//! These mappings are LOCKED. Do not add new semantic meanings without updating
//! the visual grammar reference document.
//!
//! ### Color Roles
//! - `PhysicalSignal` → blue/cyan (pressure, flow, water)
//! - `Chronicle` → amber/gold (durable civic truth, witness)
//! - `Ecology` → green/organic (growth, mycelium, living signal)
//! - `Memory` → violet/purple (replay, hidden structure)
//! - `Danger` → red/orange (heat, damage, instability)
//! - `MachineTruth` → white/clean (diagnostic truth)
//! - `FalseGreen` → sterile over-clean white/green (Null, suspicious)
//! - `ArchiveDamage` → grey/static (missing evidence)
//!
//! ### Line Styles
//! - `Crisp` → verified signal
//! - `Dashed` → inferred signal
//! - `Broken` → missing data
//! - `Trembling` → high variance / unstable
//! - `TooSmooth` → suspicious artificial consistency (Null masking)
//! - `Braided` → multi-source agreement
//! - `Diverging` → contradiction
//!
//! ### Opacity States
//! - High → current or high-confidence
//! - Low → past, uncertain, weak evidence
//! - Flickering → unstable sensor or archive damage
//! - FadingOut → decaying relevance
//! - HardDisappearance → data loss
//!
//! ### Motion Types (slow, readable, state-driven — no decorative pulsing)
//! - `Ripple` → perturbation
//! - `Contraction` → collapse
//! - `Expansion` → growing uncertainty
//! - `Spiral` → recursive/recurrent process
//! - `Drift` → slow bias accumulation
//! - `Snap` → discrete authority change
//! - `Bloom` → ecological response
//! - `Fracture` → MIP cut or trust break

use serde::{Deserialize, Serialize};

/// Color role — semantic mapping from state to visual hue.
///
/// Do NOT use these as decoration. Each role must correspond to a real semantic state.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ColorRole {
    /// Physical signal, water, pressure, flow. → blue/cyan
    PhysicalSignal,
    /// Chronicle, civic truth, witness, durable event. → amber/gold
    Chronicle,
    /// Ecology, growth, mycelium, living signal. → green/organic
    Ecology,
    /// Memory, dream, replay, hidden structure. → violet/purple
    Memory,
    /// Danger, heat, damage, unstable pressure. → red/orange
    Danger,
    /// Machine diagnostic truth. → white/clean
    MachineTruth,
    /// Suspicious false-green / Null masking. → sterile over-clean
    FalseGreen,
    /// Archive damage or missing evidence. → grey/static
    ArchiveDamage,
    /// Unknown or not yet classified.
    Unknown,
}

impl ColorRole {
    /// Canonical hex color for this role (for direct use in renderers).
    pub fn hex_color(&self) -> &'static str {
        match self {
            ColorRole::PhysicalSignal => "#38BDF8", // sky-400
            ColorRole::Chronicle => "#F59E0B",      // amber-500
            ColorRole::Ecology => "#34D399",        // emerald-400
            ColorRole::Memory => "#A78BFA",         // violet-400
            ColorRole::Danger => "#F87171",         // red-400
            ColorRole::MachineTruth => "#F8FAFC",   // slate-50
            ColorRole::FalseGreen => "#ECFDF5",     // over-bright green-50 (suspicious)
            ColorRole::ArchiveDamage => "#6B7280",  // grey-500
            ColorRole::Unknown => "#94A3B8",        // slate-400
        }
    }

    /// RGBA tuple (0.0–1.0) for use in Bevy rendering.
    pub fn rgba(&self) -> (f32, f32, f32, f32) {
        match self {
            ColorRole::PhysicalSignal => (0.22, 0.74, 0.97, 1.0),
            ColorRole::Chronicle => (0.96, 0.62, 0.04, 1.0),
            ColorRole::Ecology => (0.20, 0.83, 0.60, 1.0),
            ColorRole::Memory => (0.65, 0.55, 0.98, 1.0),
            ColorRole::Danger => (0.97, 0.44, 0.44, 1.0),
            ColorRole::MachineTruth => (0.97, 0.98, 0.99, 1.0),
            ColorRole::FalseGreen => (0.93, 0.99, 0.96, 0.9),
            ColorRole::ArchiveDamage => (0.42, 0.45, 0.50, 1.0),
            ColorRole::Unknown => (0.58, 0.64, 0.72, 1.0),
        }
    }
}

/// Line style — semantic mapping from evidence quality to line rendering.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LineStyle {
    /// Verified signal. Solid, crisp.
    Crisp,
    /// Inferred signal. Dashed.
    Dashed,
    /// Missing data. Broken.
    Broken,
    /// High variance / unstable. Trembling / noisy.
    Trembling,
    /// Suspicious artificial consistency. Too-smooth — may indicate Null masking.
    TooSmooth,
    /// Multi-source agreement. Braided / thick.
    Braided,
    /// Contradiction. Diverging / split.
    Diverging,
}

impl LineStyle {
    /// Whether this line style indicates an anomaly that should be flagged.
    pub fn is_anomaly(&self) -> bool {
        matches!(
            self,
            LineStyle::Broken | LineStyle::Trembling | LineStyle::TooSmooth | LineStyle::Diverging
        )
    }

    /// Stroke width multiplier for this style.
    pub fn stroke_multiplier(&self) -> f32 {
        match self {
            LineStyle::Crisp => 1.0,
            LineStyle::Dashed => 0.8,
            LineStyle::Broken => 0.5,
            LineStyle::Trembling => 0.7,
            LineStyle::TooSmooth => 1.2, // slightly thicker to make it visible
            LineStyle::Braided => 2.0,
            LineStyle::Diverging => 0.6,
        }
    }
}

/// Opacity state — semantic mapping from confidence to transparency.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum OpacityState {
    /// Current or high-confidence. Full opacity.
    HighConfidence,
    /// Past or uncertain. Partially transparent.
    LowConfidence,
    /// Unstable sensor or archive damage. Flickering.
    Flickering { frequency_hz: f32 },
    /// Decaying relevance. Fading toward zero.
    FadingOut { rate: f32 },
    /// Data loss. Hard disappearance.
    HardDisappearance,
    /// Explicit opacity value [0.0, 1.0].
    Explicit(f32),
}

impl OpacityState {
    /// Base opacity value for this state.
    pub fn base_opacity(&self) -> f32 {
        match self {
            OpacityState::HighConfidence => 1.0,
            OpacityState::LowConfidence => 0.4,
            OpacityState::Flickering { .. } => 0.7,
            OpacityState::FadingOut { .. } => 0.5,
            OpacityState::HardDisappearance => 0.0,
            OpacityState::Explicit(v) => *v,
        }
    }
}

/// What depth axis means in this projection mode.
///
/// Depth MUST always mean one of these three things. Never use depth as decoration.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DepthMeaning {
    /// Depth = time. Front = present. Rear = past. (Time-Waterfall)
    Time,
    /// Depth = abstraction layer. Low = physical. High = civic/metacognitive. (Stratified Stack)
    AbstractionLayer,
    /// Depth = evidence chain depth. Front = primary. Rear = source. (Cross-Section)
    EvidenceChain,
}

/// Allowed motion types. Motion must be slow, readable, and state-driven.
///
/// Decorative pulsing is forbidden.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MotionType {
    /// Perturbation signal entering the system.
    Ripple,
    /// Collapse — system contracting.
    Contraction,
    /// Growing uncertainty or graph complexity.
    Expansion,
    /// Recursive or recurrent process.
    Spiral,
    /// Slow bias accumulation.
    Drift,
    /// Discrete authority/transaction change.
    Snap,
    /// Ecological response (living system signal).
    Bloom,
    /// MIP cut or trust break.
    Fracture,
    /// No motion — static frame.
    None,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_color_roles_have_hex() {
        let roles = [
            ColorRole::PhysicalSignal,
            ColorRole::Chronicle,
            ColorRole::Ecology,
            ColorRole::Memory,
            ColorRole::Danger,
            ColorRole::MachineTruth,
            ColorRole::FalseGreen,
            ColorRole::ArchiveDamage,
            ColorRole::Unknown,
        ];
        for role in roles {
            let hex = role.hex_color();
            assert!(hex.starts_with('#'), "hex should start with #: {hex}");
            assert_eq!(hex.len(), 7, "hex should be 7 chars: {hex}");
        }
    }

    #[test]
    fn anomaly_lines_flagged() {
        assert!(LineStyle::Broken.is_anomaly());
        assert!(LineStyle::TooSmooth.is_anomaly());
        assert!(!LineStyle::Crisp.is_anomaly());
        assert!(!LineStyle::Braided.is_anomaly());
    }

    #[test]
    fn opacity_ordering() {
        assert_eq!(OpacityState::HighConfidence.base_opacity(), 1.0);
        assert!(OpacityState::LowConfidence.base_opacity() < 0.5);
        assert_eq!(OpacityState::HardDisappearance.base_opacity(), 0.0);
    }
}
