// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
//
//! Bridge between the Nix codegen pipeline and the Broca SSM language
//! backend (Phase 2 / M6 of the coding-AI roadmap).
//!
//! The goal: make the `NixChannels` struct produced by
//! `nix_codegen::build_nix_channels` shape-compatible with Broca's
//! `ThoughtChannels` representation so that when M7 (distillation
//! training) lands, the scaffolding for concatenating Nix intent into
//! the 43-channel thought vector is already in place.
//!
//! M6 is intentionally scaffolding-only: no Broca imports, no training,
//! no backend wiring. It proves the flat-vector layout round-trips
//! through the intent classifier and provides the extraction
//! primitives that M7 will consume.
//!
//! Layout (17 channels):
//! - [0..10] — intent one-hot (matches `NixIntent::ALL` order)
//! - [10]    — language id (0=none, 1=rust, 2=python, 3=node, 4=go, 5=haskell)
//! - [11]    — item count (distinct packages/services mentioned)
//! - [12]    — has_extras (plugins / buildInputs / extensions)
//! - [13]    — has_network_spec (port/firewall/listen/address)
//! - [14]    — has_hardware (gpu/nvidia/amd/intel/kernel)
//! - [15]    — has_permission (user/group/sudo)
//! - [16]    — has_wayland (wayland/sway specific)
//!
//! The count of 17 is the `NIX_CHANNEL_COUNT` constant — bump if
//! `NixChannels` grows new fields.

use crate::language::nix_codegen::{build_nix_channels, NixChannels, NixIntent};

/// Number of channels in the flat Nix → Broca intent vector.
/// Matches the byte-layout of `nix_channels_as_slice`.
pub const NIX_CHANNEL_COUNT: usize = 17;

/// Convert a `NixChannels` struct into a flat `f32` array suitable
/// for concatenation with Broca's ThoughtChannels. Channel order is
/// documented on the module.
pub fn nix_channels_as_slice(channels: &NixChannels) -> [f32; NIX_CHANNEL_COUNT] {
    let mut out = [0.0_f32; NIX_CHANNEL_COUNT];
    out[0..10].copy_from_slice(&channels.intent);
    out[10] = channels.language;
    out[11] = channels.item_count;
    out[12] = channels.has_extras;
    out[13] = channels.has_network_spec;
    out[14] = channels.has_hardware;
    out[15] = channels.has_permission;
    out[16] = channels.has_wayland;
    out
}

/// Convenience: run `build_nix_channels` on the prompt and return the
/// flat vector. This is the one entrypoint M7's distillation pipeline
/// will call per (prompt, code) training pair.
pub fn nix_channels_flat(prompt: &str) -> [f32; NIX_CHANNEL_COUNT] {
    nix_channels_as_slice(&build_nix_channels(prompt))
}

/// Inverse: read back which `NixIntent` the first-10 one-hot block
/// encodes. Returns `None` if the block is all zero (shouldn't happen
/// in practice — `build_nix_channels` always sets one bit).
/// Primary use: tests + the `--deep` benchmark mode to confirm the
/// intent signal survives round-trip.
pub fn intent_from_channels(channels: &[f32]) -> Option<NixIntent> {
    if channels.len() < 10 {
        return None;
    }
    let mut best_idx = 0usize;
    let mut best = f32::NEG_INFINITY;
    for (i, &v) in channels[0..10].iter().enumerate() {
        if v > best {
            best = v;
            best_idx = i;
        }
    }
    if best <= 0.0 {
        return None;
    }
    NixIntent::ALL.get(best_idx).copied()
}

/// Same as `intent_from_channels` but takes the fixed-size array —
/// saves a length check when calling with `nix_channels_flat` output.
pub fn intent_from_flat(channels: &[f32; NIX_CHANNEL_COUNT]) -> Option<NixIntent> {
    intent_from_channels(channels)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layout_is_17_channels() {
        assert_eq!(NIX_CHANNEL_COUNT, 17);
        let flat = nix_channels_flat("enable nginx");
        assert_eq!(flat.len(), 17);
    }

    #[test]
    fn service_prompt_encodes_service_intent() {
        let flat = nix_channels_flat("enable nginx web server");
        let intent = intent_from_flat(&flat).expect("service prompt should encode");
        assert_eq!(intent, NixIntent::Service);
    }

    #[test]
    fn dev_shell_prompt_encodes_devshell_intent() {
        let flat = nix_channels_flat("set up a rust dev environment with rust-analyzer");
        assert_eq!(intent_from_flat(&flat), Some(NixIntent::DevShell));
        // Rust prompts should also set language = 1.0
        assert_eq!(flat[10], 1.0, "rust prompt → language channel = 1.0");
    }

    #[test]
    fn hardware_prompt_sets_hardware_flag() {
        let flat = nix_channels_flat("configure nvidia gpu drivers");
        assert_eq!(intent_from_flat(&flat), Some(NixIntent::Hardware));
        assert_eq!(flat[14], 1.0, "hardware prompt → has_hardware channel");
    }

    #[test]
    fn networking_prompt_sets_network_flag() {
        let flat = nix_channels_flat("open firewall ports 80 and 443");
        assert_eq!(intent_from_flat(&flat), Some(NixIntent::Networking));
        assert_eq!(flat[13], 1.0, "networking prompt → has_network_spec");
    }

    #[test]
    fn distinct_intents_produce_distinct_one_hots() {
        let a = nix_channels_flat("enable nginx web server");
        let b = nix_channels_flat("configure nvidia gpu drivers");
        let c = nix_channels_flat("set up a rust dev environment");
        // One-hot bit positions must differ across the three intents.
        let a_idx = a[0..10]
            .iter()
            .position(|&v| v == 1.0)
            .expect("a must be one-hot");
        let b_idx = b[0..10]
            .iter()
            .position(|&v| v == 1.0)
            .expect("b must be one-hot");
        let c_idx = c[0..10]
            .iter()
            .position(|&v| v == 1.0)
            .expect("c must be one-hot");
        assert_ne!(a_idx, b_idx);
        assert_ne!(b_idx, c_idx);
        assert_ne!(a_idx, c_idx);
    }

    #[test]
    fn intent_from_channels_handles_short_input() {
        assert!(intent_from_channels(&[1.0_f32; 5]).is_none());
        assert!(intent_from_channels(&[0.0_f32; 10]).is_none());
    }

    #[test]
    fn intent_from_channels_picks_argmax() {
        // Simulate a soft-output layer: largest value wins, not
        // strictly one-hot. Generic is idx 9 in NixIntent::ALL.
        let mut chans = [0.0_f32; NIX_CHANNEL_COUNT];
        chans[0] = 0.3; // DevShell
        chans[2] = 0.6; // Hardware
        chans[9] = 0.1; // Generic
        let recovered = intent_from_flat(&chans);
        assert_eq!(recovered, Some(NixIntent::ALL[2]));
    }

    #[test]
    fn channel_layout_preserves_context_scalars() {
        // Direct-construct a NixChannels with known values and verify
        // they land at the expected flat-vector slots. Guards against
        // layout drift if NixChannels grows new fields and
        // nix_channels_as_slice is not updated in lockstep.
        let mut chans = NixChannels::default();
        chans.intent[0] = 1.0;
        chans.language = 42.0;
        chans.item_count = 7.0;
        chans.has_extras = 1.0;
        chans.has_network_spec = 1.0;
        chans.has_hardware = 1.0;
        chans.has_permission = 1.0;
        chans.has_wayland = 1.0;
        let flat = nix_channels_as_slice(&chans);
        assert_eq!(flat[10], 42.0);
        assert_eq!(flat[11], 7.0);
        assert_eq!(flat[12], 1.0);
        assert_eq!(flat[13], 1.0);
        assert_eq!(flat[14], 1.0);
        assert_eq!(flat[15], 1.0);
        assert_eq!(flat[16], 1.0);
    }
}
