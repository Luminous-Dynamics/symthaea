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

use crate::language::nix_codegen::{NixChannels, NixIntent, build_nix_channels};

/// Broca's default `NUM_CHANNELS`. Intentionally duplicated rather than
/// imported from symthaea-broca — this main-crate module deliberately
/// avoids a broca dep (the ssm_language/broca_lite mutual-exclusion
/// keeps broca optional). Must stay in lockstep with broca's
/// `encoder.rs::NUM_CHANNELS` (currently 43 for the default build,
/// 47 with `therapeutic` — we target 43).
pub const BROCA_CHANNEL_COUNT: usize = 43;

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

// ─── Broca-aligned channel layout (M7.c channel-alignment fix) ──────────
//
// The 17-channel flat layout above OVERLAPS Broca's native channel
// semantics: positions 0-7 are Broca's general 8-way intent, 8-19 are
// emotional + consciousness + relational, 20-23 V3 context, 24-27 code
// channels, 28-42 epistemic cube. Putting our 10-way Nix intent at
// positions 0-9 bleeds two intents into Broca's emotional block.
// Putting our 7 context scalars at 10-16 collides with emotional tone.
//
// The aligned function below maps our 17-channel Nix signal into
// Broca's existing 43-channel ThoughtChannels without stomping:
//
//   broca[0..8]   ← projected 10-way Nix intent (lossy projection; see
//                   below) one-hot.
//   broca[8..24]  ← zero (emotional + consciousness + relational + V3
//                   context — Broca's existing semantics).
//   broca[24]     ← syntax_complexity ← nix.item_count / 5.0 (clamped)
//   broca[25]     ← type_confidence   ← nix.has_extras
//   broca[26]     ← algorithm_pattern ← nix.language / 6.0 (6 language
//                                        ids; clamped to [0,1])
//   broca[27]     ← error_likelihood  ← has_hardware * 0.5 +
//                                        has_network * 0.25 +
//                                        has_permission * 0.125 +
//                                        has_wayland * 0.0625
//                   (bit-packed 4 flags in [0, 15/16])
//   broca[28..43] ← zero (epistemic cube — Broca's hallucination gate;
//                   we don't perturb it).
//
// The lossy 10→8 intent projection (FlakeTemplate → DevShell,
// Generic → DevShell) is deliberate: FlakeTemplate prompts produce
// dev-shell-ish scaffolding, Generic is the fallback — both can
// plausibly share DevShell's Broca intent slot.

pub const BROCA_INTENT_DIM: usize = 8;

/// Map a `NixIntent` (10-way) into Broca's 8-way intent index.
/// FlakeTemplate and Generic fold into DevShell (index 0).
fn project_intent(intent: NixIntent) -> usize {
    match intent {
        NixIntent::DevShell => 0,
        NixIntent::Service => 1,
        NixIntent::Hardware => 2,
        NixIntent::Desktop => 3,
        NixIntent::User => 4,
        NixIntent::Networking => 5,
        NixIntent::HomeManager => 6,
        NixIntent::Secrets => 7,
        NixIntent::FlakeTemplate => 0, // collapse with DevShell
        NixIntent::Generic => 0,       // default/fallback
    }
}

/// Convert a `NixChannels` struct into a Broca-aligned 43-channel
/// array. This is what the harvester + trainer should use going
/// forward — the old `nix_channels_as_slice` remains for backward
/// compat / tests.
pub fn nix_channels_as_broca(channels: &NixChannels) -> [f32; BROCA_CHANNEL_COUNT] {
    let mut out = [0.0_f32; BROCA_CHANNEL_COUNT];

    // 0..8 — projected Nix intent, one-hot into Broca's 8-way slot.
    // Find which NixIntent index was set (first 1.0 in the 10D block),
    // then project to 0-7.
    if let Some(src_idx) = channels.intent.iter().position(|&v| v > 0.5) {
        if let Some(src_intent) = NixIntent::ALL.get(src_idx).copied() {
            let dst = project_intent(src_intent);
            if dst < BROCA_INTENT_DIM {
                out[dst] = 1.0;
            }
        }
    }

    // 24..28 — Nix context, packed into Broca's code-channel slots.
    let item_norm = (channels.item_count / 5.0).clamp(0.0, 1.0);
    out[24] = item_norm;
    out[25] = channels.has_extras.clamp(0.0, 1.0);
    let lang_norm = (channels.language / 6.0).clamp(0.0, 1.0);
    out[26] = lang_norm;
    // Bit-pack four flags into [0, ~0.94] range.
    let flags = channels.has_hardware * 0.5
        + channels.has_network_spec * 0.25
        + channels.has_permission * 0.125
        + channels.has_wayland * 0.0625;
    out[27] = flags.clamp(0.0, 1.0);

    out
}

/// Convenience: prompt → Broca-aligned 43-channel array. This is what
/// `harvest_nix_distillation` and the generation demo should call
/// instead of `nix_channels_flat`.
pub fn broca_channels_for_nix_prompt(prompt: &str) -> [f32; BROCA_CHANNEL_COUNT] {
    nix_channels_as_broca(&build_nix_channels(prompt))
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

    // ── Broca-aligned layout tests ────────────────────────────────

    #[test]
    fn broca_aligned_has_43_channels() {
        assert_eq!(BROCA_CHANNEL_COUNT, 43);
        let out = broca_channels_for_nix_prompt("enable nginx web server");
        assert_eq!(out.len(), 43);
    }

    #[test]
    fn broca_aligned_intent_one_hot_in_first_8() {
        let out = broca_channels_for_nix_prompt("enable nginx web server");
        // Service projects to Broca index 1.
        assert_eq!(out[1], 1.0);
        // Rest of the 0-7 block must be zero.
        for i in 0..BROCA_INTENT_DIM {
            if i != 1 {
                assert_eq!(out[i], 0.0, "position {} must be 0", i);
            }
        }
    }

    #[test]
    fn broca_aligned_flaketemplate_and_generic_collapse_to_devshell() {
        // Both FlakeTemplate and Generic project to 0 (same slot as
        // DevShell) — lossy but acceptable per the documented
        // projection.
        let flake =
            broca_channels_for_nix_prompt("complete flake template for rust and python project");
        assert_eq!(
            flake[0], 1.0,
            "flake template should land on Broca intent 0"
        );

        let generic = broca_channels_for_nix_prompt("hello world");
        assert_eq!(generic[0], 1.0, "generic should land on Broca intent 0");
    }

    #[test]
    fn broca_aligned_does_not_touch_emotional_block() {
        // Positions 8-23 must stay at 0.0 — those are Broca's
        // emotional/consciousness/relational/V3 channels. Our Nix
        // signal has no business perturbing them.
        let out = broca_channels_for_nix_prompt("configure nvidia gpu drivers");
        for i in 8..24 {
            assert_eq!(out[i], 0.0, "Broca semantic channel {} stomped", i);
        }
    }

    #[test]
    fn broca_aligned_code_channels_encode_nix_context() {
        // Hardware prompt → has_hardware flag → error_likelihood
        // channel carries 0.5 (top bit of the 4-flag pack).
        let hw = broca_channels_for_nix_prompt("configure nvidia gpu drivers");
        assert!(
            hw[27] >= 0.5,
            "hardware prompt should set error_likelihood ≥ 0.5; got {}",
            hw[27]
        );

        // Networking → has_network_spec → error_likelihood carries 0.25.
        let net = broca_channels_for_nix_prompt("open firewall ports 80 and 443");
        assert!(
            (net[27] - 0.25).abs() < 1e-6,
            "networking should set error_likelihood to 0.25; got {}",
            net[27]
        );

        // Rust dev-shell → language=1.0 → algorithm_pattern at
        // position 26 gets 1.0/6.0 ≈ 0.167.
        let rust =
            broca_channels_for_nix_prompt("set up a rust dev environment with rust-analyzer");
        let expected = 1.0 / 6.0;
        assert!(
            (rust[26] - expected).abs() < 1e-3,
            "rust → algorithm_pattern ≈ 0.167; got {}",
            rust[26]
        );
    }

    #[test]
    fn broca_aligned_preserves_epistemic_cube_block() {
        // Positions 28-42 are Broca's Epistemic Cube — the
        // hallucination-prevention gate. Our bridge must NEVER
        // perturb these or we'd defeat the gate's purpose.
        let out = broca_channels_for_nix_prompt("configure postgresql service");
        for i in 28..BROCA_CHANNEL_COUNT {
            assert_eq!(out[i], 0.0, "epistemic cube channel {} perturbed", i);
        }
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
