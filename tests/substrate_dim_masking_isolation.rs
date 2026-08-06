// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Step 0 properties for the recurrent-dimension masking lever.
//!
//! # What the mechanism under test actually is
//!
//! A **currently-dormant, fixed-suffix recurrent-state lesion applied after a
//! full-width CfC step**. It is deliberately *not* described as adaptive
//! dimensionality, because:
//!
//! - `temporal_network.step()` always runs at full width. The mask reads the
//!   resulting hidden state, zeroes a contiguous tail, and re-injects it. A
//!   fraction below 1.0 therefore costs strictly *more* compute than 1.0.
//!   **No metabolic-efficiency claim can be derived from this mechanism.**
//! - The same trailing dimensions are amputated every time. The surviving
//!   prefix was never trained to absorb what the suffix carried, so this probes
//!   capacity restriction / regularization, not reallocation.
//!
//! Before 2026-07-29 the whole pathway was unreachable on the default path by
//! three independent routes: the shared gate defaulted to false, `SiliconDigital`
//! (the default substrate) yields positive scale pressure so the fraction was
//! exactly 1.0, and speed modulation being off zeroed scale pressure outright.
//! These tests pin the isolation and override that make it reachable, so a later
//! controller comparison compares controllers rather than bundles.
//!
//! Deliberately absent: any prediction-error controller. `EffectiveDimSource`
//! reserves a variant for one, and nothing constructs it.

use symthaea::cognitive_loop::types::EffectiveDimSource;
use symthaea::cognitive_loop::{CognitiveLoopConfig, CognitiveLoopService};
use symthaea_core::hdc::substrate_independence::SubstrateType;

/// Base config with the CfC temporal network present, so the mask site is
/// actually reached. Validation overlay off to keep feasibility comparisons
/// unentangled from these tests.
fn base_config() -> CognitiveLoopConfig {
    let mut config = CognitiveLoopConfig::with_cfc();
    config.enable_validation_overlay = false;
    config
}

fn service(config: CognitiveLoopConfig) -> CognitiveLoopService {
    CognitiveLoopService::new(config).expect("service constructs")
}

/// Run a few cycles and return the last recorded mask event, if any.
fn last_mask_event(
    service: &mut CognitiveLoopService,
    cycles: usize,
) -> Option<symthaea::cognitive_loop::types::RecurrentMaskEvent> {
    let mut last = None;
    for _ in 0..cycles {
        let result = service.cycle("dimension masking isolation probe");
        if let Some(event) = result.metadata.substrate.recurrent_mask.clone() {
            last = Some(event);
        }
    }
    last
}

// ═══════════════════════════════════════════════════════════════════════════
// Isolation: one flag per mechanism
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn default_config_never_masks() {
    let config = base_config();
    assert!(!config.enable_recurrent_dim_masking);
    assert!(!config.enable_spectral_entropy_masking);
    assert!(!config.enable_substrate_encoding_noise);
    assert_eq!(config.effective_dim_fraction_override, None);

    let mut svc = service(config);
    assert_eq!(
        svc.substrate_effective_dim_source(),
        EffectiveDimSource::Disabled,
        "no masking flag set — source must report Disabled"
    );
    assert_eq!(
        svc.substrate_effective_dim_fraction(),
        1.0,
        "default substrate must retain full recurrent width"
    );
    assert!(
        last_mask_event(&mut svc, 3).is_none(),
        "default configuration must never record a mask event"
    );
}

#[test]
fn dim_masking_does_not_enable_encoding_noise() {
    let mut config = base_config();
    config.enable_recurrent_dim_masking = true;
    config.effective_dim_fraction_override = Some(0.5);

    let mut svc = service(config);
    let mut saw_noise = false;
    let mut saw_mask = false;
    for _ in 0..3 {
        let result = svc.cycle("masking without noise");
        if result.metadata.substrate.substrate_encoding_noise != 0.0 {
            saw_noise = true;
        }
        if result.metadata.substrate.recurrent_mask.is_some() {
            saw_mask = true;
        }
    }
    assert!(saw_mask, "recurrent masking flag must produce mask events");
    assert!(
        !saw_noise,
        "enabling dimension masking must NOT enable substrate encoding noise"
    );
}

#[test]
fn encoding_noise_does_not_mask_dimensions() {
    let mut config = base_config();
    config.enable_substrate_encoding_noise = true;
    // An override is present but must stay inert while masking is off.
    config.effective_dim_fraction_override = Some(0.25);

    let mut svc = service(config);
    assert_eq!(
        svc.substrate_effective_dim_source(),
        EffectiveDimSource::Disabled,
        "encoding noise alone must not make a dimension controller active"
    );
    assert!(
        last_mask_event(&mut svc, 3).is_none(),
        "enabling encoding noise must NOT mask recurrent dimensions"
    );
}

#[test]
fn spectral_masking_flag_does_not_trigger_substrate_mask() {
    let mut config = base_config();
    config.enable_spectral_entropy_masking = true;
    config.enable_recurrent_dim_masking = false;
    config.effective_dim_fraction_override = Some(0.5);

    let mut svc = service(config);
    // Any event that appears must come from the spectral controller, never from
    // the substrate-pressure path, which is switched off.
    if let Some(event) = last_mask_event(&mut svc, 3) {
        assert_eq!(
            event.source,
            EffectiveDimSource::SpectralEntropy,
            "only the spectral controller may mask when substrate masking is off"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Override semantics
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn fixed_fraction_one_is_exact_noop() {
    let mut config = base_config();
    config.enable_recurrent_dim_masking = true;
    config.effective_dim_fraction_override = Some(1.0);

    let mut svc = service(config);
    let event = last_mask_event(&mut svc, 2).expect("mask site reached and recorded");
    assert_eq!(event.applied_fraction, 1.0);
    assert!(
        !event.executed,
        "fraction 1.0 must be an exact no-op — nothing read, zeroed, or injected"
    );
    assert_eq!(event.dims_zeroed, 0);
    assert_eq!(
        event.mask_overhead_us, 0,
        "the no-op path must not even start the overhead timer"
    );
}

#[test]
fn fraction_half_zeros_intended_suffix() {
    let mut config = base_config();
    config.enable_recurrent_dim_masking = true;
    config.effective_dim_fraction_override = Some(0.5);

    let mut svc = service(config);
    let event = last_mask_event(&mut svc, 2).expect("mask event recorded");
    assert!(event.executed, "fraction 0.5 must actually mask");
    assert!(event.dims_total > 0, "recurrent state must be non-empty");

    let expected_start = (0.5 * event.dims_total as f32) as usize;
    assert_eq!(
        event.dims_zeroed,
        event.dims_total - expected_start,
        "exactly the trailing (1 - frac) of dimensions must be zeroed"
    );
    assert!(
        event.post_mask_norm <= event.pre_mask_norm,
        "zeroing a suffix cannot increase the state norm ({} -> {})",
        event.pre_mask_norm,
        event.post_mask_norm
    );
}

#[test]
fn full_lesion_fraction_zero_is_reachable() {
    // 0.0 is permitted through the override (unlike the substrate path, which
    // floors at SUBSTRATE_MIN_DIM_FRACTION) so a full-lesion negative control
    // exists for the later ladder.
    let mut config = base_config();
    config.enable_recurrent_dim_masking = true;
    config.effective_dim_fraction_override = Some(0.0);

    let mut svc = service(config);
    assert_eq!(svc.substrate_effective_dim_fraction(), 0.0);
    let event = last_mask_event(&mut svc, 2).expect("mask event recorded");
    assert!(event.executed);
    assert_eq!(
        event.dims_zeroed, event.dims_total,
        "fraction 0.0 must zero the entire recurrent state"
    );
    assert_eq!(event.post_mask_norm, 0.0);
}

#[test]
fn out_of_range_override_is_clamped_not_accepted() {
    for (requested, expected) in [(1.7_f32, 1.0_f32), (-0.4, 0.0)] {
        let mut config = base_config();
        config.enable_recurrent_dim_masking = true;
        config.effective_dim_fraction_override = Some(requested);
        let svc = service(config);
        assert_eq!(
            svc.substrate_effective_dim_fraction(),
            expected,
            "override {requested} must clamp to {expected}, not index out of bounds"
        );
    }
}

#[test]
fn non_finite_override_is_ignored() {
    let mut config = base_config();
    config.enable_recurrent_dim_masking = true;
    config.effective_dim_fraction_override = Some(f32::NAN);
    let svc = service(config);
    assert_eq!(
        svc.substrate_effective_dim_fraction(),
        1.0,
        "a non-finite override must be dropped, falling back to substrate pressure"
    );
}

#[test]
fn scale_pressure_cannot_override_fixed_fraction() {
    // Quantum has strongly negative scale pressure; silicon positive. Neither
    // may displace an explicit override, in either speed-modulation mode.
    for substrate in [
        SubstrateType::QuantumComputer,
        SubstrateType::SiliconDigital,
        SubstrateType::BiologicalNeurons,
    ] {
        for speed_modulation in [false, true] {
            let mut config = base_config();
            config.substrate_type = substrate;
            config.enable_substrate_speed_modulation = speed_modulation;
            config.enable_recurrent_dim_masking = true;
            config.effective_dim_fraction_override = Some(0.25);

            let svc = service(config);
            assert_eq!(
                svc.substrate_effective_dim_fraction(),
                0.25,
                "{substrate:?} (speed_modulation={speed_modulation}) must not \
                 displace an explicit fraction override"
            );
            assert_eq!(
                svc.substrate_effective_dim_source(),
                EffectiveDimSource::FixedOverride,
                "provenance must name the override as the source"
            );
        }
    }
}

#[test]
fn substrate_pressure_is_the_source_when_no_override() {
    let mut config = base_config();
    config.enable_recurrent_dim_masking = true;
    config.enable_substrate_speed_modulation = true;
    config.substrate_type = SubstrateType::QuantumComputer;

    let svc = service(config);
    assert_eq!(
        svc.substrate_effective_dim_source(),
        EffectiveDimSource::SubstratePressure
    );
    assert!(
        svc.substrate_scale_pressure() < 0.0,
        "quantum is expected to be scale-constrained — precondition for this test"
    );
    let frac = svc.substrate_effective_dim_fraction();
    assert!(
        (0.1..1.0).contains(&frac),
        "scale-constrained substrate must yield a floored sub-unit fraction, got {frac}"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Determinism and cost honesty
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn mask_geometry_is_deterministic() {
    // Asserts the geometry the mask mechanism itself controls. State *norms* are
    // deliberately NOT asserted across runs: loop-level determinism is not
    // established on this host (consciousness_level was demonstrated to be
    // wall-clock-time dependent, 2026-07-28), so a norm equality here would be
    // testing something this patch does not control.
    let make = || {
        let mut config = base_config();
        config.enable_recurrent_dim_masking = true;
        config.effective_dim_fraction_override = Some(0.375);
        service(config)
    };

    let a = last_mask_event(&mut make(), 2).expect("run A recorded a mask");
    let b = last_mask_event(&mut make(), 2).expect("run B recorded a mask");

    assert_eq!(a.applied_fraction, b.applied_fraction);
    assert_eq!(a.source, b.source);
    assert_eq!(a.dims_total, b.dims_total);
    assert_eq!(a.dims_zeroed, b.dims_zeroed);
    assert_eq!(a.executed, b.executed);
}

#[test]
fn masking_claims_no_computational_discount() {
    // The full-width step is timed on both paths and recorded on the event. A
    // fraction below 1.0 cannot shorten it — the mask runs afterwards, on the
    // step's output. This asserts the structural fact (the step ran and was
    // measured in both configurations) and prints the measured durations for
    // the record. It deliberately does not assert a timing *inequality*: this
    // is a shared host under heavy concurrent load, where such a comparison
    // would be noise, not evidence.
    let run = |frac: f32| -> (u64, u64, bool) {
        let mut config = base_config();
        config.enable_recurrent_dim_masking = true;
        config.effective_dim_fraction_override = Some(frac);
        let event = last_mask_event(&mut service(config), 3).expect("mask event recorded");
        (
            event.step_duration_us,
            event.mask_overhead_us,
            event.executed,
        )
    };

    let (full_step, full_overhead, full_executed) = run(1.0);
    let (masked_step, masked_overhead, masked_executed) = run(0.25);

    println!(
        "frac=1.00: step={full_step}us overhead={full_overhead}us executed={full_executed}\n\
         frac=0.25: step={masked_step}us overhead={masked_overhead}us executed={masked_executed}"
    );

    assert!(
        full_step > 0 && masked_step > 0,
        "the full-width step must be timed and non-trivial on both paths \
         (full={full_step}us masked={masked_step}us)"
    );
    assert!(!full_executed, "frac 1.0 is a no-op");
    assert!(masked_executed, "frac 0.25 must mask");
    assert_eq!(
        full_overhead, 0,
        "the no-op path must add no masking overhead"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// Characterization: is the amputated suffix special?
// ═══════════════════════════════════════════════════════════════════════════

#[test]
fn characterize_suffix_energy_share() {
    // The mask always removes the SAME trailing dimensions. If recurrent state
    // energy is unevenly distributed across the dimension ordering, the lesion's
    // severity is confounded with that ordering rather than with the fraction.
    // This records the share for several fractions rather than asserting a
    // threshold — it is a characterization, not a gate.
    let mut rows = Vec::new();
    for frac in [0.75_f32, 0.5, 0.25] {
        let mut config = base_config();
        config.enable_recurrent_dim_masking = true;
        config.effective_dim_fraction_override = Some(frac);
        let event = last_mask_event(&mut service(config), 3).expect("mask event recorded");

        // Suffix energy share = 1 - (post/pre)^2, since norms are L2.
        let share = if event.pre_mask_norm > 0.0 {
            1.0 - (event.post_mask_norm / event.pre_mask_norm).powi(2)
        } else {
            0.0
        };
        let dim_share = event.dims_zeroed as f32 / event.dims_total.max(1) as f32;
        println!(
            "frac={frac:.2}: dims {}/{} removed ({dim_share:.3}), energy share removed {share:.4}",
            event.dims_zeroed, event.dims_total
        );
        assert!(
            share.is_finite() && (-1e-4..=1.0 + 1e-4).contains(&share),
            "suffix energy share must be a finite proportion, got {share}"
        );
        rows.push((dim_share, share));
    }

    // Report the deviation from a flat-energy expectation. A large systematic
    // gap between dimension share and energy share means the ordering matters
    // and any later fraction sweep must control for it.
    for (dim_share, energy_share) in rows {
        println!(
            "  ordering skew: energy_share - dim_share = {:+.4}",
            energy_share - dim_share
        );
    }
}
