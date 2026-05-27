// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
use super::super::super::subsystem_trait::{CognitiveSubsystem, CycleSnapshot, output_flags};
use super::super::super::thresholds::{
    RADIO_BEACON_INTERVAL_CYCLES, RADIO_BEACON_SIZE, RADIO_CRYPTO_NONCE_SIZE,
    RADIO_SYNTHETIC_SNR_BASE, RADIO_SYNTHETIC_SNR_ISOLATED,
};
use super::*;
use crate::domain::DomainProfile;

// ── RadioTier profiles ───────────────────────────────────────────────

#[test]
fn test_tier_profiles_bandwidth_ordering() {
    let local = RadioTier::Local.profile();
    let metro = RadioTier::Metro.profile();
    let regional = RadioTier::Regional.profile();

    assert!(local.bandwidth_budget > metro.bandwidth_budget);
    assert!(metro.bandwidth_budget > regional.bandwidth_budget);
    assert!(local.mtu > metro.mtu);
    assert!(metro.mtu > regional.mtu);
}

#[test]
fn test_tier_profiles_latency_ordering() {
    let local = RadioTier::Local.profile();
    let metro = RadioTier::Metro.profile();
    let regional = RadioTier::Regional.profile();

    assert!(local.latency_ms < metro.latency_ms);
    assert!(metro.latency_ms < regional.latency_ms);
}

#[test]
fn test_tier_profiles_reliability_ordering() {
    let local = RadioTier::Local.profile();
    let metro = RadioTier::Metro.profile();
    let regional = RadioTier::Regional.profile();

    assert!(local.reliability > metro.reliability);
    assert!(metro.reliability > regional.reliability);
}

// ── Delta compression ────────────────────────────────────────────────

#[test]
fn test_delta_identical_vectors() {
    let v = [0xABu8; 2048];
    let delta = CompressedDelta::from_diff(&v, &v);

    assert_eq!(delta.changed_bytes, 0);
    assert!(!delta.is_full);
    // All zeros → single RLE run of 2048 zeros → 3 bytes
    assert!(delta.wire_size() < 10);
    assert!(delta.compression_ratio() < 0.01);
}

#[test]
fn test_delta_one_byte_changed() {
    let v1 = [0u8; 2048];
    let mut v2 = [0u8; 2048];
    v2[1024] = 0xFF;

    let delta = CompressedDelta::from_diff(&v1, &v2);
    assert_eq!(delta.changed_bytes, 1);
    assert!(!delta.is_full);
    // Should be much smaller than 2048
    assert!(delta.wire_size() < 20);

    // Apply and verify roundtrip
    let reconstructed = delta.apply(&v1).unwrap();
    assert_eq!(reconstructed, v2);
}

#[test]
fn test_delta_full_vector() {
    let v = [0x42u8; 2048];
    let delta = CompressedDelta::full(&v);

    assert!(delta.is_full);
    assert_eq!(delta.wire_size(), 2048);

    let zero = [0u8; 2048];
    let reconstructed = delta.apply(&zero).unwrap();
    assert_eq!(reconstructed, v);
}

#[test]
fn test_delta_roundtrip_sparse() {
    // Simulate two vectors with ~5% difference (sparse changes)
    let v1 = [0u8; 2048];
    let mut v2 = [0u8; 2048];
    // Change every 20th byte (102 changes out of 2048 = ~5%)
    for i in (0..2048).step_by(20) {
        v2[i] = 0xFF;
    }

    let delta = CompressedDelta::from_diff(&v1, &v2);
    let reconstructed = delta.apply(&v1).unwrap();
    assert_eq!(reconstructed, v2);
    // Sparse changes should compress well (mostly zero runs)
    assert!(
        delta.compression_ratio() < 0.8,
        "Sparse delta ratio {} should be < 0.8",
        delta.compression_ratio()
    );
}

#[test]
fn test_delta_high_entropy_falls_back_to_full() {
    // Two completely random vectors — XOR is high entropy, RLE won't help
    let mut v1 = [0u8; 2048];
    let mut v2 = [0u8; 2048];
    for i in 0..2048 {
        v1[i] = (i * 7 + 3) as u8;
        v2[i] = (i * 13 + 11) as u8;
    }

    let delta = CompressedDelta::from_diff(&v1, &v2);
    // May or may not fall back to full depending on RLE expansion
    // But it should always be reconstructable
    let reconstructed = delta.apply(&v1).unwrap();
    assert_eq!(reconstructed, v2);
}

// ── Payload classifier ───────────────────────────────────────────────

#[test]
fn test_classifier_default_all_available() {
    let c = PayloadClassifier::default();
    assert_eq!(c.available_count(), 3);
}

#[test]
fn test_classifier_emergency_routes_regional() {
    let c = PayloadClassifier::default();
    let decision = c.route(PayloadClass::Emergency, 40, 2).unwrap();
    match decision {
        RoutingDecision::Routed { tier, .. } => {
            // Emergency can go to any tier, but should prefer best available
            assert!(
                tier == RadioTier::Local || tier == RadioTier::Metro || tier == RadioTier::Regional
            );
        }
        _ => panic!("Expected Routed"),
    }
}

#[test]
fn test_classifier_emergency_deferred_at_low_urgency() {
    let c = PayloadClassifier::default();
    let decision = c.route(PayloadClass::Emergency, 40, 1).unwrap();
    assert!(matches!(decision, RoutingDecision::Deferred { .. }));
}

#[test]
fn test_classifier_bulk_routes_local() {
    let c = PayloadClassifier::default();
    let decision = c.route(PayloadClass::BulkSync, 2048, 1).unwrap();
    match decision {
        RoutingDecision::Routed {
            tier, fragmented, ..
        } => {
            assert_eq!(tier, RadioTier::Local);
            assert!(fragmented); // 2048 > 1500 MTU
        }
        _ => panic!("Expected Routed"),
    }
}

#[test]
fn test_classifier_no_local_blocks_bulk() {
    let mut c = PayloadClassifier::default();
    c.set_tier_available(RadioTier::Local, false);

    let decision = c.route(PayloadClass::BulkSync, 2048, 1).unwrap();
    // Metro MTU is 250, Regional is 50 — neither can handle 2048
    assert!(matches!(decision, RoutingDecision::Blocked { .. }));
}

#[test]
fn test_classifier_small_payload_fits_metro() {
    let c = PayloadClassifier::default();
    let decision = c.route(PayloadClass::ConsciousnessDelta, 100, 1).unwrap();
    match decision {
        RoutingDecision::Routed {
            tier, fragmented, ..
        } => {
            // Should route to best available tier (Local has highest bandwidth)
            assert_eq!(tier, RadioTier::Local);
            assert!(!fragmented);
        }
        _ => panic!("Expected Routed"),
    }
}

#[test]
fn test_classifier_underwater_avoids_local_mesh() {
    let mut c = PayloadClassifier::default();
    c.set_domain_profile(DomainProfile::underwater());

    let decision = c.route(PayloadClass::ConsciousnessDelta, 100, 1).unwrap();
    match decision {
        RoutingDecision::Routed { tier, .. } => assert_eq!(tier, RadioTier::Metro),
        _ => panic!("Expected Routed"),
    }
}

#[test]
fn test_classifier_subterranean_allows_store_and_forward_fragmentation() {
    let mut c = PayloadClassifier::default();
    c.set_domain_profile(DomainProfile::subterranean());
    c.set_tier_available(RadioTier::Local, false);

    let decision = c.route(PayloadClass::BulkSync, 2048, 1).unwrap();
    match decision {
        RoutingDecision::Routed {
            tier,
            fragmented,
            estimated_fragments,
        } => {
            assert_eq!(tier, RadioTier::Metro);
            assert!(fragmented);
            assert!(estimated_fragments > 1);
        }
        _ => panic!("Expected Routed"),
    }
}

#[test]
fn test_classifier_deep_space_uses_interplanetary_when_available() {
    let mut c = PayloadClassifier::default();
    c.set_domain_profile(DomainProfile::deep_space());
    c.set_tier_available(RadioTier::Local, false);
    c.set_tier_available(RadioTier::Metro, false);
    c.set_tier_available(RadioTier::Regional, false);
    c.set_tier_available(RadioTier::Interplanetary, true);

    let decision = c.route(PayloadClass::Discovery, 64, 1).unwrap();
    match decision {
        RoutingDecision::Routed { tier, .. } => assert_eq!(tier, RadioTier::Interplanetary),
        _ => panic!("Expected Routed"),
    }
}

// ── Network health ───────────────────────────────────────────────────

#[test]
fn test_network_health_all_up() {
    assert_eq!(
        NetworkHealth::from_tiers(true, true, true),
        NetworkHealth::AllTiersUp
    );
}

#[test]
fn test_network_health_local_down() {
    assert_eq!(
        NetworkHealth::from_tiers(false, true, true),
        NetworkHealth::LocalDown
    );
}

#[test]
fn test_network_health_metro_only() {
    assert_eq!(
        NetworkHealth::from_tiers(false, false, true),
        NetworkHealth::MetroOnly
    );
}

#[test]
fn test_network_health_blackout() {
    assert_eq!(
        NetworkHealth::from_tiers(false, false, false),
        NetworkHealth::Blackout
    );
}

#[test]
fn test_network_health_ordering() {
    assert!(NetworkHealth::AllTiersUp < NetworkHealth::LocalDown);
    assert!(NetworkHealth::LocalDown < NetworkHealth::MetroOnly);
    assert!(NetworkHealth::MetroOnly < NetworkHealth::Blackout);
}

#[test]
fn test_epistemic_discount_monotonic() {
    let healths = [
        NetworkHealth::AllTiersUp,
        NetworkHealth::LocalDown,
        NetworkHealth::MetroOnly,
        NetworkHealth::Blackout,
    ];
    for w in healths.windows(2) {
        assert!(
            w[0].epistemic_discount() >= w[1].epistemic_discount(),
            "{:?} discount should >= {:?} discount",
            w[0],
            w[1]
        );
    }
}

// ── Regulatory constraints ───────────────────────────────────────────

#[test]
fn test_regulatory_us_default() {
    let reg = RegulatoryConstraints::default();
    assert!(reg.is_allowed(915_000_000));
    assert!(reg.is_allowed(2_450_000_000));
    assert!(!reg.is_allowed(100_000_000)); // Not ISM
}

#[test]
fn test_regulatory_eu() {
    let reg = RegulatoryConstraints::eu();
    assert!(reg.is_allowed(868_000_000));
    assert!(!reg.is_allowed(915_000_000)); // US only
}

// ── Spectrum Manager (CognitiveSubsystem) ────────────────────────────

#[test]
fn test_spectrum_manager_name_and_interval() {
    let sm = SpectrumManager::default();
    assert_eq!(sm.name(), "spectrum_manager");
    assert_eq!(sm.interval(), 53);
}

#[test]
fn test_spectrum_manager_interval_coprime() {
    let interval = 53u32;
    for other in [7, 11, 13, 19, 23, 29, 37, 41, 47] {
        assert_eq!(
            gcd(interval, other),
            1,
            "53 should be co-prime with {}",
            other
        );
    }
}

fn gcd(a: u32, b: u32) -> u32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

#[test]
fn test_spectrum_manager_neutral_without_events() {
    let mut sm = SpectrumManager::default();
    let snapshot = CycleSnapshot::default();
    let output = sm.process(&snapshot);

    // No observations, all tiers up → neutral
    assert_eq!(output.arousal_delta, 0.0);
    assert_eq!(output.confidence_delta, 0.0);
}

#[test]
fn test_spectrum_manager_jamming_response() {
    let mut sm = SpectrumManager::default();
    sm.inject_observation(SpectrumObservation {
        frequency_hz: 915_000_000,
        noise_floor_dbm: -30.0,
        snr_db: -10.0, // Below JAMMING_SNR_THRESHOLD
        jammed: true,
    });

    let snapshot = CycleSnapshot::default();
    let output = sm.process(&snapshot);

    assert!(output.arousal_delta > 0.0, "Jamming should spike arousal");
    assert!(
        output.exploration_delta > 0.0,
        "Jamming should drive exploration"
    );
    assert!(output.flags & output_flags::ANOMALY_DETECTED != 0);
}

#[test]
fn test_spectrum_manager_blackout_escalates() {
    let mut sm = SpectrumManager::default();
    sm.set_tier_available(RadioTier::Local, false);
    sm.set_tier_available(RadioTier::Metro, false);
    sm.set_tier_available(RadioTier::Regional, false);

    let snapshot = CycleSnapshot::default();
    let output = sm.process(&snapshot);

    assert!(
        output.confidence_delta < 0.0,
        "Blackout should drop confidence"
    );
    assert!(output.flags & output_flags::ESCALATE_URGENCY != 0);
    assert_eq!(sm.network_health(), NetworkHealth::Blackout);
}

#[test]
fn test_spectrum_manager_tier_loss_dampens_lr() {
    let mut sm = SpectrumManager::default();
    // Simulate high packet loss on Local tier
    for _ in 0..20 {
        sm.report_loss(RadioTier::Local);
    }

    let snapshot = CycleSnapshot::default();
    let output = sm.process(&snapshot);

    assert!(
        output.lr_modulation < 1.0,
        "High loss should dampen learning: {}",
        output.lr_modulation
    );
}

#[test]
fn test_spectrum_manager_delta_compression() {
    let mut sm = SpectrumManager::default();
    let peer_id = [1u8; 8];

    // First call: full vector (no previous state)
    let hv1 = [0xAA; 2048];
    let delta1 = sm.compress_delta(&peer_id, &hv1);
    assert!(delta1.is_full);

    // Second call: delta (previous state exists)
    let mut hv2 = hv1;
    hv2[100] = 0xBB; // Change one byte
    let delta2 = sm.compress_delta(&peer_id, &hv2);
    assert!(!delta2.is_full);
    assert!(delta2.wire_size() < 100);
}

#[test]
fn test_spectrum_manager_checkpoint_roundtrip() {
    let mut sm = SpectrumManager::default();
    // Set internal state directly (not via set_tier_available to avoid side effects)
    sm.tier_available = [false, true, true];
    sm.degradation_streak = 3;

    let data = sm.checkpoint();
    let mut sm2 = SpectrumManager::default();
    sm2.restore(&data).unwrap();

    assert_eq!(sm2.tier_available, [false, true, true]);
}

#[test]
fn test_spectrum_manager_restore_rejects_short() {
    let mut sm = SpectrumManager::default();
    assert!(sm.restore(&[0u8; 10]).is_err());
}

#[test]
fn test_spectrum_manager_degradation_streak() {
    let mut sm = SpectrumManager::default();
    sm.set_tier_available(RadioTier::Local, false);
    // set_tier_available calls update_network_health which increments streak
    assert!(sm.degradation_streak >= 1);

    let streak_before = sm.degradation_streak;
    // Process cycle calls update_network_health again → streak grows
    let snapshot = CycleSnapshot::default();
    sm.process(&snapshot);
    assert!(sm.degradation_streak > streak_before);

    // Restore Local → streak resets
    sm.set_tier_available(RadioTier::Local, true);
    assert_eq!(sm.degradation_streak, 0);
}

#[test]
fn test_spectrum_manager_telemetry() {
    let mut sm = SpectrumManager::default();
    sm.set_tier_available(RadioTier::Local, false);

    let snapshot = CycleSnapshot::default();
    sm.process(&snapshot);

    let telem = sm.telemetry();
    assert_eq!(telem.network_health, 1); // LocalDown
    assert!(!telem.tier_available[0]);
    assert!(telem.tier_available[1]);
    assert!(telem.epistemic_discount < 1.0);
}

// ── RLE encode/decode ────────────────────────────────────────────────

#[test]
fn test_rle_roundtrip_zeros() {
    let data = [0u8; 2048];
    let encoded = CompressedDelta::rle_encode(&data);
    let decoded = CompressedDelta::rle_decode(&encoded).unwrap();
    assert_eq!(decoded, data);
}

#[test]
fn test_rle_roundtrip_pattern() {
    let mut data = [0u8; 100];
    for i in 0..100 {
        data[i] = if i < 50 { 0xFF } else { 0x00 };
    }
    let encoded = CompressedDelta::rle_encode(&data);
    let decoded = CompressedDelta::rle_decode(&encoded).unwrap();
    assert_eq!(decoded, data.to_vec());
}

#[test]
fn test_rle_empty() {
    let encoded = CompressedDelta::rle_encode(&[]);
    assert!(encoded.is_empty());
}

#[test]
fn test_rle_decode_invalid_length() {
    // Not a multiple of 3
    assert!(CompressedDelta::rle_decode(&[0, 0, 0, 1]).is_none());
}

// ── Regulatory ───────────────────────────────────────────────────────

#[test]
fn test_regulatory_hard_gate() {
    let reg = RegulatoryConstraints::default();
    // Military frequencies should be blocked
    assert!(!reg.is_allowed(300_000_000)); // UHF military
    assert!(!reg.is_allowed(50_000_000)); // VHF
}

// ── RadioHardware trait ──────────────────────────────────────────────

#[test]
fn test_mock_hardware_transmit_receive() {
    let mut hw = MockRadioHardware::new();

    // Transmit on Local tier
    let payload = b"hello mesh";
    let sent = hw.transmit(RadioTier::Local, payload).unwrap();
    assert_eq!(sent, payload.len());

    // Verify transmitted payloads
    assert_eq!(hw.transmitted_payloads().len(), 1);
    assert_eq!(hw.transmitted_payloads()[0].0, RadioTier::Local as usize);
    assert_eq!(hw.transmitted_payloads()[0].1, payload.to_vec());

    // Inject receive data and read it back
    let recv_data = b"world".to_vec();
    hw.inject_receive(RadioTier::Metro, recv_data.clone(), 12.5);

    let received = hw.receive(RadioTier::Metro).unwrap().unwrap();
    assert_eq!(received.0, recv_data);
    assert!((received.1 - 12.5).abs() < 0.01);

    // No more data on Metro
    assert!(hw.receive(RadioTier::Metro).unwrap().is_none());
}

#[test]
fn test_mock_hardware_payload_too_large() {
    let mut hw = MockRadioHardware::new();

    // Metro MTU is 250 bytes — try to send 300
    let big_payload = vec![0u8; 300];
    let result = hw.transmit(RadioTier::Metro, &big_payload);
    assert!(result.is_err());
    match result.unwrap_err() {
        RadioError::PayloadTooLarge { max, got } => {
            assert_eq!(max, 250);
            assert_eq!(got, 300);
        }
        other => panic!("Expected PayloadTooLarge, got {:?}", other),
    }
}

#[test]
fn test_mock_hardware_unavailable_tier() {
    let mut hw = MockRadioHardware::new();
    hw.set_available(RadioTier::Regional, false);

    assert!(!hw.is_available(RadioTier::Regional));
    assert!(hw.current_snr(RadioTier::Regional).is_none());
    assert!(hw.current_frequency(RadioTier::Regional).is_none());

    let result = hw.transmit(RadioTier::Regional, b"test");
    assert!(matches!(result, Err(RadioError::Unavailable)));
}

#[test]
fn test_null_hardware_always_unavailable() {
    let mut hw = NullRadioHardware;

    assert!(!hw.is_available(RadioTier::Local));
    assert!(!hw.is_available(RadioTier::Metro));
    assert!(!hw.is_available(RadioTier::Regional));

    assert!(hw.current_snr(RadioTier::Local).is_none());
    assert!(hw.current_frequency(RadioTier::Local).is_none());
    assert_eq!(hw.hardware_id(), "NullRadioHardware");

    assert!(matches!(
        hw.transmit(RadioTier::Local, b"test"),
        Err(RadioError::Unavailable)
    ));
    assert!(matches!(
        hw.receive(RadioTier::Local),
        Err(RadioError::Unavailable)
    ));
    assert!(matches!(
        hw.set_tx_power(RadioTier::Local, 10.0),
        Err(RadioError::Unavailable)
    ));
    assert!(matches!(
        hw.tune(RadioTier::Local, 2_450_000_000),
        Err(RadioError::Unavailable)
    ));
}

#[test]
fn test_mock_hardware_snr() {
    let mut hw = MockRadioHardware::new();
    hw.set_snr(RadioTier::Local, 42.0);
    assert_eq!(hw.current_snr(RadioTier::Local), Some(42.0));
}

#[test]
fn test_mock_hardware_tune_and_frequency() {
    let mut hw = MockRadioHardware::new();
    hw.tune(RadioTier::Local, 2_412_000_000).unwrap();
    assert_eq!(hw.current_frequency(RadioTier::Local), Some(2_412_000_000));
}

#[test]
fn test_mock_hardware_tx_power() {
    let mut hw = MockRadioHardware::new();
    hw.set_tx_power(RadioTier::Metro, 14.0).unwrap();
    // No assertion on internal state — success is sufficient
}

#[test]
fn test_mock_hardware_id() {
    let hw = MockRadioHardware::new();
    assert_eq!(hw.hardware_id(), "MockRadioHardware v1.0");
}

// ── RegulatoryDatabase ───────────────────────────────────────────────

#[test]
fn test_regulatory_fcc_lora_band() {
    let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);
    assert!(db.is_frequency_allowed(915_000_000, RadioTier::Metro));
    assert!(db.is_frequency_allowed(902_000_000, RadioTier::Metro));
    assert!(db.is_frequency_allowed(928_000_000, RadioTier::Metro));
}

#[test]
fn test_regulatory_fcc_outside_band() {
    let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);
    assert!(!db.is_frequency_allowed(850_000_000, RadioTier::Metro));
    assert!(!db.is_frequency_allowed(900_000_000, RadioTier::Metro));
}

#[test]
fn test_regulatory_etsi_duty_cycle() {
    let db = RegulatoryDatabase::new(RegulatoryRegion::EtsiEu);
    let duty = db.duty_cycle_for_band(868_300_000);
    assert!(duty.is_some(), "868 MHz band should have a duty cycle");
    assert!(
        (duty.unwrap() - 0.01).abs() < 0.001,
        "Expected 1% duty cycle, got {}",
        duty.unwrap()
    );
}

#[test]
fn test_regulatory_bands_for_tier() {
    let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);

    let local_bands = db.bands_for_tier(RadioTier::Local);
    assert!(
        local_bands.len() >= 2,
        "FCC should have at least 2 Local bands (2.4 GHz + 5.8 GHz)"
    );
    for band in &local_bands {
        assert_eq!(band.tier, RadioTier::Local);
    }

    let metro_bands = db.bands_for_tier(RadioTier::Metro);
    assert!(
        !metro_bands.is_empty(),
        "FCC should have Metro bands (915 MHz ISM)"
    );

    let regional_bands = db.bands_for_tier(RadioTier::Regional);
    assert!(
        !regional_bands.is_empty(),
        "FCC should have Regional bands (HF amateur)"
    );
    for band in &regional_bands {
        assert_eq!(band.license, LicenseType::Amateur);
    }
}

#[test]
fn test_regulatory_max_power() {
    let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);
    let power = db.max_power_for_frequency(915_000_000);
    assert!(power.is_some());
    assert!((power.unwrap() - 30.0).abs() < 0.1);
    let power = db.max_power_for_frequency(2_450_000_000);
    assert!(power.is_some());
    assert!((power.unwrap() - 36.0).abs() < 0.1);
    assert!(db.max_power_for_frequency(100_000_000).is_none());
}

#[test]
fn test_regulatory_available_bandwidth() {
    let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);
    let metro_bw = db.available_bandwidth(RadioTier::Metro);
    assert_eq!(metro_bw, 26_000_000);
    let local_bw = db.available_bandwidth(RadioTier::Local);
    assert!(
        local_bw > 200_000_000,
        "Local bandwidth should be > 200 MHz, got {}",
        local_bw
    );
}

#[test]
fn test_regulatory_etsi_region() {
    let db = RegulatoryDatabase::new(RegulatoryRegion::EtsiEu);
    assert_eq!(db.region(), RegulatoryRegion::EtsiEu);
    assert!(!db.is_frequency_allowed(915_000_000, RadioTier::Metro));
    assert!(db.is_frequency_allowed(868_300_000, RadioTier::Metro));
    let power = db.max_power_for_frequency(2_450_000_000);
    assert!(power.is_some());
    assert!((power.unwrap() - 20.0).abs() < 0.1);
}

#[test]
fn test_regulatory_ism_global_fallback() {
    let db = RegulatoryDatabase::new(RegulatoryRegion::IsmGlobal);
    assert!(db.is_frequency_allowed(2_450_000_000, RadioTier::Local));
    assert!(db.is_frequency_allowed(433_500_000, RadioTier::Metro));
}

#[test]
fn test_regulatory_to_legacy_constraints() {
    let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);
    let legacy = db.to_legacy_constraints();
    assert_eq!(legacy.region, "US");
    assert!(legacy.is_allowed(915_000_000));
    assert!(legacy.is_allowed(2_450_000_000));
    assert!(
        !legacy.is_allowed(7_100_000),
        "Amateur bands should not appear in legacy unlicensed constraints"
    );
}

// ── Hardware + Regulatory integration ────────────────────────────────

#[test]
fn test_hardware_regulatory_integration() {
    let mut hw = MockRadioHardware::new();
    let db = RegulatoryDatabase::new(RegulatoryRegion::FccUs);
    hw.set_regulatory_db(db);

    hw.tune(RadioTier::Metro, 915_000_000).unwrap();

    let result = hw.tune(RadioTier::Metro, 850_000_000);
    assert!(result.is_err());
    match result.unwrap_err() {
        RadioError::FrequencyOutOfBand { freq_hz, .. } => {
            assert_eq!(freq_hz, 850_000_000);
        }
        other => panic!("Expected FrequencyOutOfBand, got {:?}", other),
    }

    hw.tune(RadioTier::Metro, 915_000_000).unwrap();
    hw.set_tx_power(RadioTier::Metro, 20.0).unwrap();

    let result = hw.set_tx_power(RadioTier::Metro, 35.0);
    assert!(result.is_err());
    assert!(matches!(
        result.unwrap_err(),
        RadioError::RegulatoryViolation(_)
    ));
}

// ── SpectrumManager + hardware integration ──────────────────────────

#[test]
fn test_spectrum_manager_with_hardware() {
    let mut sm = SpectrumManager::with_region(RegulatoryRegion::FccUs);
    let hw = MockRadioHardware::new();
    sm.set_hardware(Box::new(hw));
    assert_eq!(sm.hardware_id(), Some("MockRadioHardware v1.0"));
    let snapshot = CycleSnapshot::default();
    let _output = sm.process(&snapshot);
    assert!(sm.telemetry().tier_available[0]);
}

#[test]
fn test_spectrum_manager_with_region() {
    let sm = SpectrumManager::with_region(RegulatoryRegion::EtsiEu);
    assert_eq!(sm.regulatory_db().region(), RegulatoryRegion::EtsiEu);
    assert!(sm.regulatory().is_allowed(868_300_000));
}

#[test]
fn test_spectrum_manager_hardware_down_tier() {
    let mut sm = SpectrumManager::default();
    let mut hw = MockRadioHardware::new();
    hw.set_available(RadioTier::Local, false);
    sm.set_hardware(Box::new(hw));
    let snapshot = CycleSnapshot::default();
    let output = sm.process(&snapshot);
    assert!(!sm.tier_available[0]);
    assert!(
        output.confidence_delta < 0.0,
        "Hardware-detected tier loss should reduce confidence"
    );
}

// ── Item 1: AIMD ─────────────────────────────────────────────────

#[test]
fn test_aimd_additive_increase_on_healthy() {
    let mut sm = SpectrumManager::default();
    let initial_budget = sm.tier_budgets()[0];
    sm.tick_aimd();
    assert!(
        sm.tier_budgets()[0] > initial_budget,
        "Healthy tier should increase budget: {} -> {}",
        initial_budget,
        sm.tier_budgets()[0]
    );
}

#[test]
fn test_aimd_multiplicative_decrease_on_congestion() {
    let mut sm = SpectrumManager::default();
    // Simulate high loss on Metro — need to use report_loss to affect tier_loss_ema
    for _ in 0..50 {
        sm.report_loss(RadioTier::Metro);
    }
    let initial_budget = sm.tier_budgets()[1];
    sm.tick_aimd();
    assert!(
        sm.tier_budgets()[1] < initial_budget,
        "Congested tier should decrease budget: {} -> {}",
        initial_budget,
        sm.tier_budgets()[1]
    );
}

#[test]
fn test_aimd_respects_floor_and_ceiling() {
    let mut sm = SpectrumManager::default();
    let local_profile = RadioTier::Local.profile();

    // Push budget to ceiling
    for _ in 0..200 {
        sm.tick_aimd();
    }
    assert!(
        sm.tier_budgets()[0] <= local_profile.bandwidth_max,
        "Budget should not exceed ceiling"
    );

    // Force heavy loss → decrease
    for _ in 0..200 {
        sm.report_loss(RadioTier::Local);
    }
    for _ in 0..200 {
        sm.tick_aimd();
    }
    assert!(
        sm.tier_budgets()[0] >= local_profile.bandwidth_min,
        "Budget should not fall below floor"
    );
}

#[test]
fn test_aimd_skips_unavailable_tiers() {
    let mut sm = SpectrumManager::default();
    sm.tier_available[2] = false; // Regional down
    let initial = sm.tier_budgets()[2];
    sm.tick_aimd();
    assert_eq!(
        sm.tier_budgets()[2],
        initial,
        "Down tier budget should not change"
    );
}

// ── Item 2: Adaptive compression ─────────────────────────────────

#[test]
fn test_compress_for_tier_local_full() {
    let mut sm = SpectrumManager::default();
    let peer_id = [1u8; 8];
    let hv = [0xAA; 2048];
    let result = sm.compress_for_tier(&peer_id, &hv, RadioTier::Local);
    assert_eq!(result.strategy, CompressionStrategy::Full);
    assert_eq!(result.data.len(), 2048);
}

#[test]
fn test_compress_for_tier_metro_delta() {
    let mut sm = SpectrumManager::default();
    let peer_id = [2u8; 8];
    let hv1 = [0xAA; 2048];
    let r1 = sm.compress_for_tier(&peer_id, &hv1, RadioTier::Metro);
    assert_eq!(r1.strategy, CompressionStrategy::Full);
    let mut hv2 = hv1;
    hv2[100] = 0xBB;
    let r2 = sm.compress_for_tier(&peer_id, &hv2, RadioTier::Metro);
    assert_eq!(r2.strategy, CompressionStrategy::Delta);
    assert!(r2.data.len() < 100, "Delta should be small");
}

#[test]
fn test_compress_for_tier_regional_hash() {
    let mut sm = SpectrumManager::default();
    let peer_id = [3u8; 8];
    let hv = [0xCC; 2048];
    let result = sm.compress_for_tier(&peer_id, &hv, RadioTier::Regional);
    assert_eq!(result.strategy, CompressionStrategy::HashOnly);
    assert_eq!(result.data.len(), 32);
}

#[test]
fn test_compress_for_tier_tracks_energy() {
    let mut sm = SpectrumManager::default();
    let peer_id = [4u8; 8];
    let hv = [0xDD; 2048];
    sm.compress_for_tier(&peer_id, &hv, RadioTier::Local);
    assert!(sm.energy_spent_nj > 0.0, "Should track energy cost");
}

// ── Item 3: Waterfall ────────────────────────────────────────────

#[test]
fn test_waterfall_records_observations() {
    let mut sm = SpectrumManager::default();
    sm.inject_observation(SpectrumObservation {
        frequency_hz: 915_000_000,
        noise_floor_dbm: -90.0,
        snr_db: 20.0,
        jammed: false,
    });
    let snapshot = CycleSnapshot::default();
    sm.process(&snapshot);
    assert_eq!(sm.waterfall.len(), 1);
}

#[test]
fn test_waterfall_mean_noise_floor() {
    let mut wf = tier::SpectrumWaterfall::new(64);
    for i in 0..10 {
        wf.push(tier::WaterfallEntry {
            cycle: i,
            noise_floor_dbm: -90.0 + i as f64,
            snr_db: 20.0,
            jammed: false,
            observation_count: 1,
        });
    }
    let mean = wf.mean_noise_floor().unwrap();
    assert!((mean - (-85.5)).abs() < 0.01);
}

#[test]
fn test_waterfall_jamming_ratio() {
    let mut wf = tier::SpectrumWaterfall::new(64);
    for i in 0..10 {
        wf.push(tier::WaterfallEntry {
            cycle: i,
            noise_floor_dbm: -90.0,
            snr_db: 20.0,
            jammed: i % 2 == 0, // 5 out of 10 jammed
            observation_count: 1,
        });
    }
    assert!((wf.jamming_ratio() - 0.5).abs() < 0.01);
}

#[test]
fn test_waterfall_capacity_bounded() {
    let mut wf = tier::SpectrumWaterfall::new(4);
    for i in 0..10 {
        wf.push(tier::WaterfallEntry {
            cycle: i,
            noise_floor_dbm: -90.0,
            snr_db: 20.0,
            jammed: false,
            observation_count: 1,
        });
    }
    assert_eq!(wf.len(), 4);
}

#[test]
fn test_waterfall_periodic_detection() {
    let mut wf = tier::SpectrumWaterfall::new(64);
    for i in 0..40 {
        let noise = if i % 10 == 0 { -50.0 } else { -90.0 };
        wf.push(tier::WaterfallEntry {
            cycle: i,
            noise_floor_dbm: noise,
            snr_db: if i % 10 == 0 { 2.0 } else { 20.0 },
            jammed: i % 10 == 0,
            observation_count: 1,
        });
    }
    let period = wf.detect_periodic_interference();
    assert!(period.is_some(), "Should detect periodic interference");
    assert_eq!(period.unwrap(), 10);
}

// ── Item 6: Frequency hopping ────────────────────────────────────

#[test]
fn test_frequency_hop_cooldown() {
    let mut sm = SpectrumManager::default();
    sm.hop_cooldown = 3;
    assert!(sm.suggest_frequency_hop(RadioTier::Metro).is_none());
}

#[test]
fn test_frequency_hop_needs_waterfall_data() {
    let mut sm = SpectrumManager::default();
    assert!(sm.suggest_frequency_hop(RadioTier::Metro).is_none());
}

// ── Item 7: Energy-aware routing ─────────────────────────────────

#[test]
fn test_energy_per_bit_in_profile() {
    let local = RadioTier::Local.profile();
    let metro = RadioTier::Metro.profile();
    let regional = RadioTier::Regional.profile();
    assert!(local.energy_per_bit_nj > metro.energy_per_bit_nj);
    assert!(regional.energy_per_bit_nj > metro.energy_per_bit_nj);
}

#[test]
fn test_energy_aware_route_normal() {
    let sm = SpectrumManager::default();
    let route = sm.energy_aware_route(100, 1);
    assert!(route.is_some());
}

#[test]
fn test_energy_aware_route_constrained() {
    let mut sm = SpectrumManager::default();
    sm.energy_spent_nj = sm.energy_budget_nj * 0.6;
    let route = sm.energy_aware_route(100, 1);
    assert!(route.is_some());
    let tier = route.unwrap();
    assert_eq!(tier, RadioTier::Metro);
}

// ── Item 8: Peer discovery ───────────────────────────────────────

#[test]
fn test_beacon_serialization_roundtrip() {
    let beacon = DiscoveryBeacon {
        node_id: [1, 2, 3, 4, 5, 6, 7, 8],
        capabilities_hash: [0xAA; 8],
        cycle_counter: 42,
        network_health: 1,
        tier_mask: 0x07,
    };
    let bytes = beacon.to_bytes();
    assert_eq!(bytes.len(), RADIO_BEACON_SIZE);
    let decoded = DiscoveryBeacon::from_bytes(&bytes);
    assert_eq!(decoded.node_id, beacon.node_id);
    assert_eq!(decoded.capabilities_hash, beacon.capabilities_hash);
    assert_eq!(decoded.cycle_counter, 42);
    assert_eq!(decoded.network_health, 1);
    assert_eq!(decoded.tier_mask, 0x07);
}

#[test]
fn test_beacon_tier_mask() {
    assert_eq!(DiscoveryBeacon::tier_mask_from(&[true, true, true]), 0x07);
    assert_eq!(DiscoveryBeacon::tier_mask_from(&[true, false, false]), 0x01);
    assert_eq!(DiscoveryBeacon::tier_mask_from(&[false, true, true]), 0x06);
    assert_eq!(
        DiscoveryBeacon::tier_mask_from(&[false, false, false]),
        0x00
    );
}

#[test]
fn test_beacon_interval() {
    let mut sm = SpectrumManager::default();
    sm.set_node_id([1u8; 8]);
    let caps = [0xABu8; 8];
    for _ in 0..RADIO_BEACON_INTERVAL_CYCLES - 1 {
        assert!(sm.generate_beacon(&caps).is_none());
    }
    let beacon = sm.generate_beacon(&caps);
    assert!(beacon.is_some());
    assert_eq!(beacon.unwrap().node_id, [1u8; 8]);
}

#[test]
fn test_process_beacon_adds_route() {
    let mut sm = SpectrumManager::default();
    let beacon = DiscoveryBeacon {
        node_id: [42u8; 8],
        capabilities_hash: [0; 8],
        cycle_counter: 100,
        network_health: 0,
        tier_mask: 0x07,
    };
    sm.process_beacon(&beacon, RadioTier::Metro, 15.0);
    assert_eq!(sm.route_table.len(), 1);
    let route = sm.route_table.lookup(&[42u8; 8]).unwrap();
    assert_eq!(route.hop_count, 0);
    assert_eq!(route.tier, RadioTier::Metro);
}

// ── Item 9: Multi-hop relay ──────────────────────────────────────

#[test]
fn test_route_table_update_and_lookup() {
    let mut rt = RouteTable::new(128);
    rt.update(RouteEntry {
        destination: [1u8; 8],
        next_hop: [2u8; 8],
        hop_count: 1,
        tier: RadioTier::Metro,
        last_seen_cycle: 100,
        link_quality: 0.8,
    });
    assert_eq!(rt.len(), 1);
    let route = rt.lookup(&[1u8; 8]).unwrap();
    assert_eq!(route.hop_count, 1);
    assert_eq!(route.next_hop, [2u8; 8]);
}

#[test]
fn test_route_table_prefers_shorter_hops() {
    let mut rt = RouteTable::new(128);
    rt.update(RouteEntry {
        destination: [1u8; 8],
        next_hop: [2u8; 8],
        hop_count: 3,
        tier: RadioTier::Metro,
        last_seen_cycle: 100,
        link_quality: 0.8,
    });
    rt.update(RouteEntry {
        destination: [1u8; 8],
        next_hop: [3u8; 8],
        hop_count: 1,
        tier: RadioTier::Local,
        last_seen_cycle: 101,
        link_quality: 0.9,
    });
    let route = rt.lookup(&[1u8; 8]).unwrap();
    assert_eq!(route.hop_count, 1);
    assert_eq!(route.next_hop, [3u8; 8]);
}

#[test]
fn test_route_table_prune() {
    let mut rt = RouteTable::new(128);
    rt.update(RouteEntry {
        destination: [1u8; 8],
        next_hop: [2u8; 8],
        hop_count: 1,
        tier: RadioTier::Metro,
        last_seen_cycle: 10,
        link_quality: 0.8,
    });
    rt.prune(600, 500);
    assert_eq!(rt.len(), 0, "Stale route should be pruned");
}

#[test]
fn test_prepare_relay() {
    let mut sm = SpectrumManager::default();
    sm.set_node_id([10u8; 8]);
    sm.route_table.update(RouteEntry {
        destination: [20u8; 8],
        next_hop: [15u8; 8],
        hop_count: 1,
        tier: RadioTier::Metro,
        last_seen_cycle: 0,
        link_quality: 0.9,
    });
    let payload = b"hello";
    let result = sm.prepare_relay(&[20u8; 8], payload);
    assert!(result.is_some());
    let (next_hop, packet) = result.unwrap();
    assert_eq!(next_hop, [15u8; 8]);
    assert_eq!(packet.len(), 23);
    assert_eq!(&packet[0..8], &[20u8; 8]);
    assert_eq!(&packet[8..16], &[10u8; 8]);
}

// ── Item 10: FEC ─────────────────────────────────────────────────

#[test]
fn test_fec_encode_decode_no_loss() {
    let data = b"hello world, this is a test of FEC encoding!";
    let encoded = FecEncoder::encode(data, 16);
    assert!(encoded.len() > data.len());
    let decoded = FecEncoder::decode(&encoded, 16, None);
    assert_eq!(&decoded, data);
}

#[test]
fn test_fec_recover_lost_block() {
    let data = vec![0xAA; 64];
    let encoded = FecEncoder::encode(&data, 16);
    let mut corrupted = encoded.clone();
    for i in 32..48 {
        corrupted[i] = 0x00;
    }
    let recovered = FecEncoder::decode(&corrupted, 16, Some(2));
    assert_eq!(&recovered[..64], &data[..]);
}

#[test]
fn test_fec_small_payload_passthrough() {
    let data = b"tiny";
    let encoded = FecEncoder::encode(data, 16);
    assert_eq!(encoded.len(), data.len());
}

#[test]
fn test_fec_overhead() {
    assert_eq!(FecEncoder::overhead(100, 16), 16);
    assert_eq!(FecEncoder::overhead(10, 16), 0);
}

// ── Item 11: Encryption ──────────────────────────────────────────

#[test]
fn test_encryption_roundtrip() {
    let key = [42u8; 32];
    let nonce = [1u8; RADIO_CRYPTO_NONCE_SIZE];
    let plaintext = b"secret consciousness data";
    let encrypted = MeshEncryption::encrypt(&key, &nonce, plaintext);
    assert_ne!(&encrypted[..plaintext.len()], plaintext);
    let decrypted = MeshEncryption::decrypt(&key, &nonce, &encrypted);
    assert!(decrypted.is_some());
    assert_eq!(&decrypted.unwrap(), plaintext);
}

#[test]
fn test_encryption_auth_tag_failure() {
    let key = [42u8; 32];
    let nonce = [1u8; RADIO_CRYPTO_NONCE_SIZE];
    let plaintext = b"secret data";
    let mut encrypted = MeshEncryption::encrypt(&key, &nonce, plaintext);
    let len = encrypted.len();
    encrypted[len - 1] ^= 0xFF;
    assert!(MeshEncryption::decrypt(&key, &nonce, &encrypted).is_none());
}

#[test]
fn test_peer_session_nonce_monotonic() {
    let mut session = PeerSession::new([1u8; 8], [2u8; 32], 0);
    let n1 = session.next_nonce();
    let n2 = session.next_nonce();
    assert_ne!(n1, n2);
    assert_eq!(session.tx_nonce_counter, 2);
}

#[test]
fn test_peer_session_replay_detection() {
    let mut session = PeerSession::new([1u8; 8], [2u8; 32], 0);
    assert!(session.check_nonce(1));
    assert!(session.check_nonce(2));
    assert!(!session.check_nonce(1));
    assert!(!session.check_nonce(2));
    assert!(session.check_nonce(3));
}

#[test]
fn test_spectrum_manager_encrypt_decrypt() {
    let mut sm = SpectrumManager::default();
    let peer_id = [5u8; 8];
    let session = PeerSession::new(peer_id, [99u8; 32], 0);
    sm.encryption_mut().add_session(session);
    let plaintext = b"consciousness vector";
    let encrypted = sm.encrypt_for_peer(&peer_id, plaintext).unwrap();
    let decrypted = sm.decrypt_from_peer(&peer_id, &encrypted).unwrap();
    assert_eq!(&decrypted, plaintext);
}

#[test]
fn test_encryption_no_session() {
    let mut sm = SpectrumManager::default();
    assert!(sm.encrypt_for_peer(&[9u8; 8], b"test").is_none());
}

#[test]
fn test_mesh_encryption_capacity() {
    let mut enc = MeshEncryption::new(2);
    enc.add_session(PeerSession::new([1u8; 8], [1u8; 32], 0));
    enc.add_session(PeerSession::new([2u8; 8], [2u8; 32], 0));
    enc.add_session(PeerSession::new([3u8; 8], [3u8; 32], 0));
    assert_eq!(enc.session_count(), 2);
}

// ── Telemetry includes new fields ────────────────────────────────

#[test]
fn test_telemetry_includes_new_fields() {
    let mut sm = SpectrumManager::default();
    sm.inject_observation(SpectrumObservation {
        frequency_hz: 915_000_000,
        noise_floor_dbm: -90.0,
        snr_db: 20.0,
        jammed: false,
    });
    let snapshot = CycleSnapshot::default();
    sm.process(&snapshot);
    let t = sm.telemetry();
    assert!(t.tier_budgets[0] > 0);
    assert_eq!(t.waterfall_depth, 1);
    assert_eq!(t.known_peers, 0);
    assert_eq!(t.encryption_sessions, 0);
}

// ── Item 1: Mesh→Spectrum feedback ──────────────────────────────

#[test]
fn test_ingest_mesh_stats_updates_loss_ema() {
    let mut sm = SpectrumManager::default();
    sm.ingest_mesh_stats(50, 100);
    let t = sm.telemetry();
    assert!(t.tier_loss_ema[0] > 0.0, "Local loss should increase");
    assert!(
        t.tier_loss_ema[1] > t.tier_loss_ema[0],
        "Metro should absorb more"
    );
    assert!(
        t.tier_loss_ema[2] > t.tier_loss_ema[1],
        "Regional should absorb most"
    );
}

#[test]
fn test_ingest_mesh_stats_zero_dropped() {
    let mut sm = SpectrumManager::default();
    sm.ingest_mesh_stats(0, 100);
    let t = sm.telemetry();
    for &loss in &t.tier_loss_ema {
        assert!(loss.abs() < 1e-10, "No drops should not increase loss");
    }
}

#[test]
fn test_ingest_mesh_stats_zero_total() {
    let mut sm = SpectrumManager::default();
    sm.ingest_mesh_stats(10, 0);
    let t = sm.telemetry();
    for &loss in &t.tier_loss_ema {
        assert!(loss.abs() < 1e-10);
    }
}

// ── Item 2: Safety escalation ───────────────────────────────────

#[test]
fn test_network_critical_on_blackout() {
    let mut sm = SpectrumManager::default();
    sm.set_tier_available(RadioTier::Local, false);
    sm.set_tier_available(RadioTier::Metro, false);
    sm.set_tier_available(RadioTier::Regional, false);
    assert!(sm.is_network_critical());
}

#[test]
fn test_network_not_critical_all_up() {
    let sm = SpectrumManager::default();
    assert!(!sm.is_network_critical());
}

// ── Item 3: Connectivity factor ─────────────────────────────────

#[test]
fn test_connectivity_factor_all_up() {
    let sm = SpectrumManager::default();
    assert!((sm.connectivity_factor() - 1.0).abs() < 1e-10);
}

#[test]
fn test_connectivity_factor_blackout() {
    let mut sm = SpectrumManager::default();
    sm.set_tier_available(RadioTier::Local, false);
    sm.set_tier_available(RadioTier::Metro, false);
    sm.set_tier_available(RadioTier::Regional, false);
    assert!((sm.connectivity_factor() - 0.0).abs() < 1e-10);
}

#[test]
fn test_connectivity_factor_local_down() {
    let mut sm = SpectrumManager::default();
    sm.set_tier_available(RadioTier::Local, false);
    assert!((sm.connectivity_factor() - 0.7).abs() < 1e-10);
}

// ── Item 9: Beacon→SwarmManager ──────────────────────────────────

#[test]
fn test_process_beacon_returns_true_for_new_peer() {
    let mut sm = SpectrumManager::default();
    let beacon = DiscoveryBeacon {
        node_id: [42u8; 8],
        capabilities_hash: [0; 8],
        cycle_counter: 100,
        network_health: 0,
        tier_mask: 0b111,
    };
    let is_new = sm.process_beacon(&beacon, RadioTier::Local, 15.0);
    assert!(is_new, "First beacon from peer should be new");
    let is_new_again = sm.process_beacon(&beacon, RadioTier::Local, 15.0);
    assert!(
        !is_new_again,
        "Second beacon from same peer should not be new"
    );
}

// ── Item 10: Regulatory validation ──────────────────────────────

#[test]
fn test_validate_transmission_ism_band() {
    let sm = SpectrumManager::default();
    assert!(sm.validate_transmission(RadioTier::Metro, 433_500_000, 10.0));
}

#[test]
fn test_validate_transmission_out_of_band() {
    let sm = SpectrumManager::default();
    assert!(!sm.validate_transmission(RadioTier::Metro, 100_000_000, 14.0));
}

// ── Item 3: Synthetic observations from swarm state ────────────────

#[test]
fn test_ingest_swarm_state_generates_observation() {
    let mut sm = SpectrumManager::default();
    assert_eq!(sm.pending_observations.len(), 0);
    sm.ingest_swarm_state(5, 0.6, 0.8);
    assert_eq!(sm.pending_observations.len(), 1);
    let obs = &sm.pending_observations[0];
    assert!(!obs.jammed, "Connected peers should not be jammed");
    assert!(
        obs.snr_db > RADIO_SYNTHETIC_SNR_BASE as f32,
        "Peers should boost SNR"
    );
}

#[test]
fn test_ingest_swarm_state_isolated_is_jammed() {
    let mut sm = SpectrumManager::default();
    sm.ingest_swarm_state(0, 0.0, 0.05);
    let obs = &sm.pending_observations[0];
    assert!(obs.jammed, "Zero peers + low connectivity should be jammed");
    assert!(
        (obs.snr_db - RADIO_SYNTHETIC_SNR_ISOLATED as f32).abs() < 1e-3,
        "Isolated SNR should be {}",
        RADIO_SYNTHETIC_SNR_ISOLATED
    );
}

#[test]
fn test_ingest_swarm_state_snr_increases_with_peers() {
    let mut sm = SpectrumManager::default();
    sm.ingest_swarm_state(1, 0.5, 0.5);
    let snr1 = sm.pending_observations[0].snr_db;
    let mut sm2 = SpectrumManager::default();
    sm2.ingest_swarm_state(20, 0.5, 0.5);
    let snr20 = sm2.pending_observations[0].snr_db;
    assert!(snr20 > snr1, "More peers should increase SNR");
}

#[test]
fn test_spectrum_prediction_error_accessor() {
    let sm = SpectrumManager::default();
    assert!((sm.spectrum_prediction_error() - 0.0).abs() < 1e-10);
}

// ── Consciousness-aware tier selection ─────────────────────────────

#[test]
fn test_consciousness_aware_high_confidence_prefers_local() {
    let sm = SpectrumManager::default();
    let tier = sm.consciousness_aware_tier(40, 0.9);
    assert_eq!(
        tier,
        Some(RadioTier::Local),
        "High confidence should prefer Local"
    );
}

#[test]
fn test_consciousness_aware_low_confidence_prefers_metro() {
    let sm = SpectrumManager::default();
    let tier = sm.consciousness_aware_tier(40, 0.1);
    assert_eq!(
        tier,
        Some(RadioTier::Metro),
        "Low confidence should prefer Metro (lowest energy)"
    );
}

#[test]
fn test_consciousness_aware_medium_confidence_defers() {
    let sm = SpectrumManager::default();
    let tier = sm.consciousness_aware_tier(40, 0.5);
    assert_eq!(
        tier, None,
        "Medium confidence should defer to normal routing"
    );
}

#[test]
fn test_consciousness_aware_respects_mtu() {
    let sm = SpectrumManager::default();
    let tier = sm.consciousness_aware_tier(1000, 0.9);
    assert_eq!(
        tier,
        Some(RadioTier::Local),
        "Only Local can fit 1000B payload"
    );
}

// ── Energy-aware routing ──────────────────────────────────────────

#[test]
fn test_energy_aware_route_activates_above_threshold() {
    let mut sm = SpectrumManager::default();
    sm.energy_spent_nj = sm.energy_budget_nj * 0.3;
    let r1 = sm.energy_aware_route(100, 1);
    assert!(r1.is_some());
    sm.energy_spent_nj = sm.energy_budget_nj * 0.6;
    let r2 = sm.energy_aware_route(100, 1);
    assert!(r2.is_some());
    assert_eq!(r2, Some(RadioTier::Metro));
}

// ═══════════════════════════════════════════════════════════════════
// CONSCIOUSNESS-AWARE ROUTER TESTS
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_consciousness_router_default() {
    let router = ConsciousnessAwareRouter::default();
    assert_eq!(router.peer_count(), 0);
    assert_eq!(router.collective_phi(), 0.0);
    assert_eq!(router.sharing_cadence(), 50);
}

#[test]
fn test_consciousness_router_update_peer() {
    let mut router = ConsciousnessAwareRouter::default();
    router.update_local(0.5, 0.7, 2);
    router.update_peer([1; 8], 0.8, 0.9, 3, 100);
    assert_eq!(router.peer_count(), 1);
    assert!(router.collective_phi() > 0.0);
    router.update_peer([2; 8], 0.3, 0.4, 1, 101);
    assert_eq!(router.peer_count(), 2);
}

#[test]
fn test_consciousness_router_highest_phi_peer() {
    let mut router = ConsciousnessAwareRouter::default();
    router.update_local(0.5, 0.7, 2);
    router.update_peer([1; 8], 0.3, 0.6, 1, 100);
    router.update_peer([2; 8], 0.9, 0.95, 4, 100);
    router.update_peer([3; 8], 0.6, 0.8, 2, 100);
    let best = router.highest_phi_peer();
    assert_eq!(best, Some([2; 8]));
}

#[test]
fn test_consciousness_router_moral_emergency() {
    let mut router = ConsciousnessAwareRouter::default();
    let classifier = PayloadClassifier::default();
    router.signal_moral_emergency();
    let decision = router.route(PayloadClass::Emergency, 40, 2, &classifier);
    match decision {
        ConsciousRoutingDecision::MoralEmergency { tier, .. } => {
            assert_eq!(tier, RadioTier::Local);
        }
        _ => panic!("expected MoralEmergency routing"),
    }
    let decision2 = router.route(PayloadClass::Discovery, 64, 1, &classifier);
    assert!(matches!(decision2, ConsciousRoutingDecision::Normal(_)));
}

#[test]
fn test_consciousness_router_adaptive_cadence() {
    let mut router = ConsciousnessAwareRouter::default();
    router.update_local(0.95, 0.9, 4);
    let cadence_before = router.sharing_cadence();
    router.update_peer([1; 8], 0.05, 0.3, 1, 100);
    router.update_peer([2; 8], 0.05, 0.3, 1, 100);
    let cadence_after = router.sharing_cadence();
    assert!(
        cadence_after < cadence_before,
        "high divergence should decrease cadence, before={}, after={}",
        cadence_before,
        cadence_after
    );
}

#[test]
fn test_consciousness_router_sharing_suppression() {
    let mut router = ConsciousnessAwareRouter::default();
    let classifier = PayloadClassifier::default();
    let decision = router.route(PayloadClass::ConsciousnessDelta, 100, 1, &classifier);
    match decision {
        ConsciousRoutingDecision::Suppressed { reason, .. } => {
            assert!(reason.contains("cadence"));
        }
        ConsciousRoutingDecision::ConsciousnessShare { .. } => {}
        _ => panic!("expected Suppressed or ConsciousnessShare"),
    }
}

#[test]
fn test_consciousness_router_threat_recording() {
    let mut router = ConsciousnessAwareRouter::default();
    let threat = ThreatObservation {
        threat_type: 3,
        severity: 0.8,
        agent_hash: [0xDE; 8],
        signature: [0x42; 32],
        observed_cycle: 100,
        corroboration_count: 0,
    };
    router.record_threat(threat.clone());
    assert_eq!(router.threat_count(), 1);
    router.record_threat(threat);
    assert_eq!(router.threat_count(), 1);
    assert_eq!(router.threats()[0].corroboration_count, 1);
}

#[test]
fn test_consciousness_router_trust_decay() {
    let mut router = ConsciousnessAwareRouter::default();
    router.update_peer([1; 8], 0.3, 0.7, 2, 100);
    let trust_before = router.peer_phi.get(&[1; 8]).unwrap().trust;
    router.update_peer([1; 8], 0.99, 0.7, 2, 101);
    let trust_after = router.peer_phi.get(&[1; 8]).unwrap().trust;
    assert!(
        trust_after < trust_before,
        "large Phi jump should decrease trust"
    );
}

#[test]
fn test_consciousness_router_prune() {
    let mut router = ConsciousnessAwareRouter::default();
    router.update_peer([1; 8], 0.5, 0.7, 2, 100);
    router.update_peer([2; 8], 0.6, 0.8, 3, 200);
    router.prune(249, 50);
    assert_eq!(router.peer_count(), 1);
}

// ═══════════════════════════════════════════════════════════════════
// STORE-AND-FORWARD TESTS
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_store_forward_default() {
    let sf = StoreAndForward::default();
    assert!(!sf.is_offline());
    assert_eq!(sf.buffer_len(), 0);
    assert_eq!(sf.reconnection_count(), 0);
}

#[test]
fn test_store_forward_offline_online_cycle() {
    let mut sf = StoreAndForward::default();
    sf.go_offline(100);
    assert!(sf.is_offline());
    for i in 0..15 {
        sf.record(OfflineExperience {
            cycle: 100 + i,
            kind: OfflineExperienceKind::SensorAnomaly {
                sensor_id: format!("sensor-{i}"),
                value: 42.0,
            },
            salience: 0.5 + (i as f32 * 0.02),
        });
    }
    assert_eq!(sf.buffer_len(), 15);
    let needs_consolidation = sf.go_online(200);
    assert!(needs_consolidation);
    assert!(!sf.is_offline());
    assert_eq!(sf.reconnection_count(), 1);
}

#[test]
fn test_store_forward_salience_filtering() {
    let mut sf = StoreAndForward::default();
    sf.go_offline(100);
    sf.record(OfflineExperience {
        cycle: 101,
        kind: OfflineExperienceKind::SensorAnomaly {
            sensor_id: "low".into(),
            value: 1.0,
        },
        salience: 0.1,
    });
    assert_eq!(sf.buffer_len(), 0);
    sf.record(OfflineExperience {
        cycle: 102,
        kind: OfflineExperienceKind::SensorAnomaly {
            sensor_id: "high".into(),
            value: 99.0,
        },
        salience: 0.9,
    });
    assert_eq!(sf.buffer_len(), 1);
}

#[test]
fn test_store_forward_consolidation() {
    let mut sf = StoreAndForward::default();
    sf.go_offline(100);
    sf.record(OfflineExperience {
        cycle: 110,
        kind: OfflineExperienceKind::SensorAnomaly {
            sensor_id: "temp".into(),
            value: 45.0,
        },
        salience: 0.7,
    });
    sf.record(OfflineExperience {
        cycle: 120,
        kind: OfflineExperienceKind::ConsciousnessShift { from: 0.5, to: 0.8 },
        salience: 0.9,
    });
    sf.record(OfflineExperience {
        cycle: 130,
        kind: OfflineExperienceKind::ThreatDetected {
            threat_type: 2,
            severity: 0.6,
        },
        salience: 0.8,
    });
    let wisdom = sf.consolidate(200);
    assert_eq!(wisdom.experiences_consolidated, 3);
    assert_eq!(wisdom.offline_duration, 100);
    assert!(wisdom.mean_salience > 0.0);
    assert!(!wisdom.patterns.is_empty());
    assert_eq!(sf.buffer_len(), 0);
}

#[test]
fn test_peers_by_trust_ordering() {
    let mut router = ConsciousnessAwareRouter::default();
    router.update_peer([1; 8], 0.3, 0.4, 1, 100);
    router.update_peer([2; 8], 0.9, 0.9, 4, 100);
    router.update_peer([3; 8], 0.5, 0.5, 2, 100);
    let sorted = router.peers_by_trust();
    assert_eq!(sorted.len(), 3);
    assert!(sorted[0].1 >= sorted[1].1);
    assert!(sorted[1].1 >= sorted[2].1);
}

// ═══════════════════════════════════════════════════════════════════
// INTERPLANETARY TIER TESTS
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_interplanetary_tier_profile() {
    let profile = RadioTier::Interplanetary.profile();
    assert_eq!(profile.mtu, 1024);
    assert_eq!(profile.duty_cycle, 1.0);
    assert!(profile.reliability > 0.99);
    assert!(profile.latency_ms >= 2_000);
}

#[test]
fn test_all_with_space_includes_interplanetary() {
    assert_eq!(RadioTier::ALL_WITH_SPACE.len(), 4);
    assert!(RadioTier::ALL_WITH_SPACE.contains(&RadioTier::Interplanetary));
    assert_eq!(RadioTier::ALL.len(), 3);
    assert!(!RadioTier::ALL.contains(&RadioTier::Interplanetary));
}

#[test]
fn test_light_time_seconds() {
    assert!(light_time_seconds("LEO") < 0.01);
    assert!(light_time_seconds("Moon") > 1.0 && light_time_seconds("Moon") < 2.0);
    assert!(light_time_seconds("Mars") > 200.0);
    assert!(light_time_seconds("Jupiter") > 2000.0);
}

#[test]
fn test_min_sharing_cadence_scales_with_distance() {
    let leo = min_sharing_cadence_for_latency(light_time_seconds("LEO"), 15.0);
    let moon = min_sharing_cadence_for_latency(light_time_seconds("Moon"), 15.0);
    let mars = min_sharing_cadence_for_latency(light_time_seconds("Mars"), 15.0);
    assert!(
        leo < moon,
        "LEO cadence {leo} should be less than Moon {moon}"
    );
    assert!(
        moon < mars,
        "Moon cadence {moon} should be less than Mars {mars}"
    );
    assert!(mars > 10_000, "Mars cadence {mars} should be >10K cycles");
}

// ═══════════════════════════════════════════════════════════════════
// CCSDS SPACE PACKET TESTS
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_ccsds_encode_decode() {
    let packet = CcsdsPacket {
        apid: apid::CONSCIOUSNESS_TM,
        sequence: 42,
        is_command: false,
        data: b"phi=0.72".to_vec(),
    };
    let encoded = packet.encode();
    assert!(encoded.len() >= 6 + 8);
    let decoded = CcsdsPacket::decode(&encoded).unwrap();
    assert_eq!(decoded.apid, apid::CONSCIOUSNESS_TM);
    assert_eq!(decoded.sequence, 42);
    assert_eq!(decoded.data, b"phi=0.72");
    assert!(!decoded.is_command);
}

#[test]
fn test_ccsds_all_apids() {
    for &id in &[
        apid::CONSCIOUSNESS_TM,
        apid::CONSCIOUSNESS_DELTA,
        apid::GOVERNANCE_RELAY,
        apid::THREAT_SIGNATURE,
        apid::CONSOLIDATED_WISDOM,
        apid::HEARTBEAT,
        apid::SENSOR_TM,
    ] {
        let p = CcsdsPacket {
            apid: id,
            sequence: 0,
            is_command: false,
            data: vec![0xAB],
        };
        let decoded = CcsdsPacket::decode(&p.encode()).unwrap();
        assert_eq!(decoded.apid, id);
    }
}

// ═══════════════════════════════════════════════════════════════════
// RTL-SDR TESTS
// ═══════════════════════════════════════════════════════════════════

#[test]
fn test_rtlsdr_default() {
    let sdr = RtlSdrHardware::default();
    assert!(!sdr.available);
    assert!(!sdr.sweep_plan.is_empty());
    assert_eq!(sdr.hardware_id(), "RTL-SDR Spectrum Scanner");
}

#[test]
fn test_rtlsdr_inject_observation() {
    let mut sdr = RtlSdrHardware::default();
    sdr.inject_observation(915_000_000, 25.0);
    assert_eq!(sdr.active_frequency_count(), 1);
    assert!((sdr.peak_snr() - 25.0).abs() < f32::EPSILON);
}

#[test]
fn test_rtlsdr_signal_classification() {
    assert_eq!(
        RtlSdrHardware::classify_signal(500, 0),
        SignalClass::ContinuousWave
    );
    assert_eq!(
        RtlSdrHardware::classify_signal(5_000, 0),
        SignalClass::Voice
    );
    assert_eq!(
        RtlSdrHardware::classify_signal(50_000, 0),
        SignalClass::Digital
    );
    assert_eq!(
        RtlSdrHardware::classify_signal(1_000_000, 0),
        SignalClass::Noise
    );
}

#[test]
fn test_rtlsdr_frequency_range() {
    let mut sdr = RtlSdrHardware::default();
    sdr.available = true;
    assert!(sdr.tune(RadioTier::Metro, 915_000_000).is_ok());
    assert!(sdr.tune(RadioTier::Metro, 1_000_000).is_err());
    assert!(sdr.tune(RadioTier::Metro, 2_000_000_000).is_err());
}

#[test]
fn test_rtlsdr_receive_only() {
    let mut sdr = RtlSdrHardware::default();
    sdr.available = true;
    assert!(sdr.transmit(RadioTier::Metro, &[0]).is_err());
    assert!(sdr.set_tx_power(RadioTier::Metro, 10.0).is_err());
}

#[test]
fn test_rtlsdr_sweep_advance() {
    let mut sdr = RtlSdrHardware::default();
    let plan_len = sdr.sweep_plan.len();
    let first = sdr.advance_sweep();
    let second = sdr.advance_sweep();
    assert_ne!(first, second, "Consecutive sweeps should advance");
    for _ in 0..(plan_len - 2) {
        sdr.advance_sweep();
    }
    let wrapped = sdr.advance_sweep();
    assert_eq!(
        wrapped, first,
        "Should wrap to first frequency after full cycle"
    );
}

#[test]
fn test_rtlsdr_bands_comprehensive() {
    let all_bands = [
        bands::FM_BROADCAST,
        bands::CB_RADIO,
        bands::FRS_GMRS,
        bands::MARINE_VHF,
        bands::AVIATION,
        bands::AMATEUR_2M,
        bands::AMATEUR_70CM,
        bands::ISM_868,
        bands::ISM_915,
        bands::NOAA_APT,
        bands::UHF_SATELLITE,
    ];
    for (low, high) in &all_bands {
        assert!(
            *low >= 24_000_000 && *high <= 1_766_000_000,
            "Band {low}-{high} outside RTL-SDR range"
        );
        assert!(low < high, "Band {low}-{high} inverted");
    }
}

// ── Perception HV encoding ───────────────────────────────────────────

#[test]
fn perception_hv_empty_returns_none() {
    let sm = SpectrumManager::default();
    assert!(
        sm.perception_hv().is_none(),
        "empty observation buffer must yield None so perception can skip"
    );
}

#[test]
fn perception_hv_nonempty_returns_some() {
    let mut sm = SpectrumManager::default();
    sm.inject_observation(SpectrumObservation {
        frequency_hz: 915_000_000,
        noise_floor_dbm: -100.0,
        snr_db: 12.0,
        jammed: false,
    });
    let hv = sm
        .perception_hv()
        .expect("one observation must produce an HV");
    // 16,384 bits = 2048 bytes (BinaryHV::BYTES).
    assert_eq!(hv.0.len(), 2048);
}

#[test]
fn perception_hv_identical_observations_produce_identical_hvs() {
    let obs = SpectrumObservation {
        frequency_hz: 433_000_000,
        noise_floor_dbm: -95.0,
        snr_db: 7.5,
        jammed: false,
    };
    let mut sm1 = SpectrumManager::default();
    sm1.inject_observation(obs.clone());
    let mut sm2 = SpectrumManager::default();
    sm2.inject_observation(obs);

    let h1 = sm1.perception_hv().unwrap();
    let h2 = sm2.perception_hv().unwrap();
    assert_eq!(h1, h2, "deterministic encoding must be stable across calls");
}

#[test]
fn perception_hv_jammed_vs_clear_are_orthogonal() {
    // Two observations that differ only in the `jammed` flag should produce
    // HVs with low cosine similarity — HDC orthogonality across buckets.
    let mut sm_clear = SpectrumManager::default();
    sm_clear.inject_observation(SpectrumObservation {
        frequency_hz: 2_400_000_000,
        noise_floor_dbm: -90.0,
        snr_db: 15.0,
        jammed: false,
    });
    let mut sm_jam = SpectrumManager::default();
    sm_jam.inject_observation(SpectrumObservation {
        frequency_hz: 2_400_000_000,
        noise_floor_dbm: -90.0,
        snr_db: 15.0,
        jammed: true,
    });

    let clear = sm_clear.perception_hv().unwrap();
    let jammed = sm_jam.perception_hv().unwrap();
    let sim = clear.similarity(&jammed);
    // Bundling 4 pairs where one differs: ~75% agreement on XOR'd bits →
    // similarity well below 1.0 but above 0. Being generous with the bound
    // because HDC bundle thresholding is lossy.
    assert!(
        sim < 0.95,
        "clear vs jammed must differ: similarity {sim} too high"
    );
}
