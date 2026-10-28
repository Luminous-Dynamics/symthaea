// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Test-only event-time model for deadline-aware fragment recovery.
//!
//! This is an analytical research model, not a production ACK protocol or radio
//! simulator. Unlike the fixed-slot baseline, it uses one exogenous 100-ms
//! good/bad-state timeline shared by strategies and charges PHY airtime, a
//! synthetic 100-ms inter-frame gap, ACK turnaround, a synthetic 1-s timeout,
//! a hypothetical 24-byte ACK PHY payload, and any retransmissions.
//!
//! A transmitted frame's error probability is the duration-weighted mean of the
//! configured good/bad loss rates during its airtime. This deliberately simple
//! model does not reproduce LoRa demodulator/SNR packet-error curves, collisions,
//! duty-cycle wait, energy, or multihop. Parameters are experimental fixtures,
//! not deployment recommendations.

use super::fragment_channel_model::lora_time_on_air_us_for_sf;
use super::{fragment, HEADER_SIZE, LORA_MTU};

const TICK_US: u64 = 100_000;
const HORIZON_US: u64 = 90_000_000;
const TRACE_US: u64 = HORIZON_US + 10_000_000;
const INTER_FRAME_GAP_US: u64 = 100_000;
const ACK_TURNAROUND_US: u64 = 100_000;
const ACK_TIMEOUT_US: u64 = 1_000_000;
// Hypothetical 8-byte outer header plus 16-byte fragment bitmap/control body.
const ACK_PHY_BYTES: usize = 24;
// Production uncompressed-envelope baseline: 1-byte compression tag + full WisdomPacket.
const BASELINE_ENVELOPE_BYTES: usize = super::WISDOM_PACKET_SIZE + 1;
// Packet AEAD adds the 12-byte nonce and 16-byte Poly1305 tag to that envelope.
const PACKET_AEAD_ENVELOPE_BYTES: usize = BASELINE_ENVELOPE_BYTES + 12 + 16;
const RAW_HV_BYTES: usize = 2048;
const HASH_BUCKETS: u64 = 1_000_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChannelModel {
    Iid {
        loss_per_mille: u16,
    },
    GilbertElliott {
        good_loss_per_mille: u16,
        bad_loss_per_mille: u16,
        good_to_bad_per_mille: u16,
        bad_to_good_per_mille: u16,
    },
}

impl ChannelModel {
    fn is_burst(self) -> bool {
        matches!(self, Self::GilbertElliott { .. })
    }

    fn loss_per_mille(self, in_bad_state: bool) -> u16 {
        match self {
            Self::Iid { loss_per_mille } => loss_per_mille,
            Self::GilbertElliott {
                good_loss_per_mille,
                bad_loss_per_mille,
                ..
            } => {
                if in_bad_state {
                    bad_loss_per_mille
                } else {
                    good_loss_per_mille
                }
            }
        }
    }

    fn transition(self, rng: &mut TraceRng, in_bad_state: &mut bool) {
        if let Self::GilbertElliott {
            good_to_bad_per_mille,
            bad_to_good_per_mille,
            ..
        } = self
        {
            if *in_bad_state {
                if rng.event(bad_to_good_per_mille) {
                    *in_bad_state = false;
                }
            } else if rng.event(good_to_bad_per_mille) {
                *in_bad_state = true;
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Strategy {
    NoRecovery,
    XorFec,
    SelectiveRetry,
    Hybrid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Outcome {
    delivered_us: Option<u64>,
    airtime_us: u64,
    bytes_sent: usize,
    frame_attempts: u32,
    ack_attempts: u32,
    retry_frames: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TraceRng(u64);

impl TraceRng {
    fn new(seed: u64) -> Self {
        Self(if seed == 0 {
            0x9E37_79B9_7F4A_7C15
        } else {
            seed
        })
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    fn event(&mut self, per_mille: u16) -> bool {
        debug_assert!(per_mille <= 1000);
        self.next_u64() % 1000 < u64::from(per_mille)
    }
}

fn make_state_trace(model: ChannelModel, seed: u64) -> Vec<bool> {
    let bins = (TRACE_US / TICK_US + 1) as usize;
    let mut trace = Vec::with_capacity(bins);
    let mut rng = TraceRng::new(seed);
    let mut in_bad_state = false;
    for _ in 0..bins {
        trace.push(in_bad_state);
        model.transition(&mut rng, &mut in_bad_state);
    }
    trace
}

fn mix64(mut value: u64) -> u64 {
    value = value.wrapping_add(0x9E37_79B9_7F4A_7C15);
    value = (value ^ (value >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    value = (value ^ (value >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    value ^ (value >> 31)
}

fn event_delivered(seed: u64, start_us: u64, is_ack: bool, loss_threshold_ppm: u64) -> bool {
    let direction = if is_ack {
        0xD1B5_4A32_D192_ED03
    } else {
        0x94D0_49BB_1331_11EB
    };
    let random_bucket = mix64(
        seed ^ start_us.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ direction,
    ) % HASH_BUCKETS;
    random_bucket >= loss_threshold_ppm
}

fn loss_threshold_ppm(
    model: ChannelModel,
    states: &[bool],
    start_us: u64,
    duration_us: u64,
    is_ack: bool,
    ack_loss_override_per_mille: Option<u16>,
) -> u64 {
    if is_ack {
        if let Some(ack_loss) = ack_loss_override_per_mille {
            return u64::from(ack_loss) * 1000;
        }
    }

    if !model.is_burst() {
        return u64::from(model.loss_per_mille(false)) * 1000;
    }

    let end_us = start_us.saturating_add(duration_us);
    let mut tick = start_us / TICK_US;
    let mut weighted_loss = 0u128;
    let mut covered_us = 0u128;
    while tick.saturating_mul(TICK_US) < end_us {
        let tick_start = tick * TICK_US;
        let tick_end = tick_start.saturating_add(TICK_US);
        let overlap_start = start_us.max(tick_start);
        let overlap_end = end_us.min(tick_end);
        if overlap_end > overlap_start {
            let state_index = (tick as usize).min(states.len().saturating_sub(1));
            let bad = states.get(state_index).copied().unwrap_or(false);
            let loss = model.loss_per_mille(bad);
            let overlap = u128::from(overlap_end - overlap_start);
            weighted_loss += u128::from(loss) * overlap;
            covered_us += overlap;
        }
        tick += 1;
    }

    if covered_us == 0 {
        u64::from(model.loss_per_mille(false)) * 1000
    } else {
        // Convert averaged per-mille probability to parts per million, without
        // quantizing the weighted good/bad exposure to a whole per-mille.
        ((weighted_loss * 1000) / covered_us).min(u128::from(HASH_BUCKETS)) as u64
    }
}


fn encode_gap_varints(indices: &[usize]) -> Vec<u8> {
    let mut encoded = Vec::new();
    let mut previous_exclusive = 0usize;
    for &index in indices {
        assert!(index >= previous_exclusive);
        let mut gap = index - previous_exclusive;
        loop {
            let low = (gap & 0x7f) as u8;
            gap >>= 7;
            if gap == 0 {
                encoded.push(low);
                break;
            }
            encoded.push(low | 0x80);
        }
        previous_exclusive = index + 1;
    }
    encoded
}

/// Strict test codec for a proposed delta-index format. Gaps are bounded by the
/// 16,384-bit vector and non-canonical, truncated, overflowing or count-mismatched
/// input is rejected. This is research code, not a deployed wire decoder.
fn decode_gap_varints(encoded: &[u8], expected_count: usize) -> Option<Vec<usize>> {
    const HV_BITS: usize = 16_384;
    let mut decoded = Vec::with_capacity(expected_count.min(HV_BITS));
    let mut previous_exclusive = 0usize;
    let mut cursor = 0usize;

    while cursor < encoded.len() {
        if decoded.len() >= expected_count {
            return None;
        }
        let mut value = 0usize;
        let mut shift = 0u32;
        loop {
            if cursor >= encoded.len() || shift >= 14 {
                return None;
            }
            let byte = encoded[cursor];
            cursor += 1;
            let low = usize::from(byte & 0x7f);
            value |= low << shift;
            if byte & 0x80 == 0 {
                if shift > 0 && low == 0 {
                    return None;
                }
                break;
            }
            shift += 7;
        }
        let index = previous_exclusive.checked_add(value)?;
        if index >= HV_BITS {
            return None;
        }
        decoded.push(index);
        previous_exclusive = index.checked_add(1)?;
    }

    (decoded.len() == expected_count).then_some(decoded)
}

fn deterministic_flip_positions(hamming_bits: usize, seed: u64) -> Vec<usize> {
    const HV_BITS: usize = 16_384;
    assert!(hamming_bits <= HV_BITS);
    let mut positions: Vec<usize> = (0..HV_BITS).collect();
    let mut rng = TraceRng::new(seed);
    for i in (1..positions.len()).rev() {
        let j = (rng.next_u64() % (i as u64 + 1)) as usize;
        positions.swap(i, j);
    }
    positions.truncate(hamming_bits);
    positions.sort_unstable();
    positions
}

fn frame_lengths_for_bytes(payload_len: usize) -> Vec<usize> {
    let payload = vec![0xA5; payload_len];
    let frames = fragment(0xC0DE, &payload);
    frames
        .iter()
        .map(|frame| HEADER_SIZE + usize::from(frame.payload_len))
        .collect()
}

fn frame_lengths() -> Vec<usize> {
    frame_lengths_for_bytes(BASELINE_ENVELOPE_BYTES)
}

struct EventSimulation<'a> {
    model: ChannelModel,
    states: &'a [bool],
    seed: u64,
    spreading_factor: u8,
    ack_loss_override_per_mille: Option<u16>,
    now_us: u64,
    airtime_us: u64,
    bytes_sent: usize,
    frame_attempts: u32,
    ack_attempts: u32,
    retry_frames: u32,
}

impl<'a> EventSimulation<'a> {
    fn new(
        model: ChannelModel,
        states: &'a [bool],
        seed: u64,
        spreading_factor: u8,
        ack_loss_override_per_mille: Option<u16>,
    ) -> Self {
        Self {
            model,
            states,
            seed,
            spreading_factor,
            ack_loss_override_per_mille,
            now_us: 0,
            airtime_us: 0,
            bytes_sent: 0,
            frame_attempts: 0,
            ack_attempts: 0,
            retry_frames: 0,
        }
    }

    fn send(
        &mut self,
        phy_payload_len: usize,
        is_ack: bool,
        is_retry: bool,
    ) -> (bool, Option<u64>) {
        if self.now_us >= HORIZON_US {
            return (false, None);
        }
        let start_us = self.now_us;
        let toa_us = lora_time_on_air_us_for_sf(phy_payload_len, self.spreading_factor);
        let threshold = loss_threshold_ppm(
            self.model,
            self.states,
            start_us,
            toa_us,
            is_ack,
            self.ack_loss_override_per_mille,
        );
        let delivered = event_delivered(self.seed, start_us, is_ack, threshold);
        let end_us = start_us.saturating_add(toa_us);
        self.now_us = end_us.saturating_add(INTER_FRAME_GAP_US);
        self.airtime_us = self.airtime_us.saturating_add(toa_us);
        self.bytes_sent = self.bytes_sent.saturating_add(phy_payload_len);
        if is_ack {
            self.ack_attempts = self.ack_attempts.saturating_add(1);
        } else {
            self.frame_attempts = self.frame_attempts.saturating_add(1);
            if is_retry {
                self.retry_frames = self.retry_frames.saturating_add(1);
            }
        }
        (delivered, Some(end_us))
    }

    fn wait_until(&mut self, requested_us: u64) {
        self.now_us = self.now_us.max(requested_us);
    }

    fn outcome(&self, delivered_us: Option<u64>) -> Outcome {
        Outcome {
            delivered_us,
            airtime_us: self.airtime_us,
            bytes_sent: self.bytes_sent,
            frame_attempts: self.frame_attempts,
            ack_attempts: self.ack_attempts,
            retry_frames: self.retry_frames,
        }
    }
}

fn complete_time(
    data_received_us: &[Option<u64>],
    parity_end_us: Option<u64>,
    parity_received: bool,
) -> Option<u64> {
    if data_received_us.iter().all(|item| item.is_some()) {
        return data_received_us.iter().flatten().copied().max();
    }
    let missing = data_received_us.iter().filter(|item| item.is_none()).count();
    if missing == 1 && parity_received {
        return parity_end_us;
    }
    None
}

fn simulate(
    strategy: Strategy,
    model: ChannelModel,
    states: &[bool],
    seed: u64,
    spreading_factor: u8,
    ack_loss_override_per_mille: Option<u16>,
) -> Outcome {
    let frame_sizes = frame_lengths();
    let data_count = frame_sizes.len() - 1;
    assert!(frame_sizes.iter().all(|size| *size <= LORA_MTU));

    let mut sim = EventSimulation::new(
        model,
        states,
        seed,
        spreading_factor,
        ack_loss_override_per_mille,
    );
    let mut received_us = vec![None; data_count];
    let mut parity_end_us = None;
    let mut parity_received = false;

    for (index, frame_size) in frame_sizes[..data_count].iter().copied().enumerate() {
        let (received, end_us) = sim.send(frame_size, false, false);
        if received {
            received_us[index] = end_us;
        }
    }

    if matches!(strategy, Strategy::XorFec | Strategy::Hybrid) {
        (parity_received, parity_end_us) = sim.send(frame_sizes[data_count], false, false);
    }

    let mut delivered_us = complete_time(&received_us, parity_end_us, parity_received);
    if matches!(strategy, Strategy::NoRecovery | Strategy::XorFec) {
        return sim.outcome(delivered_us);
    }

    // The sender's ACK turnaround starts after its last attempted initial frame,
    // whether or not that frame reached the receiver.
    let last_end_us = sim.now_us.saturating_sub(INTER_FRAME_GAP_US);
    sim.wait_until(last_end_us.saturating_add(ACK_TURNAROUND_US));
    let (ack_received, ack_end_us) = sim.send(ACK_PHY_BYTES, true, false);

    let retry_indexes: Vec<usize> = if !ack_received {
        if let Some(ack_end) = ack_end_us {
            sim.wait_until(ack_end.saturating_add(ACK_TIMEOUT_US));
        }
        // Without a valid bitmap, perform one bounded blind fallback round.
        (0..data_count).collect()
    } else if strategy == Strategy::Hybrid && delivered_us.is_some() {
        // The bitmap can report a message already reconstructed from XOR parity.
        Vec::new()
    } else {
        received_us
            .iter()
            .enumerate()
            .filter_map(|(index, received)| if received.is_none() { Some(index) } else { None })
            .collect()
    };

    for index in retry_indexes {
        let (received, end_us) = sim.send(frame_sizes[index], false, true);
        if received && received_us[index].is_none() {
            received_us[index] = end_us;
        }
        if delivered_us.is_none() {
            delivered_us = complete_time(&received_us, parity_end_us, parity_received);
        }
    }

    sim.outcome(delivered_us)
}

fn by_deadline(outcome: Outcome, deadline_us: u64) -> bool {
    outcome.delivered_us.is_some_and(|delivered| delivered <= deadline_us)
}


#[test]
fn wire_profile_airtime_accounts_for_vector_envelope_and_packet_aead() {
    let raw_hv = frame_lengths_for_bytes(RAW_HV_BYTES);
    let plain_envelope = frame_lengths_for_bytes(BASELINE_ENVELOPE_BYTES);
    let packet_aead = frame_lengths_for_bytes(PACKET_AEAD_ENVELOPE_BYTES);

    assert_eq!(RAW_HV_BYTES, 2048);
    assert_eq!(BASELINE_ENVELOPE_BYTES, 2105);
    assert_eq!(PACKET_AEAD_ENVELOPE_BYTES, 2133);
    for sizes in [&raw_hv, &plain_envelope, &packet_aead] {
        assert_eq!(sizes.len(), 11);
        assert_eq!(sizes.iter().filter(|size| **size == LORA_MTU).count(), 10);
        assert_eq!(sizes[10], LORA_MTU, "XOR parity occupies a full MTU");
        assert!(sizes.iter().all(|size| *size <= LORA_MTU));
    }
    assert_eq!(raw_hv[9], 130);
    assert_eq!(plain_envelope[9], 187);
    assert_eq!(packet_aead[9], 215);

    // Analytical PHY reference sums, not measured radio completion latency.
    let expected = [
        (7, 3_351_040, 3_432_960, 3_473_920),
        (9, 10_659_840, 10_905_600, 11_028_480),
        (12, 77_250_560, 79_216_640, 80_035_840),
    ];
    for (sf, raw_expected, plain_expected, aead_expected) in expected {
        let sum = |sizes: &[usize]| -> u64 {
            sizes[..10]
                .iter()
                .map(|size| lora_time_on_air_us_for_sf(*size, sf))
                .sum()
        };
        let raw_airtime = sum(&raw_hv);
        let plain_airtime = sum(&plain_envelope);
        let aead_airtime = sum(&packet_aead);
        assert_eq!(raw_airtime, raw_expected, "raw-HV-only SF{sf}");
        assert_eq!(plain_airtime, plain_expected, "uncompressed envelope SF{sf}");
        assert_eq!(aead_airtime, aead_expected, "packet-AEAD envelope SF{sf}");
        assert!(raw_airtime < plain_airtime && plain_airtime < aead_airtime);
        eprintln!(
            "mesh-wire-footprint profile=raw_hv/envelope/packet_aead sf={} data_airtime_us={}/{}/{} bytes={}/{}/{} frames=10 parity_frame=222",
            sf,
            raw_airtime,
            plain_airtime,
            aead_airtime,
            RAW_HV_BYTES,
            BASELINE_ENVELOPE_BYTES,
            PACKET_AEAD_ENVELOPE_BYTES,
        );
    }
}

#[test]
fn deadline_model_uses_fragment_lengths_and_distinguishes_deadlines() {
    let sizes = frame_lengths();
    assert_eq!(sizes.len(), 11);
    assert_eq!(sizes.iter().filter(|size| **size == LORA_MTU).count(), 10);
    assert_eq!(sizes.iter().filter(|size| **size == 187).count(), 1);
    assert!(sizes.iter().all(|size| *size <= LORA_MTU));

    let model = ChannelModel::Iid { loss_per_mille: 0 };
    let trace = make_state_trace(model, 0x5EED_1234);
    let sf7 = simulate(Strategy::NoRecovery, model, &trace, 11, 7, Some(0));
    let sf12 = simulate(Strategy::NoRecovery, model, &trace, 11, 12, Some(0));
    let sf7_done = sf7.delivered_us.expect("no-loss SF7 must complete");
    let sf12_done = sf12.delivered_us.expect("no-loss SF12 must complete");

    assert!(by_deadline(sf7, 5_000_000));
    assert!(!by_deadline(sf12, 30_000_000));
    assert!(by_deadline(sf12, 90_000_000));
    assert!(sf12_done > sf7_done);
}

#[test]
fn ack_loss_is_deterministic_and_charged_as_airtime_and_retransmissions() {
    let model = ChannelModel::Iid { loss_per_mille: 100 };
    let mut selective_airtime_ack0 = 0u64;
    let mut selective_airtime_ack250 = 0u64;
    let mut selective_retry_ack0 = 0u64;
    let mut selective_retry_ack250 = 0u64;

    for trial in 0..512u64 {
        let seed = 0x4D45_5348_2026_1028u64 ^ trial.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let trace = make_state_trace(model, seed ^ 0xBADC_0FFE);
        let ack0 = simulate(Strategy::SelectiveRetry, model, &trace, seed, 7, Some(0));
        let ack0_repeat = simulate(Strategy::SelectiveRetry, model, &trace, seed, 7, Some(0));
        let ack250 = simulate(Strategy::SelectiveRetry, model, &trace, seed, 7, Some(250));
        assert_eq!(ack0, ack0_repeat);

        let no_ack0 = simulate(Strategy::NoRecovery, model, &trace, seed, 7, Some(0));
        let no_ack250 = simulate(Strategy::NoRecovery, model, &trace, seed, 7, Some(250));
        assert_eq!(no_ack0, no_ack250);

        let xor_ack0 = simulate(Strategy::XorFec, model, &trace, seed, 7, Some(0));
        let xor_ack250 = simulate(Strategy::XorFec, model, &trace, seed, 7, Some(250));
        assert_eq!(xor_ack0, xor_ack250);

        selective_airtime_ack0 += ack0.airtime_us;
        selective_airtime_ack250 += ack250.airtime_us;
        selective_retry_ack0 += u64::from(ack0.retry_frames);
        selective_retry_ack250 += u64::from(ack250.retry_frames);
    }

    assert!(selective_airtime_ack250 > selective_airtime_ack0);
    assert!(selective_retry_ack250 > selective_retry_ack0);
}

#[test]
fn burst_trace_is_reproducible_and_exercises_both_channel_states() {
    let model = ChannelModel::GilbertElliott {
        good_loss_per_mille: 20,
        bad_loss_per_mille: 750,
        good_to_bad_per_mille: 100,
        bad_to_good_per_mille: 500,
    };
    let first = make_state_trace(model, 0xBADC_0FFE);
    let second = make_state_trace(model, 0xBADC_0FFE);
    assert_eq!(first, second);
    assert!(first.iter().any(|bad| *bad), "trace should enter the bad state");
    assert!(first.iter().any(|bad| !*bad), "trace should include the good state");

    let seed = 0x1234_5678;
    let outcome_a = simulate(Strategy::Hybrid, model, &first, seed, 7, Some(100));
    let outcome_b = simulate(Strategy::Hybrid, model, &second, seed, 7, Some(100));
    assert_eq!(outcome_a, outcome_b);
}

#[test]
fn impossible_loss_probability_rejects_every_frame() {
    assert!(!event_delivered(123, 0, false, HASH_BUCKETS));
    assert!(!event_delivered(123, 500_000, true, HASH_BUCKETS));
}

#[test]
fn fixed_seed_cohort_reports_delivery_by_deadline_for_all_strategies() {
    let model = ChannelModel::Iid {
        loss_per_mille: 100,
    };
    let strategies = [
        Strategy::NoRecovery,
        Strategy::XorFec,
        Strategy::SelectiveRetry,
        Strategy::Hybrid,
    ];
    let deadlines_us = [5_000_000, 15_000_000, 30_000_000, 90_000_000];
    let mut successes = [[0u32; 4]; 4];
    let mut airtime_totals = [0u64; 4];
    const TRIALS: u64 = 512;

    for trial in 0..TRIALS {
        let seed = 0x4D45_5348_2026_1028u64
            ^ trial.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let trace = make_state_trace(model, seed ^ 0xBADC_0FFE);
        let outcomes = [
            simulate(Strategy::NoRecovery, model, &trace, seed, 7, Some(100)),
            simulate(Strategy::XorFec, model, &trace, seed, 7, Some(100)),
            simulate(Strategy::SelectiveRetry, model, &trace, seed, 7, Some(100)),
            simulate(Strategy::Hybrid, model, &trace, seed, 7, Some(100)),
        ];

        for (strategy_index, outcome) in outcomes.iter().copied().enumerate() {
            airtime_totals[strategy_index] =
                airtime_totals[strategy_index].saturating_add(outcome.airtime_us);
            for (deadline_index, deadline_us) in deadlines_us.iter().copied().enumerate() {
                if by_deadline(outcome, deadline_us) {
                    successes[strategy_index][deadline_index] += 1;
                }
            }
        }
    }

    for strategy_index in 0..strategies.len() {
        for deadline_index in 1..deadlines_us.len() {
            assert!(
                successes[strategy_index][deadline_index - 1]
                    <= successes[strategy_index][deadline_index],
                "delivery count cannot decrease when the deadline increases"
            );
        }
    }
    for deadline_index in 0..deadlines_us.len() {
        assert!(
            successes[2][deadline_index] >= successes[0][deadline_index],
            "selective retry preserves every no-recovery delivery by deadline"
        );
        assert!(
            successes[3][deadline_index] >= successes[1][deadline_index],
            "hybrid preserves every XOR-FEC delivery by deadline"
        );
    }
    assert!(airtime_totals[2] >= airtime_totals[0]);
    assert!(airtime_totals[3] >= airtime_totals[1]);

    eprintln!(
        "mesh-deadline-feedback channel=iid100 ack_loss=100‰ sf=7 trials={} delivery_by_deadline_5/15/30/90s_no_fec={}/{}/{}/{} xor={}/{}/{}/{} selective={}/{}/{}/{} hybrid={}/{}/{}/{} mean_phy_airtime_us_no_fec/xor/selective/hybrid={}/{}/{}/{}",
        TRIALS,
        successes[0][0], successes[0][1], successes[0][2], successes[0][3],
        successes[1][0], successes[1][1], successes[1][2], successes[1][3],
        successes[2][0], successes[2][1], successes[2][2], successes[2][3],
        successes[3][0], successes[3][1], successes[3][2], successes[3][3],
        airtime_totals[0] / TRIALS,
        airtime_totals[1] / TRIALS,
        airtime_totals[2] / TRIALS,
        airtime_totals[3] / TRIALS,
    );
}

fn pseudo_random_packet() -> [u8; super::WISDOM_PACKET_SIZE] {
    let mut raw = [0u8; super::WISDOM_PACKET_SIZE];
    let mut state = 0x4D45_5348_2026_1028u64;
    for byte in &mut raw {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = (state >> 24) as u8;
    }
    raw
}

#[test]
fn current_compression_path_keeps_random_like_packet_envelope_exact_size() {
    let raw = pseudo_random_packet();
    let envelope = super::compress_packet(&raw);
    assert_eq!(
        envelope.first().copied(),
        Some(super::COMPRESS_NONE),
        "random-like packet should take the non-expanding fallback"
    );
    assert_eq!(envelope.len(), BASELINE_ENVELOPE_BYTES);
}

#[cfg(feature = "mesh-encryption")]
#[test]
fn packet_aead_adds_nonce_and_tag_to_current_envelope() {
    let raw = pseudo_random_packet();
    let envelope = super::compress_packet(&raw);
    let encrypted = super::encrypt_packet_typed(
        &envelope,
        &[0x42; 32],
        &[1, 2, 3, 4, 5, 6, 7, 8],
        0,
        1,
        1,
    );
    assert_eq!(
        encrypted.len(),
        envelope.len() + 12 + 16,
        "packet AEAD prepends a 12-byte nonce and a 16-byte tag"
    );
    assert_eq!(encrypted.len(), PACKET_AEAD_ENVELOPE_BYTES);
}

#[test]
fn gap_coded_delta_reconstructs_exact_hv_and_reports_airtime_break_even() {
    const HV_BITS: usize = 16_384;
    const HV_BYTES: usize = HV_BITS / 8;
    const DELTA_FIXED_BYTES: usize = 95;
    let full_plain_airtime = frame_lengths_for_bytes(BASELINE_ENVELOPE_BYTES)
        .iter()
        .map(|size| lora_time_on_air_us_for_sf(*size, 7))
        .sum::<u64>();
    let full_aead_airtime = frame_lengths_for_bytes(PACKET_AEAD_ENVELOPE_BYTES)
        .iter()
        .map(|size| lora_time_on_air_us_for_sf(*size, 7))
        .sum::<u64>();
    let flip_rates = [0usize, 16, 82, 164, 327, 819, 1311, 1638, 2048, 2458, 3277, 8192, 16_384];

    for (sample, flips) in flip_rates.into_iter().enumerate() {
        let positions = deterministic_flip_positions(flips, 0xD311_A000 + sample as u64);
        let encoded = encode_gap_varints(&positions);
        let decoded = decode_gap_varints(&encoded, positions.len())
            .expect("valid encoded gap list must decode");
        assert_eq!(decoded, positions);

        let mut rng = TraceRng::new(0x4241_5345 + sample as u64);
        let mut base = vec![0u8; HV_BYTES];
        for byte in &mut base {
            *byte = rng.next_u64() as u8;
        }
        let mut target = base.clone();
        for index in &positions {
            target[index / 8] ^= 1u8 << (index % 8);
        }
        let mut reconstructed = base;
        for index in &decoded {
            reconstructed[index / 8] ^= 1u8 << (index % 8);
        }
        assert_eq!(reconstructed, target, "Hamming flips={flips}");

        let delta_bytes = DELTA_FIXED_BYTES + encoded.len();
        let delta_airtime = frame_lengths_for_bytes(delta_bytes)
            .iter()
            .map(|size| lora_time_on_air_us_for_sf(*size, 7))
            .sum::<u64>();
        let encrypted_delta_bytes = delta_bytes + 12 + 16;
        let encrypted_delta_airtime = frame_lengths_for_bytes(encrypted_delta_bytes)
            .iter()
            .map(|size| lora_time_on_air_us_for_sf(*size, 7))
            .sum::<u64>();
        eprintln!(
            "mesh-hdc-delta sf=7 flips={} hamming_pct_milli={} delta_bytes={} delta_airtime_us={} full_airtime_us={} encrypted_delta_airtime_us={} encrypted_full_airtime_us={}",
            flips,
            flips * 1_000_000 / HV_BITS,
            delta_bytes,
            delta_airtime,
            full_plain_airtime,
            encrypted_delta_airtime,
            full_aead_airtime,
        );
        if flips <= 819 {
            assert!(delta_airtime < full_plain_airtime);
        }
        if flips >= 2048 {
            assert!(delta_airtime > full_plain_airtime);
        }
    }
}

#[test]
fn delta_gap_decoder_rejects_malformed_and_noncanonical_inputs() {
    assert_eq!(decode_gap_varints(&[], 0), Some(Vec::new()));
    assert_eq!(decode_gap_varints(&[0x00], 1), Some(vec![0]));
    assert!(decode_gap_varints(&[0x80], 1).is_none());
    assert!(decode_gap_varints(&[0x80, 0x00], 1).is_none());
    assert!(decode_gap_varints(&[0x80, 0x80, 0x01], 1).is_none());
    assert!(decode_gap_varints(&[0x00], 0).is_none());
    assert!(decode_gap_varints(&[0xff, 0x7f], 1).is_some());
    assert!(decode_gap_varints(&[0x80, 0x7f], 1).is_none(), "decoded index out of range");
}
