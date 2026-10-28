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
const PAYLOAD_BYTES: usize = 2048;
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
        ((weighted_loss * 1000) / covered_us).min(u128::from(HASH_BUCKETS - 1)) as u64
    }
}

fn frame_lengths() -> Vec<usize> {
    let payload = vec![0xA5; PAYLOAD_BYTES];
    let frames = fragment(0xC0DE, &payload);
    frames
        .iter()
        .map(|frame| HEADER_SIZE + usize::from(frame.payload_len))
        .collect()
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
    if data_received_us.iter().all(Option::is_some) {
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
fn deadline_model_uses_fragment_lengths_and_distinguishes_deadlines() {
    let sizes = frame_lengths();
    assert_eq!(sizes.len(), 11);
    assert_eq!(sizes.iter().filter(|size| **size == LORA_MTU).count(), 10);
    assert_eq!(sizes.iter().filter(|size| **size == 130).count(), 1);
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
