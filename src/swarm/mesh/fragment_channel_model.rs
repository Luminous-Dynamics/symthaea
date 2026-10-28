// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, test-only LoRa fragment channel model.
//!
//! This module is a research baseline, not a radio simulator. It applies
//! seeded IID or two-state burst loss to the repository's actual fragment
//! output and XOR-FEC assembler, then compares no recovery, XOR-FEC, one-round
//! selective retry, and hybrid FEC+retry.
//!
//! Retry strategies assume perfect aggregated feedback and one retry per
//! initially lost data fragment. ACK loss, contention, hardware energy, and
//! multi-hop effects are intentionally not modelled. PHY airtime uses an
//! analytical SX127x equation under a frozen profile; it is not measured
//! latency or energy. ACK airtime, startup, receive windows, regulatory
//! scheduling and collision effects remain excluded.
//! The retry results are optimistic upper bounds, not claims about runtime
//! behavior.

use super::{fragment, FragmentAssembler, HEADER_SIZE, LORA_MTU, PAYLOAD_SIZE};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum LossModel {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Summary {
    trials: usize,
    no_fec_successes: usize,
    xor_fec_successes: usize,
    selective_retry_successes: usize,
    hybrid_successes: usize,
    xor_fec_recoveries: usize,
    hybrid_fec_recoveries: usize,
    xor_fec_incorrect_outputs: usize,
    hybrid_incorrect_outputs: usize,
    no_fec_frames_per_message: usize,
    xor_fec_frames_per_message: usize,
    selective_retry_frames_per_message_milli: usize,
    hybrid_frames_per_message_milli: usize,
    no_fec_wire_bytes_per_message: usize,
    xor_fec_wire_bytes_per_message: usize,
    selective_retry_wire_bytes_per_message_milli: usize,
    hybrid_wire_bytes_per_message_milli: usize,
    /// Mean PHY time-on-air for data frames alone, excluding feedback and gaps.
    no_fec_airtime_us_per_message: u64,
    /// Mean PHY time-on-air for data plus XOR parity.
    xor_fec_airtime_us_per_message: u64,
    /// Mean PHY time-on-air including modeled selective retries.
    selective_retry_airtime_us_per_message: u64,
    /// Mean PHY time-on-air including XOR parity and modeled retries.
    hybrid_airtime_us_per_message: u64,
}

impl Summary {
    fn delivery_per_mille(self, successes: usize) -> usize {
        successes * 1000 / self.trials
    }

    /// Wilson score interval for a binomial delivery rate. This describes the
    /// synthetic model only, not a confidence interval for field-radio truth.
    fn wilson_95(self, successes: usize) -> (f64, f64) {
        let n = self.trials as f64;
        let p = successes as f64 / n;
        let z = 1.959_963_984_540_054_f64;
        let z2 = z * z;
        let denominator = 1.0 + z2 / n;
        let center = (p + z2 / (2.0 * n)) / denominator;
        let margin = z * ((p * (1.0 - p) / n + z2 / (4.0 * n * n)).sqrt()) / denominator;
        ((center - margin).max(0.0), (center + margin).min(1.0))
    }
}

#[derive(Debug, Clone, Copy)]
struct DeterministicRng(u64);

impl DeterministicRng {
    fn new(seed: u64) -> Self {
        // Xorshift's all-zero state is absorbing; map it to a fixed nonzero seed.
        Self(if seed == 0 { 0x9E37_79B9_7F4A_7C15 } else { seed })
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

fn frame_delivered(
    model: LossModel,
    rng: &mut DeterministicRng,
    in_bad_state: &mut bool,
) -> bool {
    let loss_rate = match model {
        LossModel::Iid { loss_per_mille } => loss_per_mille,
        LossModel::GilbertElliott {
            good_loss_per_mille,
            bad_loss_per_mille,
            ..
        } => {
            if *in_bad_state {
                bad_loss_per_mille
            } else {
                good_loss_per_mille
            }
        }
    };

    let delivered = !rng.event(loss_rate);
    if let LossModel::GilbertElliott {
        good_to_bad_per_mille,
        bad_to_good_per_mille,
        ..
    } = model
    {
        if *in_bad_state {
            if rng.event(bad_to_good_per_mille) {
                *in_bad_state = false;
            }
        } else if rng.event(good_to_bad_per_mille) {
            *in_bad_state = true;
        }
    }
    delivered
}

fn wire_len(payload_len: u8) -> usize {
    HEADER_SIZE + payload_len as usize
}

/// Frozen PHY parameters for the SX127x Semtech time-on-air equation.
/// coding_rate follows the datasheet encoding: 1 = 4/5, ... 4 = 4/8.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct LoRaPhyProfile {
    spreading_factor: u8,
    bandwidth_hz: u32,
    coding_rate: u8,
    preamble_symbols: u16,
    explicit_header: bool,
    crc_enabled: bool,
}

const DEFAULT_LORA_PHY: LoRaPhyProfile = LoRaPhyProfile {
    spreading_factor: 7,
    bandwidth_hz: 125_000,
    coding_rate: 1,
    preamble_symbols: 8,
    explicit_header: true,
    crc_enabled: true,
};

/// Calculate PHY airtime in microseconds using Semtech SX1276/77/78/79 DS Rev. 7,
/// section 4.1.1.7. This is a PHY model only: ACKs, turnaround, contention,
/// radio startup, receive windows, and inter-frame scheduling are not included.
fn lora_time_on_air_us(payload_len: usize, profile: LoRaPhyProfile) -> u64 {
    assert!(
        (1..=255).contains(&payload_len),
        "LoRa PHY payload must be 1..=255 bytes"
    );
    assert!(
        (7..=12).contains(&profile.spreading_factor),
        "explicit-header profile requires SF7..=SF12"
    );
    assert!(profile.bandwidth_hz > 0, "bandwidth must be nonzero");
    assert!((1..=4).contains(&profile.coding_rate), "coding rate must be 1..=4");

    let sf = u32::from(profile.spreading_factor);
    let bandwidth_hz = u64::from(profile.bandwidth_hz);
    let two_to_sf = 1u64 << sf;
    // Semtech requires low-data-rate optimization when symbol duration > 16 ms.
    let de = if two_to_sf * 1_000_000 > bandwidth_hz * 16_000 {
        1u32
    } else {
        0u32
    };
    let implicit_header: i64 = if profile.explicit_header { 0 } else { 1 };
    let crc: i64 = if profile.crc_enabled { 1 } else { 0 };
    let numerator = 8 * payload_len as i64 - 4 * i64::from(sf) + 28 + 16 * crc
        - 20 * implicit_header;
    let denominator = 4 * (i64::from(sf) - 2 * i64::from(de));
    let ceil_blocks = if numerator > 0 {
        (numerator + denominator - 1) / denominator
    } else {
        0
    };
    let payload_symbols = 8 + ceil_blocks * (i64::from(profile.coding_rate) + 4);
    let quarter_symbol_count = 4 * u64::from(profile.preamble_symbols) + 17
        + 4 * payload_symbols as u64;
    // Exact quarter-symbol arithmetic, rounded upward to a microsecond.
    let numerator_us = u128::from(quarter_symbol_count) * u128::from(two_to_sf) * 1_000_000;
    let denominator_us = 4 * u128::from(profile.bandwidth_hz);
    ((numerator_us + denominator_us - 1) / denominator_us) as u64
}

/// Run four packet-delivery strategies against one exogenous time-slot trace:
///
/// - no FEC/retry: success only if all initial data fragments arrive;
/// - XOR-FEC: existing parity/assembler behavior, without retries;
/// - selective retry: one retry for each initially missing data fragment;
/// - hybrid: send parity and retry only when initial FEC cannot reconstruct.
///
/// The channel's Gilbert-Elliott state evolves on every time slot, regardless
/// of whether a strategy transmits in that slot. This avoids consuming a fake
/// parity-frame outcome in the no-parity retry strategy. The schedules are:
/// initial data at slots 0..N-1; XOR/hybrid parity at N; selective-retry feedback
/// turnaround at N; hybrid feedback turnaround at N+1; retries then consume
/// slots in actual send order.
///
/// Every paired trial reserves a fixed 2N+2-slot window. If a strategy finishes
/// before the window ends, the remaining slots are idle so every strategy's
/// next trial starts at the same exogenous channel state. This supports paired
/// per-message comparison, but is not a back-to-back throughput simulation;
/// continuous-operation throughput needs a separate event-driven workload model.
///
/// Retry strategies assume ideal feedback and at most one retry per missing
/// data frame; ACK bytes/loss are not modelled.
fn run_benchmark(
    payload: &[u8],
    model: LossModel,
    seed: u64,
    trials: usize,
) -> Summary {
    run_benchmark_with_phy(payload, model, seed, trials, DEFAULT_LORA_PHY)
}

fn run_benchmark_with_phy(
    payload: &[u8],
    model: LossModel,
    seed: u64,
    trials: usize,
    phy: LoRaPhyProfile,
) -> Summary {
    assert!(!payload.is_empty(), "benchmark payload must not be empty");
    assert!(trials > 0, "benchmark requires at least one trial");
    let frames = fragment(0xC0DE, payload);
    let data_count = frames.len() - 1;
    let fec_index = data_count;
    // Worst case is N data sends + parity slot + feedback turnaround + N retries.
    let trace_slots = 2 * data_count + 2;

    let no_fec_wire_bytes_per_message: usize = frames[..data_count]
        .iter()
        .map(|frame| wire_len(frame.payload_len))
        .sum();
    let xor_fec_wire_bytes_per_message: usize = frames
        .iter()
        .map(|frame| wire_len(frame.payload_len))
        .sum();
    let no_fec_airtime_us_per_message: u64 = frames[..data_count]
        .iter()
        .map(|frame| lora_time_on_air_us(wire_len(frame.payload_len), phy))
        .sum();
    let xor_fec_airtime_us_per_message: u64 = frames
        .iter()
        .map(|frame| lora_time_on_air_us(wire_len(frame.payload_len), phy))
        .sum();

    assert!(frames
        .iter()
        .all(|frame| wire_len(frame.payload_len) <= LORA_MTU));

    let mut rng = DeterministicRng::new(seed);
    let mut in_bad_state = false;
    let mut summary = Summary {
        trials,
        no_fec_successes: 0,
        xor_fec_successes: 0,
        selective_retry_successes: 0,
        hybrid_successes: 0,
        xor_fec_recoveries: 0,
        hybrid_fec_recoveries: 0,
        xor_fec_incorrect_outputs: 0,
        hybrid_incorrect_outputs: 0,
        no_fec_frames_per_message: data_count,
        xor_fec_frames_per_message: frames.len(),
        selective_retry_frames_per_message_milli: 0,
        hybrid_frames_per_message_milli: 0,
        no_fec_wire_bytes_per_message,
        xor_fec_wire_bytes_per_message,
        selective_retry_wire_bytes_per_message_milli: 0,
        hybrid_wire_bytes_per_message_milli: 0,
        no_fec_airtime_us_per_message,
        xor_fec_airtime_us_per_message,
        selective_retry_airtime_us_per_message: 0,
        hybrid_airtime_us_per_message: 0,
    };
    let mut selective_airtime_us_total = 0u64;
    let mut hybrid_airtime_us_total = 0u64;
    let mut selective_frames_total = 0usize;
    let mut hybrid_frames_total = 0usize;
    let mut selective_bytes_total = 0usize;
    let mut hybrid_bytes_total = 0usize;

    for _ in 0..trials {
        // One shared channel timeline: state transitions happen once per
        // elapsed slot, including idle slots during feedback turn-around.
        // trace[t] is the potential delivery result if a frame is sent at t.
        let trace: Vec<bool> = (0..trace_slots)
            .map(|_| frame_delivered(model, &mut rng, &mut in_bad_state))
            .collect();
        let initial_data_ok = trace[..data_count].to_vec();
        let all_initial_data = initial_data_ok.iter().all(|&delivered| delivered);

        if all_initial_data {
            summary.no_fec_successes += 1;
        }

        // Evaluate the actual XOR-FEC assembler after the initial data and
        // parity slots. This initial outcome also governs whether the hybrid
        // needs any retransmissions at all.
        let mut xor_assembler =
            FragmentAssembler::new(0xC0DE, frames.len() as u8, payload.len());
        for index in 0..data_count {
            if initial_data_ok[index] {
                xor_assembler.feed(&frames[index]);
            }
        }
        let parity_initially_ok = trace[fec_index];
        if parity_initially_ok {
            xor_assembler.feed(&frames[fec_index]);
        }
        let xor_initial_success = match xor_assembler.assemble() {
            Some(recovered) if recovered.as_slice() == payload => {
                if !all_initial_data {
                    summary.xor_fec_recoveries += 1;
                }
                true
            }
            Some(_) => {
                summary.xor_fec_incorrect_outputs += 1;
                false
            }
            None => false,
        };
        if xor_initial_success {
            summary.xor_fec_successes += 1;
        }

        // Selective retry sends no parity. Slot N is the ACK turnaround gap;
        // each actually missing data fragment consumes the next retry slot.
        let mut selective_data_ok = initial_data_ok.clone();
        let mut selective_frames = data_count;
        let mut selective_bytes = no_fec_wire_bytes_per_message;
        let mut selective_airtime_us = no_fec_airtime_us_per_message;
        let mut retry_slot = data_count + 1;
        for index in 0..data_count {
            if !selective_data_ok[index] {
                selective_frames += 1;
                selective_bytes += wire_len(frames[index].payload_len);
                selective_airtime_us +=
                    lora_time_on_air_us(wire_len(frames[index].payload_len), phy);
                selective_data_ok[index] = trace[retry_slot];
                retry_slot += 1;
            }
        }
        if selective_data_ok.iter().all(|&delivered| delivered) {
            summary.selective_retry_successes += 1;
        }
        selective_frames_total += selective_frames;
        selective_bytes_total += selective_bytes;
        selective_airtime_us_total += selective_airtime_us;

        // Hybrid has already transmitted parity at slot N. If XOR-FEC has
        // reconstructed exactly, do not retransmit. Otherwise request the
        // missing data blocks only, in send order after the feedback slot N+1.
        let mut hybrid_data_ok = initial_data_ok.clone();
        let mut hybrid_frames = frames.len();
        let mut hybrid_bytes = xor_fec_wire_bytes_per_message;
        let mut hybrid_airtime_us = xor_fec_airtime_us_per_message;
        if !xor_initial_success {
            let mut retry_slot = data_count + 2;
            for index in 0..data_count {
                if !hybrid_data_ok[index] {
                    hybrid_frames += 1;
                    hybrid_bytes += wire_len(frames[index].payload_len);
                    hybrid_airtime_us +=
                        lora_time_on_air_us(wire_len(frames[index].payload_len), phy);
                    hybrid_data_ok[index] = trace[retry_slot];
                    retry_slot += 1;
                }
            }
        }

        let mut hybrid_assembler =
            FragmentAssembler::new(0xC0DE, frames.len() as u8, payload.len());
        for index in 0..data_count {
            if hybrid_data_ok[index] {
                hybrid_assembler.feed(&frames[index]);
            }
        }
        if parity_initially_ok {
            hybrid_assembler.feed(&frames[fec_index]);
        }
        match hybrid_assembler.assemble() {
            Some(recovered) if recovered.as_slice() == payload => {
                summary.hybrid_successes += 1;
                if !hybrid_data_ok.iter().all(|&delivered| delivered) {
                    summary.hybrid_fec_recoveries += 1;
                }
            }
            Some(_) => summary.hybrid_incorrect_outputs += 1,
            None => {}
        }
        hybrid_frames_total += hybrid_frames;
        hybrid_bytes_total += hybrid_bytes;
        hybrid_airtime_us_total += hybrid_airtime_us;
    }

    // Fixed-point milli-units preserve deterministic, integer-only receipts.
    summary.selective_retry_frames_per_message_milli =
        selective_frames_total * 1000 / trials;
    summary.hybrid_frames_per_message_milli = hybrid_frames_total * 1000 / trials;
    summary.selective_retry_wire_bytes_per_message_milli = selective_bytes_total * 1000 / trials;
    summary.hybrid_wire_bytes_per_message_milli = hybrid_bytes_total * 1000 / trials;
    summary.selective_retry_airtime_us_per_message =
        selective_airtime_us_total / trials as u64;
    summary.hybrid_airtime_us_per_message = hybrid_airtime_us_total / trials as u64;
    summary
}

fn payload_of_len(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|index| seed.wrapping_mul(index as u8).wrapping_add((index >> 3) as u8))
        .collect()
}

fn assert_strategies_are_ordered(summary: Summary) {
    assert_eq!(summary.xor_fec_incorrect_outputs, 0);
    assert_eq!(summary.hybrid_incorrect_outputs, 0);
    assert!(
        summary.xor_fec_successes >= summary.no_fec_successes,
        "under paired traces, single XOR parity must not reduce delivery"
    );
    assert!(
        summary.selective_retry_successes >= summary.no_fec_successes,
        "a retry cannot make a delivered initial data frame disappear"
    );
    assert!(
        summary.hybrid_successes >= summary.xor_fec_successes,
        "hybrid retry must preserve every success that XOR-FEC already had"
    );
}

#[test]
fn semtech_time_on_air_equation_matches_reference_vectors_and_spreading_factor_sweep() {
    let sf7 = DEFAULT_LORA_PHY;
    assert_eq!(lora_time_on_air_us(222, sf7), 348_416);
    assert_eq!(lora_time_on_air_us(130, sf7), 215_296);

    let payload = payload_of_len(2048, 0xA5);
    let frames = fragment(17, &payload);
    let data_count = frames.len() - 1;
    let mut last_data_only_us = 0u64;
    let mut last_xor_us = 0u64;
    for sf in 7..=12 {
        let phy = LoRaPhyProfile {
            spreading_factor: sf,
            ..DEFAULT_LORA_PHY
        };
        let data_only_us: u64 = frames[..data_count]
            .iter()
            .map(|frame| lora_time_on_air_us(wire_len(frame.payload_len), phy))
            .sum();
        let xor_us: u64 = frames
            .iter()
            .map(|frame| lora_time_on_air_us(wire_len(frame.payload_len), phy))
            .sum();
        assert!(
            data_only_us > last_data_only_us,
            "data-only airtime must rise with SF"
        );
        assert!(xor_us > last_xor_us, "FEC airtime must rise with SF");
        assert!(xor_us > data_only_us, "XOR parity must add PHY airtime");
        eprintln!(
            "mesh-airtime-baseline payload=2048 sf={} bw_hz={} cr=4/5 preamble={} explicit_header=true crc=true data_only_us={} xor_fec_us={} data_frames={} xor_frames={} ack_included=false",
            sf, phy.bandwidth_hz, phy.preamble_symbols, data_only_us, xor_us, data_count, frames.len(),
        );
        last_data_only_us = data_only_us;
        last_xor_us = xor_us;
    }

    let summary = run_benchmark_with_phy(
        &payload,
        LossModel::Iid { loss_per_mille: 100 },
        0x5EED_1234,
        1000,
        DEFAULT_LORA_PHY,
    );
    assert!(summary.xor_fec_airtime_us_per_message > summary.no_fec_airtime_us_per_message);
    assert!(
        summary.hybrid_airtime_us_per_message >= summary.xor_fec_airtime_us_per_message,
        "hybrid mean airtime includes parity plus any modeled retries"
    );
    eprintln!(
        "mesh-airtime-iid loss=100‰ seed=0x5eed1234 trials={} mean_airtime_us_per_message_no_fec/xor/selective/hybrid={}/{}/{}/{} ack_airtime_excluded=true",
        summary.trials,
        summary.no_fec_airtime_us_per_message,
        summary.xor_fec_airtime_us_per_message,
        summary.selective_retry_airtime_us_per_message,
        summary.hybrid_airtime_us_per_message,
    );
}

#[test]
fn xor_fec_reconstructs_exact_bytes_after_every_single_frame_loss() {
    let lengths = [
        1,
        PAYLOAD_SIZE - 1,
        PAYLOAD_SIZE,
        PAYLOAD_SIZE + 1,
        PAYLOAD_SIZE * 2,
        2048,
    ];

    for (sample, len) in lengths.into_iter().enumerate() {
        let payload = payload_of_len(len, 31 + sample as u8);
        let frames = fragment(17, &payload);
        assert!(frames.iter().all(|frame| wire_len(frame.payload_len) <= LORA_MTU));

        for dropped_index in 0..frames.len() {
            let mut assembler = FragmentAssembler::new(17, frames.len() as u8, payload.len());
            for (index, frame) in frames.iter().enumerate() {
                if index != dropped_index {
                    assembler.feed(frame);
                }
            }
            assert_eq!(
                assembler.assemble().as_deref(),
                Some(payload.as_slice()),
                "payload length {len}, dropped frame {dropped_index}"
            );
        }
    }
}

#[test]
fn iid_loss_sweep_is_reproducible_and_reports_seeded_replicates() {
    let payload = payload_of_len(2048, 0xA5);
    let replicate_seeds = [0x1010_0001, 0x2020_0002, 0x3030_0003];

    for loss_per_mille in [0, 25, 50, 100, 200, 300, 400, 600] {
        let model = LossModel::Iid { loss_per_mille };
        let first = run_benchmark(&payload, model, 0x5EED_1234, 2000);
        let second = run_benchmark(&payload, model, 0x5EED_1234, 2000);

        assert_eq!(first, second, "same seed/config must produce same results");
        assert_strategies_are_ordered(first);

        eprintln!(
            "mesh-iid loss={}‰ seed=0x5eed1234 n={} no_fec={}‰ xor_fec={}‰ selective_retry={}‰ hybrid={}‰ frames×1000/message={}/{}/{}/{} bytes×1000/message={}/{}/{}/{}",
            loss_per_mille,
            first.trials,
            first.delivery_per_mille(first.no_fec_successes),
            first.delivery_per_mille(first.xor_fec_successes),
            first.delivery_per_mille(first.selective_retry_successes),
            first.delivery_per_mille(first.hybrid_successes),
            first.no_fec_frames_per_message * 1000,
            first.xor_fec_frames_per_message * 1000,
            first.selective_retry_frames_per_message_milli,
            first.hybrid_frames_per_message_milli,
            first.no_fec_wire_bytes_per_message * 1000,
            first.xor_fec_wire_bytes_per_message * 1000,
            first.selective_retry_wire_bytes_per_message_milli,
            first.hybrid_wire_bytes_per_message_milli,
        );

        for seed in replicate_seeds {
            let replicate = run_benchmark(&payload, model, seed, 1000);
            assert_strategies_are_ordered(replicate);
            let no_fec_ci = replicate.wilson_95(replicate.no_fec_successes);
            let xor_ci = replicate.wilson_95(replicate.xor_fec_successes);
            let retry_ci = replicate.wilson_95(replicate.selective_retry_successes);
            let hybrid_ci = replicate.wilson_95(replicate.hybrid_successes);
            eprintln!(
                "mesh-iid-replicate loss={}‰ seed={:#010x} n={} delivery‰={}/{}/{}/{} CI95-no-fec=[{:.1},{:.1}]% xor=[{:.1},{:.1}]% retry=[{:.1},{:.1}]% hybrid=[{:.1},{:.1}]%",
                loss_per_mille,
                seed,
                replicate.trials,
                replicate.delivery_per_mille(replicate.no_fec_successes),
                replicate.delivery_per_mille(replicate.xor_fec_successes),
                replicate.delivery_per_mille(replicate.selective_retry_successes),
                replicate.delivery_per_mille(replicate.hybrid_successes),
                no_fec_ci.0 * 100.0,
                no_fec_ci.1 * 100.0,
                xor_ci.0 * 100.0,
                xor_ci.1 * 100.0,
                retry_ci.0 * 100.0,
                retry_ci.1 * 100.0,
                hybrid_ci.0 * 100.0,
                hybrid_ci.1 * 100.0,
            );
        }
    }
}

#[test]
fn burst_loss_model_is_reproducible_and_compares_four_strategies() {
    let payload = payload_of_len(2048, 0x6D);
    let model = LossModel::GilbertElliott {
        good_loss_per_mille: 20,
        bad_loss_per_mille: 750,
        good_to_bad_per_mille: 80,
        bad_to_good_per_mille: 350,
    };
    let first = run_benchmark(&payload, model, 0xBADC_0FFE, 5000);
    let second = run_benchmark(&payload, model, 0xBADC_0FFE, 5000);

    assert_eq!(first, second);
    assert_strategies_are_ordered(first);
    eprintln!(
        "mesh-burst seed=0xbadc0ffe n={} delivery‰={}/{}/{}/{} frames×1000/message={}/{}/{}/{} bytes×1000/message={}/{}/{}/{}",
        first.trials,
        first.delivery_per_mille(first.no_fec_successes),
        first.delivery_per_mille(first.xor_fec_successes),
        first.delivery_per_mille(first.selective_retry_successes),
        first.delivery_per_mille(first.hybrid_successes),
        first.no_fec_frames_per_message * 1000,
        first.xor_fec_frames_per_message * 1000,
        first.selective_retry_frames_per_message_milli,
        first.hybrid_frames_per_message_milli,
        first.no_fec_wire_bytes_per_message * 1000,
        first.xor_fec_wire_bytes_per_message * 1000,
        first.selective_retry_wire_bytes_per_message_milli,
        first.hybrid_wire_bytes_per_message_milli,
    );
}
