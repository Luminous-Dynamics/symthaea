// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Deterministic, test-only LoRa fragment channel model.
//!
//! This module is deliberately a research baseline, not a radio simulator.
//! It applies seeded IID or two-state burst loss to the repository's actual
//! fragment output and XOR-FEC assembler. Airtime/energy are not inferred here;
//! those require a separately frozen PHY profile and device measurements.

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
    xor_fec_recoveries: usize,
    xor_fec_incorrect_outputs: usize,
    no_fec_frames_per_message: usize,
    xor_fec_frames_per_message: usize,
    no_fec_wire_bytes_per_message: usize,
    xor_fec_wire_bytes_per_message: usize,
}

impl Summary {
    fn no_fec_delivery_per_mille(self) -> usize {
        self.no_fec_successes * 1000 / self.trials
    }

    fn xor_fec_delivery_per_mille(self) -> usize {
        self.xor_fec_successes * 1000 / self.trials
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

/// Runs paired no-recovery and XOR-FEC variants with a shared deterministic
/// trace for each packet. The no-recovery baseline sends data fragments only;
/// the XOR strategy sends the same data fragments plus one parity fragment.
///
/// This is a packet-delivery model only: there are no ACKs, retries, radio
/// timing, contention, PHY airtime, or energy claims in this first baseline.
fn run_benchmark(
    payload: &[u8],
    model: LossModel,
    seed: u64,
    trials: usize,
) -> Summary {
    assert!(!payload.is_empty(), "benchmark payload must not be empty");
    assert!(trials > 0, "benchmark requires at least one trial");
    let frames = fragment(0xC0DE, payload);
    let data_count = frames.len() - 1;
    let no_fec_wire_bytes_per_message = frames[..data_count]
        .iter()
        .map(|frame| wire_len(frame.payload_len))
        .sum();
    let xor_fec_wire_bytes_per_message = frames.iter()
        .map(|frame| wire_len(frame.payload_len))
        .sum();

    assert!(frames.iter().all(|frame| wire_len(frame.payload_len) <= LORA_MTU));

    let mut rng = DeterministicRng::new(seed);
    let mut in_bad_state = false;
    let mut summary = Summary {
        trials,
        no_fec_successes: 0,
        xor_fec_successes: 0,
        xor_fec_recoveries: 0,
        xor_fec_incorrect_outputs: 0,
        no_fec_frames_per_message: data_count,
        xor_fec_frames_per_message: frames.len(),
        no_fec_wire_bytes_per_message,
        xor_fec_wire_bytes_per_message,
    };

    for _ in 0..trials {
        // Precompute one shared loss trace so both strategies see the same
        // data-frame outcomes. The parity slot is an extra potential frame.
        let trace: Vec<bool> = (0..frames.len())
            .map(|_| frame_delivered(model, &mut rng, &mut in_bad_state))
            .collect();

        if trace[..data_count].iter().all(|&delivered| delivered) {
            summary.no_fec_successes += 1;
        }

        let mut assembler = FragmentAssembler::new(0xC0DE, frames.len() as u8, payload.len());
        for (frame, &delivered) in frames.iter().zip(trace.iter()) {
            if delivered {
                assembler.feed(frame);
            }
        }

        if let Some(recovered) = assembler.assemble() {
            if recovered.as_slice() == payload {
                summary.xor_fec_successes += 1;
                if !trace[..data_count].iter().all(|&delivered| delivered) {
                    summary.xor_fec_recoveries += 1;
                }
            } else {
                summary.xor_fec_incorrect_outputs += 1;
            }
        }
    }

    summary
}

fn payload_of_len(len: usize, seed: u8) -> Vec<u8> {
    (0..len)
        .map(|index| seed.wrapping_mul(index as u8).wrapping_add((index >> 3) as u8))
        .collect()
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
fn iid_loss_sweep_is_reproducible_and_fec_never_fabricates_payloads() {
    let payload = payload_of_len(2048, 0xA5);
    for loss_per_mille in [0, 25, 50, 100, 200, 300, 400, 600] {
        let model = LossModel::Iid { loss_per_mille };
        let first = run_benchmark(&payload, model, 0x5EED_1234, 2000);
        let second = run_benchmark(&payload, model, 0x5EED_1234, 2000);

        assert_eq!(first, second, "same seed/config must produce same results");
        assert_eq!(first.xor_fec_incorrect_outputs, 0);
        assert!(
            first.xor_fec_successes >= first.no_fec_successes,
            "under paired traces, single XOR parity must not reduce delivery"
        );
        eprintln!(
            "mesh-iid loss={}‰ trials={} no_fec={}‰ xor_fec={}‰ fec_recoveries={} wire_bytes/message={}/{}",
            loss_per_mille,
            first.trials,
            first.no_fec_delivery_per_mille(),
            first.xor_fec_delivery_per_mille(),
            first.xor_fec_recoveries,
            first.no_fec_wire_bytes_per_message,
            first.xor_fec_wire_bytes_per_message,
        );
    }
}

#[test]
fn burst_loss_model_is_reproducible_and_exact() {
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
    assert_eq!(first.xor_fec_incorrect_outputs, 0);
    assert!(
        first.xor_fec_successes >= first.no_fec_successes,
        "paired traces require FEC delivery to dominate the no-recovery baseline"
    );
    eprintln!(
        "mesh-burst trials={} no_fec={}‰ xor_fec={}‰ fec_recoveries={} wire_bytes/message={}/{}",
        first.trials,
        first.no_fec_delivery_per_mille(),
        first.xor_fec_delivery_per_mille(),
        first.xor_fec_recoveries,
        first.no_fec_wire_bytes_per_message,
        first.xor_fec_wire_bytes_per_message,
    );
}
