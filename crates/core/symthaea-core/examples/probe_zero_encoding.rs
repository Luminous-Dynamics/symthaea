// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Diagnostic: why do specific sentences yield PE=1.0 sentinels / bits=None
//! in the cognitive loop? (Predictive Compression Amendment 3 item 3.)
//! Checks encode → attend → compress norms for the two implicated sentences
//! vs. healthy controls.

use symthaea_core::hdc::predictive_encoder::{PredictiveEncoderConfig, PredictiveHdcEncoder};

fn main() {
    let mut enc = PredictiveHdcEncoder::new(PredictiveEncoderConfig::default()).unwrap();
    let sentences = [
        "What is the meaning of a life well lived?",
        "Critical failure: coolant pressure dropping, meltdown risk rising!",
        "The water cycle moves moisture from oceans to clouds to rain.",
        "URGENT: fire detected in the server room, evacuate immediately!",
    ];
    for s in sentences {
        let r = enc.encode(s);
        let hdv_norm: f32 = r.hdv.values.iter().map(|x| x * x).sum::<f32>().sqrt();
        let compressed = enc.compress_for_ltc(&r.hdv, 256);
        let comp_norm: f32 = compressed.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nonzero = r.hdv.values.iter().filter(|x| **x != 0.0).count();
        println!(
            "hdv_norm={hdv_norm:>10.4} nonzero={nonzero:>6} comp_norm={comp_norm:>10.4} peak_attn={:.4} prims={:?} :: {s}",
            r.peak_attention, r.detected_primitives
        );
    }
}
