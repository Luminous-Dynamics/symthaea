// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![no_main]

use libfuzzer_sys::fuzz_target;
use symthaea_swarm::semantic_evidence_vds::{
    Rfc9162ConsistencyProof,
    Rfc9162InclusionProof,
    Rfc9942Vdp,
};

fuzz_target!(|data: &[u8]| {
    let _ = Rfc9942Vdp::from_cbor(data);
    let _ = Rfc9162InclusionProof::from_cbor(data);
    let _ = Rfc9162ConsistencyProof::from_cbor(data);
});
