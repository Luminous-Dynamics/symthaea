// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![no_main]

use libfuzzer_sys::fuzz_target;
use symthaea_swarm::semantic_evidence_vds::{
    Rfc9162ConsistencyProof,
    Rfc9162InclusionProof,
    Rfc9942Vdp, Rfc9942VdpError, VdsTreeHead, Rfc9162Sha256Vds,
};

fuzz_target!(|data: &[u8]| {
    let _ = Rfc9162InclusionProof::from_cbor(data);
    let _ = Rfc9162ConsistencyProof::from_cbor(data);

    // Exercise semantic proof consumption, not just structural decoding. This
    // keeps attacker-controlled proof bytes on the same derivation/verification
    // walkers used by RFC 9942 Receipt verification.
    if let Ok(vdp) = Rfc9942Vdp::from_cbor(data) {
        let _ = vdp.derive_inclusion_root(b"fuzz-candidate");
        let older = VdsTreeHead::new(1, [0u8; 32]);
        let newer = VdsTreeHead::new(2, [0u8; 32]);
        let _ = vdp.verify_consistency(older, newer);

        // Exercise the higher-level RFC9942 proof adapters with expected tree
        // heads as well, including their typed decode/error boundary.
        let vds = Rfc9162Sha256Vds;
        let _ = vds.verify_rfc9942_inclusion_cbor(
            b"fuzz-candidate",
            VdsTreeHead::new(2, [0u8; 32]),
            data,
        );
        let _ = vds.verify_rfc9942_consistency_cbor(older, newer, data);
    }

    let _ = Rfc9942VdpError::InvalidStructure;
});
