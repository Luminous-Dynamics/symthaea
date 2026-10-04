// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![no_main]

use libfuzzer_sys::fuzz_target;
use symthaea_swarm::semantic_evidence_vds::{
    Rfc9942Es256CoseKey, Rfc9942ReceiptEnvelope, Rfc9942ReceiptPayload,
    Rfc9942ProofKind, Rfc9942Vdp, Rfc9162InclusionProof, COSE_ES256_ALGORITHM_ID,
};

fuzz_target!(|data: &[u8]| {
    if let Ok(key) = Rfc9942Es256CoseKey::from_cbor(data) {
        let _ = key.public_key_sec1();

        // Carry accepted keys into the cryptographic boundary as well. The
        // fixed receipt/signature are intentionally invalid; the objective is
        // to exercise point parsing/verification without ever accepting data.
        let proof = Rfc9162InclusionProof::new(2, 0, vec![[0u8; 32]]).to_cbor();
        if let Ok(vdp) = Rfc9942Vdp::new(Rfc9942ProofKind::Inclusion, vec![proof]) {
            if let Ok(receipt) = Rfc9942ReceiptEnvelope::new(
                COSE_ES256_ALGORITHM_ID,
                vdp,
                Rfc9942ReceiptPayload::Attached([0u8; 32]),
                vec![0u8; 64],
            ) {
                let _ = receipt.verify_es256_cose_key(&key, &[], None);
            }
        }
    }
});
