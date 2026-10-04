// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![no_main]

use libfuzzer_sys::fuzz_target;
use symthaea_swarm::semantic_evidence_vds::{Rfc9942ProofKind, Rfc9942ReceiptEnvelope, Rfc9942ReceiptPayload, VdsTreeHead};

fuzz_target!(|data: &[u8]| {
    if let Ok(receipt) = Rfc9942ReceiptEnvelope::from_cbor(data) {
        // Push structurally valid receipts through their semantic proof
        // consumers as well. These calls perform no signature acceptance:
        // they exercise the RFC 9162 proof/state binding with bounded inputs.
        match receipt.vdp().kind() {
            Rfc9942ProofKind::Inclusion => {
                let _ = match receipt.payload() {
                    Rfc9942ReceiptPayload::Attached(_) => {
                        receipt.verify_inclusion(b"fuzz-candidate")
                    }
                    Rfc9942ReceiptPayload::Detached => {
                        receipt.verify_inclusion_with_detached_payload(
                            b"fuzz-candidate",
                            &[0u8; 32],
                        )
                    }
                };
            }
            Rfc9942ProofKind::Consistency => {
                let older = VdsTreeHead::new(1, [0u8; 32]);
                let _ = match receipt.payload() {
                    Rfc9942ReceiptPayload::Attached(_) => receipt.verify_consistency(older),
                    Rfc9942ReceiptPayload::Detached => {
                        receipt.verify_consistency_with_detached_payload(older, &[0u8; 32])
                    }
                };
            }
        }
    }
});
