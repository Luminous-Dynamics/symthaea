// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![no_main]

use libfuzzer_sys::fuzz_target;
use symthaea_swarm::semantic_evidence_vds::{
    Rfc9942SignaturePayload, Rfc9942SignatureWithReceipts,
};

fuzz_target!(|data: &[u8]| {
    if let Ok(outer) = Rfc9942SignatureWithReceipts::from_cbor(data) {
        // Exercise RFC 9052 Sig_structure construction after structural
        // admission, including detached and attached payload resolution.
        let payload = match outer.payload() {
            Rfc9942SignaturePayload::Attached(bytes) => Some(bytes.as_slice()),
            Rfc9942SignaturePayload::Detached => Some(&[][..]),
        };
        let _ = outer.signature1_tbs(&[], payload);
        let _ = outer.protected_algorithm_id();

        // Cross the complete RFC 9942 outer-receipt composition boundary.
        // The fixed P-256 point is intentionally invalid; this is for panic,
        // ordering, and binding coverage, never acceptance.
        let detached_outer_payload = match outer.payload() {
            Rfc9942SignaturePayload::Attached(_) => None,
            Rfc9942SignaturePayload::Detached => Some(payload.unwrap()),
        };
        if outer.receipts().is_some() {
            let _ = outer.verify_es256_inclusion_receipt_state(
                0,
                &[0x04; 65],
                &[0x04; 65],
                &[],
                &[],
                detached_outer_payload,
            );
        }

        // Cross the outer cryptographic boundary too; the fixed public point
        // is deliberately invalid and can never authenticate a fuzz input.
        let _ = outer.verify_es256(&[0x04; 65], &[], payload);
        let _ = outer.verify_ed25519(&[0u8; 32], &[], payload);
    }
});
