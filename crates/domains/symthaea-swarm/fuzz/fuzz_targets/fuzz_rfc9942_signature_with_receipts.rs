// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![no_main]

use libfuzzer_sys::fuzz_target;
use symthaea_swarm::semantic_evidence_vds::Rfc9942SignatureWithReceipts;

fuzz_target!(|data: &[u8]| {
    if let Ok(outer) = Rfc9942SignatureWithReceipts::from_cbor(data) {
        // Exercise RFC 9052 Sig_structure construction after structural
        // admission, including detached and attached payload resolution.
        let payload = match outer.payload() {
            symthaea_swarm::semantic_evidence_vds::Rfc9942SignaturePayload::Attached(bytes) => {
                Some(bytes.as_slice())
            }
            symthaea_swarm::semantic_evidence_vds::Rfc9942SignaturePayload::Detached => {
                Some(&[][..])
            }
        };
        let _ = outer.signature1_tbs(&[], payload);
    }
});
