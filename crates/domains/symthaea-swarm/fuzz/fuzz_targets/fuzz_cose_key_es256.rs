// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![no_main]

use libfuzzer_sys::fuzz_target;
use symthaea_swarm::semantic_evidence_vds::Rfc9942Es256CoseKey;

fuzz_target!(|data: &[u8]| {
    if let Ok(key) = Rfc9942Es256CoseKey::from_cbor(data) {
        let _ = key.public_key_sec1();
    }
});
