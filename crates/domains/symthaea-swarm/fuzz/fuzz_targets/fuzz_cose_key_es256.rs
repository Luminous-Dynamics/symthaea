// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

#![no_main]

use libfuzzer_sys::fuzz_target;
use symthaea_swarm::semantic_evidence_vds::Rfc9942Es256CoseKey;

/// Adversarial parser harness for the untrusted COSE_Key byte boundary.
///
/// The production parser already applies structural bounds to map size,
/// text/bstr sizes, nested skipped values, and key operation counts. This
/// harness deliberately performs no pre-validation: libFuzzer should exercise
/// the parser against arbitrary bytes, including truncation, non-canonical
/// encodings, nested unknown fields, duplicate labels, and hostile lengths.
///
/// No successful parse is treated as proof of a valid cryptographic point;
/// point validation remains at the ring verification boundary.
fuzz_target!(|data: &[u8]| {
    let _ = Rfc9942Es256CoseKey::from_cbor(data);
});
