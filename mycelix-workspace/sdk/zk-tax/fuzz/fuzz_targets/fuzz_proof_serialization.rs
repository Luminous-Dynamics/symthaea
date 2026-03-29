#![no_main]

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root

use libfuzzer_sys::fuzz_target;
use mycelix_zk_tax::TaxBracketProof;

fuzz_target!(|data: &[u8]| {
    // Try to deserialize arbitrary bytes as a proof
    // This should never panic, only return errors
    if let Ok(s) = std::str::from_utf8(data) {
        let result: Result<TaxBracketProof, _> = serde_json::from_str(s);

        // If deserialization succeeds, the proof should be usable
        if let Ok(proof) = result {
            // Serialization should roundtrip
            if let Ok(json) = serde_json::to_string(&proof) {
                let _roundtrip: Result<TaxBracketProof, _> = serde_json::from_str(&json);
            }

            // Verification should not panic (may return error)
            let _verify_result = proof.verify();

            // Accessing fields should not panic
            let _bracket = proof.bracket_index;
            let _rate = proof.rate_bps;
            let _year = proof.tax_year;
            let _lower = proof.bracket_lower;
            let _upper = proof.bracket_upper;
        }
    }
});
