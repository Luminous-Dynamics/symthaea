#![no_main]

use libfuzzer_sys::fuzz_target;
use mycelix_zk_tax::proof::CompressedProof;

fuzz_target!(|data: &[u8]| {
    // Try to deserialize arbitrary bytes as compressed proof
    if let Ok(s) = std::str::from_utf8(data) {
        let result: Result<CompressedProof, _> = serde_json::from_str(s);

        if let Ok(compressed) = result {
            // Decompression should not panic (may return error)
            let decompress_result = compressed.to_bracket_proof();

            if let Ok(proof) = decompress_result {
                // Re-compression should work
                if let Ok(recompressed) = CompressedProof::from_bracket_proof(&proof) {
                    // Decompress again
                    if let Ok(proof2) = recompressed.to_bracket_proof() {
                        // Key fields should match
                        assert_eq!(proof.bracket_index, proof2.bracket_index);
                        assert_eq!(proof.rate_bps, proof2.rate_bps);
                        assert_eq!(proof.tax_year, proof2.tax_year);
                    }
                }
            }
        }
    }
});
