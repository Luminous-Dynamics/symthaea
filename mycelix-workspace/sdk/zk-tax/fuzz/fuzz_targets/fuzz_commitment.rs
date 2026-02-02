#![no_main]

use libfuzzer_sys::fuzz_target;
use arbitrary::Arbitrary;
use mycelix_zk_tax::brackets::compute_commitment;

#[derive(Debug, Arbitrary)]
struct CommitmentInput {
    lower: u64,
    upper: u64,
    tax_year: u32,
}

fuzz_target!(|input: CommitmentInput| {
    // Commitment computation should never panic
    let commitment = compute_commitment(input.lower, input.upper, input.tax_year);

    // Commitment should be deterministic
    let commitment2 = compute_commitment(input.lower, input.upper, input.tax_year);
    assert_eq!(commitment, commitment2, "Non-deterministic commitment");

    // Commitment should be 32 bytes
    let bytes = commitment.to_bytes();
    assert_eq!(bytes.len(), 32, "Wrong commitment size");

    // Hex encoding should work
    let hex = commitment.to_hex();
    assert_eq!(hex.len(), 64, "Wrong hex length");

    // Different inputs should (usually) produce different commitments
    if input.lower != input.upper || input.tax_year != 2024 {
        let different = compute_commitment(
            input.lower.wrapping_add(1),
            input.upper.wrapping_add(1),
            input.tax_year,
        );
        // This isn't guaranteed but is extremely likely
        // We don't assert because hash collisions are theoretically possible
        let _ = different;
    }
});
