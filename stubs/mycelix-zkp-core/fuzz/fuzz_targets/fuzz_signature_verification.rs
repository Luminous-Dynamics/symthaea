#![no_main]

use libfuzzer_sys::fuzz_target;
use mycelix_zkp_core::dilithium::{
    verify_signature, PUBLIC_KEY_SIZE, SIGNATURE_SIZE,
};
use sha2::{Digest, Sha256};

const MAX_INPUT_BYTES: usize = 8 * 1024;
const MAX_MESSAGE_SUFFIX_BYTES: usize = 1024;

/// Construct a bounded, correctly sized byte sequence from fuzz input.
/// This takes arbitrary inputs beyond the wrapper's cheap length rejection.
fn expand_to_len(input: &[u8], len: usize, offset: usize) -> Vec<u8> {
    let mut seed_hasher = Sha256::new();
    seed_hasher.update(b"SYMTHAEA-CRYPTO-FUZZ/SIGNATURE-EXPANSION/v1");
    seed_hasher.update((input.len() as u64).to_le_bytes());
    seed_hasher.update(input);
    seed_hasher.update((offset as u64).to_le_bytes());
    let seed = seed_hasher.finalize();

    let mut output = Vec::with_capacity(len);
    let mut counter = 0_u64;
    while output.len() < len {
        let mut block_hasher = Sha256::new();
        block_hasher.update(b"SYMTHAEA-CRYPTO-FUZZ/BLOCK/v1");
        block_hasher.update(seed);
        block_hasher.update(counter.to_le_bytes());
        let block = block_hasher.finalize();
        let count = (len - output.len()).min(block.len());
        output.extend_from_slice(&block[..count]);
        counter = counter.wrapping_add(1);
    }
    output
}

fuzz_target!(|input: &[u8]| {
    let input = &input[..input.len().min(MAX_INPUT_BYTES)];
    let public_key = expand_to_len(input, PUBLIC_KEY_SIZE, 0);
    let suffix_len = input.len().min(MAX_MESSAGE_SUFFIX_BYTES);
    let signed_message = expand_to_len(input, SIGNATURE_SIZE + suffix_len, 1);

    // This exercises the actual public-key and signed-message parsers, then
    // the underlying verifier when parsing succeeds. Fuzzing asserts absence
    // of panics/crashes; it does not equate an arbitrary invalid input with
    // a cryptographic forgery or a conformance test.
    let _ = verify_signature(input, &signed_message, &public_key);
});
