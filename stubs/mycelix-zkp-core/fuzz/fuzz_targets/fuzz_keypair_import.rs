#![no_main]

use libfuzzer_sys::fuzz_target;
use mycelix_zkp_core::dilithium::{
    DilithiumKeypair, PUBLIC_KEY_SIZE, SECRET_KEY_SIZE,
};
use sha2::{Digest, Sha256};

const MAX_INPUT_BYTES: usize = 8 * 1024;

/// Expand arbitrary input into correctly sized key encodings so the target
/// explores representation parsing and the key-pair consistency check, rather
/// than only returning at the public length guard.
fn expand_to_len(input: &[u8], len: usize, offset: usize) -> Vec<u8> {
    let mut seed_hasher = Sha256::new();
    seed_hasher.update(b"SYMTHAEA-CRYPTO-FUZZ/KEY-EXPANSION/v1");
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
    // Bound fixture construction independently from libFuzzer's command-line
    // settings. The target must remain cheap even for oversized inputs.
    let input = &input[..input.len().min(MAX_INPUT_BYTES)];
    let public_key = expand_to_len(input, PUBLIC_KEY_SIZE, 0);
    let secret_key = expand_to_len(input, SECRET_KEY_SIZE, 1);

    // Invalid inputs should return an error, not panic. A valid matching key
    // pair is not required from arbitrary fuzz bytes; unit tests cover that
    // positive control and explicit cross-pair negative control.
    let _ = DilithiumKeypair::from_bytes(public_key, secret_key);
});
