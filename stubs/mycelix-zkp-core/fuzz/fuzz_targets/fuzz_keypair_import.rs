#![no_main]

use libfuzzer_sys::fuzz_target;
use mycelix_zkp_core::dilithium::{
    DilithiumKeypair, PUBLIC_KEY_SIZE, SECRET_KEY_SIZE,
};

const MAX_INPUT_BYTES: usize = 8 * 1024;

/// Expand arbitrary input into correctly sized key encodings so the target
/// explores representation parsing and the key-pair consistency check, rather
/// than only returning at the public length guard.
fn expand_to_len(input: &[u8], len: usize, offset: usize) -> Vec<u8> {
    if input.is_empty() {
        return vec![0; len];
    }

    (0..len)
        .map(|index| input[(index + offset) % input.len()])
        .collect()
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
