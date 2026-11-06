#![no_main]

use libfuzzer_sys::fuzz_target;
use rea_claim_replay_verifier::replay_claim_evaluation;

const MAX_INPUT: usize = 1024 * 1024;

fuzz_target!(|input: &[u8]| {
    if input.len() > MAX_INPUT {
        return;
    }
    let _ = replay_claim_evaluation(input, &[], b"{}");
});
