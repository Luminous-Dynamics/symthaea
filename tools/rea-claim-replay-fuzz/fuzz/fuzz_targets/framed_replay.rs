#![no_main]

use libfuzzer_sys::fuzz_target;
use rea_claim_replay_fuzz::parse_frame;
use rea_claim_replay_verifier::replay_claim_evaluation;

fuzz_target!(|input: &[u8]| {
    if let Some(frame) = parse_frame(input) {
        let evidence: Vec<&[u8]> = frame.evidence;
        let _ = replay_claim_evaluation(frame.claim, &evidence, frame.result);
    }
});
