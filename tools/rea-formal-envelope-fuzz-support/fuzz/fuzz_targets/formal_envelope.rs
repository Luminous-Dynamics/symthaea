#![no_main]

use libfuzzer_sys::fuzz_target;
use rea_formal_envelope_fuzz_support::rea_compat::formal::parse_formal_provider_observation_json;

const MAX_INPUT: usize = 1024 * 1024;

fuzz_target!(|input: &[u8]| {
    if input.len() > MAX_INPUT {
        return;
    }
    let _ = parse_formal_provider_observation_json(input);
});
