#![no_main]

use libfuzzer_sys::fuzz_target;
use rea_numeric_lexeme_fuzz_support::exercise_raw_input;

fuzz_target!(|input: &[u8]| {
    let _ = exercise_raw_input(input);
});
