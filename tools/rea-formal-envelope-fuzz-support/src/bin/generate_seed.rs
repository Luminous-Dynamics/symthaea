fn main() {
    let seed = rea_formal_envelope_fuzz_support::generate_valid_seed();
    print!("{}", String::from_utf8(seed).expect("formal envelope is UTF-8"));
}
