use std::{env, fs, path::Path};

fn main() {
    // Deliberate defect: this file is read, but is NOT named in rerun-if-changed.
    let input = fs::read_to_string("build.rs-input.txt")
        .expect("fixture build-script input must be readable");
    let value = input.trim().to_string();
    let generated = format!("pub const BASALT_BUILD_INPUT: &str = {value:?};\n");
    let out_dir = env::var_os("OUT_DIR").expect("Cargo provides OUT_DIR");
    fs::write(Path::new(&out_dir).join("generated.rs"), generated)
        .expect("generated Rust source must be writable");
    println!("cargo:rerun-if-changed=build.rs");
}
