use std::{env, fs, path::Path};

fn main() {
    let manifest = env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    let input = Path::new(&manifest).join("build.rs-input.txt");
    let value = fs::read_to_string(&input).expect("fixture input is readable");
    let generated = format!("pub const BASALT_BUILD_INPUT: &str = {:?};\n", value.trim());
    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR");
    fs::write(Path::new(&out_dir).join("generated.rs"), generated).expect("generated source write");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=build.rs-input.txt");
}
