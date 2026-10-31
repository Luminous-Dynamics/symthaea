use std::{env, fs, path::Path};

fn main() {
    let value = env::var("BASALT_BUILD_INPUT").expect("fixture environment input must exist");
    let generated = format!("pub const BASALT_BUILD_INPUT_VALUE: &str = {:?};\\n", value);
    let out_dir = env::var_os("OUT_DIR").expect("OUT_DIR");
    fs::write(Path::new(&out_dir).join("generated.rs"), generated).expect("generated source write");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=BASALT_BUILD_INPUT");
}
