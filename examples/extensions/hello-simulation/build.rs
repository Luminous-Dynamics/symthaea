use sha2::{Digest, Sha256};
use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=manifest.json");

    let manifest = fs::read("manifest.json").expect("read manifest.json");
    let digest = Sha256::digest(manifest);

    let mut literal = String::from("[");
    for (index, byte) in digest.iter().enumerate() {
        if index != 0 {
            literal.push_str(", ");
        }
        literal.push_str(&format!("0x{byte:02x}"));
    }
    literal.push(']');

    let generated = format!("pub const MANIFEST_DIGEST: [u8; 32] = {literal};\n");
    let out_dir = PathBuf::from(env::var_os("OUT_DIR").expect("OUT_DIR"));
    fs::write(out_dir.join("manifest_digest.rs"), generated)
        .expect("write generated manifest digest");
}
