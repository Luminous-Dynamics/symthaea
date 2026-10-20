use sha2::{Digest, Sha256};
use std::env;
use std::ffi::OsString;
use std::fs;
use std::path::Path;
use std::process::Command;

fn command_line(program: OsString, args: &[&str], cwd: &Path) -> Option<String> {
    let output = Command::new(program)
        .args(args)
        .current_dir(cwd)
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&output.stdout);
    Some(
        text.lines()
            .map(str::trim)
            .filter(|line| !line.is_empty())
            .collect::<Vec<_>>()
            .join(" | "),
    )
}

fn sha256_hex(bytes: &[u8]) -> String {
    let digest = Sha256::digest(bytes);
    let mut output = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut output, "{byte:02x}").expect("writing to String cannot fail");
    }
    output
}

fn hash_evaluator_sources(manifest_dir: &Path) -> Option<String> {
    let mut hasher = Sha256::new();
    for relative_path in ["Cargo.toml", "build.rs", "src/lib.rs", "src/main.rs"] {
        let bytes = fs::read(manifest_dir.join(relative_path)).ok()?;
        hasher.update(relative_path.as_bytes());
        hasher.update([0]);
        hasher.update((bytes.len() as u64).to_le_bytes());
        hasher.update(&bytes);
    }
    Some(sha256_hex(&hasher.finalize()))
}

fn main() {
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=CARGO");
    for path in [
        "Cargo.toml",
        "Cargo.lock",
        "build.rs",
        "src/lib.rs",
        "src/main.rs",
    ] {
        println!("cargo:rerun-if-changed={path}");
    }

    let manifest_dir = env::var_os("CARGO_MANIFEST_DIR")
        .map(std::path::PathBuf::from)
        .expect("Cargo sets CARGO_MANIFEST_DIR");

    let rustc = env::var_os("RUSTC").unwrap_or_else(|| OsString::from("rustc"));
    let rustc_identity =
        command_line(rustc, &["-Vv"], &manifest_dir).unwrap_or_else(|| "unavailable".to_owned());

    let cargo = env::var_os("CARGO").unwrap_or_else(|| OsString::from("cargo"));
    let cargo_identity = command_line(cargo, &["--version"], &manifest_dir)
        .unwrap_or_else(|| "unavailable".to_owned());

    let source_commit = command_line(OsString::from("git"), &["rev-parse", "HEAD"], &manifest_dir)
        .unwrap_or_else(|| "unavailable".to_owned());

    let source_tree_state = match command_line(
        OsString::from("git"),
        &["status", "--porcelain"],
        &manifest_dir,
    ) {
        Some(status) if status.is_empty() => "clean",
        Some(_) => "dirty",
        None => "unavailable",
    };

    let source_sha256 =
        hash_evaluator_sources(&manifest_dir).unwrap_or_else(|| "unavailable".to_owned());

    let lock_sha256 = fs::read(manifest_dir.join("Cargo.lock"))
        .map(|bytes| sha256_hex(&bytes))
        .map(|digest| format!("sha256:{digest}"))
        .unwrap_or_else(|_| "missing".to_owned());

    println!("cargo:rustc-env=OPEN_WORLD_RUSTC_ID={rustc_identity}");
    println!("cargo:rustc-env=OPEN_WORLD_CARGO_ID={cargo_identity}");
    println!("cargo:rustc-env=OPEN_WORLD_SOURCE_COMMIT={source_commit}");
    println!("cargo:rustc-env=OPEN_WORLD_SOURCE_TREE_STATE={source_tree_state}");
    println!("cargo:rustc-env=OPEN_WORLD_SOURCE_SHA256={source_sha256}");
    println!("cargo:rustc-env=OPEN_WORLD_LOCK_SHA256={lock_sha256}");
}
