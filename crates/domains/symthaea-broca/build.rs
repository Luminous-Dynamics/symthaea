// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Build-time identities for the source-backed UniMorph compiler.

use std::{
    env,
    fs,
    path::{Path, PathBuf},
    process::Command,
};

fn canonical_source_bytes(bytes: &[u8]) -> Vec<u8> {
    let mut canonical = Vec::with_capacity(bytes.len());
    let mut index = 0;
    while index < bytes.len() {
        if bytes[index] == b'\r' {
            if bytes.get(index + 1) == Some(&b'\n') {
                index += 1;
            }
            canonical.push(b'\n');
        } else {
            canonical.push(bytes[index]);
        }
        index += 1;
    }
    canonical
}

fn domain_digest(domain: &[u8], surfaces: &[&[u8]]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(b"\0");
    for surface in surfaces {
        let canonical = canonical_source_bytes(surface);
        hasher.update(&(canonical.len() as u64).to_le_bytes());
        hasher.update(&canonical);
    }
    hasher.finalize().to_hex().to_string()
}

fn delimited_surface<'a>(
    source: &'a [u8],
    begin: &[u8],
    end: &[u8],
) -> Result<&'a [u8], String> {
    let start = source
        .windows(begin.len())
        .position(|window| window == begin)
        .ok_or_else(|| format!("missing compiler revision marker: {}", String::from_utf8_lossy(begin)))?;
    let content_start = start + begin.len();
    let relative_end = source[content_start..]
        .windows(end.len())
        .position(|window| window == end)
        .ok_or_else(|| format!("missing compiler revision marker: {}", String::from_utf8_lossy(end)))?;
    let content_end = content_start + relative_end;
    if content_start >= content_end {
        return Err("compiler revision marker surface is empty".to_string());
    }
    Ok(&source[content_start..content_end])
}

fn read_required(path: &Path) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        panic!("failed to read compiler identity input {}: {error}", path.display())
    })
}

fn rustc_identity() -> Vec<u8> {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(&rustc)
        .arg("--version")
        .arg("--verbose")
        .output()
        .unwrap_or_else(|error| panic!("failed to execute rustc for compiler identity: {error}"));
    if !output.status.success() {
        panic!(
            "rustc identity command failed with status {}",
            output.status
        );
    }
    output.stdout
}

fn cargo_identity() -> Vec<u8> {
    let cargo = env::var_os("CARGO").unwrap_or_else(|| "cargo".into());
    let output = Command::new(&cargo)
        .arg("--version")
        .arg("--verbose")
        .output()
        .unwrap_or_else(|error| panic!("failed to execute cargo for compiler identity: {error}"));
    if !output.status.success() {
        panic!(
            "cargo identity command failed with status {}",
            output.status
        );
    }
    output.stdout
}

fn cargo_feature_identity() -> Vec<u8> {
    let mut features = env::vars()
        .filter_map(|(key, value)| {
            key.strip_prefix("CARGO_FEATURE_")
                .map(|feature| format!("{feature}={value}"))
        })
        .collect::<Vec<_>>();
    features.sort_unstable();
    features.join("\n").into_bytes()
}

fn main() {
    println!("cargo:rerun-if-changed=src/lexical_binding.rs");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=../../../Cargo.toml");
    println!("cargo:rerun-if-changed=../../../Cargo.lock");
    println!("cargo:rerun-if-changed=../../../rust-toolchain.toml");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=CARGO");
    println!("cargo:rerun-if-env-changed=RUSTC_WRAPPER");
    println!("cargo:rerun-if-env-changed=RUSTC_WORKSPACE_WRAPPER");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    for feature in [
        "CARGO_FEATURE_GPU",
        "CARGO_FEATURE_PARALLEL",
        "CARGO_FEATURE_NETWORKING",
        "CARGO_FEATURE_WASM_SANDBOX",
        "CARGO_FEATURE_MAMBA_CPU",
        "CARGO_FEATURE_MAMBA",
        "CARGO_FEATURE_GPU_LOGITS",
        "CARGO_FEATURE_CUDA",
        "CARGO_FEATURE_HF_TOKENIZER",
        "CARGO_FEATURE_COLLECT",
        "CARGO_FEATURE_SIMD",
        "CARGO_FEATURE_TEST_HELPERS",
        "CARGO_FEATURE_THERAPEUTIC",
        "CARGO_FEATURE_SPEECH_DATA",
        "CARGO_FEATURE_CODE_SHEAF_EVAL",
        "CARGO_FEATURE_HIGHWAY_PROJECTION",
    ] {
        println!("cargo:rerun-if-env-changed={feature}");
    }

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    let workspace_root = manifest_dir.join("../../..");
    let source = read_required(&manifest_dir.join("src/lexical_binding.rs"));
    let build_script = read_required(&manifest_dir.join("build.rs"));
    let crate_manifest = read_required(&manifest_dir.join("Cargo.toml"));
    let workspace_manifest = read_required(&workspace_root.join("Cargo.toml"));
    let cargo_lock = read_required(&workspace_root.join("Cargo.lock"));
    let rust_toolchain = read_required(&workspace_root.join("rust-toolchain.toml"));
    let rustc_identity = rustc_identity();
    let cargo_identity = cargo_identity();
    let rustc_wrapper = env::var("RUSTC_WRAPPER").unwrap_or_default().into_bytes();
    let rustc_workspace_wrapper =
        env::var("RUSTC_WORKSPACE_WRAPPER").unwrap_or_default().into_bytes();
    let cargo_features = cargo_feature_identity();
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS").unwrap_or_default().into_bytes();
    let target = env::var("TARGET").unwrap_or_default().into_bytes();
    let host = env::var("HOST").unwrap_or_default().into_bytes();

    // The implementation revision is an exact content identity of the compiler's source module
    // plus this build-time identity mechanism. It is deliberately not a manually maintained
    // version label and therefore cannot silently survive a source edit.
    let implementation_revision = domain_digest(
        b"symthaea-broca-unimorph-compiler-implementation-revision-v1",
        &[&source, &build_script],
    );

    // The parser revision is narrower: only the explicitly delimited source-format parser
    // surface is hashed, plus the build identity mechanism that interprets the markers.
    let parser_surface = delimited_surface(
        &source,
        b"// BEGIN UNIMORPH_TSV_SOURCE_PARSER_SURFACE_V1",
        b"// END UNIMORPH_TSV_SOURCE_PARSER_SURFACE_V1",
    )
    .unwrap_or_else(|error| panic!("{error}"));
    let parser_revision = domain_digest(
        b"symthaea-broca-unimorph-source-parser-revision-v1",
        &[parser_surface, &build_script],
    );

    println!(
        "cargo:rustc-env=SYMTHAEA_UNIMORPH_TSV_COMPILER_IMPLEMENTATION_REVISION={implementation_revision}"
    );
    // This is intentionally separate from compiler source identity: the same checked-in source
    // can have different dependency/toolchain semantics if its build context changes.
    let build_context_revision = domain_digest(
        b"symthaea-broca-unimorph-compiler-build-context-revision-v3",
        &[
            &crate_manifest,
            &workspace_manifest,
            &cargo_lock,
            &rust_toolchain,
            &rustc_identity,
            &cargo_identity,
            &rustc_wrapper,
            &rustc_workspace_wrapper,
            &cargo_features,
            &rustflags,
            &target,
            &host,
        ],
    );

    println!(
        "cargo:rustc-env=SYMTHAEA_UNIMORPH_TSV_SOURCE_PARSER_REVISION={parser_revision}"
    );
    println!(
        "cargo:rustc-env=SYMTHAEA_UNIMORPH_TSV_COMPILER_BUILD_CONTEXT_REVISION={build_context_revision}"
    );
}
