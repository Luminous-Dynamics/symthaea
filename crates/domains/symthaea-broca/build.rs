// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
//! Build-time identities for the source-backed UniMorph compiler.

use std::{
    env,
    fs,
    path::{Path, PathBuf},
};

fn domain_digest(domain: &[u8], surfaces: &[&[u8]]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(domain);
    hasher.update(b"\0");
    for surface in surfaces {
        hasher.update(&(surface.len() as u64).to_le_bytes());
        hasher.update(surface);
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

fn main() {
    println!("cargo:rerun-if-changed=src/lexical_binding.rs");
    println!("cargo:rerun-if-changed=build.rs");

    let manifest_dir = PathBuf::from(
        env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR must be set by Cargo"),
    );
    let source = read_required(&manifest_dir.join("src/lexical_binding.rs"));
    let build_script = read_required(&manifest_dir.join("build.rs"));

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
    println!(
        "cargo:rustc-env=SYMTHAEA_UNIMORPH_TSV_SOURCE_PARSER_REVISION={parser_revision}"
    );
}
