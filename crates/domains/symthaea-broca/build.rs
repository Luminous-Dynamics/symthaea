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
        .ok_or_else(|| {
            format!(
                "missing compiler revision marker: {}",
                String::from_utf8_lossy(begin)
            )
        })?;
    let content_start = start + begin.len();
    let relative_end = source[content_start..]
        .windows(end.len())
        .position(|window| window == end)
        .ok_or_else(|| {
            format!(
                "missing compiler revision marker: {}",
                String::from_utf8_lossy(end)
            )
        })?;
    let content_end = content_start + relative_end;
    if content_start >= content_end {
        return Err("compiler revision marker surface is empty".to_string());
    }
    Ok(&source[content_start..content_end])
}

fn read_required(path: &Path) -> Vec<u8> {
    fs::read(path).unwrap_or_else(|error| {
        panic!(
            "failed to read compiler identity input {}: {error}",
            path.display()
        )
    })
}

fn read_optional(path: &Path) -> Vec<u8> {
    match fs::read(path) {
        Ok(bytes) => {
            let mut surface = path.to_string_lossy().as_bytes().to_vec();
            surface.extend_from_slice(b"\0present\0");
            surface.extend_from_slice(&bytes);
            surface
        }
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            let mut surface = path.to_string_lossy().as_bytes().to_vec();
            surface.extend_from_slice(b"\0absent");
            surface
        }
        Err(error) => {
            panic!(
                "failed to read optional compiler identity input {}: {error}",
                path.display()
            )
        }
    }
}

fn rustc_identity() -> Vec<u8> {
    let rustc = env::var_os("RUSTC").unwrap_or_else(|| "rustc".into());
    let output = Command::new(&rustc)
        .arg("--version")
        .arg("--verbose")
        .output()
        .unwrap_or_else(|error| {
            panic!("failed to execute rustc for compiler identity: {error}")
        });
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
        .unwrap_or_else(|error| {
            panic!("failed to execute cargo for compiler identity: {error}")
        });
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

fn cargo_cfg_identity() -> Vec<u8> {
    let mut cfg = env::vars()
        .filter_map(|(key, value)| {
            key.strip_prefix("CARGO_CFG_")
                .map(|feature| format!("{feature}={value}"))
        })
        .collect::<Vec<_>>();
    cfg.sort_unstable();
    cfg.join("\n").into_bytes()
}

fn native_build_environment_identity() -> Vec<u8> {
    const VARIABLES: &[&str] = &[
        "CC",
        "CXX",
        "AR",
        "RANLIB",
        "LD",
        "RUSTC_LINKER",
        "CFLAGS",
        "CXXFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "PKG_CONFIG",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_SYSROOT_DIR",
        "OPENSSL_DIR",
        "OPENSSL_LIB_DIR",
        "OPENSSL_INCLUDE_DIR",
        "LIBCLANG_PATH",
        "BINDGEN_EXTRA_CLANG_ARGS",
        "CMAKE_PREFIX_PATH",
        "CMAKE_GENERATOR",
        "CMAKE_TOOLCHAIN_FILE",
        "CUDA_HOME",
        "CUDA_PATH",
        "VULKAN_SDK",
    ];
    let mut entries = Vec::with_capacity(VARIABLES.len());
    for variable in VARIABLES {
        let value = env::var(variable).unwrap_or_default();
        entries.push(format!("{variable}={value}"));
    }
    entries.join("
").into_bytes()
}

fn system_package_identity() -> Vec<u8> {
    let packages = ["pkg-config", "libssl-dev", "libclang-dev", "cmake"];
    let mut entries = Vec::with_capacity(packages.len());
    for package in packages {
        let output = Command::new("dpkg-query")
            .args(["-W", "-f=${Package}=${Version}", package])
            .output();
        let value = match output {
            Ok(output) if output.status.success() => String::from_utf8_lossy(&output.stdout).trim().to_owned(),
            _ => format!("{package}=unavailable"),
        };
        entries.push(value);
    }
    entries.sort_unstable();
    entries.join("\n").into_bytes()
}

fn main() {
    println!("cargo:rerun-if-changed=src/lexical_binding.rs");
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-changed=Cargo.toml");
    println!("cargo:rerun-if-changed=../../../Cargo.toml");
    println!("cargo:rerun-if-changed=../../../Cargo.lock");
    println!("cargo:rerun-if-changed=../../../rust-toolchain.toml");
    println!("cargo:rerun-if-changed=../../../.cargo/config.toml");
    println!("cargo:rerun-if-changed=../../../.cargo/config");
    println!("cargo:rerun-if-env-changed=RUSTC");
    println!("cargo:rerun-if-env-changed=CARGO");
    println!("cargo:rerun-if-env-changed=RUSTC_WRAPPER");
    println!("cargo:rerun-if-env-changed=RUSTC_WORKSPACE_WRAPPER");
    println!("cargo:rerun-if-env-changed=CARGO_ENCODED_RUSTFLAGS");
    println!("cargo:rerun-if-env-changed=BROCA_NATIVE_PACKAGE_CONTEXT");
    println!("cargo:rerun-if-env-changed=BROCA_RUSTUP_VERSION");
    println!("cargo:rerun-if-env-changed=BROCA_NATIVE_TOOLCHAIN_CONTEXT");
    for variable in [
        "CC",
        "CXX",
        "AR",
        "RANLIB",
        "LD",
        "RUSTC_LINKER",
        "CFLAGS",
        "CXXFLAGS",
        "CPPFLAGS",
        "LDFLAGS",
        "PKG_CONFIG",
        "PKG_CONFIG_PATH",
        "PKG_CONFIG_LIBDIR",
        "PKG_CONFIG_SYSROOT_DIR",
        "OPENSSL_DIR",
        "OPENSSL_LIB_DIR",
        "OPENSSL_INCLUDE_DIR",
        "LIBCLANG_PATH",
        "BINDGEN_EXTRA_CLANG_ARGS",
        "CMAKE_PREFIX_PATH",
        "CMAKE_GENERATOR",
        "CMAKE_TOOLCHAIN_FILE",
        "CUDA_HOME",
        "CUDA_PATH",
        "VULKAN_SDK",
    ] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    for variable in ["PROFILE", "DEBUG", "OPT_LEVEL", "NUM_JOBS", "RUNNER_OS", "RUNNER_ARCH", "ImageOS", "ImageVersion"] {
        println!("cargo:rerun-if-env-changed={variable}");
    }
    for (key, _) in env::vars() {
        if key.starts_with("CARGO_CFG_") {
            println!("cargo:rerun-if-env-changed={key}");
        }
    }
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
    let cargo_config_toml = read_optional(&workspace_root.join(".cargo/config.toml"));
    let cargo_config = read_optional(&workspace_root.join(".cargo/config"));
    let rustc_identity = rustc_identity();
    let cargo_identity = cargo_identity();
    let rustc_wrapper = env::var("RUSTC_WRAPPER").unwrap_or_default().into_bytes();
    let rustc_workspace_wrapper =
        env::var("RUSTC_WORKSPACE_WRAPPER").unwrap_or_default().into_bytes();
    let cargo_features = cargo_feature_identity();
    let cargo_cfg = cargo_cfg_identity();
    let system_packages = env::var("BROCA_NATIVE_PACKAGE_CONTEXT")
        .map(String::into_bytes)
        .unwrap_or_else(|_| system_package_identity());
    let rustup_version = env::var("BROCA_RUSTUP_VERSION")
        .unwrap_or_default()
        .into_bytes();
    let native_toolchain_context = env::var("BROCA_NATIVE_TOOLCHAIN_CONTEXT")
        .unwrap_or_default()
        .into_bytes();
    let native_build_environment = native_build_environment_identity();
    let profile = env::var("PROFILE").unwrap_or_default().into_bytes();
    let debug = env::var("DEBUG").unwrap_or_default().into_bytes();
    let opt_level = env::var("OPT_LEVEL").unwrap_or_default().into_bytes();
    let num_jobs = env::var("NUM_JOBS").unwrap_or_default().into_bytes();
    let rustflags = env::var("CARGO_ENCODED_RUSTFLAGS")
        .unwrap_or_default()
        .into_bytes();
    let target = env::var("TARGET").unwrap_or_default().into_bytes();
    let host = env::var("HOST").unwrap_or_default().into_bytes();
    let runner_os = env::var("RUNNER_OS").unwrap_or_default().into_bytes();
    let runner_arch = env::var("RUNNER_ARCH").unwrap_or_default().into_bytes();
    let image_os = env::var("ImageOS").unwrap_or_default().into_bytes();
    let image_version = env::var("ImageVersion").unwrap_or_default().into_bytes();

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
        b"symthaea-broca-unimorph-compiler-build-context-revision-v9",
        &[
            &crate_manifest,
            &workspace_manifest,
            &cargo_lock,
            &rust_toolchain,
            &cargo_config_toml,
            &cargo_config,
            &rustc_identity,
            &cargo_identity,
            &rustc_wrapper,
            &rustc_workspace_wrapper,
            &cargo_features,
            &cargo_cfg,
            &system_packages,
            &rustup_version,
            &native_toolchain_context,
            &native_build_environment,
            &profile,
            &debug,
            &opt_level,
            &num_jobs,
            &rustflags,
            &target,
            &host,
            &runner_os,
            &runner_arch,
            &image_os,
            &image_version,
        ],
    );

    println!(
        "cargo:rustc-env=SYMTHAEA_UNIMORPH_TSV_SOURCE_PARSER_REVISION={parser_revision}"
    );
    println!(
        "cargo:rustc-env=SYMTHAEA_UNIMORPH_TSV_COMPILER_BUILD_CONTEXT_REVISION={build_context_revision}"
    );
}
