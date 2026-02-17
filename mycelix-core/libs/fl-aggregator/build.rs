//! Build script for fl-aggregator.
//!
//! This build script is currently minimal but can be extended to compile
//! Protocol Buffer definitions when the `grpc` feature is enabled.
//!
//! ## Enabling Proto Compilation
//!
//! To compile the proto files in `proto/mycelix_fl.proto`, uncomment the
//! tonic_build section below and add the build dependencies:
//!
//! ```toml
//! [build-dependencies]
//! tonic-build = "0.12"
//! ```
//!
//! The current implementation uses manually-defined stub types that mirror
//! the proto schema, allowing the module to compile without proto compilation.
//! This approach is useful for:
//! - Faster iteration during development
//! - Avoiding protoc dependency in CI/CD
//! - Cross-compilation scenarios
//!
//! When ready to switch to generated code:
//! 1. Uncomment the proto compilation section below
//! 2. Add `include!(concat!(env!("OUT_DIR"), "/mycelix.fl.v1.rs"));` in mod.rs
//! 3. Remove the manually-defined stub types

fn main() {
    // Currently using manually-defined proto stub types.
    // Uncomment below to enable proto compilation:
    //
    // #[cfg(feature = "grpc")]
    // {
    //     let proto_file = "proto/mycelix_fl.proto";
    //
    //     // Only rebuild if proto file changes
    //     println!("cargo:rerun-if-changed={}", proto_file);
    //
    //     tonic_build::configure()
    //         .build_server(true)
    //         .build_client(true)
    //         .out_dir("src/grpc/generated")  // Or use OUT_DIR for build artifacts
    //         .compile(&[proto_file], &["proto/"])
    //         .expect("Failed to compile proto file");
    // }

    // Trigger rebuild if build.rs changes
    println!("cargo:rerun-if-changed=build.rs");
}
