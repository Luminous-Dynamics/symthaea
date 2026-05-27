// Robust, forward-compatible build script for CUDA 12.9 environments
fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    println!("cargo:rerun-if-env-changed=CUDA_ROOT");
    println!("cargo:rerun-if-env-changed=CUDA_PATH");
    println!("cargo:rerun-if-env-changed=CUDA_TOOLKIT_ROOT_DIR");

    // Pass the compile-time environment string variables expected by src/lib.rs
    println!("cargo:rustc-env=CUDA_MAJOR_VERSION=12");
    println!("cargo:rustc-env=CUDA_MINOR_VERSION=2");

    // Emit exactly ONE feature flag mapping to clear ambiguity collisions
    println!("cargo:rustc-cfg=feature=\"cuda-12020\"");
}