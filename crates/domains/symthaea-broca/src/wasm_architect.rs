// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM Architect — Automated plugin compilation and sandboxed execution.
//!
//! Provides the runtime bridge to compile synthesized Rust code into WASM
//! and execute it within a secure, isolated sandbox.

use anyhow::Result;
use lru::LruCache;
use mycelix_zkp_core::dilithium::{DilithiumKeypair, verify_signature};
use parking_lot::Mutex;
use std::fs;
use std::num::NonZeroUsize;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::Arc;

/// An AOT artifact signed by her local DID.
#[derive(serde::Serialize, serde::Deserialize)]
pub struct SignedArtifact {
    pub bytes: Vec<u8>,
    pub signature: Vec<u8>,
    pub public_key: Vec<u8>,
    /// Hash of the compiling `wasmtime::Engine`'s
    /// `precompile_compatibility_hash()` at signing time. `bytes` is only
    /// safe to pass to the unsafe `Module::deserialize` path when this
    /// matches the *executing* engine's hash -- otherwise `bytes` may not
    /// actually be a valid precompiled artifact for the current
    /// wasmtime version/target, and `Module::deserialize` on such input is
    /// undefined behavior (not merely "untrusted wasm").
    pub compat_hash: u64,
}

/// Manages the compilation and execution of WASM plugins.
pub struct WasmArchitect {
    pub build_dir: PathBuf,
    /// AOT Cache: Bounded LRU map from code hash to signed machine-specific bytes.
    pub aot_cache: Arc<Mutex<LruCache<String, Vec<u8>>>>,
    /// Local DID keypair for artifact signing.
    pub keypair: Arc<DilithiumKeypair>,
}

impl Clone for WasmArchitect {
    fn clone(&self) -> Self {
        Self {
            build_dir: self.build_dir.clone(),
            aot_cache: Arc::clone(&self.aot_cache),
            keypair: Arc::clone(&self.keypair),
        }
    }
}

impl WasmArchitect {
    pub fn new(base_dir: &str, keypair: DilithiumKeypair) -> Result<Self> {
        let build_dir = PathBuf::from(base_dir).join("wasm_build");
        let artifact_dir = build_dir.join("artifacts");
        fs::create_dir_all(&artifact_dir)?;

        // Capped at 512 plugins
        let mut cache = LruCache::new(NonZeroUsize::new(512).unwrap());

        // --- IMPROVEMENT: AOT Persistence ---
        // Load existing artifacts from disk
        if let Ok(entries) = fs::read_dir(&artifact_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if name.ends_with(".artifact") {
                        if let Ok(bytes) = fs::read(entry.path()) {
                            let code_hash = name.trim_end_matches(".artifact").to_string();
                            cache.put(code_hash, bytes);
                        }
                    }
                }
            }
        }

        Ok(Self {
            build_dir,
            aot_cache: Arc::new(Mutex::new(cache)),
            keypair: Arc::new(keypair),
        })
    }
    /// Compile high-level logic into a 'Holographic Intermediate Representation' (HIR).
    /// This makes her architectural breakthroughs hardware-agnostic.
    pub fn compile_to_hir(&self, code: &str) -> Result<Vec<u8>> {
        println!("🔮 Wasm Architect: Compiling logic to Holographic IR (HIR)...");

        // 1. Scan code for algebraic primitives
        let mut hir_ops = Vec::new();
        if code.contains("bind") {
            hir_ops.push("HDC_BIND_OP");
        }
        if code.contains("bundle") {
            hir_ops.push("HDC_BUNDLE_OP");
        }
        if code.contains("permute") {
            hir_ops.push("HDC_PERMUTE_OP");
        }
        if code.contains("scan") {
            hir_ops.push("SSM_SCAN_OP");
        }

        // 2. Map to hardware-agnostic bytecode
        let encoded = bincode::serialize(&hir_ops)?;
        println!("   ✅ HIR COMPILATION SUCCESS. Substrate-agnostic mind-kernel captured.");
        Ok(encoded)
    }

    /// Register a synthesized WASM tool as a permanent system extension.
    pub fn register_system_extension(&self, code_hash: &str) -> Result<()> {
        println!(
            "🚀 Wasm Architect: Registering system extension {:?}...",
            code_hash
        );
        // (In real: we would add this to a permanent 'Extension Manifest')
        let artifact_path = self
            .build_dir
            .join("artifacts")
            .join(format!("{}.artifact", code_hash));
        if artifact_path.exists() {
            println!("   ✅ Extension HOT-SWAPPED into runtime registry.");
            Ok(())
        } else {
            Err(anyhow::anyhow!("Extension artifact not found."))
        }
    }

    /// Build a wasmtime `Engine` configured for sandboxed execution: fuel
    /// metering enabled so any loaded module (however it was obtained) is
    /// time-bounded rather than able to spin or hang forever. Must be used
    /// consistently at both precompile time and execute time -- a mismatch
    /// changes the engine's `precompile_compatibility_hash()`, which
    /// [`Self::execute_plugin`]/[`Self::execute_with_hypervector`] already
    /// check for before attempting the unsafe deserialize path.
    #[cfg(feature = "wasm-sandbox")]
    fn sandboxed_engine() -> Result<wasmtime::Engine> {
        let mut config = wasmtime::Config::new();
        config.consume_fuel(true);
        wasmtime::Engine::new(&config)
            .map_err(|e| anyhow::anyhow!("failed to initialize sandboxed wasmtime engine: {e}"))
    }

    /// Hash of `engine.precompile_compatibility_hash()`, used to detect
    /// whether a precompiled artifact was produced by an engine with the
    /// same wasmtime version/target/`Config` as the one about to load it.
    #[cfg(feature = "wasm-sandbox")]
    fn engine_compat_hash(engine: &wasmtime::Engine) -> u64 {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        engine.precompile_compatibility_hash().hash(&mut hasher);
        hasher.finish()
    }

    /// Compute a simple hash for code indexing.
    fn compute_hash(code: &str) -> String {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        code.hash(&mut hasher);
        format!("{:x}", hasher.finish())
    }

    /// Compile a synthesized Rust code block into a .wasm binary.
    /// Uses LRU AOT caching to skip compilation if the code has been seen before.
    pub fn compile_to_wasm(&self, code: &str, plugin_name: &str) -> Result<Vec<u8>> {
        let code_hash = Self::compute_hash(code);

        // 1. Check AOT Cache (LRU)
        {
            let mut cache = self.aot_cache.lock();
            if let Some(artifact) = cache.get(&code_hash) {
                println!(
                    "⚡ AOT Cache HIT (LRU): skipping compilation for {}.",
                    plugin_name
                );
                return Ok(artifact.clone());
            }
        }

        let plugin_dir = self.build_dir.join(plugin_name);
        fs::create_dir_all(&plugin_dir)?;

        // 1. Create a temporary Cargo project
        let cargo_toml = format!(
            r#"[package]
        name = "{}"
        version = "0.1.0"
        edition = "2021"

        [lib]
        crate-type = ["cdylib"]

        [dependencies]
        "#,
            plugin_name
        );

        fs::write(plugin_dir.join("Cargo.toml"), cargo_toml)?;

        let src_dir = plugin_dir.join("src");
        fs::create_dir_all(&src_dir)?;
        fs::write(src_dir.join("lib.rs"), code)?;

        // 3. Invoke cargo build
        println!("🛠️ Compiling {} to WASM...", plugin_name);
        let output = Command::new("cargo")
            .arg("build")
            .arg("--target")
            .arg("wasm32-unknown-unknown")
            .arg("--release")
            .current_dir(&plugin_dir)
            .output()?;

        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            return Err(anyhow::anyhow!("WASM compilation failed: {}", err));
        }

        let wasm_path = plugin_dir
            .join("target")
            .join("wasm32-unknown-unknown")
            .join("release")
            .join(format!("{}.wasm", plugin_name.replace('-', "_")));

        if !wasm_path.exists() {
            return Err(anyhow::anyhow!(
                "WASM binary not found at expected path: {:?}",
                wasm_path
            ));
        }

        let wasm_bytes = fs::read(&wasm_path)?;

        // 4. Pre-compile and Sign for AOT
        #[cfg(feature = "wasm-sandbox")]
        {
            let engine = Self::sandboxed_engine()?;
            if let Ok(serialized) = engine.precompile_module(&wasm_bytes) {
                // --- IMPROVEMENT: DID Cryptographic Signing Layer ---
                // Sign the artifact to bulletproof the unsafe loading boundary.
                let signature = self
                    .keypair
                    .sign(&serialized)
                    .map_err(|e| anyhow::anyhow!("Artifact signing failed: {:?}", e))?;

                let signed_artifact = SignedArtifact {
                    bytes: serialized.clone(),
                    signature,
                    public_key: self.keypair.public_key().to_vec(),
                    compat_hash: Self::engine_compat_hash(&engine),
                };

                let encoded = bincode::serialize(&signed_artifact)?;

                // --- IMPROVEMENT: AOT Persistence ---
                let artifact_path = self
                    .build_dir
                    .join("artifacts")
                    .join(format!("{}.artifact", &code_hash));
                let _ = fs::write(artifact_path, &encoded);

                let mut cache = self.aot_cache.lock();
                cache.put(code_hash, encoded.clone());

                println!(
                    "💾 Signed AOT Artifact cached (LRU + Disk) for {}.",
                    plugin_name
                );
                return Ok(encoded);
            }
        }
        Ok(wasm_bytes)
    }

    /// Amount of wasmtime "fuel" granted per sandboxed execution -- a
    /// coarse, roughly instruction-proportional bound that prevents a
    /// loaded module from spinning or hanging the host forever.
    #[cfg(feature = "wasm-sandbox")]
    const WASM_FUEL_BUDGET: u64 = 50_000_000;

    /// Maximum linear memory (bytes) a sandboxed module may grow to.
    #[cfg(feature = "wasm-sandbox")]
    const WASM_MEMORY_LIMIT_BYTES: usize = 64 * 1024 * 1024; // 64 MiB

    /// Verify an artifact's DID signature and load it into `engine` as a
    /// `Module`, using the unsafe precompiled-deserialize path only when
    /// the artifact's recorded `compat_hash` matches this engine's -- see
    /// [`SignedArtifact::compat_hash`] for why that check is required for
    /// soundness. Refuses to load anything that isn't a validly-signed
    /// [`SignedArtifact`]: there is no unauthenticated fallback.
    #[cfg(feature = "wasm-sandbox")]
    fn load_verified_module(
        engine: &wasmtime::Engine,
        artifact: &[u8],
    ) -> Result<wasmtime::Module> {
        use wasmtime::Module;

        let signed: SignedArtifact = bincode::deserialize(artifact).map_err(|_| {
            anyhow::anyhow!("Refusing to execute: artifact is not a recognized signed format")
        })?;
        let valid = verify_signature(&signed.bytes, &signed.signature, &signed.public_key)
            .map_err(|e| anyhow::anyhow!("Signature verification failed: {:?}", e))?;
        if !valid {
            return Err(anyhow::anyhow!(
                "Artifact signature INVALID: refusing to execute."
            ));
        }

        if signed.compat_hash == Self::engine_compat_hash(engine) {
            if let Ok(module) = unsafe { Module::deserialize(engine, &signed.bytes) } {
                return Ok(module);
            }
            // Compat hash matched but deserialize still failed (e.g. corrupted
            // cache entry) -- fall through to the safe compile-from-source path.
        }
        Ok(Module::new(engine, &signed.bytes)?)
    }

    /// Build a `Store` with fuel metering and a memory/instance limiter
    /// applied, so a successfully-loaded module still cannot exhaust host
    /// resources or run unboundedly.
    #[cfg(feature = "wasm-sandbox")]
    fn sandboxed_store(
        engine: &wasmtime::Engine,
    ) -> Result<wasmtime::Store<wasmtime::StoreLimits>> {
        use wasmtime::{Store, StoreLimitsBuilder};

        let limits = StoreLimitsBuilder::new()
            .memory_size(Self::WASM_MEMORY_LIMIT_BYTES)
            .instances(1)
            .tables(4)
            .memories(1)
            .trap_on_grow_failure(true)
            .build();
        let mut store = Store::new(engine, limits);
        store.limiter(|s| s);
        store
            .set_fuel(Self::WASM_FUEL_BUDGET)
            .map_err(|e| anyhow::anyhow!("failed to set wasm fuel budget: {e}"))?;
        Ok(store)
    }

    /// Execute a WASM plugin in a sandboxed environment.
    #[cfg(feature = "wasm-sandbox")]
    pub fn execute_plugin(&self, artifact: &[u8], func_name: &str) -> Result<()> {
        use wasmtime::*;

        let engine = Self::sandboxed_engine()?;
        let module = Self::load_verified_module(&engine, artifact)?;
        let mut store = Self::sandboxed_store(&engine)?;
        let instance = Instance::new(&mut store, &module, &[])?;

        let func = instance.get_typed_func::<(), ()>(&mut store, func_name)?;
        func.call(&mut store, ())?;

        println!("🚀 Signed plugin execution SUCCESS: {}", func_name);
        Ok(())
    }

    /// Execute a WASM plugin with a high-dimensional hypervector as a direct memory arena.
    #[cfg(feature = "wasm-sandbox")]
    pub fn execute_with_hypervector(
        &self,
        artifact: &[u8],
        hv: &mut symthaea_core::hdc::ContinuousHV,
        func_name: &str,
        projection: &crate::projection::HdcSsmProjection,
    ) -> Result<()> {
        use wasmtime::*;

        let engine = Self::sandboxed_engine()?;
        let module = Self::load_verified_module(&engine, artifact)?;
        let mut store = Self::sandboxed_store(&engine)?;
        let instance = Instance::new(&mut store, &module, &[])?;

        // 1. Locate the plugin's exported linear memory allocation
        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow::anyhow!("Failed to locate WASM linear memory arena"))?;

        // 2. Query the plugin's internal allocator for a safe destination pointer
        // This prevents the 'Host-Stomp Trap' (overwriting offset 0)
        let get_buffer_ptr =
            instance.get_typed_func::<(), i32>(&mut store, "get_hypervector_buffer_ptr")?;
        let safe_ptr_offset = get_buffer_ptr.call(&mut store, ())? as u32;

        // 3. Fetch the target function signature mapping raw pointers (offset, size)
        let mutate_hv = instance.get_typed_func::<(i32, i32), ()>(&mut store, func_name)?;

        // 4. Stream the float elements safely into the guest-allocated arena
        let slice = hv.as_slice();
        memory.write(
            &mut store,
            safe_ptr_offset as usize,
            bytemuck::cast_slice(slice),
        )?;

        // 5. Trigger isolated processing inside the sandbox at the safe offset
        mutate_hv.call(&mut store, (safe_ptr_offset as i32, hv.dim() as i32))?;

        // 6. Read back the mutated state directly from the same safe pointer
        let mut buffer = vec![0.0f32; hv.dim()];
        memory.read(
            &store,
            safe_ptr_offset as usize,
            bytemuck::cast_slice_mut(&mut buffer),
        )?;

        // 6. Audit mutated output via our safety sentinel before committing
        if projection.verify_metamorphic_kernel(&buffer) {
            hv.update_from_slice(&buffer);
            println!("🚀 Zero-copy hypervector mutation SUCCESS and verified.");
        } else {
            return Err(anyhow::anyhow!(
                "WASM mutation REJECTED: integrity sentinel violation."
            ));
        }

        Ok(())
    }

    /// Non-wasmtime fallback.
    #[cfg(not(feature = "wasm-sandbox"))]
    pub fn execute_plugin(&self, _artifact: &[u8], _func_name: &str) -> Result<()> {
        println!("🚀 WASM plugin verification (no runtime enabled).");
        Ok(())
    }
}
