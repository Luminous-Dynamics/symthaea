// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! WASM Architect — automated plugin compilation and sandboxed execution.
//!
//! Synthesized Rust is compiled to portable WebAssembly, wrapped in a signed
//! artifact, and executed only after the artifact is verified against the
//! architect's configured Dilithium signer. Persisted artifacts deliberately
//! contain portable WASM rather than serialized native/AOT code, so loading never
//! requires Wasmtime's unsafe precompiled-deserialization boundary.

use anyhow::Result;
use lru::LruCache;
use mycelix_zkp_core::dilithium::{DilithiumKeypair, verify_signature};
use parking_lot::Mutex;
use sha2::{Digest, Sha256};
use std::fmt::Write as _;
use std::fs;
use std::num::NonZeroUsize;
use std::path::PathBuf;
use std::process::Command;
use std::sync::Arc;

const SIGNED_ARTIFACT_FORMAT_VERSION: u16 = 2;
const SIGNING_DOMAIN: &[u8] = b"symthaea.wasm-artifact.v2\0";
const CACHE_DOMAIN: &[u8] = b"symthaea.wasm-source-cache.v2\0";
const MAX_PLUGIN_NAME_LEN: usize = 64;

/// Portable WASM artifact signed by the architect's configured local signer.
///
/// The artifact intentionally does **not** carry a public key that can authorize
/// itself. Verification is performed against the trusted key configured on the
/// [`WasmArchitect`] instance that consumes it.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub struct SignedArtifact {
    format_version: u16,
    wasm_bytes: Vec<u8>,
    wasm_sha256: [u8; 32],
    signature: Vec<u8>,
}

/// Manages compilation and execution of locally synthesized WASM plugins.
pub struct WasmArchitect {
    pub build_dir: PathBuf,
    /// Bounded cache of signed portable artifacts. Disk entries are treated as
    /// untrusted bytes until `verify_signed_artifact` succeeds at use time.
    pub artifact_cache: Arc<Mutex<LruCache<String, Vec<u8>>>>,
    /// Local signer and trust root for synthesized artifact verification.
    pub keypair: Arc<DilithiumKeypair>,
}

impl Clone for WasmArchitect {
    fn clone(&self) -> Self {
        Self {
            build_dir: self.build_dir.clone(),
            artifact_cache: Arc::clone(&self.artifact_cache),
            keypair: Arc::clone(&self.keypair),
        }
    }
}

impl WasmArchitect {
    pub fn new(base_dir: &str, keypair: DilithiumKeypair) -> Result<Self> {
        let build_dir = PathBuf::from(base_dir).join("wasm_build");
        let artifact_dir = build_dir.join("artifacts");
        fs::create_dir_all(&artifact_dir)?;

        // Capped at 512 plugins. Persistence is only an optimization: every
        // cached entry is cryptographically revalidated before it is trusted.
        let mut cache = LruCache::new(NonZeroUsize::new(512).unwrap());
        if let Ok(entries) = fs::read_dir(&artifact_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(code_hash) = name.strip_suffix(".artifact") {
                        if Self::valid_cache_key(code_hash) {
                            if let Ok(bytes) = fs::read(entry.path()) {
                                cache.put(code_hash.to_string(), bytes);
                            }
                        }
                    }
                }
            }
        }

        Ok(Self {
            build_dir,
            artifact_cache: Arc::new(Mutex::new(cache)),
            keypair: Arc::new(keypair),
        })
    }

    /// Compile high-level logic into a 'Holographic Intermediate Representation' (HIR).
    /// This makes her architectural breakthroughs hardware-agnostic.
    pub fn compile_to_hir(&self, code: &str) -> Result<Vec<u8>> {
        println!("🔮 Wasm Architect: Compiling logic to Holographic IR (HIR)...");

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

        let encoded = bincode::serialize(&hir_ops)?;
        println!("   ✅ HIR COMPILATION SUCCESS. Substrate-agnostic mind-kernel captured.");
        Ok(encoded)
    }

    /// Register a synthesized WASM tool as a permanent system extension.
    ///
    /// This remains a placeholder for the future extension manifest, but an
    /// artifact must now be structurally valid and signed by this architect's
    /// configured signer before registration can succeed.
    pub fn register_system_extension(&self, code_hash: &str) -> Result<()> {
        Self::validate_cache_key(code_hash)?;
        println!(
            "🚀 Wasm Architect: Registering system extension {:?}...",
            code_hash
        );
        let artifact_path = self
            .build_dir
            .join("artifacts")
            .join(format!("{}.artifact", code_hash));
        let encoded = fs::read(&artifact_path)
            .map_err(|_| anyhow::anyhow!("Extension artifact not found."))?;
        self.verify_signed_artifact(&encoded)?;
        println!("   ✅ Signed extension artifact accepted for runtime registration.");
        Ok(())
    }

    /// Build the one Wasmtime engine configuration used by this host.
    #[cfg(feature = "wasm-sandbox")]
    fn sandboxed_engine() -> Result<wasmtime::Engine> {
        let mut config = wasmtime::Config::new();
        config.consume_fuel(true);
        wasmtime::Engine::new(&config)
            .map_err(|e| anyhow::anyhow!("failed to initialize sandboxed wasmtime engine: {e}"))
    }

    fn sha256(bytes: &[u8]) -> [u8; 32] {
        Sha256::digest(bytes).into()
    }

    fn artifact_signing_message(wasm_sha256: [u8; 32]) -> Vec<u8> {
        let mut message = Vec::with_capacity(SIGNING_DOMAIN.len() + 2 + 32);
        message.extend_from_slice(SIGNING_DOMAIN);
        message.extend_from_slice(&SIGNED_ARTIFACT_FORMAT_VERSION.to_le_bytes());
        message.extend_from_slice(&wasm_sha256);
        message
    }

    fn encode_signed_artifact(&self, wasm_bytes: &[u8]) -> Result<Vec<u8>> {
        let wasm_sha256 = Self::sha256(wasm_bytes);
        let message = Self::artifact_signing_message(wasm_sha256);
        let signature = self
            .keypair
            .sign(&message)
            .map_err(|e| anyhow::anyhow!("Artifact signing failed: {e:?}"))?;
        let artifact = SignedArtifact {
            format_version: SIGNED_ARTIFACT_FORMAT_VERSION,
            wasm_bytes: wasm_bytes.to_vec(),
            wasm_sha256,
            signature,
        };
        Ok(bincode::serialize(&artifact)?)
    }

    /// Verify an artifact against the signer configured on this host.
    ///
    /// The artifact cannot nominate its own trust root. Old self-signed/AOT
    /// artifact formats therefore fail closed and are rebuilt from source when a
    /// cache hit encounters them.
    fn verify_signed_artifact(&self, artifact: &[u8]) -> Result<SignedArtifact> {
        let signed: SignedArtifact = bincode::deserialize(artifact).map_err(|_| {
            anyhow::anyhow!("Refusing artifact: unrecognized signed WASM format")
        })?;
        if signed.format_version != SIGNED_ARTIFACT_FORMAT_VERSION {
            return Err(anyhow::anyhow!(
                "Refusing artifact format version {}; expected {}",
                signed.format_version,
                SIGNED_ARTIFACT_FORMAT_VERSION
            ));
        }

        let actual_wasm_sha256 = Self::sha256(&signed.wasm_bytes);
        if actual_wasm_sha256 != signed.wasm_sha256 {
            return Err(anyhow::anyhow!(
                "Refusing artifact: portable WASM digest mismatch"
            ));
        }

        let message = Self::artifact_signing_message(signed.wasm_sha256);
        let valid = verify_signature(&message, &signed.signature, self.keypair.public_key())
            .map_err(|e| anyhow::anyhow!("Artifact signature verification failed: {e:?}"))?;
        if !valid {
            return Err(anyhow::anyhow!(
                "Artifact signer is not trusted by this WASM architect"
            ));
        }
        Ok(signed)
    }

    fn validate_plugin_name(plugin_name: &str) -> Result<()> {
        if plugin_name.is_empty() || plugin_name.len() > MAX_PLUGIN_NAME_LEN {
            return Err(anyhow::anyhow!(
                "plugin name must contain 1..={} characters",
                MAX_PLUGIN_NAME_LEN
            ));
        }
        if !plugin_name
            .bytes()
            .next()
            .is_some_and(|byte| byte.is_ascii_alphabetic())
        {
            return Err(anyhow::anyhow!(
                "plugin name must start with an ASCII letter"
            ));
        }
        if !plugin_name
            .bytes()
            .all(|byte| byte.is_ascii_alphanumeric() || matches!(byte, b'-' | b'_'))
        {
            return Err(anyhow::anyhow!(
                "plugin name may contain only ASCII letters, digits, '-' and '_'"
            ));
        }
        Ok(())
    }

    fn valid_cache_key(code_hash: &str) -> bool {
        code_hash.len() == 64 && code_hash.bytes().all(|byte| byte.is_ascii_hexdigit())
    }

    fn validate_cache_key(code_hash: &str) -> Result<()> {
        if Self::valid_cache_key(code_hash) {
            Ok(())
        } else {
            Err(anyhow::anyhow!(
                "extension cache key must be exactly 64 hexadecimal characters"
            ))
        }
    }

    /// Cryptographic cache key binding both the package name and exact source.
    fn compute_hash(code: &str, plugin_name: &str) -> String {
        let mut hasher = Sha256::new();
        hasher.update(CACHE_DOMAIN);
        hasher.update((plugin_name.len() as u64).to_le_bytes());
        hasher.update(plugin_name.as_bytes());
        hasher.update((code.len() as u64).to_le_bytes());
        hasher.update(code.as_bytes());
        let digest = hasher.finalize();
        let mut encoded = String::with_capacity(64);
        for byte in digest {
            write!(&mut encoded, "{byte:02x}").expect("writing to String cannot fail");
        }
        encoded
    }

    /// Compile synthesized Rust into portable WASM and return a signed artifact.
    ///
    /// The return format is identical whether or not `wasm-sandbox` is enabled;
    /// build features no longer change an artifact from signed bytes into raw
    /// executable input. A cached artifact is reused only after signature and
    /// digest verification against this host's configured signer.
    pub fn compile_to_wasm(&self, code: &str, plugin_name: &str) -> Result<Vec<u8>> {
        Self::validate_plugin_name(plugin_name)?;
        let code_hash = Self::compute_hash(code, plugin_name);

        let cached = {
            let mut cache = self.artifact_cache.lock();
            cache.get(&code_hash).cloned()
        };
        if let Some(artifact) = cached {
            match self.verify_signed_artifact(&artifact) {
                Ok(_) => {
                    println!(
                        "⚡ Signed WASM cache HIT: skipping compilation for {}.",
                        plugin_name
                    );
                    return Ok(artifact);
                }
                Err(error) => {
                    eprintln!(
                        "⚠️ Ignoring invalid/stale WASM cache entry for {}: {}",
                        plugin_name, error
                    );
                    self.artifact_cache.lock().pop(&code_hash);
                }
            }
        }

        let plugin_dir = self.build_dir.join(plugin_name);
        fs::create_dir_all(&plugin_dir)?;

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

        println!("🛠️ Compiling {} to portable WASM...", plugin_name);
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
        let encoded = self.encode_signed_artifact(&wasm_bytes)?;

        let artifact_path = self
            .build_dir
            .join("artifacts")
            .join(format!("{}.artifact", &code_hash));
        fs::write(artifact_path, &encoded)?;
        self.artifact_cache
            .lock()
            .put(code_hash, encoded.clone());

        println!("💾 Signed portable WASM artifact cached for {}.", plugin_name);
        Ok(encoded)
    }

    /// Amount of Wasmtime fuel granted per sandboxed execution.
    #[cfg(feature = "wasm-sandbox")]
    const WASM_FUEL_BUDGET: u64 = 50_000_000;

    /// Maximum linear memory a sandboxed module may grow to.
    #[cfg(feature = "wasm-sandbox")]
    const WASM_MEMORY_LIMIT_BYTES: usize = 64 * 1024 * 1024;

    /// Verify the signed portable artifact and compile it through Wasmtime's safe
    /// portable-WASM path. Persisted native/AOT deserialization is deliberately
    /// not used here.
    #[cfg(feature = "wasm-sandbox")]
    fn load_verified_module(
        &self,
        engine: &wasmtime::Engine,
        artifact: &[u8],
    ) -> Result<wasmtime::Module> {
        let signed = self.verify_signed_artifact(artifact)?;
        wasmtime::Module::new(engine, &signed.wasm_bytes)
            .map_err(|e| anyhow::anyhow!("verified portable WASM failed to compile: {e}"))
    }

    /// Build a Store with fuel metering and strict instance/memory limits.
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

    /// Execute a verified WASM plugin in the sandboxed host.
    #[cfg(feature = "wasm-sandbox")]
    pub fn execute_plugin(&self, artifact: &[u8], func_name: &str) -> Result<()> {
        use wasmtime::Instance;

        let engine = Self::sandboxed_engine()?;
        let module = self.load_verified_module(&engine, artifact)?;
        let mut store = Self::sandboxed_store(&engine)?;
        let instance = Instance::new(&mut store, &module, &[])?;

        let func = instance.get_typed_func::<(), ()>(&mut store, func_name)?;
        func.call(&mut store, ())?;

        println!("🚀 Signed plugin execution SUCCESS: {}", func_name);
        Ok(())
    }

    /// Execute a verified WASM plugin with a high-dimensional hypervector arena.
    #[cfg(feature = "wasm-sandbox")]
    pub fn execute_with_hypervector(
        &self,
        artifact: &[u8],
        hv: &mut symthaea_core::hdc::ContinuousHV,
        func_name: &str,
        projection: &symthaea_broca::projection::HdcSsmProjection,
    ) -> Result<()> {
        use wasmtime::Instance;

        let engine = Self::sandboxed_engine()?;
        let module = self.load_verified_module(&engine, artifact)?;
        let mut store = Self::sandboxed_store(&engine)?;
        let instance = Instance::new(&mut store, &module, &[])?;

        let memory = instance
            .get_memory(&mut store, "memory")
            .ok_or_else(|| anyhow::anyhow!("Failed to locate WASM linear memory arena"))?;

        let get_buffer_ptr =
            instance.get_typed_func::<(), i32>(&mut store, "get_hypervector_buffer_ptr")?;
        let safe_ptr_offset = get_buffer_ptr.call(&mut store, ())? as u32;

        let mutate_hv = instance.get_typed_func::<(i32, i32), ()>(&mut store, func_name)?;

        let slice = hv.as_slice();
        memory.write(
            &mut store,
            safe_ptr_offset as usize,
            bytemuck::cast_slice(slice),
        )?;

        mutate_hv.call(&mut store, (safe_ptr_offset as i32, hv.dim() as i32))?;

        let mut buffer = vec![0.0f32; hv.dim()];
        memory.read(
            &store,
            safe_ptr_offset as usize,
            bytemuck::cast_slice_mut(&mut buffer),
        )?;

        if projection.verify_metamorphic_kernel(&buffer) {
            hv.update_from_slice(&buffer);
            println!("🚀 Hypervector mutation SUCCESS and verified.");
        } else {
            return Err(anyhow::anyhow!(
                "WASM mutation REJECTED: integrity sentinel violation."
            ));
        }

        Ok(())
    }

    /// No runtime means no execution. Verification-only builds fail closed rather
    /// than reporting a synthetic successful execution.
    #[cfg(not(feature = "wasm-sandbox"))]
    pub fn execute_plugin(&self, _artifact: &[u8], _func_name: &str) -> Result<()> {
        Err(anyhow::anyhow!(
            "wasm-sandbox feature is not enabled; refusing to execute plugin"
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_base(label: &str) -> PathBuf {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        std::env::temp_dir().join(format!(
            "symthaea-wasm-architect-{label}-{}-{unique}",
            std::process::id()
        ))
    }

    fn architect(label: &str) -> WasmArchitect {
        WasmArchitect::new(
            temp_base(label).to_str().unwrap(),
            DilithiumKeypair::generate(),
        )
        .unwrap()
    }

    #[test]
    fn cache_key_is_cryptographic_and_binds_plugin_name() {
        let first = WasmArchitect::compute_hash("pub fn x() {}", "plugin-a");
        let same = WasmArchitect::compute_hash("pub fn x() {}", "plugin-a");
        let renamed = WasmArchitect::compute_hash("pub fn x() {}", "plugin-b");
        assert_eq!(first.len(), 64);
        assert_eq!(first, same);
        assert_ne!(first, renamed);
    }

    #[test]
    fn invalid_plugin_names_fail_before_filesystem_or_cargo_use() {
        for invalid in [
            "../escape",
            "nested/path",
            "plugin\n[dependencies]\nevil = \"*\"",
            "9starts-with-digit",
            "",
        ] {
            assert!(WasmArchitect::validate_plugin_name(invalid).is_err());
        }
        assert!(WasmArchitect::validate_plugin_name("safe-plugin_2").is_ok());
    }

    #[test]
    fn arbitrary_cache_paths_are_rejected() {
        assert!(WasmArchitect::validate_cache_key("../outside").is_err());
        assert!(WasmArchitect::validate_cache_key(&"a".repeat(64)).is_ok());
    }

    #[test]
    fn artifact_rejects_foreign_signer() {
        let trusted = architect("trusted");
        let foreign = architect("foreign");
        let encoded = foreign
            .encode_signed_artifact(b"\0asm\x01\0\0\0")
            .unwrap();
        let error = trusted.verify_signed_artifact(&encoded).unwrap_err();
        assert!(error.to_string().contains("not trusted"));
    }

    #[test]
    fn artifact_rejects_tampered_portable_wasm() {
        let host = architect("tamper");
        let encoded = host
            .encode_signed_artifact(b"\0asm\x01\0\0\0")
            .unwrap();
        let mut decoded: SignedArtifact = bincode::deserialize(&encoded).unwrap();
        decoded.wasm_bytes[0] ^= 0xff;
        let tampered = bincode::serialize(&decoded).unwrap();
        let error = host.verify_signed_artifact(&tampered).unwrap_err();
        assert!(error.to_string().contains("digest mismatch"));
    }

    #[test]
    fn artifact_rejects_tampered_digest_even_with_untouched_signature() {
        let host = architect("digest");
        let encoded = host
            .encode_signed_artifact(b"\0asm\x01\0\0\0")
            .unwrap();
        let mut decoded: SignedArtifact = bincode::deserialize(&encoded).unwrap();
        decoded.wasm_sha256[0] ^= 0x01;
        let tampered = bincode::serialize(&decoded).unwrap();
        assert!(host.verify_signed_artifact(&tampered).is_err());
    }

    #[cfg(feature = "wasm-sandbox")]
    #[test]
    fn verified_portable_wasm_loads_without_native_deserialization() {
        let host = architect("portable");
        let encoded = host
            .encode_signed_artifact(b"\0asm\x01\0\0\0")
            .unwrap();
        let engine = WasmArchitect::sandboxed_engine().unwrap();
        host.load_verified_module(&engine, &encoded).unwrap();
    }

    #[cfg(not(feature = "wasm-sandbox"))]
    #[test]
    fn execution_without_runtime_fails_closed() {
        let host = architect("no-runtime");
        assert!(host.execute_plugin(&[], "run").is_err());
    }
}
