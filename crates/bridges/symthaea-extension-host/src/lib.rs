// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! Fail-closed control-plane host for public Symthaea WebAssembly Components.
//!
//! A successful [`ControlPlaneHost::inspect`] means only that a component is
//! technically compatible with the presented manifest bytes and can execute its
//! zero-authority control interface within the configured **guest execution**
//! envelope. It does **not** mean that the component or signer is trusted,
//! admitted for a capability, or authorized to perform an action.
//!
//! The initial host intentionally links no WASI or Symthaea host imports. Any
//! component requiring ambient filesystem, network, clock, randomness, sensor,
//! actuator, or other host authority therefore fails before instantiation.
//!
//! JIT compilation occurs before a [`Store`] exists; store limits, fuel, and
//! epoch deadlines therefore bound guest execution, not compiler CPU/memory/time.
//! Arbitrary-public-package promotion requires a separately supervised compiler
//! worker rather than pretending these controls contain compilation.

#![deny(unsafe_code)]

use sha2::{Digest, Sha256};
use std::sync::mpsc;
use std::thread;
use std::time::Duration;
use symthaea_extension_core::{
    AbiVersion, ExtensionManifest, FilesystemPermission, NetworkPermission, PermissionSet,
    RuntimeKind,
};
use thiserror::Error;
use wasmtime::component::{Component, ComponentType, Lift, Linker};
use wasmtime::{Config, Engine, Store, StoreLimits, StoreLimitsBuilder};

/// Canonical exported control interface for ABI v1.
pub const CONTROL_INTERFACE_V1: &str = "luminous:symthaea-extension/control@1.0.0";
pub const CONTROL_ABI_V1: AbiVersion = AbiVersion { major: 1, minor: 0 };

/// Stable identity for the Wasm proposal/codegen profile used by this host.
///
/// Changing any accepted/rejected language feature or determinism setting must
/// mint a new profile identity rather than silently widening v1.
pub const CONTROL_WASM_PROFILE_V1: &str = "symthaea.extension.control-wasm-profile.v1";

/// Host-side ceiling independent of extension-requested resource budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ControlHostPolicy {
    pub max_manifest_bytes: usize,
    pub max_component_bytes: usize,
    pub max_memory_bytes: u64,
    pub max_fuel: u64,
    pub max_wall_time_ms: u64,
    pub max_output_bytes: u64,
    pub max_concurrency: u16,
}

impl Default for ControlHostPolicy {
    fn default() -> Self {
        Self {
            max_manifest_bytes: 256 * 1024,
            max_component_bytes: 16 * 1024 * 1024,
            max_memory_bytes: 64 * 1024 * 1024,
            max_fuel: 50_000_000,
            max_wall_time_ms: 1_000,
            max_output_bytes: 4 * 1024 * 1024,
            max_concurrency: 1,
        }
    }
}

/// Identity returned by the guest's WIT control interface.
#[derive(Debug, Clone, PartialEq, Eq, ComponentType, Lift)]
#[component(record)]
pub struct GuestExtensionIdentity {
    pub id: String,
    pub version: String,
    #[component(name = "abi-major")]
    pub abi_major: u16,
    #[component(name = "abi-minor")]
    pub abi_minor: u16,
    #[component(name = "manifest-digest")]
    pub manifest_digest: Vec<u8>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ComponentType, Lift)]
#[component(enum)]
#[repr(u8)]
pub enum GuestHealthState {
    #[component(name = "ready")]
    Ready,
    #[component(name = "degraded")]
    Degraded,
    #[component(name = "unavailable")]
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Eq, ComponentType, Lift)]
#[component(record)]
pub struct GuestHealthReport {
    pub state: GuestHealthState,
    pub message: Option<String>,
}

/// Result of technical control-plane inspection.
///
/// This deliberately contains no `trusted` or `admitted` boolean. Signer trust
/// and capability admission are separate policy layers.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ControlInspection {
    pub manifest: ExtensionManifest,
    pub identity: GuestExtensionIdentity,
    pub health: GuestHealthReport,
    pub manifest_sha256: [u8; 32],
    pub component_sha256: [u8; 32],
    pub wasm_profile: &'static str,
}

#[derive(Debug, Error)]
pub enum ControlHostError {
    #[error("manifest exceeds host size limit")]
    ManifestTooLarge,
    #[error("component exceeds host size limit")]
    ComponentTooLarge,
    #[error("manifest JSON is invalid: {0}")]
    ManifestJson(String),
    #[error("manifest structural validation failed: {0}")]
    ManifestInvalid(String),
    #[error("control host accepts only wasm runtime extensions")]
    RuntimeNotWasm,
    #[error("unsupported extension ABI {major}.{minor}")]
    UnsupportedAbi { major: u16, minor: u16 },
    #[error("zero-authority control host cannot grant requested permissions")]
    PermissionRequestNotSupported,
    #[error("manifest resource request exceeds host policy: {0}")]
    ResourcePolicy(String),
    #[error("component compilation failed: {0}")]
    ComponentCompile(String),
    #[error("component has imports or types not satisfied by the zero-authority linker: {0}")]
    UnexpectedImportsOrTypeMismatch(String),
    #[error("required control interface export is missing")]
    MissingControlInterface,
    #[error("required control function {0:?} is missing")]
    MissingControlFunction(&'static str),
    #[error("control function type does not match WIT ABI: {0}")]
    ControlType(String),
    #[error("control component instantiation/execution failed: {0}")]
    Execution(String),
    #[error("guest extension id {guest:?} does not match manifest id {manifest:?}")]
    IdentityMismatch { guest: String, manifest: String },
    #[error("guest version {guest:?} does not match manifest version {manifest:?}")]
    VersionMismatch { guest: String, manifest: String },
    #[error("guest ABI {guest_major}.{guest_minor} does not match manifest ABI {manifest_major}.{manifest_minor}")]
    GuestAbiMismatch {
        guest_major: u16,
        guest_minor: u16,
        manifest_major: u16,
        manifest_minor: u16,
    },
    #[error("guest manifest digest does not match the exact presented manifest bytes")]
    ManifestDigestMismatch,
    #[error("control output exceeds the declared/host output budget")]
    ControlOutputTooLarge,
    #[error("host memory limit does not fit this platform")]
    MemoryLimitOverflow,
}

/// Zero-authority technical compatibility host.
#[derive(Debug, Clone, Copy)]
pub struct ControlPlaneHost {
    policy: ControlHostPolicy,
}

impl Default for ControlPlaneHost {
    fn default() -> Self {
        Self::new(ControlHostPolicy::default())
    }
}

impl ControlPlaneHost {
    pub fn new(policy: ControlHostPolicy) -> Self {
        Self { policy }
    }

    pub fn policy(&self) -> ControlHostPolicy {
        self.policy
    }

    pub const fn wasm_profile(&self) -> &'static str {
        CONTROL_WASM_PROFILE_V1
    }

    /// Compile and run only the extension-control-v1 interface with zero host
    /// imports. Success is technical compatibility, not signer authorization.
    pub fn inspect(
        &self,
        manifest_bytes: &[u8],
        component_bytes: &[u8],
    ) -> Result<ControlInspection, ControlHostError> {
        if manifest_bytes.len() > self.policy.max_manifest_bytes {
            return Err(ControlHostError::ManifestTooLarge);
        }
        if component_bytes.len() > self.policy.max_component_bytes {
            return Err(ControlHostError::ComponentTooLarge);
        }

        let manifest: ExtensionManifest = serde_json::from_slice(manifest_bytes)
            .map_err(|error| ControlHostError::ManifestJson(error.to_string()))?;
        manifest.validate().map_err(|problems| {
            ControlHostError::ManifestInvalid(format!("{problems:?}"))
        })?;
        self.validate_manifest_policy(&manifest)?;

        let manifest_sha256 = sha256(manifest_bytes);
        let component_sha256 = sha256(component_bytes);
        let engine = control_engine()?;
        let component = Component::new(&engine, component_bytes)
            .map_err(|error| ControlHostError::ComponentCompile(error.to_string()))?;

        // Empty linker is a deliberate capability boundary. Pre-instantiation
        // type-checks all imports and fails if the component expects anything.
        let linker = Linker::<HostState>::new(&engine);
        let instance_pre = linker
            .instantiate_pre(&component)
            .map_err(|error| ControlHostError::UnexpectedImportsOrTypeMismatch(error.to_string()))?;

        let control_index = component
            .get_export_index(None, CONTROL_INTERFACE_V1)
            .ok_or(ControlHostError::MissingControlInterface)?;
        let identity_index = component
            .get_export_index(Some(&control_index), "identity")
            .ok_or(ControlHostError::MissingControlFunction("identity"))?;
        let health_index = component
            .get_export_index(Some(&control_index), "health")
            .ok_or(ControlHostError::MissingControlFunction("health"))?;

        let memory_size = usize::try_from(manifest.resources.memory_bytes)
            .map_err(|_| ControlHostError::MemoryLimitOverflow)?;
        let limits = StoreLimitsBuilder::new()
            .memory_size(memory_size)
            .instances(8)
            .tables(4)
            .memories(1)
            .trap_on_grow_failure(true)
            .build();
        let mut store = Store::new(&engine, HostState { limits });
        store.limiter(|state| &mut state.limits);
        store
            .set_fuel(manifest.resources.fuel)
            .map_err(|error| ControlHostError::Execution(error.to_string()))?;
        store.set_epoch_deadline(1);
        store.epoch_deadline_trap();

        let deadline = Duration::from_millis(manifest.resources.max_wall_time_ms);
        let (cancel_tx, cancel_rx) = mpsc::channel::<()>();
        let deadline_engine = engine.clone();
        let timer = thread::spawn(move || {
            if cancel_rx.recv_timeout(deadline).is_err() {
                deadline_engine.increment_epoch();
            }
        });

        let execution = (|| {
            let instance = instance_pre
                .instantiate(&mut store)
                .map_err(|error| ControlHostError::Execution(error.to_string()))?;

            let identity_func = instance
                .get_func(&mut store, &identity_index)
                .ok_or(ControlHostError::MissingControlFunction("identity"))?;
            let identity_func = identity_func
                .typed::<(), (GuestExtensionIdentity,)>(&store)
                .map_err(|error| ControlHostError::ControlType(error.to_string()))?;
            let (identity,) = identity_func
                .call(&mut store, ())
                .map_err(|error| ControlHostError::Execution(error.to_string()))?;

            let health_func = instance
                .get_func(&mut store, &health_index)
                .ok_or(ControlHostError::MissingControlFunction("health"))?;
            let health_func = health_func
                .typed::<(), (GuestHealthReport,)>(&store)
                .map_err(|error| ControlHostError::ControlType(error.to_string()))?;
            let (health,) = health_func
                .call(&mut store, ())
                .map_err(|error| ControlHostError::Execution(error.to_string()))?;

            Ok::<_, ControlHostError>((identity, health))
        })();

        let _ = cancel_tx.send(());
        let _ = timer.join();
        let (identity, health) = execution?;

        self.validate_identity(&manifest, manifest_sha256, &identity)?;
        self.validate_output_budget(&manifest, &identity, &health)?;

        Ok(ControlInspection {
            manifest,
            identity,
            health,
            manifest_sha256,
            component_sha256,
            wasm_profile: CONTROL_WASM_PROFILE_V1,
        })
    }

    fn validate_manifest_policy(
        &self,
        manifest: &ExtensionManifest,
    ) -> Result<(), ControlHostError> {
        if manifest.runtime != RuntimeKind::Wasm {
            return Err(ControlHostError::RuntimeNotWasm);
        }
        if manifest.abi.major != CONTROL_ABI_V1.major
            || manifest.abi.minor > CONTROL_ABI_V1.minor
        {
            return Err(ControlHostError::UnsupportedAbi {
                major: manifest.abi.major,
                minor: manifest.abi.minor,
            });
        }
        if !is_zero_authority(&manifest.permissions) {
            return Err(ControlHostError::PermissionRequestNotSupported);
        }

        let resources = manifest.resources;
        let policy = self.policy;
        if resources.memory_bytes > policy.max_memory_bytes {
            return Err(ControlHostError::ResourcePolicy("memory_bytes".into()));
        }
        if resources.fuel > policy.max_fuel {
            return Err(ControlHostError::ResourcePolicy("fuel".into()));
        }
        if resources.max_wall_time_ms > policy.max_wall_time_ms {
            return Err(ControlHostError::ResourcePolicy("max_wall_time_ms".into()));
        }
        if resources.max_output_bytes > policy.max_output_bytes {
            return Err(ControlHostError::ResourcePolicy("max_output_bytes".into()));
        }
        if resources.max_concurrency > policy.max_concurrency {
            return Err(ControlHostError::ResourcePolicy("max_concurrency".into()));
        }
        Ok(())
    }

    fn validate_identity(
        &self,
        manifest: &ExtensionManifest,
        manifest_sha256: [u8; 32],
        identity: &GuestExtensionIdentity,
    ) -> Result<(), ControlHostError> {
        if identity.id != manifest.id.as_str() {
            return Err(ControlHostError::IdentityMismatch {
                guest: identity.id.clone(),
                manifest: manifest.id.as_str().to_string(),
            });
        }
        if identity.version != manifest.version {
            return Err(ControlHostError::VersionMismatch {
                guest: identity.version.clone(),
                manifest: manifest.version.clone(),
            });
        }
        if identity.abi_major != manifest.abi.major || identity.abi_minor != manifest.abi.minor {
            return Err(ControlHostError::GuestAbiMismatch {
                guest_major: identity.abi_major,
                guest_minor: identity.abi_minor,
                manifest_major: manifest.abi.major,
                manifest_minor: manifest.abi.minor,
            });
        }
        if identity.manifest_digest.as_slice() != manifest_sha256 {
            return Err(ControlHostError::ManifestDigestMismatch);
        }
        Ok(())
    }

    fn validate_output_budget(
        &self,
        manifest: &ExtensionManifest,
        identity: &GuestExtensionIdentity,
        health: &GuestHealthReport,
    ) -> Result<(), ControlHostError> {
        let bytes = identity.id.len()
            + identity.version.len()
            + identity.manifest_digest.len()
            + health.message.as_deref().map_or(0, str::len);
        let bytes = u64::try_from(bytes).unwrap_or(u64::MAX);
        if bytes > manifest.resources.max_output_bytes || bytes > self.policy.max_output_bytes {
            return Err(ControlHostError::ControlOutputTooLarge);
        }
        Ok(())
    }
}

/// Create the exact v1 engine profile.
///
/// Standard SIMD remains available for useful high-performance extensions, but
/// relaxed SIMD is rejected because its specified results may vary by host.
/// Experimental/expanded memory and control-flow proposals are unnecessary for
/// the tiny control plane and stay out of the accepted baseline.
fn control_engine() -> Result<Engine, ControlHostError> {
    let mut config = Config::new();
    config
        .wasm_component_model(true)
        .wasm_relaxed_simd(false)
        .relaxed_simd_deterministic(true)
        .wasm_memory64(false)
        .wasm_multi_memory(false)
        .wasm_tail_call(false)
        .wasm_stack_switching(false)
        .cranelift_nan_canonicalization(true)
        .consume_fuel(true)
        .epoch_interruption(true);
    Engine::new(&config).map_err(|error| ControlHostError::ComponentCompile(error.to_string()))
}

struct HostState {
    limits: StoreLimits,
}

fn sha256(bytes: &[u8]) -> [u8; 32] {
    Sha256::digest(bytes).into()
}

fn is_zero_authority(permissions: &PermissionSet) -> bool {
    permissions.network == NetworkPermission::None
        && permissions.filesystem == FilesystemPermission::None
        && !permissions.gpu
        && !permissions.wall_clock
        && !permissions.randomness
        && permissions.sensors.is_empty()
        && permissions.actuators.is_empty()
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_extension_core::{ExtensionId, ExtensionKind, PermissionSet, ResourceBudget};

    fn manifest() -> ExtensionManifest {
        ExtensionManifest {
            id: ExtensionId::new("org.example.control-test"),
            name: "Control Test".into(),
            version: "1.0.0".into(),
            abi: CONTROL_ABI_V1,
            kind: ExtensionKind::Tool,
            runtime: RuntimeKind::Wasm,
            description: String::new(),
            provides: vec![],
            requires: vec![],
            permissions: PermissionSet::default(),
            resources: ResourceBudget::default(),
        }
    }

    fn bytes(manifest: &ExtensionManifest) -> Vec<u8> {
        serde_json::to_vec(manifest).unwrap()
    }

    #[test]
    fn control_wasm_profile_is_versioned_and_engine_builds() {
        let host = ControlPlaneHost::default();
        assert_eq!(host.wasm_profile(), CONTROL_WASM_PROFILE_V1);
        control_engine().unwrap();
    }

    #[test]
    fn native_runtime_is_rejected_before_component_compilation() {
        let host = ControlPlaneHost::default();
        let mut manifest = manifest();
        manifest.runtime = RuntimeKind::Native;
        let error = host.inspect(&bytes(&manifest), b"not wasm").unwrap_err();
        assert!(matches!(error, ControlHostError::RuntimeNotWasm));
    }

    #[test]
    fn permission_request_is_rejected_by_zero_authority_host() {
        let host = ControlPlaneHost::default();
        let mut manifest = manifest();
        manifest.permissions.wall_clock = true;
        let error = host.inspect(&bytes(&manifest), b"not wasm").unwrap_err();
        assert!(matches!(
            error,
            ControlHostError::PermissionRequestNotSupported
        ));
    }

    #[test]
    fn resource_request_above_host_policy_is_rejected() {
        let host = ControlPlaneHost::default();
        let mut manifest = manifest();
        manifest.resources.memory_bytes = host.policy().max_memory_bytes + 1;
        let error = host.inspect(&bytes(&manifest), b"not wasm").unwrap_err();
        assert!(matches!(error, ControlHostError::ResourcePolicy(_)));
    }

    #[test]
    fn malformed_component_is_rejected_after_manifest_checks() {
        let host = ControlPlaneHost::default();
        let manifest = manifest();
        let error = host.inspect(&bytes(&manifest), b"not a component").unwrap_err();
        assert!(matches!(error, ControlHostError::ComponentCompile(_)));
    }

    #[test]
    fn oversized_manifest_fails_before_parsing() {
        let policy = ControlHostPolicy {
            max_manifest_bytes: 4,
            ..ControlHostPolicy::default()
        };
        let host = ControlPlaneHost::new(policy);
        let error = host.inspect(b"12345", b"").unwrap_err();
        assert!(matches!(error, ControlHostError::ManifestTooLarge));
    }
}
