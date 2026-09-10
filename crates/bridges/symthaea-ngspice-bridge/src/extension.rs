use crate::NgspiceBridge;
use symthaea_extension_core::{
    AbiVersion, CapabilityDescriptor, EffectClass, ExtensionId, ExtensionKind,
    ExtensionManifest, PermissionSet, ResourceBudget, RuntimeKind,
};
use symthaea_sim_bridge::{SimulationBackend, SimulationError, SolverKind};
use symthaea_sim_extension_routing::{
    SimulationBackendFactory, SimulationProviderDescriptor, solver_capability,
};

/// Stable extension identity for the built-in ngspice adapter.
pub const NGSPICE_EXTENSION_ID: &str = "io.luminous.symthaea.ngspice";

/// Lazy factory exposing `NgspiceBridge` through the generic extension router.
///
/// The factory itself is cheap: no solver executable is spawned or probed until
/// the selected provider is instantiated and its backend is actually run.
#[derive(Debug, Clone)]
pub struct NgspiceExtensionFactory {
    descriptor: SimulationProviderDescriptor,
    dry_run: bool,
    solver_cmd: String,
}

impl Default for NgspiceExtensionFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl NgspiceExtensionFactory {
    pub fn new() -> Self {
        Self {
            descriptor: descriptor(),
            dry_run: false,
            solver_cmd: "ngspice".into(),
        }
    }

    /// Construct a factory whose selected backend returns deterministic dry-run
    /// fixtures. Registration remains cold in either mode.
    pub fn dry_run() -> Self {
        Self {
            dry_run: true,
            ..Self::new()
        }
    }

    pub fn with_solver_cmd(mut self, command: impl Into<String>) -> Self {
        self.solver_cmd = command.into();
        self
    }
}

impl SimulationBackendFactory for NgspiceExtensionFactory {
    fn descriptor(&self) -> &SimulationProviderDescriptor {
        &self.descriptor
    }

    fn create(&self) -> Result<Box<dyn SimulationBackend>, SimulationError> {
        Ok(Box::new(NgspiceBridge {
            dry_run: self.dry_run,
            solver_cmd: self.solver_cmd.clone(),
        }))
    }
}

fn descriptor() -> SimulationProviderDescriptor {
    let capability = solver_capability(SolverKind::Circuit);

    SimulationProviderDescriptor {
        manifest: ExtensionManifest {
            id: ExtensionId::new(NGSPICE_EXTENSION_ID),
            name: "ngspice circuit simulation".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            abi: AbiVersion::V1,
            kind: ExtensionKind::Simulation,
            runtime: RuntimeKind::Native,
            description: "Built-in native ngspice circuit simulation adapter".into(),
            provides: vec![CapabilityDescriptor {
                id: capability,
                description: "Run normalized circuit simulation requests through ngspice".into(),
                effect: EffectClass::Pure,
            }],
            requires: vec![],
            permissions: PermissionSet::default(),
            // The current native adapter has no OS-level resource supervisor.
            // Report no finite enforceable envelope rather than pretending the
            // WASM defaults apply. Calls requiring hard memory/fuel ceilings
            // therefore reject this provider at the router.
            resources: ResourceBudget {
                memory_bytes: u64::MAX,
                fuel: u64::MAX,
                max_wall_time_ms: u64::MAX,
                max_output_bytes: u64::MAX,
                max_concurrency: 1,
            },
        },
        backend_name: "ngspice".into(),
        supported_solvers: vec![SolverKind::Circuit],
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use symthaea_extension_core::RuntimeKind;

    #[test]
    fn descriptor_is_valid_and_maps_circuit_capability() {
        let factory = NgspiceExtensionFactory::new();
        let descriptor = factory.descriptor();
        descriptor.validate().unwrap();
        assert_eq!(descriptor.manifest.id.as_str(), NGSPICE_EXTENSION_ID);
        assert_eq!(descriptor.manifest.runtime, RuntimeKind::Native);
        assert_eq!(descriptor.supported_solvers, vec![SolverKind::Circuit]);
        assert!(descriptor
            .manifest
            .provides
            .iter()
            .any(|cap| cap.id == solver_capability(SolverKind::Circuit)));
    }

    #[test]
    fn factory_preserves_backend_contract() {
        let factory = NgspiceExtensionFactory::dry_run().with_solver_cmd("custom-ngspice");
        let backend = factory.create().unwrap();
        assert_eq!(backend.name(), "ngspice");
        assert_eq!(backend.supported_solvers(), &[SolverKind::Circuit]);
    }

    #[test]
    fn native_factory_does_not_claim_wasm_resource_enforcement() {
        let resources = factory_resources();
        assert_eq!(resources.memory_bytes, u64::MAX);
        assert_eq!(resources.fuel, u64::MAX);
        assert_eq!(resources.max_wall_time_ms, u64::MAX);
    }

    fn factory_resources() -> ResourceBudget {
        NgspiceExtensionFactory::new()
            .descriptor()
            .manifest
            .resources
    }
}
