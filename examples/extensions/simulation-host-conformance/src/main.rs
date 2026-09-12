use sha2::{Digest, Sha256};
use std::{env, fs};
use wasmtime::component::{Component, Linker};
use wasmtime::{Config, Engine, Store};

wasmtime::component::bindgen!({
    world: "simulation-provider-v1",
    path: "../../../crates/core/symthaea-extension-core/wit",
});

use exports::luminous::symthaea_extension::control::HealthState;
use exports::luminous::symthaea_extension::simulation_types::{
    ModelParameter, SimulationDomain, SimulationProviderError, SimulationRequest, SolverKind,
};

fn main() -> Result<(), wasmtime::Error> {
    let mut args = env::args_os().skip(1);
    let component_path = args
        .next()
        .ok_or_else(|| wasmtime::Error::msg("expected component path"))?;
    let manifest_path = args
        .next()
        .ok_or_else(|| wasmtime::Error::msg("expected manifest path"))?;
    if args.next().is_some() {
        return Err(wasmtime::Error::msg("unexpected extra arguments"));
    }

    let component_bytes = fs::read(&component_path)
        .map_err(|error| wasmtime::Error::msg(format!("read component: {error}")))?;
    let manifest_bytes = fs::read(&manifest_path)
        .map_err(|error| wasmtime::Error::msg(format!("read manifest: {error}")))?;
    let manifest_digest: [u8; 32] = Sha256::digest(&manifest_bytes).into();

    let engine = engine()?;
    let component = Component::new(&engine, &component_bytes)?;

    // Deliberately empty: a guest that imports WASI, filesystem, network,
    // clocks, randomness, or any other host capability must fail here.
    let linker = Linker::<()>::new(&engine);
    let mut store = Store::new(&engine, ());
    store.set_fuel(50_000_000)?;

    let bindings = SimulationProviderV1::instantiate(&mut store, &component, &linker)?;

    let identity = bindings
        .luminous_symthaea_extension_control()
        .call_identity(&mut store)?;
    if identity.id != "org.example.hello-simulation" {
        return Err(wasmtime::Error::msg(format!(
            "unexpected guest id {:?}",
            identity.id
        )));
    }
    if identity.version != "0.1.0" || identity.abi_major != 1 || identity.abi_minor != 0 {
        return Err(wasmtime::Error::msg("guest version/ABI identity mismatch"));
    }
    if identity.manifest_digest.as_slice() != &manifest_digest[..] {
        return Err(wasmtime::Error::msg(
            "guest identity is not bound to the exact manifest bytes",
        ));
    }

    let health = bindings
        .luminous_symthaea_extension_control()
        .call_health(&mut store)?;
    if !matches!(health.state, HealthState::Ready) {
        return Err(wasmtime::Error::msg("guest did not report ready health"));
    }

    let request = SimulationRequest {
        id: "fixture-request-1".into(),
        domain: SimulationDomain::Systems,
        solver: SolverKind::Custom,
        objective: "prove typed zero-import simulation invocation".into(),
        parameters: vec![
            ModelParameter {
                name: "a".into(),
                value: 2.5,
                unit: "fixture-unit".into(),
                provenance: "conformance-harness".into(),
                uncertainty: None,
            },
            ModelParameter {
                name: "b".into(),
                value: 1.5,
                unit: "fixture-unit".into(),
                provenance: "conformance-harness".into(),
                uncertainty: None,
            },
        ],
        requested_metrics: vec!["fixture.parameter-sum".into()],
    };

    let output = match bindings
        .luminous_symthaea_extension_simulation_provider()
        .call_simulate(&mut store, &request)?
    {
        Ok(output) => output,
        Err(_) => {
            return Err(wasmtime::Error::msg(
                "guest unexpectedly rejected the valid custom-solver request",
            ));
        }
    };

    if output.request_id != request.id {
        return Err(wasmtime::Error::msg("guest returned the wrong request id"));
    }
    if !output.converged || output.confidence != 0.0 {
        return Err(wasmtime::Error::msg(
            "fixture convergence/confidence contract changed",
        ));
    }
    if output.uncertainty.epistemic != 1.0 || output.uncertainty.aleatoric != 0.0 {
        return Err(wasmtime::Error::msg(
            "fixture uncertainty contract changed",
        ));
    }
    if output.metrics.len() != 1
        || output.metrics[0].name != "fixture.parameter-sum"
        || output.metrics[0].value != 4.0
    {
        return Err(wasmtime::Error::msg(
            "typed simulation metric did not round-trip as expected",
        ));
    }
    if !output
        .warnings
        .iter()
        .any(|warning| warning.contains("not engineering evidence"))
    {
        return Err(wasmtime::Error::msg(
            "fixture lost its non-evidence warning",
        ));
    }

    let unsupported = SimulationRequest {
        id: "fixture-request-unsupported".into(),
        domain: SimulationDomain::Electrical,
        solver: SolverKind::Circuit,
        objective: "prove typed provider error transport".into(),
        parameters: vec![],
        requested_metrics: vec![],
    };
    let unsupported_result = bindings
        .luminous_symthaea_extension_simulation_provider()
        .call_simulate(&mut store, &unsupported)?;
    if !matches!(unsupported_result, Err(SimulationProviderError::UnsupportedSolver)) {
        return Err(wasmtime::Error::msg(
            "non-custom request did not return typed unsupported-solver",
        ));
    }

    println!("typed simulation component passed empty-linker conformance");
    Ok(())
}

fn engine() -> Result<Engine, wasmtime::Error> {
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
        .consume_fuel(true);
    Engine::new(&config)
}
