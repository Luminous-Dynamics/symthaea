wit_bindgen::generate!({
    path: "wit",
    world: "simulation-provider-v1",
    generate_unused_types: true,
});

use exports::luminous::symthaea_extension::control::{
    ExtensionIdentity, Guest as ControlGuest, HealthReport, HealthState,
};
use exports::luminous::symthaea_extension::simulation_provider::Guest as SimulationGuest;
use exports::luminous::symthaea_extension::simulation_types::{
    SimulationMetric, SimulationOutput, SimulationProviderError, SimulationRequest, SolverKind,
    Uncertainty,
};

include!(concat!(env!("OUT_DIR"), "/manifest_digest.rs"));

struct HelloSimulation;

impl ControlGuest for HelloSimulation {
    fn identity() -> ExtensionIdentity {
        ExtensionIdentity {
            id: "org.example.hello-simulation".into(),
            version: env!("CARGO_PKG_VERSION").into(),
            abi_major: 1,
            abi_minor: 0,
            manifest_digest: MANIFEST_DIGEST.to_vec(),
        }
    }

    fn health() -> HealthReport {
        HealthReport {
            state: HealthState::Ready,
            message: Some("hello-simulation is ready".into()),
        }
    }
}

impl SimulationGuest for HelloSimulation {
    fn simulate(
        request: SimulationRequest,
    ) -> Result<SimulationOutput, SimulationProviderError> {
        if request.id.trim().is_empty() {
            return Err(SimulationProviderError::InvalidRequest(
                "request id must not be empty".into(),
            ));
        }
        if request.objective.trim().is_empty() {
            return Err(SimulationProviderError::InvalidRequest(
                "objective must not be empty".into(),
            ));
        }
        if !matches!(request.solver, SolverKind::Custom) {
            return Err(SimulationProviderError::UnsupportedSolver);
        }
        if request.parameters.iter().any(|parameter| !parameter.value.is_finite()) {
            return Err(SimulationProviderError::InvalidRequest(
                "all fixture parameters must be finite".into(),
            ));
        }

        let parameter_sum = request
            .parameters
            .iter()
            .map(|parameter| parameter.value)
            .sum::<f64>();
        if !parameter_sum.is_finite() {
            return Err(SimulationProviderError::DomainFailure(
                "synthetic parameter sum is not finite".into(),
            ));
        }

        let metrics = vec![SimulationMetric {
            name: "fixture.parameter-sum".into(),
            value: parameter_sum,
            unit: "fixture-unit".into(),
            uncertainty: None,
        }];

        Ok(SimulationOutput {
            request_id: request.id,
            converged: true,
            confidence: 0.0,
            uncertainty: Uncertainty {
                epistemic: 1.0,
                aleatoric: 0.0,
                interval: None,
            },
            metrics,
            warnings: vec![
                "synthetic authoring fixture only; not engineering evidence".into(),
            ],
        })
    }
}

export!(HelloSimulation);
