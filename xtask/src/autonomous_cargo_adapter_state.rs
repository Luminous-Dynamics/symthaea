use anyhow::bail;
use serde::Serialize;

use crate::cargo_adapter_state::{
    AdapterRunReport, AdmissionPersistence, CargoProcessBackend, EffectEntryGate,
    PreSpawnFreshnessGate, RepositorySourceProbe, ValidatedPreflightBindings, run_with_backend,
};
use crate::cargo_execution_contract::{CargoExecutionIntent, validate_intent};

/// Opaque positive evidence that an external owning verifier admitted this exact
/// Cargo execution subject for autonomous execution.
///
/// The deterministic adapter model does not recreate #5099 validation logic. A
/// production implementation must obtain this evidence from the canonical
/// autonomous-eligibility verifier.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AutonomousEligibilityEvidence {
    pub eligibility_id: String,
    pub prepared_intent_id: String,
    pub execution_intent_id: String,
}

/// External autonomous-eligibility denial before any effect boundary is reached.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct AutonomousEligibilityRejection {
    pub reason: String,
}

/// Injected autonomous-eligibility boundary.
///
/// Production wiring is expected to validate #5099 and project its exact lineage
/// through this seam rather than minting eligibility inside the adapter state model.
pub(crate) trait AutonomousEligibilityGate {
    fn verify(
        &mut self,
        intent: &CargoExecutionIntent,
    ) -> Result<AutonomousEligibilityEvidence, AutonomousEligibilityRejection>;
}

/// Autonomous orchestration outcome preserving eligibility lineage separately from
/// the existing effect-admission/persistence/freshness/process evidence.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum AutonomousAdapterRunReport {
    /// The owning eligibility boundary rejected before the inner adapter state
    /// machine became reachable. No positive eligibility receipt exists here.
    EligibilityRejected {
        intent_id: String,
        reason: String,
    },
    /// Well-formed positive eligibility evidence named a different execution
    /// intent. Its eligibility/prepared lineage is retained for audit while all
    /// inner boundaries remain unreachable.
    EligibilityEvidenceRejected {
        intent_id: String,
        eligibility_id: String,
        prepared_intent_id: String,
        reason: String,
    },
    /// Autonomous eligibility matched the exact intent and the existing adapter
    /// state machine was allowed to proceed. `inner` retains the complete
    /// preflight/effect/persistence/freshness/backend/postflight outcome.
    AfterEligibility {
        intent_id: String,
        eligibility_id: String,
        prepared_intent_id: String,
        inner: AdapterRunReport,
    },
}

/// Require autonomous eligibility before entering the existing trusted Cargo
/// orchestration model.
///
/// Ordering theorem:
///
/// 1. validate the immutable execution intent locally;
/// 2. obtain positive autonomous-eligibility evidence from the owning boundary;
/// 3. require that evidence to name this exact execution-intent id;
/// 4. only then enter [`run_with_backend`], which owns source preflight, effect
///    admission, persistence, immediate freshness, backend entry, and postflight.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_autonomous_with_backend<E, P, G, S, F, B>(
    mut intent: CargoExecutionIntent,
    bindings: ValidatedPreflightBindings,
    eligibility_gate: &mut E,
    source_probe: &mut P,
    effect_gate: &mut G,
    persistence: &mut S,
    freshness: &mut F,
    backend: &mut B,
) -> anyhow::Result<AutonomousAdapterRunReport>
where
    E: AutonomousEligibilityGate,
    P: RepositorySourceProbe,
    G: EffectEntryGate,
    S: AdmissionPersistence,
    F: PreSpawnFreshnessGate,
    B: CargoProcessBackend,
{
    validate_intent(&mut intent)?;
    let intent_id = intent.intent_id.clone();

    let mut eligibility = match eligibility_gate.verify(&intent) {
        Ok(evidence) => evidence,
        Err(rejection) => {
            return Ok(AutonomousAdapterRunReport::EligibilityRejected {
                intent_id,
                reason: rejection.reason,
            });
        }
    };
    normalize_digest("eligibility.eligibility_id", &mut eligibility.eligibility_id)?;
    normalize_digest(
        "eligibility.prepared_intent_id",
        &mut eligibility.prepared_intent_id,
    )?;
    normalize_digest(
        "eligibility.execution_intent_id",
        &mut eligibility.execution_intent_id,
    )?;

    if eligibility.execution_intent_id != intent_id {
        return Ok(AutonomousAdapterRunReport::EligibilityEvidenceRejected {
            intent_id,
            eligibility_id: eligibility.eligibility_id,
            prepared_intent_id: eligibility.prepared_intent_id,
            reason: "autonomous eligibility does not target this exact execution intent".into(),
        });
    }

    let inner = run_with_backend(
        intent,
        bindings,
        source_probe,
        effect_gate,
        persistence,
        freshness,
        backend,
    )?;
    Ok(AutonomousAdapterRunReport::AfterEligibility {
        intent_id,
        eligibility_id: eligibility.eligibility_id,
        prepared_intent_id: eligibility.prepared_intent_id,
        inner,
    })
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character hex digest");
    }
    value.make_ascii_lowercase();
    Ok(())
}
