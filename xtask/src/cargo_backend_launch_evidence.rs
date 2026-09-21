use anyhow::{Context, bail};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

use crate::cargo_adapter_state::ProcessTerminal;
use crate::cargo_execution_contract::{CargoExecutionIntent, validate_intent};

const SCHEMA: &str = "symthaea.cargo-backend-launch-evidence.v1";
const HASH_DOMAIN: &[u8] = b"symthaea.cargo-backend-launch-evidence.v1\0";

/// Terminal state that is only legal after positive child-creation evidence.
///
/// `TimedOut` and `Cancelled` record the observed control-plane condition. They
/// do not by themselves prove the child was fully terminated/reaped.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum ChildTerminalState {
    Exited { code: i32 },
    TimedOut,
    Cancelled,
    OutcomeUnknown { evidence_digest: String },
}

/// Outcome of one host process-launch request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "kind")]
pub(crate) enum LaunchAttemptOutcome {
    /// The host crossed its process-launch API boundary but did not obtain
    /// positive child-creation evidence.
    LaunchFailed { error_digest: String },
    /// The host obtained positive evidence that one child process was created.
    ChildCreated {
        child_creation_evidence_digest: String,
        terminal: ChildTerminalState,
    },
}

/// Lifecycle evidence after the trusted process-backend boundary was entered.
///
/// This deliberately distinguishes failure while preparing a process/sandbox
/// specification from failure of an actual host launch request.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case", tag = "stage")]
pub(crate) enum BackendLaunchState {
    PreparationFailed { error_digest: String },
    LaunchAttempted {
        launch_attempt_evidence_digest: String,
        outcome: LaunchAttemptOutcome,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CargoBackendLaunchEvidenceInput {
    pub backend_entry_evidence_digest: String,
    pub state: BackendLaunchState,
}

/// Content-addressed launch-lifecycle evidence for one exact admitted Cargo
/// execution intent. This is execution evidence only, never qualification.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub(crate) struct CargoBackendLaunchEvidence {
    pub launch_evidence_id: String,
    pub schema: String,
    pub intent_id: String,
    pub backend_entry_evidence_digest: String,
    pub state: BackendLaunchState,
}

#[derive(Serialize)]
struct LaunchIdentity<'a> {
    schema: &'static str,
    intent_id: &'a str,
    backend_entry_evidence_digest: &'a str,
    state: &'a BackendLaunchState,
}

/// Compatibility projection into the older deterministic `ProcessTerminal`
/// surface. Some truthful v2 states intentionally cannot be squeezed into v1.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum LegacyProcessProjection {
    Representable(ProcessTerminal),
    NotRepresentable { reason: &'static str },
}

pub(crate) fn build_launch_evidence(
    mut intent: CargoExecutionIntent,
    mut input: CargoBackendLaunchEvidenceInput,
) -> anyhow::Result<CargoBackendLaunchEvidence> {
    validate_intent(&mut intent)?;
    normalize_digest(
        "backend_entry_evidence_digest",
        &mut input.backend_entry_evidence_digest,
    )?;
    normalize_state(&mut input.state)?;

    let identity = LaunchIdentity {
        schema: SCHEMA,
        intent_id: &intent.intent_id,
        backend_entry_evidence_digest: &input.backend_entry_evidence_digest,
        state: &input.state,
    };
    let bytes = serde_json::to_vec(&identity)
        .context("serialize Cargo backend launch-evidence identity")?;
    let launch_evidence_id = domain_sha256(HASH_DOMAIN, &bytes);

    Ok(CargoBackendLaunchEvidence {
        launch_evidence_id,
        schema: SCHEMA.into(),
        intent_id: intent.intent_id,
        backend_entry_evidence_digest: input.backend_entry_evidence_digest,
        state: input.state,
    })
}

/// Rebuild one stored receipt against the independently supplied exact intent.
pub(crate) fn verify_launch_evidence(
    intent: CargoExecutionIntent,
    stored: CargoBackendLaunchEvidence,
) -> anyhow::Result<CargoBackendLaunchEvidence> {
    if stored.schema != SCHEMA {
        bail!("unexpected Cargo backend launch-evidence schema");
    }
    let rebuilt = build_launch_evidence(
        intent,
        CargoBackendLaunchEvidenceInput {
            backend_entry_evidence_digest: stored.backend_entry_evidence_digest.clone(),
            state: stored.state.clone(),
        },
    )?;
    if rebuilt != stored {
        bail!("stored Cargo backend launch evidence is not the canonical receipt for this intent");
    }
    Ok(rebuilt)
}

pub(crate) fn project_legacy_terminal(
    evidence: &CargoBackendLaunchEvidence,
) -> LegacyProcessProjection {
    match &evidence.state {
        BackendLaunchState::PreparationFailed { .. } => LegacyProcessProjection::NotRepresentable {
            reason: "backend preparation failed before any host launch request was attempted",
        },
        BackendLaunchState::LaunchAttempted { outcome, .. } => match outcome {
            LaunchAttemptOutcome::LaunchFailed { error_digest } => {
                LegacyProcessProjection::Representable(ProcessTerminal::SpawnFailed {
                    error_digest: error_digest.clone(),
                })
            }
            LaunchAttemptOutcome::ChildCreated { terminal, .. } => match terminal {
                ChildTerminalState::Exited { code } => {
                    LegacyProcessProjection::Representable(ProcessTerminal::Exited { code: *code })
                }
                ChildTerminalState::TimedOut => {
                    LegacyProcessProjection::Representable(ProcessTerminal::TimedOut)
                }
                ChildTerminalState::Cancelled => {
                    LegacyProcessProjection::Representable(ProcessTerminal::Cancelled)
                }
                ChildTerminalState::OutcomeUnknown { .. } => {
                    LegacyProcessProjection::NotRepresentable {
                        reason: "child creation is known but the old terminal model has no unknown-outcome state",
                    }
                }
            },
        },
    }
}

fn normalize_state(state: &mut BackendLaunchState) -> anyhow::Result<()> {
    match state {
        BackendLaunchState::PreparationFailed { error_digest } => {
            normalize_digest("state.preparation.error_digest", error_digest)
        }
        BackendLaunchState::LaunchAttempted {
            launch_attempt_evidence_digest,
            outcome,
        } => {
            normalize_digest(
                "state.launch_attempt_evidence_digest",
                launch_attempt_evidence_digest,
            )?;
            match outcome {
                LaunchAttemptOutcome::LaunchFailed { error_digest } => {
                    normalize_digest("state.launch_failed.error_digest", error_digest)
                }
                LaunchAttemptOutcome::ChildCreated {
                    child_creation_evidence_digest,
                    terminal,
                } => {
                    normalize_digest(
                        "state.child_creation_evidence_digest",
                        child_creation_evidence_digest,
                    )?;
                    if let ChildTerminalState::OutcomeUnknown { evidence_digest } = terminal {
                        normalize_digest("state.child_unknown.evidence_digest", evidence_digest)?;
                    }
                    Ok(())
                }
            }
        }
    }
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character hex digest");
    }
    value.make_ascii_lowercase();
    Ok(())
}

fn domain_sha256(domain: &[u8], bytes: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(domain);
    hasher.update(bytes);
    let digest = hasher.finalize();
    let mut out = String::with_capacity(digest.len() * 2);
    for byte in digest {
        use std::fmt::Write as _;
        write!(&mut out, "{byte:02x}").expect("writing to String cannot fail");
    }
    out
}
