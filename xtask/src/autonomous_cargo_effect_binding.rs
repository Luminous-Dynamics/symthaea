use std::cell::RefCell;
use std::rc::Rc;

use anyhow::bail;
use serde::Serialize;

use crate::autonomous_cargo_adapter_state::{
    AutonomousAdapterRunReport, AutonomousEligibilityEvidence, AutonomousEligibilityGate,
    AutonomousEligibilityRejection, run_autonomous_with_backend,
};
use crate::cargo_adapter_state::{
    AdmissionPersistence, CargoProcessBackend, EffectAdmission, EffectEntryGate, EffectRejection,
    PreSpawnFreshnessGate, RepositorySourceProbe, ValidatedPreflightBindings,
};
use crate::cargo_execution_contract::{CargoExecutionIntent, validate_intent};

/// Host-validated expectation for the exact AI-Assurance admission commitment
/// applicable to one autonomous Cargo execution subject.
///
/// This value is evidence/configuration, not authority. Production wiring must
/// derive it from the trusted host/AI-Assurance composition rather than allowing
/// cognition or caller JSON to mint the commitment semantics.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ExpectedEffectAdmissionBinding {
    pub execution_intent_id: String,
    pub eligibility_id: String,
    pub prepared_intent_id: String,
    pub action_binding: String,
    pub authority_snapshot_digest: String,
    pub adapter_semantics_digest: String,
    pub effect_admission_commitment_digest: String,
}

/// Projection of one positive AI-Assurance admission receipt plus the exact
/// commitment it carries. The deterministic engineering model compares this
/// projection but does not recreate AI-Assurance's commitment hash constructor.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct CommitmentAwareEffectAdmission {
    pub receipt_digest: String,
    pub action_binding: String,
    pub authority_snapshot_digest: String,
    pub adapter_semantics_digest: String,
    pub effect_admission_commitment_digest: String,
}

/// Effect-entry boundary that exposes enough AI-Assurance evidence to prove that
/// the returned receipt belongs to the host-expected commitment before it is
/// reduced to the older receipt-digest-only adapter seam.
pub(crate) trait CommitmentAwareEffectEntryGate {
    fn admit(
        &mut self,
        intent: &CargoExecutionIntent,
    ) -> Result<CommitmentAwareEffectAdmission, EffectRejection>;
}

/// Result of the commitment-binding composition layer.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum CommitmentBoundAutonomousRunReport {
    /// The host expectation was well formed but targets another immutable Cargo
    /// execution/adapter subject. No external boundary was called.
    ExpectationRejectedBeforeEligibility {
        intent_id: String,
        reason: String,
    },
    /// Positive eligibility evidence was well formed but did not equal the exact
    /// eligibility/prepared/intent lineage committed by the host expectation.
    EligibilityBindingRejected {
        intent_id: String,
        expected_eligibility_id: String,
        actual_eligibility_id: String,
        expected_prepared_intent_id: String,
        actual_prepared_intent_id: String,
        reason: String,
    },
    /// Positive effect-admission evidence carried a different AI-Assurance
    /// commitment. The returned evidence is retained, while persistence and all
    /// later boundaries remain unreachable.
    AdmissionBindingRejected {
        intent_id: String,
        eligibility_id: String,
        prepared_intent_id: String,
        expected: ExpectedEffectAdmissionBinding,
        actual: CommitmentAwareEffectAdmission,
        reason: String,
    },
    /// The host expectation was accepted and the existing autonomous state
    /// machine ran unchanged. `accepted_admission` is present only when a
    /// commitment-aware positive admission actually matched the expectation.
    AfterCommitmentBinding {
        intent_id: String,
        expected: ExpectedEffectAdmissionBinding,
        accepted_admission: Option<CommitmentAwareEffectAdmission>,
        inner: AutonomousAdapterRunReport,
    },
}

#[derive(Debug, Clone)]
struct EligibilityMismatch {
    actual: AutonomousEligibilityEvidence,
    reason: String,
}

#[derive(Debug, Clone)]
struct AdmissionMismatch {
    actual: CommitmentAwareEffectAdmission,
    reason: String,
}

struct BindingEligibilityGate<'a, E> {
    inner: &'a mut E,
    expected: &'a ExpectedEffectAdmissionBinding,
    mismatch: Rc<RefCell<Option<EligibilityMismatch>>>,
    fatal: Rc<RefCell<Option<String>>>,
}

impl<E> AutonomousEligibilityGate for BindingEligibilityGate<'_, E>
where
    E: AutonomousEligibilityGate,
{
    fn verify(
        &mut self,
        intent: &CargoExecutionIntent,
    ) -> Result<AutonomousEligibilityEvidence, AutonomousEligibilityRejection> {
        let mut actual = self.inner.verify(intent)?;
        if let Err(error) = normalize_eligibility(&mut actual) {
            *self.fatal.borrow_mut() = Some(error.to_string());
            return Err(AutonomousEligibilityRejection {
                reason: "malformed autonomous eligibility evidence".into(),
            });
        }

        if actual.execution_intent_id != self.expected.execution_intent_id
            || actual.eligibility_id != self.expected.eligibility_id
            || actual.prepared_intent_id != self.expected.prepared_intent_id
        {
            *self.mismatch.borrow_mut() = Some(EligibilityMismatch {
                actual,
                reason: "autonomous eligibility does not match the host-expected lineage".into(),
            });
            return Err(AutonomousEligibilityRejection {
                reason: "autonomous eligibility binding mismatch".into(),
            });
        }

        Ok(actual)
    }
}

struct BindingEffectGate<'a, G> {
    inner: &'a mut G,
    expected: &'a ExpectedEffectAdmissionBinding,
    mismatch: Rc<RefCell<Option<AdmissionMismatch>>>,
    accepted: Rc<RefCell<Option<CommitmentAwareEffectAdmission>>>,
    fatal: Rc<RefCell<Option<String>>>,
}

impl<G> EffectEntryGate for BindingEffectGate<'_, G>
where
    G: CommitmentAwareEffectEntryGate,
{
    fn admit(&mut self, intent: &CargoExecutionIntent) -> Result<EffectAdmission, EffectRejection> {
        let mut actual = self.inner.admit(intent)?;
        if let Err(error) = normalize_admission(&mut actual) {
            *self.fatal.borrow_mut() = Some(error.to_string());
            return Err(EffectRejection {
                reason: "malformed effect-admission evidence".into(),
            });
        }

        let reason = if actual.action_binding != self.expected.action_binding {
            Some("effect admission action binding does not match host expectation")
        } else if actual.authority_snapshot_digest != self.expected.authority_snapshot_digest {
            Some("effect admission authority snapshot does not match host expectation")
        } else if actual.adapter_semantics_digest != self.expected.adapter_semantics_digest {
            Some("effect admission adapter semantics do not match host expectation")
        } else if actual.effect_admission_commitment_digest
            != self.expected.effect_admission_commitment_digest
        {
            Some("effect admission commitment digest does not match host expectation")
        } else {
            None
        };

        if let Some(reason) = reason {
            *self.mismatch.borrow_mut() = Some(AdmissionMismatch {
                actual,
                reason: reason.into(),
            });
            return Err(EffectRejection {
                reason: "effect admission binding mismatch".into(),
            });
        }

        *self.accepted.borrow_mut() = Some(actual.clone());
        Ok(EffectAdmission {
            receipt_digest: actual.receipt_digest,
        })
    }
}

/// Require a host-validated AI-Assurance commitment binding around the existing
/// autonomous Cargo orchestration model.
///
/// The outer layer owns only cross-binding. It intentionally does not recreate
/// AI-Assurance's commitment hashing, admission authority, or durability policy.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_commitment_bound_autonomous<E, G, P, S, F, B>(
    mut intent: CargoExecutionIntent,
    bindings: ValidatedPreflightBindings,
    mut expected: ExpectedEffectAdmissionBinding,
    eligibility_gate: &mut E,
    effect_gate: &mut G,
    source_probe: &mut P,
    persistence: &mut S,
    freshness: &mut F,
    backend: &mut B,
) -> anyhow::Result<CommitmentBoundAutonomousRunReport>
where
    E: AutonomousEligibilityGate,
    G: CommitmentAwareEffectEntryGate,
    P: RepositorySourceProbe,
    S: AdmissionPersistence,
    F: PreSpawnFreshnessGate,
    B: CargoProcessBackend,
{
    validate_intent(&mut intent)?;
    normalize_expected(&mut expected)?;
    let intent_id = intent.intent_id.clone();

    if expected.execution_intent_id != intent.intent_id {
        return Ok(
            CommitmentBoundAutonomousRunReport::ExpectationRejectedBeforeEligibility {
                intent_id,
                reason: "expected effect admission targets another execution intent".into(),
            },
        );
    }
    if expected.adapter_semantics_digest != intent.adapter_semantics_digest {
        return Ok(
            CommitmentBoundAutonomousRunReport::ExpectationRejectedBeforeEligibility {
                intent_id,
                reason: "expected effect admission targets another adapter-semantics subject".into(),
            },
        );
    }

    let eligibility_mismatch = Rc::new(RefCell::new(None));
    let admission_mismatch = Rc::new(RefCell::new(None));
    let accepted_admission = Rc::new(RefCell::new(None));
    let fatal = Rc::new(RefCell::new(None));

    let inner = {
        let mut eligibility = BindingEligibilityGate {
            inner: eligibility_gate,
            expected: &expected,
            mismatch: Rc::clone(&eligibility_mismatch),
            fatal: Rc::clone(&fatal),
        };
        let mut effect = BindingEffectGate {
            inner: effect_gate,
            expected: &expected,
            mismatch: Rc::clone(&admission_mismatch),
            accepted: Rc::clone(&accepted_admission),
            fatal: Rc::clone(&fatal),
        };

        run_autonomous_with_backend(
            intent,
            bindings,
            &mut eligibility,
            source_probe,
            &mut effect,
            persistence,
            freshness,
            backend,
        )?
    };

    if let Some(error) = fatal.borrow_mut().take() {
        bail!("{error}");
    }

    if let Some(mismatch) = eligibility_mismatch.borrow_mut().take() {
        return Ok(CommitmentBoundAutonomousRunReport::EligibilityBindingRejected {
            intent_id,
            expected_eligibility_id: expected.eligibility_id,
            actual_eligibility_id: mismatch.actual.eligibility_id,
            expected_prepared_intent_id: expected.prepared_intent_id,
            actual_prepared_intent_id: mismatch.actual.prepared_intent_id,
            reason: mismatch.reason,
        });
    }

    if let Some(mismatch) = admission_mismatch.borrow_mut().take() {
        return Ok(CommitmentBoundAutonomousRunReport::AdmissionBindingRejected {
            intent_id,
            eligibility_id: expected.eligibility_id.clone(),
            prepared_intent_id: expected.prepared_intent_id.clone(),
            expected,
            actual: mismatch.actual,
            reason: mismatch.reason,
        });
    }

    let accepted_admission = accepted_admission.borrow_mut().take();
    Ok(CommitmentBoundAutonomousRunReport::AfterCommitmentBinding {
        intent_id,
        expected,
        accepted_admission,
        inner,
    })
}

fn normalize_expected(value: &mut ExpectedEffectAdmissionBinding) -> anyhow::Result<()> {
    normalize_digest("expected.execution_intent_id", &mut value.execution_intent_id)?;
    normalize_digest("expected.eligibility_id", &mut value.eligibility_id)?;
    normalize_digest("expected.prepared_intent_id", &mut value.prepared_intent_id)?;
    normalize_digest("expected.action_binding", &mut value.action_binding)?;
    normalize_digest(
        "expected.authority_snapshot_digest",
        &mut value.authority_snapshot_digest,
    )?;
    normalize_digest(
        "expected.adapter_semantics_digest",
        &mut value.adapter_semantics_digest,
    )?;
    normalize_digest(
        "expected.effect_admission_commitment_digest",
        &mut value.effect_admission_commitment_digest,
    )?;
    Ok(())
}

fn normalize_eligibility(value: &mut AutonomousEligibilityEvidence) -> anyhow::Result<()> {
    normalize_digest("eligibility.eligibility_id", &mut value.eligibility_id)?;
    normalize_digest(
        "eligibility.prepared_intent_id",
        &mut value.prepared_intent_id,
    )?;
    normalize_digest(
        "eligibility.execution_intent_id",
        &mut value.execution_intent_id,
    )?;
    Ok(())
}

fn normalize_admission(value: &mut CommitmentAwareEffectAdmission) -> anyhow::Result<()> {
    normalize_digest("admission.receipt_digest", &mut value.receipt_digest)?;
    normalize_digest("admission.action_binding", &mut value.action_binding)?;
    normalize_digest(
        "admission.authority_snapshot_digest",
        &mut value.authority_snapshot_digest,
    )?;
    normalize_digest(
        "admission.adapter_semantics_digest",
        &mut value.adapter_semantics_digest,
    )?;
    normalize_digest(
        "admission.effect_admission_commitment_digest",
        &mut value.effect_admission_commitment_digest,
    )?;
    Ok(())
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character hex digest");
    }
    value.make_ascii_lowercase();
    Ok(())
}
