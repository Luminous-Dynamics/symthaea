use std::cell::RefCell;
use std::rc::Rc;

use anyhow::bail;
use serde::Serialize;

use crate::autonomous_cargo_adapter_state::AutonomousEligibilityGate;
use crate::autonomous_cargo_effect_binding::{
    CommitmentAwareEffectEntryGate, CommitmentBoundAutonomousRunReport,
    ExpectedEffectAdmissionBinding, run_commitment_bound_autonomous,
};
use crate::cargo_adapter_state::{
    AdmissionPersistence, AdmissionPersistenceAck, CargoProcessBackend, EffectAdmission,
    PreSpawnFreshnessApproval, PreSpawnFreshnessGate, PreSpawnFreshnessRejection,
    RepositorySourceProbe, ValidatedPreflightBindings,
};
use crate::cargo_execution_contract::{CargoExecutionIntent, validate_intent};

/// Canonical evidence projection from a verified `PreSpawnFreshnessReceiptV2`.
///
/// Production wiring must construct this projection only after the stored receipt
/// has passed the canonical verifier from #5249. The constituent IDs remain here
/// for auditability; prepared-intent identity is the cross-layer subject binding.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct CanonicalFreshnessV2Evidence {
    pub freshness_receipt_id: String,
    pub prepared_intent_id: String,
    pub runtime_binding_id: String,
    pub materialization_receipt_id: String,
    pub tool_attestation_id: String,
    pub git_worktree_state_id: String,
}

/// Immediate freshness boundary backed in production by the canonical v2 verifier.
///
/// The provider receives the actual admitted receipt + persistence acknowledgement
/// only so its fresh observation occurs at the correct ordered boundary. It does
/// not return those IDs: the wrapper below constructs that binding itself.
pub(crate) trait CanonicalFreshnessV2Gate {
    fn verify(
        &mut self,
        intent: &CargoExecutionIntent,
        admission: &EffectAdmission,
        persistence: &AdmissionPersistenceAck,
    ) -> Result<CanonicalFreshnessV2Evidence, PreSpawnFreshnessRejection>;
}

/// Result of joining canonical freshness evidence to the already-admitted prepared
/// Cargo subject while preserving the complete inherited execution report.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum FreshnessBoundAutonomousRunReport {
    /// A well-formed canonical freshness receipt targeted another prepared Cargo
    /// subject. The inherited report retains all eligibility/admission/persistence
    /// evidence established before the freshness boundary rejected.
    FreshnessBindingRejected {
        intent_id: String,
        expected_prepared_intent_id: String,
        actual: CanonicalFreshnessV2Evidence,
        reason: String,
        inner: CommitmentBoundAutonomousRunReport,
    },
    /// No freshness-subject substitution occurred. `accepted_freshness` is present
    /// only if the canonical freshness provider actually returned matching evidence.
    AfterFreshnessBinding {
        intent_id: String,
        accepted_freshness: Option<CanonicalFreshnessV2Evidence>,
        inner: CommitmentBoundAutonomousRunReport,
    },
}

#[derive(Debug, Clone)]
struct FreshnessMismatch {
    actual: CanonicalFreshnessV2Evidence,
    reason: String,
}

struct BindingFreshnessGate<'a, F> {
    inner: &'a mut F,
    expected_prepared_intent_id: &'a str,
    mismatch: Rc<RefCell<Option<FreshnessMismatch>>>,
    accepted: Rc<RefCell<Option<CanonicalFreshnessV2Evidence>>>,
    fatal: Rc<RefCell<Option<String>>>,
}

impl<F> PreSpawnFreshnessGate for BindingFreshnessGate<'_, F>
where
    F: CanonicalFreshnessV2Gate,
{
    fn verify(
        &mut self,
        intent: &CargoExecutionIntent,
        admission: &EffectAdmission,
        persistence: &AdmissionPersistenceAck,
    ) -> Result<PreSpawnFreshnessApproval, PreSpawnFreshnessRejection> {
        let mut actual = self.inner.verify(intent, admission, persistence)?;
        if let Err(error) = normalize_freshness(&mut actual) {
            *self.fatal.borrow_mut() = Some(error.to_string());
            return Err(PreSpawnFreshnessRejection {
                reason: "malformed canonical freshness-v2 evidence".into(),
            });
        }

        if actual.prepared_intent_id != self.expected_prepared_intent_id {
            *self.mismatch.borrow_mut() = Some(FreshnessMismatch {
                actual,
                reason: "canonical freshness v2 targets another prepared Cargo subject".into(),
            });
            return Err(PreSpawnFreshnessRejection {
                reason: "freshness-v2 prepared-subject binding mismatch".into(),
            });
        }

        *self.accepted.borrow_mut() = Some(actual.clone());
        Ok(PreSpawnFreshnessApproval {
            admission_receipt_digest: admission.receipt_digest.clone(),
            persistence_ack_digest: persistence.persistence_ack_digest.clone(),
            freshness_evidence_digest: actual.freshness_receipt_id,
        })
    }
}

/// Join canonical freshness-v2 evidence to the prepared subject already committed
/// by the autonomous eligibility + AI-Assurance admission expectation.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_freshness_bound_autonomous<E, G, P, S, F, B>(
    mut intent: CargoExecutionIntent,
    bindings: ValidatedPreflightBindings,
    expected: ExpectedEffectAdmissionBinding,
    eligibility_gate: &mut E,
    effect_gate: &mut G,
    source_probe: &mut P,
    persistence: &mut S,
    freshness: &mut F,
    backend: &mut B,
) -> anyhow::Result<FreshnessBoundAutonomousRunReport>
where
    E: AutonomousEligibilityGate,
    G: CommitmentAwareEffectEntryGate,
    P: RepositorySourceProbe,
    S: AdmissionPersistence,
    F: CanonicalFreshnessV2Gate,
    B: CargoProcessBackend,
{
    validate_intent(&mut intent)?;
    let intent_id = intent.intent_id.clone();
    let mut expected_prepared_intent_id = expected.prepared_intent_id.clone();
    normalize_digest(
        "expected.prepared_intent_id",
        &mut expected_prepared_intent_id,
    )?;

    let mismatch = Rc::new(RefCell::new(None));
    let accepted = Rc::new(RefCell::new(None));
    let fatal = Rc::new(RefCell::new(None));

    let inner = {
        let mut freshness_gate = BindingFreshnessGate {
            inner: freshness,
            expected_prepared_intent_id: &expected_prepared_intent_id,
            mismatch: Rc::clone(&mismatch),
            accepted: Rc::clone(&accepted),
            fatal: Rc::clone(&fatal),
        };
        run_commitment_bound_autonomous(
            intent,
            bindings,
            expected,
            eligibility_gate,
            effect_gate,
            source_probe,
            persistence,
            &mut freshness_gate,
            backend,
        )?
    };

    if let Some(error) = fatal.borrow_mut().take() {
        bail!("{error}");
    }

    if let Some(mismatch) = mismatch.borrow_mut().take() {
        return Ok(FreshnessBoundAutonomousRunReport::FreshnessBindingRejected {
            intent_id,
            expected_prepared_intent_id,
            actual: mismatch.actual,
            reason: mismatch.reason,
            inner,
        });
    }

    let accepted_freshness = accepted.borrow_mut().take();
    Ok(FreshnessBoundAutonomousRunReport::AfterFreshnessBinding {
        intent_id,
        accepted_freshness,
        inner,
    })
}

fn normalize_freshness(value: &mut CanonicalFreshnessV2Evidence) -> anyhow::Result<()> {
    for (name, digest) in [
        ("freshness.freshness_receipt_id", &mut value.freshness_receipt_id),
        ("freshness.prepared_intent_id", &mut value.prepared_intent_id),
        ("freshness.runtime_binding_id", &mut value.runtime_binding_id),
        (
            "freshness.materialization_receipt_id",
            &mut value.materialization_receipt_id,
        ),
        ("freshness.tool_attestation_id", &mut value.tool_attestation_id),
        (
            "freshness.git_worktree_state_id",
            &mut value.git_worktree_state_id,
        ),
    ] {
        normalize_digest(name, digest)?;
    }
    Ok(())
}

fn normalize_digest(name: &str, value: &mut String) -> anyhow::Result<()> {
    if value.len() != 64 || !value.bytes().all(|byte| byte.is_ascii_hexdigit()) {
        bail!("{name} must be a 64-character hex digest");
    }
    value.make_ascii_lowercase();
    Ok(())
}
