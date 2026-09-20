use std::cell::RefCell;
use std::rc::Rc;

use anyhow::bail;
use serde::Serialize;

use crate::autonomous_cargo_adapter_state::AutonomousEligibilityGate;
use crate::autonomous_cargo_effect_binding::{
    CommitmentAwareEffectEntryGate, ExpectedEffectAdmissionBinding,
};
use crate::autonomous_cargo_freshness_binding::{
    CanonicalFreshnessV2Gate, FreshnessBoundAutonomousRunReport,
    run_freshness_bound_autonomous,
};
use crate::cargo_adapter_state::{
    AdmissionPersistence, AdmissionPersistenceAck, AdmissionPersistenceRejection,
    CargoProcessBackend, EffectAdmission, RepositorySourceProbe, ValidatedPreflightBindings,
};
use crate::cargo_execution_contract::{CargoExecutionIntent, validate_intent};

/// Host-selected persistence requirement that belongs to the same authority
/// snapshot already committed by AI Assurance for this effect admission.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct ExpectedAdmissionPersistenceBinding {
    pub authority_snapshot_digest: String,
    pub required_persistence_profile_id: String,
}

/// Evidence projection produced by the configured persistence provider.
///
/// These fields identify what the provider says it actually enforced. They are
/// evidence commitments, not independent proof that a backend truthfully
/// implements fsync, replication, hardware monotonicity, or anti-rollback.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
pub(crate) struct CanonicalAdmissionPersistenceEvidence {
    pub persistence_ack_digest: String,
    pub persistence_profile_id: String,
    pub persistence_record_id: String,
    pub control_plane_id: String,
    pub control_plane_generation: u64,
    pub evidence_root_digest: Option<String>,
    pub anti_rollback_evidence_digest: Option<String>,
}

/// Persistence boundary used after one exact effect admission has already won.
///
/// The provider receives the required profile explicitly but does not return or
/// choose an admission-receipt id. The wrapper binds persistence evidence to the
/// actual accepted admission itself.
pub(crate) trait CanonicalAdmissionPersistenceGate {
    fn persist(
        &mut self,
        intent: &CargoExecutionIntent,
        admission: &EffectAdmission,
        required_persistence_profile_id: &str,
    ) -> Result<CanonicalAdmissionPersistenceEvidence, AdmissionPersistenceRejection>;
}

/// Result of joining persistence-profile evidence to the same AI-Assurance
/// authority snapshot and admission lineage used by the autonomous Cargo path.
#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[allow(clippy::large_enum_variant)]
#[serde(rename_all = "snake_case", tag = "state")]
pub(crate) enum PersistenceBoundAutonomousRunReport {
    /// The persistence requirement belongs to another authority snapshot. No
    /// eligibility/effect boundary is called.
    PersistenceExpectationRejectedBeforeEligibility {
        intent_id: String,
        effect_authority_snapshot_digest: String,
        persistence_authority_snapshot_digest: String,
        reason: String,
    },
    /// Persistence completed under a well-formed but different profile. Earlier
    /// admission evidence is retained through the inherited report; freshness and
    /// backend remain unreachable.
    PersistenceBindingRejected {
        intent_id: String,
        required_persistence_profile_id: String,
        actual: CanonicalAdmissionPersistenceEvidence,
        reason: String,
        inner: FreshnessBoundAutonomousRunReport,
    },
    /// No persistence-profile substitution occurred. `accepted_persistence` is
    /// present only after positive evidence matched the required profile.
    AfterPersistenceBinding {
        intent_id: String,
        accepted_persistence: Option<CanonicalAdmissionPersistenceEvidence>,
        inner: FreshnessBoundAutonomousRunReport,
    },
}

#[derive(Debug, Clone)]
struct PersistenceMismatch {
    actual: CanonicalAdmissionPersistenceEvidence,
    reason: String,
}

struct BindingPersistenceGate<'a, S> {
    inner: &'a mut S,
    required_persistence_profile_id: &'a str,
    mismatch: Rc<RefCell<Option<PersistenceMismatch>>>,
    accepted: Rc<RefCell<Option<CanonicalAdmissionPersistenceEvidence>>>,
    fatal: Rc<RefCell<Option<String>>>,
}

impl<S> AdmissionPersistence for BindingPersistenceGate<'_, S>
where
    S: CanonicalAdmissionPersistenceGate,
{
    fn persist(
        &mut self,
        intent: &CargoExecutionIntent,
        admission: &EffectAdmission,
    ) -> Result<AdmissionPersistenceAck, AdmissionPersistenceRejection> {
        let mut actual = self.inner.persist(
            intent,
            admission,
            self.required_persistence_profile_id,
        )?;
        if let Err(error) = normalize_persistence(&mut actual) {
            *self.fatal.borrow_mut() = Some(error.to_string());
            return Err(AdmissionPersistenceRejection {
                reason: "malformed canonical admission-persistence evidence".into(),
            });
        }

        if actual.persistence_profile_id != self.required_persistence_profile_id {
            *self.mismatch.borrow_mut() = Some(PersistenceMismatch {
                actual,
                reason: "persistence provider enforced another profile".into(),
            });
            return Err(AdmissionPersistenceRejection {
                reason: "required persistence profile was not satisfied".into(),
            });
        }

        *self.accepted.borrow_mut() = Some(actual.clone());
        Ok(AdmissionPersistenceAck {
            admission_receipt_digest: admission.receipt_digest.clone(),
            persistence_ack_digest: actual.persistence_ack_digest,
        })
    }
}

/// Require exact persistence-profile satisfaction before the already-bound
/// freshness-v2 stage and backend can become reachable.
#[allow(clippy::too_many_arguments)]
pub(crate) fn run_persistence_bound_autonomous<E, G, P, S, F, B>(
    mut intent: CargoExecutionIntent,
    bindings: ValidatedPreflightBindings,
    expected_effect: ExpectedEffectAdmissionBinding,
    mut expected_persistence: ExpectedAdmissionPersistenceBinding,
    eligibility_gate: &mut E,
    effect_gate: &mut G,
    source_probe: &mut P,
    persistence: &mut S,
    freshness: &mut F,
    backend: &mut B,
) -> anyhow::Result<PersistenceBoundAutonomousRunReport>
where
    E: AutonomousEligibilityGate,
    G: CommitmentAwareEffectEntryGate,
    P: RepositorySourceProbe,
    S: CanonicalAdmissionPersistenceGate,
    F: CanonicalFreshnessV2Gate,
    B: CargoProcessBackend,
{
    validate_intent(&mut intent)?;
    let intent_id = intent.intent_id.clone();

    normalize_digest(
        "persistence_expectation.authority_snapshot_digest",
        &mut expected_persistence.authority_snapshot_digest,
    )?;
    normalize_digest(
        "persistence_expectation.required_persistence_profile_id",
        &mut expected_persistence.required_persistence_profile_id,
    )?;
    let mut effect_authority_snapshot_digest = expected_effect.authority_snapshot_digest.clone();
    normalize_digest(
        "effect_expectation.authority_snapshot_digest",
        &mut effect_authority_snapshot_digest,
    )?;

    if expected_persistence.authority_snapshot_digest != effect_authority_snapshot_digest {
        return Ok(
            PersistenceBoundAutonomousRunReport::PersistenceExpectationRejectedBeforeEligibility {
                intent_id,
                effect_authority_snapshot_digest,
                persistence_authority_snapshot_digest: expected_persistence.authority_snapshot_digest,
                reason: "persistence requirement belongs to another authority snapshot".into(),
            },
        );
    }

    let required_persistence_profile_id = expected_persistence.required_persistence_profile_id;
    let mismatch = Rc::new(RefCell::new(None));
    let accepted = Rc::new(RefCell::new(None));
    let fatal = Rc::new(RefCell::new(None));

    let inner = {
        let mut persistence_gate = BindingPersistenceGate {
            inner: persistence,
            required_persistence_profile_id: &required_persistence_profile_id,
            mismatch: Rc::clone(&mismatch),
            accepted: Rc::clone(&accepted),
            fatal: Rc::clone(&fatal),
        };
        run_freshness_bound_autonomous(
            intent,
            bindings,
            expected_effect,
            eligibility_gate,
            effect_gate,
            source_probe,
            &mut persistence_gate,
            freshness,
            backend,
        )?
    };

    if let Some(error) = fatal.borrow_mut().take() {
        bail!("{error}");
    }

    if let Some(mismatch) = mismatch.borrow_mut().take() {
        return Ok(PersistenceBoundAutonomousRunReport::PersistenceBindingRejected {
            intent_id,
            required_persistence_profile_id,
            actual: mismatch.actual,
            reason: mismatch.reason,
            inner,
        });
    }

    let accepted_persistence = accepted.borrow_mut().take();
    Ok(PersistenceBoundAutonomousRunReport::AfterPersistenceBinding {
        intent_id,
        accepted_persistence,
        inner,
    })
}

fn normalize_persistence(value: &mut CanonicalAdmissionPersistenceEvidence) -> anyhow::Result<()> {
    normalize_digest(
        "persistence.persistence_ack_digest",
        &mut value.persistence_ack_digest,
    )?;
    normalize_digest(
        "persistence.persistence_profile_id",
        &mut value.persistence_profile_id,
    )?;
    validate_canonical_text("persistence.persistence_record_id", &value.persistence_record_id)?;
    validate_canonical_text("persistence.control_plane_id", &value.control_plane_id)?;
    normalize_optional_digest(
        "persistence.evidence_root_digest",
        &mut value.evidence_root_digest,
    )?;
    normalize_optional_digest(
        "persistence.anti_rollback_evidence_digest",
        &mut value.anti_rollback_evidence_digest,
    )?;
    Ok(())
}

fn validate_canonical_text(name: &str, value: &str) -> anyhow::Result<()> {
    if value.is_empty() || value.trim() != value {
        bail!("{name} must be non-empty canonical text without surrounding whitespace");
    }
    if value.bytes().any(|byte| matches!(byte, b'\r' | b'\n' | 0)) {
        bail!("{name} must not contain line separators or NUL");
    }
    if value.len() > 256 {
        bail!("{name} exceeds the 256-byte deterministic-model limit");
    }
    Ok(())
}

fn normalize_optional_digest(name: &str, value: &mut Option<String>) -> anyhow::Result<()> {
    if let Some(value) = value {
        normalize_digest(name, value)?;
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
