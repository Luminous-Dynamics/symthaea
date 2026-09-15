// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later

//! PIE-002F evidence-bearing electrical supply/recovery context.
//!
//! This module mirrors the independently qualified PIE-002F oracle while using
//! qualified CORE-ID-001 lexical admission for evidence identifiers. It accepts
//! historical [`EvidenceRef`] DTOs for compatibility, validates them at this
//! boundary, preserves their exact provenance bytes, and projects the numerical
//! values losslessly into the existing PIE-002D [`ElectricalSupplyRecoveryContext`].
//!
//! Evidence-bearing does not imply truth, freshness, independence, applicability,
//! feasibility, dispatch correctness, or execution authority. In particular, the
//! same evidence item may support several facts and remains visibly the same
//! provenance rather than becoming independent corroboration.

use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;
use std::error::Error;
use std::fmt;

use symthaea_identity::{CanonicalIdentifier, CanonicalIdentifierError};

use crate::{
    DurationRangeS, ElectricalSupplyRecoveryContext, EnergyRangeJ, EvidenceRef, FractionRange,
    OntologyError, PowerRangeW,
};

/// Maximum number of evidence references admitted for one external utility fact.
pub const MAX_UTILITY_FACT_EVIDENCE_REFS: usize = 64;

/// Maximum UTF-8 byte length admitted for an evidence identifier, source, or note.
pub const MAX_UTILITY_EVIDENCE_TEXT_BYTES: usize = 4096;

type UtilityEvidenceIdentifier = CanonicalIdentifier<MAX_UTILITY_EVIDENCE_TEXT_BYTES>;

/// One explicit value together with the evidence records offered for that fact.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceBound<T> {
    /// Numerical value or range being supported.
    pub value: T,
    /// Explicit provenance attached to this one fact.
    pub evidence: Vec<EvidenceRef>,
}

/// Evidence-bearing form of all nine external facts required by PIE-002D.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceBearingElectricalSupplyRecoveryContext {
    /// Recoverable process energy before storage limits.
    pub recoverable_energy_j: EvidenceBound<EnergyRangeJ>,
    /// Recovery duration; lower bound must be strictly positive.
    pub recovery_duration_s: EvidenceBound<DurationRangeS>,
    /// Remaining storage energy acceptance.
    pub storage_acceptance_j: EvidenceBound<EnergyRangeJ>,
    /// Storage charge-power acceptance.
    pub storage_charge_power_w: EvidenceBound<PowerRangeW>,
    /// Storage discharge-power capability.
    pub storage_discharge_power_w: EvidenceBound<PowerRangeW>,
    /// Recoverable-energy delivery fraction after conversion/storage loss.
    pub recovery_delivery_fraction: EvidenceBound<FractionRange>,
    /// Electrical energy capacity available to the process basis.
    pub available_energy_capacity_j: EvidenceBound<EnergyRangeJ>,
    /// Sustained electrical power available to the process basis.
    pub available_sustained_power_w: EvidenceBound<PowerRangeW>,
    /// Peak electrical power available to the process basis.
    pub available_peak_power_w: EvidenceBound<PowerRangeW>,
}

/// Lossless PIE-002F validation receipt.
///
/// This is portable evidence, not an opaque authority capability. Deserializing
/// it does not recreate the validation event; stronger compositions must
/// revalidate their inputs.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct EvidenceBearingContextReceipt {
    /// Exact numerical context accepted by PIE-002D.
    pub bare_context: ElectricalSupplyRecoveryContext,
    /// Provenance for recoverable energy.
    pub recoverable_energy_evidence: Vec<EvidenceRef>,
    /// Provenance for recovery duration.
    pub recovery_duration_evidence: Vec<EvidenceRef>,
    /// Provenance for storage energy acceptance.
    pub storage_acceptance_evidence: Vec<EvidenceRef>,
    /// Provenance for storage charge power.
    pub storage_charge_power_evidence: Vec<EvidenceRef>,
    /// Provenance for storage discharge power.
    pub storage_discharge_power_evidence: Vec<EvidenceRef>,
    /// Provenance for recovery delivery fraction.
    pub recovery_delivery_fraction_evidence: Vec<EvidenceRef>,
    /// Provenance for available energy capacity.
    pub available_energy_capacity_evidence: Vec<EvidenceRef>,
    /// Provenance for available sustained power.
    pub available_sustained_power_evidence: Vec<EvidenceRef>,
    /// Provenance for available peak power.
    pub available_peak_power_evidence: Vec<EvidenceRef>,
}

/// Failures specific to PIE-002F provenance validation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UtilityEvidenceContextError {
    /// Historical PIE ontology validation failed.
    Ontology(OntologyError),
    /// An evidence identifier violates qualified CORE-ID-001 lexical admission.
    CanonicalIdentifier(CanonicalIdentifierError),
    /// One numerical fact carries no explicit provenance.
    MissingEvidence(&'static str),
    /// One fact exceeds the frozen evidence-reference budget.
    TooManyEvidenceRefs {
        /// Fact being validated.
        field: &'static str,
        /// Actual number of evidence references.
        count: usize,
    },
    /// A source or note exceeds the frozen UTF-8 byte budget.
    EvidenceTextTooLarge {
        /// Evidence text field being validated.
        field: &'static str,
        /// Maximum admitted UTF-8 bytes.
        max_utf8_bytes: usize,
        /// Actual UTF-8 byte length.
        actual_utf8_bytes: usize,
    },
    /// One fact repeats the same canonical evidence identifier.
    DuplicateEvidenceId {
        /// Fact whose provenance list contains the duplicate.
        field: &'static str,
        /// Exact duplicate identifier bytes represented as text.
        evidence_id: String,
    },
}

impl fmt::Display for UtilityEvidenceContextError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ontology(error) => write!(f, "{error}"),
            Self::CanonicalIdentifier(error) => write!(f, "{error}"),
            Self::MissingEvidence(field) => write!(f, "evidence required for {field}"),
            Self::TooManyEvidenceRefs { field, count } => write!(
                f,
                "too many evidence references for {field}: {count} > {MAX_UTILITY_FACT_EVIDENCE_REFS}"
            ),
            Self::EvidenceTextTooLarge {
                field,
                max_utf8_bytes,
                actual_utf8_bytes,
            } => write!(
                f,
                "evidence text exceeds {max_utf8_bytes} UTF-8 bytes for {field}: {actual_utf8_bytes}"
            ),
            Self::DuplicateEvidenceId { field, evidence_id } => {
                write!(f, "duplicate evidence id for {field}: {evidence_id}")
            }
        }
    }
}

impl Error for UtilityEvidenceContextError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match self {
            Self::Ontology(error) => Some(error),
            Self::CanonicalIdentifier(error) => Some(error),
            _ => None,
        }
    }
}

impl From<OntologyError> for UtilityEvidenceContextError {
    fn from(value: OntologyError) -> Self {
        Self::Ontology(value)
    }
}

impl From<CanonicalIdentifierError> for UtilityEvidenceContextError {
    fn from(value: CanonicalIdentifierError) -> Self {
        Self::CanonicalIdentifier(value)
    }
}

fn validate_text_budget(
    value: &str,
    field: &'static str,
) -> Result<(), UtilityEvidenceContextError> {
    let actual_utf8_bytes = value.len();
    if actual_utf8_bytes <= MAX_UTILITY_EVIDENCE_TEXT_BYTES {
        Ok(())
    } else {
        Err(UtilityEvidenceContextError::EvidenceTextTooLarge {
            field,
            max_utf8_bytes: MAX_UTILITY_EVIDENCE_TEXT_BYTES,
            actual_utf8_bytes,
        })
    }
}

fn validate_fact_evidence(
    evidence: &[EvidenceRef],
    field: &'static str,
) -> Result<(), UtilityEvidenceContextError> {
    if evidence.is_empty() {
        return Err(UtilityEvidenceContextError::MissingEvidence(field));
    }
    if evidence.len() > MAX_UTILITY_FACT_EVIDENCE_REFS {
        return Err(UtilityEvidenceContextError::TooManyEvidenceRefs {
            field,
            count: evidence.len(),
        });
    }

    let mut ids = BTreeSet::<UtilityEvidenceIdentifier>::new();
    for item in evidence {
        item.validate()?;
        let canonical_id = UtilityEvidenceIdentifier::new(item.evidence_id.clone())?;
        validate_text_budget(&item.source, "evidence_source")?;
        if let Some(note) = &item.note {
            validate_text_budget(note, "evidence_note")?;
        }
        if !ids.insert(canonical_id) {
            return Err(UtilityEvidenceContextError::DuplicateEvidenceId {
                field,
                evidence_id: item.evidence_id.clone(),
            });
        }
    }
    Ok(())
}

fn validate_positive_duration(
    value: DurationRangeS,
    field: &'static str,
) -> Result<(), UtilityEvidenceContextError> {
    value.validate()?;
    if value.min.value() > 0.0 {
        Ok(())
    } else {
        Err(OntologyError::InvalidRange(field).into())
    }
}

/// Validate all nine evidence-bearing external utility facts and project their
/// numerical values losslessly into the existing PIE-002D context.
///
/// Success establishes only explicit, structurally valid provenance for this
/// invocation. It does not establish evidence independence, applicability,
/// freshness, factual truth, feasibility, or authority.
pub fn validate_evidence_bearing_electrical_context(
    context: &EvidenceBearingElectricalSupplyRecoveryContext,
) -> Result<EvidenceBearingContextReceipt, UtilityEvidenceContextError> {
    context.recoverable_energy_j.value.validate()?;
    validate_fact_evidence(
        &context.recoverable_energy_j.evidence,
        "recoverable_energy_j",
    )?;

    validate_positive_duration(
        context.recovery_duration_s.value,
        "recovery_duration_s",
    )?;
    validate_fact_evidence(
        &context.recovery_duration_s.evidence,
        "recovery_duration_s",
    )?;

    context.storage_acceptance_j.value.validate()?;
    validate_fact_evidence(
        &context.storage_acceptance_j.evidence,
        "storage_acceptance_j",
    )?;

    context.storage_charge_power_w.value.validate()?;
    validate_fact_evidence(
        &context.storage_charge_power_w.evidence,
        "storage_charge_power_w",
    )?;

    context.storage_discharge_power_w.value.validate()?;
    validate_fact_evidence(
        &context.storage_discharge_power_w.evidence,
        "storage_discharge_power_w",
    )?;

    FractionRange::new(
        context.recovery_delivery_fraction.value.min(),
        context.recovery_delivery_fraction.value.max(),
    )?;
    validate_fact_evidence(
        &context.recovery_delivery_fraction.evidence,
        "recovery_delivery_fraction",
    )?;

    context.available_energy_capacity_j.value.validate()?;
    validate_fact_evidence(
        &context.available_energy_capacity_j.evidence,
        "available_energy_capacity_j",
    )?;

    context.available_sustained_power_w.value.validate()?;
    validate_fact_evidence(
        &context.available_sustained_power_w.evidence,
        "available_sustained_power_w",
    )?;

    context.available_peak_power_w.value.validate()?;
    validate_fact_evidence(
        &context.available_peak_power_w.evidence,
        "available_peak_power_w",
    )?;

    let bare_context = ElectricalSupplyRecoveryContext {
        recoverable_energy_j: context.recoverable_energy_j.value,
        recovery_duration_s: context.recovery_duration_s.value,
        storage_acceptance_j: context.storage_acceptance_j.value,
        storage_charge_power_w: context.storage_charge_power_w.value,
        storage_discharge_power_w: context.storage_discharge_power_w.value,
        recovery_delivery_fraction: context.recovery_delivery_fraction.value,
        available_energy_capacity_j: context.available_energy_capacity_j.value,
        available_sustained_power_w: context.available_sustained_power_w.value,
        available_peak_power_w: context.available_peak_power_w.value,
    };

    Ok(EvidenceBearingContextReceipt {
        bare_context,
        recoverable_energy_evidence: context.recoverable_energy_j.evidence.clone(),
        recovery_duration_evidence: context.recovery_duration_s.evidence.clone(),
        storage_acceptance_evidence: context.storage_acceptance_j.evidence.clone(),
        storage_charge_power_evidence: context.storage_charge_power_w.evidence.clone(),
        storage_discharge_power_evidence: context.storage_discharge_power_w.evidence.clone(),
        recovery_delivery_fraction_evidence: context.recovery_delivery_fraction.evidence.clone(),
        available_energy_capacity_evidence: context.available_energy_capacity_j.evidence.clone(),
        available_sustained_power_evidence: context.available_sustained_power_w.evidence.clone(),
        available_peak_power_evidence: context.available_peak_power_w.evidence.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EvidenceClass;

    fn evidence(id: &str, class: EvidenceClass, source: &str) -> EvidenceRef {
        EvidenceRef {
            evidence_id: id.into(),
            class,
            source: source.into(),
            note: Some("synthetic fixture".into()),
        }
    }

    fn hypothesis(id: &str) -> EvidenceRef {
        evidence(id, EvidenceClass::Hypothesis, "")
    }

    fn bound<T>(value: T, evidence: Vec<EvidenceRef>) -> EvidenceBound<T> {
        EvidenceBound { value, evidence }
    }

    fn baseline_context() -> EvidenceBearingElectricalSupplyRecoveryContext {
        let shared = evidence(
            "study-a",
            EvidenceClass::LiteratureModel,
            "doi:study-a",
        );
        let measured = evidence("lab-a", EvidenceClass::LabMeasured, "report:lab-a");
        let hypothesized = hypothesis("hyp-a");

        EvidenceBearingElectricalSupplyRecoveryContext {
            recoverable_energy_j: bound(
                EnergyRangeJ::new(30.0, 40.0).unwrap(),
                vec![shared.clone()],
            ),
            recovery_duration_s: bound(
                DurationRangeS::new(4.0, 5.0).unwrap(),
                vec![measured.clone()],
            ),
            storage_acceptance_j: bound(
                EnergyRangeJ::new(50.0, 60.0).unwrap(),
                vec![shared.clone()],
            ),
            storage_charge_power_w: bound(
                PowerRangeW::new(10.0, 12.0).unwrap(),
                vec![measured.clone()],
            ),
            storage_discharge_power_w: bound(
                PowerRangeW::new(8.0, 9.0).unwrap(),
                vec![measured.clone()],
            ),
            recovery_delivery_fraction: bound(
                FractionRange::new(0.8, 0.9).unwrap(),
                vec![shared.clone()],
            ),
            available_energy_capacity_j: bound(
                EnergyRangeJ::new(120.0, 140.0).unwrap(),
                vec![hypothesized],
            ),
            available_sustained_power_w: bound(
                PowerRangeW::new(15.0, 20.0).unwrap(),
                vec![shared],
            ),
            available_peak_power_w: bound(
                PowerRangeW::new(30.0, 35.0).unwrap(),
                vec![measured],
            ),
        }
    }

    #[test]
    fn exact_values_and_provenance_are_preserved() {
        let context = baseline_context();
        let receipt = validate_evidence_bearing_electrical_context(&context).unwrap();

        assert_eq!(
            receipt.bare_context.recoverable_energy_j,
            context.recoverable_energy_j.value
        );
        assert_eq!(
            receipt.bare_context.recovery_duration_s,
            context.recovery_duration_s.value
        );
        assert_eq!(
            receipt.bare_context.recovery_delivery_fraction,
            context.recovery_delivery_fraction.value
        );
        assert_eq!(
            receipt.recoverable_energy_evidence,
            context.recoverable_energy_j.evidence
        );
        assert_eq!(
            receipt.storage_acceptance_evidence,
            context.storage_acceptance_j.evidence
        );
    }

    #[test]
    fn missing_evidence_fails_closed() {
        let mut context = baseline_context();
        context.available_peak_power_w.evidence.clear();
        assert_eq!(
            validate_evidence_bearing_electrical_context(&context),
            Err(UtilityEvidenceContextError::MissingEvidence(
                "available_peak_power_w"
            ))
        );
    }

    #[test]
    fn measured_evidence_requires_source() {
        let mut context = baseline_context();
        context.recovery_duration_s.evidence = vec![evidence(
            "lab-b",
            EvidenceClass::LabMeasured,
            "",
        )];
        assert_eq!(
            validate_evidence_bearing_electrical_context(&context),
            Err(UtilityEvidenceContextError::Ontology(
                OntologyError::MissingEvidenceSource
            ))
        );
    }

    #[test]
    fn duplicate_id_within_one_fact_fails_closed() {
        let mut context = baseline_context();
        context.recoverable_energy_j.evidence = vec![
            evidence("dup", EvidenceClass::LiteratureModel, "doi:a"),
            evidence("dup", EvidenceClass::LabMeasured, "report:b"),
        ];
        assert_eq!(
            validate_evidence_bearing_electrical_context(&context),
            Err(UtilityEvidenceContextError::DuplicateEvidenceId {
                field: "recoverable_energy_j",
                evidence_id: "dup".into(),
            })
        );
    }

    #[test]
    fn reused_source_across_facts_remains_shared_not_independent() {
        let mut context = baseline_context();
        let shared = evidence(
            "shared",
            EvidenceClass::LiteratureModel,
            "doi:shared",
        );
        context.recoverable_energy_j.evidence = vec![shared.clone()];
        context.storage_acceptance_j.evidence = vec![shared];

        let receipt = validate_evidence_bearing_electrical_context(&context).unwrap();
        assert_eq!(
            receipt.recoverable_energy_evidence[0],
            receipt.storage_acceptance_evidence[0]
        );
    }

    #[test]
    fn zero_inclusive_recovery_duration_fails_closed() {
        let mut context = baseline_context();
        context.recovery_duration_s.value = DurationRangeS::new(0.0, 5.0).unwrap();
        assert_eq!(
            validate_evidence_bearing_electrical_context(&context),
            Err(UtilityEvidenceContextError::Ontology(
                OntologyError::InvalidRange("recovery_duration_s")
            ))
        );
    }

    #[test]
    fn noncanonical_evidence_id_fails_through_core_id() {
        let mut context = baseline_context();
        context.recoverable_energy_j.evidence = vec![hypothesis(" study-a")];
        assert!(matches!(
            validate_evidence_bearing_electrical_context(&context),
            Err(UtilityEvidenceContextError::CanonicalIdentifier(
                CanonicalIdentifierError::EdgeWhitespace { .. }
            ))
        ));
    }

    #[test]
    fn evidence_budget_is_enforced_per_fact() {
        let mut context = baseline_context();
        context.recoverable_energy_j.evidence = (0..=MAX_UTILITY_FACT_EVIDENCE_REFS)
            .map(|index| hypothesis(&format!("e-{index}")))
            .collect();
        assert_eq!(
            validate_evidence_bearing_electrical_context(&context),
            Err(UtilityEvidenceContextError::TooManyEvidenceRefs {
                field: "recoverable_energy_j",
                count: MAX_UTILITY_FACT_EVIDENCE_REFS + 1,
            })
        );
    }

    #[test]
    fn oversized_source_or_note_fails_closed() {
        let mut source_context = baseline_context();
        source_context.recoverable_energy_j.evidence = vec![EvidenceRef {
            evidence_id: "hyp-long-source".into(),
            class: EvidenceClass::Hypothesis,
            source: "x".repeat(MAX_UTILITY_EVIDENCE_TEXT_BYTES + 1),
            note: None,
        }];
        assert!(matches!(
            validate_evidence_bearing_electrical_context(&source_context),
            Err(UtilityEvidenceContextError::EvidenceTextTooLarge {
                field: "evidence_source",
                ..
            })
        ));

        let mut note_context = baseline_context();
        note_context.recoverable_energy_j.evidence = vec![EvidenceRef {
            evidence_id: "hyp-long-note".into(),
            class: EvidenceClass::Hypothesis,
            source: String::new(),
            note: Some("x".repeat(MAX_UTILITY_EVIDENCE_TEXT_BYTES + 1)),
        }];
        assert!(matches!(
            validate_evidence_bearing_electrical_context(&note_context),
            Err(UtilityEvidenceContextError::EvidenceTextTooLarge {
                field: "evidence_note",
                ..
            })
        ));
    }

    #[test]
    fn hypothesis_may_have_blank_source_and_evidence_order_is_preserved() {
        let mut context = baseline_context();
        let first = hypothesis("hyp-1");
        let second = hypothesis("hyp-2");
        context.available_energy_capacity_j.evidence = vec![first.clone(), second.clone()];

        let receipt = validate_evidence_bearing_electrical_context(&context).unwrap();
        assert_eq!(
            receipt.available_energy_capacity_evidence,
            vec![first, second]
        );
    }
}
