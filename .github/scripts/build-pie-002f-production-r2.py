#!/usr/bin/env python3
"""Preparation-only source transformation for the PIE-002F R2 candidate.

This script is not product code. The builder checks out the exact qualified
convergence parent, imports the frozen failed PIE-002F subject, applies this
narrow structural repair, and lets Rust 1.96 formatting/tests/doctests/clippy
prove whether a publishable candidate exists.
"""

from pathlib import Path

SOURCE = Path(
    "crates/domains/symthaea-planetary-industry/src/utility_evidence_context.rs"
)

source = SOURCE.read_text()
start_marker = "/// One explicit value together with the evidence records offered for that fact.\n"
end_marker = "/// Lossless PIE-002F validation receipt.\n"
start = source.index(start_marker)
end = source.index(end_marker)

replacement = r'''/// One explicit value together with the evidence records offered for that fact.
///
/// The state is intentionally private. Instances exposed by an
/// [`EvidenceBearingElectricalSupplyRecoveryContext`] have already crossed that
/// context's validation boundary; external code receives read-only accessors.
#[derive(Debug, Clone, Serialize)]
pub struct EvidenceBound<T> {
    value: T,
    evidence: Vec<EvidenceRef>,
}

impl<T> EvidenceBound<T> {
    fn from_parts(value: T, evidence: Vec<EvidenceRef>) -> Self {
        Self { value, evidence }
    }

    /// Borrow the exact value admitted for this fact.
    pub fn value(&self) -> &T {
        &self.value
    }

    /// Borrow provenance in the exact caller-supplied order.
    pub fn evidence(&self) -> &[EvidenceRef] {
        &self.evidence
    }
}

/// Evidence-bearing form of all nine external facts required by PIE-002D.
///
/// This type is a validated in-memory witness for PIE-002F provenance binding,
/// not a portable authority token. Its state is private, it deliberately does
/// not implement `Deserialize`, and construction routes through [`Self::try_new`].
/// Serialization is one-way and exists only for evidence/reporting surfaces.
///
/// External code cannot read or reconstruct private state directly:
///
/// ```compile_fail
/// use symthaea_planetary_industry::EvidenceBearingElectricalSupplyRecoveryContext;
/// fn cannot_read_private_state(context: &EvidenceBearingElectricalSupplyRecoveryContext) {
///     let _ = &context.recoverable_energy_j;
/// }
/// ```
///
/// Deserialization cannot manufacture a validated witness:
///
/// ```compile_fail
/// use serde::de::DeserializeOwned;
/// use symthaea_planetary_industry::EvidenceBearingElectricalSupplyRecoveryContext;
/// fn require_deserialize<T: DeserializeOwned>() {}
/// fn proof() {
///     require_deserialize::<EvidenceBearingElectricalSupplyRecoveryContext>();
/// }
/// ```
#[derive(Debug, Clone, Serialize)]
pub struct EvidenceBearingElectricalSupplyRecoveryContext {
    recoverable_energy_j: EvidenceBound<EnergyRangeJ>,
    recovery_duration_s: EvidenceBound<DurationRangeS>,
    storage_acceptance_j: EvidenceBound<EnergyRangeJ>,
    storage_charge_power_w: EvidenceBound<PowerRangeW>,
    storage_discharge_power_w: EvidenceBound<PowerRangeW>,
    recovery_delivery_fraction: EvidenceBound<FractionRange>,
    available_energy_capacity_j: EvidenceBound<EnergyRangeJ>,
    available_sustained_power_w: EvidenceBound<PowerRangeW>,
    available_peak_power_w: EvidenceBound<PowerRangeW>,
}

impl EvidenceBearingElectricalSupplyRecoveryContext {
    /// Validate and bind all nine external numerical facts to explicit provenance.
    ///
    /// The intentionally explicit signature prevents field omission or implicit
    /// defaults. Success establishes provenance structure only; it does not prove
    /// truth, freshness, independence, applicability, feasibility, or authority.
    #[allow(clippy::too_many_arguments)]
    pub fn try_new(
        recoverable_energy_j: EnergyRangeJ,
        recoverable_energy_evidence: Vec<EvidenceRef>,
        recovery_duration_s: DurationRangeS,
        recovery_duration_evidence: Vec<EvidenceRef>,
        storage_acceptance_j: EnergyRangeJ,
        storage_acceptance_evidence: Vec<EvidenceRef>,
        storage_charge_power_w: PowerRangeW,
        storage_charge_power_evidence: Vec<EvidenceRef>,
        storage_discharge_power_w: PowerRangeW,
        storage_discharge_power_evidence: Vec<EvidenceRef>,
        recovery_delivery_fraction: FractionRange,
        recovery_delivery_fraction_evidence: Vec<EvidenceRef>,
        available_energy_capacity_j: EnergyRangeJ,
        available_energy_capacity_evidence: Vec<EvidenceRef>,
        available_sustained_power_w: PowerRangeW,
        available_sustained_power_evidence: Vec<EvidenceRef>,
        available_peak_power_w: PowerRangeW,
        available_peak_power_evidence: Vec<EvidenceRef>,
    ) -> Result<Self, UtilityEvidenceContextError> {
        let context = Self {
            recoverable_energy_j: EvidenceBound::from_parts(
                recoverable_energy_j,
                recoverable_energy_evidence,
            ),
            recovery_duration_s: EvidenceBound::from_parts(
                recovery_duration_s,
                recovery_duration_evidence,
            ),
            storage_acceptance_j: EvidenceBound::from_parts(
                storage_acceptance_j,
                storage_acceptance_evidence,
            ),
            storage_charge_power_w: EvidenceBound::from_parts(
                storage_charge_power_w,
                storage_charge_power_evidence,
            ),
            storage_discharge_power_w: EvidenceBound::from_parts(
                storage_discharge_power_w,
                storage_discharge_power_evidence,
            ),
            recovery_delivery_fraction: EvidenceBound::from_parts(
                recovery_delivery_fraction,
                recovery_delivery_fraction_evidence,
            ),
            available_energy_capacity_j: EvidenceBound::from_parts(
                available_energy_capacity_j,
                available_energy_capacity_evidence,
            ),
            available_sustained_power_w: EvidenceBound::from_parts(
                available_sustained_power_w,
                available_sustained_power_evidence,
            ),
            available_peak_power_w: EvidenceBound::from_parts(
                available_peak_power_w,
                available_peak_power_evidence,
            ),
        };
        validate_evidence_bearing_electrical_context(&context)?;
        Ok(context)
    }

    /// Recoverable process energy and its provenance.
    pub fn recoverable_energy_j(&self) -> &EvidenceBound<EnergyRangeJ> {
        &self.recoverable_energy_j
    }

    /// Recovery duration and its provenance.
    pub fn recovery_duration_s(&self) -> &EvidenceBound<DurationRangeS> {
        &self.recovery_duration_s
    }

    /// Storage energy acceptance and its provenance.
    pub fn storage_acceptance_j(&self) -> &EvidenceBound<EnergyRangeJ> {
        &self.storage_acceptance_j
    }

    /// Storage charge-power acceptance and its provenance.
    pub fn storage_charge_power_w(&self) -> &EvidenceBound<PowerRangeW> {
        &self.storage_charge_power_w
    }

    /// Storage discharge-power capability and its provenance.
    pub fn storage_discharge_power_w(&self) -> &EvidenceBound<PowerRangeW> {
        &self.storage_discharge_power_w
    }

    /// Recovery delivery fraction and its provenance.
    pub fn recovery_delivery_fraction(&self) -> &EvidenceBound<FractionRange> {
        &self.recovery_delivery_fraction
    }

    /// Available energy capacity and its provenance.
    pub fn available_energy_capacity_j(&self) -> &EvidenceBound<EnergyRangeJ> {
        &self.available_energy_capacity_j
    }

    /// Available sustained power and its provenance.
    pub fn available_sustained_power_w(&self) -> &EvidenceBound<PowerRangeW> {
        &self.available_sustained_power_w
    }

    /// Available peak power and its provenance.
    pub fn available_peak_power_w(&self) -> &EvidenceBound<PowerRangeW> {
        &self.available_peak_power_w
    }

    /// Explicitly erase provenance and return the numerical PIE-002D context.
    ///
    /// This conversion is lossless for the nine numerical values but deliberately
    /// drops provenance; callers that need provenance authority must retain this
    /// witness or a validation receipt alongside the numerical context.
    pub fn to_numeric_context(&self) -> ElectricalSupplyRecoveryContext {
        ElectricalSupplyRecoveryContext {
            recoverable_energy_j: self.recoverable_energy_j.value,
            recovery_duration_s: self.recovery_duration_s.value,
            storage_acceptance_j: self.storage_acceptance_j.value,
            storage_charge_power_w: self.storage_charge_power_w.value,
            storage_discharge_power_w: self.storage_discharge_power_w.value,
            recovery_delivery_fraction: self.recovery_delivery_fraction.value,
            available_energy_capacity_j: self.available_energy_capacity_j.value,
            available_sustained_power_w: self.available_sustained_power_w.value,
            available_peak_power_w: self.available_peak_power_w.value,
        }
    }

    /// Revalidate the witness and emit a portable provenance receipt.
    ///
    /// The receipt records what passed validation; deserializing that receipt is
    /// not equivalent to reconstructing this validated in-memory witness.
    pub fn validation_receipt(
        &self,
    ) -> Result<EvidenceBearingContextReceipt, UtilityEvidenceContextError> {
        validate_evidence_bearing_electrical_context(self)
    }
}

'''
source = source[:start] + replacement + source[end:]

old_bare = '''    let bare_context = ElectricalSupplyRecoveryContext {
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
'''
if source.count(old_bare) != 1:
    raise SystemExit("expected exactly one bare-context construction block")
source = source.replace(old_bare, "    let bare_context = context.to_numeric_context();\n", 1)

insert = r'''

    #[test]
    fn public_constructor_routes_through_the_same_validation_boundary() {
        let raw = baseline_context();
        let context = EvidenceBearingElectricalSupplyRecoveryContext::try_new(
            raw.recoverable_energy_j.value,
            raw.recoverable_energy_j.evidence.clone(),
            raw.recovery_duration_s.value,
            raw.recovery_duration_s.evidence.clone(),
            raw.storage_acceptance_j.value,
            raw.storage_acceptance_j.evidence.clone(),
            raw.storage_charge_power_w.value,
            raw.storage_charge_power_w.evidence.clone(),
            raw.storage_discharge_power_w.value,
            raw.storage_discharge_power_w.evidence.clone(),
            raw.recovery_delivery_fraction.value,
            raw.recovery_delivery_fraction.evidence.clone(),
            raw.available_energy_capacity_j.value,
            raw.available_energy_capacity_j.evidence.clone(),
            raw.available_sustained_power_w.value,
            raw.available_sustained_power_w.evidence.clone(),
            raw.available_peak_power_w.value,
            raw.available_peak_power_w.evidence.clone(),
        )
        .unwrap();

        assert_eq!(
            context.recoverable_energy_j().value().min.value(),
            30.0
        );
        assert_eq!(
            context.recoverable_energy_j().evidence()[0].evidence_id,
            "study-a"
        );
        assert_eq!(
            context.to_numeric_context().recovery_delivery_fraction,
            FractionRange::new(0.8, 0.9).unwrap()
        );
    }

    #[test]
    fn public_constructor_rejects_missing_provenance() {
        let raw = baseline_context();
        let result = EvidenceBearingElectricalSupplyRecoveryContext::try_new(
            raw.recoverable_energy_j.value,
            Vec::new(),
            raw.recovery_duration_s.value,
            raw.recovery_duration_s.evidence.clone(),
            raw.storage_acceptance_j.value,
            raw.storage_acceptance_j.evidence.clone(),
            raw.storage_charge_power_w.value,
            raw.storage_charge_power_w.evidence.clone(),
            raw.storage_discharge_power_w.value,
            raw.storage_discharge_power_w.evidence.clone(),
            raw.recovery_delivery_fraction.value,
            raw.recovery_delivery_fraction.evidence.clone(),
            raw.available_energy_capacity_j.value,
            raw.available_energy_capacity_j.evidence.clone(),
            raw.available_sustained_power_w.value,
            raw.available_sustained_power_w.evidence.clone(),
            raw.available_peak_power_w.value,
            raw.available_peak_power_w.evidence.clone(),
        );

        assert!(matches!(
            result,
            Err(UtilityEvidenceContextError::MissingEvidence(
                "recoverable_energy_j"
            ))
        ));
    }
'''
closing = source.rfind("\n}")
if closing < 0:
    raise SystemExit("tests module closing brace not found")
source = source[:closing] + insert + source[closing:]

strong = source[
    source.index("pub struct EvidenceBound<T>") : source.index(
        "/// Lossless PIE-002F validation receipt."
    )
]
if "Deserialize" in strong:
    raise SystemExit("strong PIE-002F wrappers must not implement Deserialize")
for forbidden in ("pub value:", "pub evidence:", "pub recoverable_energy_j:"):
    if forbidden in strong:
        raise SystemExit(f"public validated state leaked: {forbidden}")
if "pub fn try_new(" not in strong:
    raise SystemExit("validated public construction path missing")

SOURCE.write_text(source)
