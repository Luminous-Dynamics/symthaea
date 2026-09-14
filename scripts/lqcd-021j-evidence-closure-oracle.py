#!/usr/bin/env python3
"""Independent cross-receipt evidence-closure oracle for LQCD-021J."""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass, replace
from typing import Optional

ORACLE_ID = "lqcd_beta6_evidence_closure_oracle_v1"
CLOSURE_TAG = b"symthaea.lqcd.beta6.evidence-closure.v1\x00"
COMPARISON_TAG = b"symthaea.lqcd.beta6.comparison-closure.v1\x00"
EXPECTED_CONFIGS = 4000
EXPECTED_KEYS_PER_CONFIG = 24 * 8
EXPECTED_MEASUREMENT_KEYS = EXPECTED_CONFIGS * EXPECTED_KEYS_PER_CONFIG

QUALIFIED = "RustQualified"
SOURCE_REVIEWED = "SourceReviewedCandidate"

@dataclass(frozen=True)
class ProductionQualification:
    component: str
    campaign_subject: bytes
    state: str
    subject_digest: bytes

@dataclass(frozen=True)
class FinalExecutionReceipt:
    campaign_subject: bytes
    final_campaign_predecessor: bytes
    retained_configuration_count: int
    configuration_set_commitment: bytes

@dataclass(frozen=True)
class AdmissionReceipt:
    campaign_subject: bytes
    configuration_set_commitment: bytes
    disposition: str
    receipt_digest: bytes

@dataclass(frozen=True)
class MeasurementReceipt:
    campaign_subject: bytes
    configuration_set_commitment: bytes
    measurement_key_count: int
    measurement_set_commitment: bytes
    provenance_qualification: str

@dataclass(frozen=True)
class OperatorReceipt:
    campaign_subject: bytes
    disposition: str
    receipt_digest: bytes
    unresolved_systematic: bool

@dataclass(frozen=True)
class HistoricalFidelityReceipt:
    campaign_subject: bytes
    ledger_digest: bytes
    claim_ceiling: str

@dataclass(frozen=True)
class SealedAnalysisReceipt:
    campaign_subject: bytes
    configuration_set_commitment: bytes
    measurement_set_commitment: bytes
    analysis_subject_digest: bytes
    sealed_result_digest: bytes
    target_isolated: bool
    benchmark_digest: Optional[bytes]

@dataclass(frozen=True)
class SystematicLedgerReceipt:
    campaign_subject: bytes
    sealed_result_digest: bytes
    operator_receipt_digest: bytes
    historical_ledger_digest: bytes
    ledger_digest: bytes

@dataclass(frozen=True)
class EvidenceBundle:
    final_campaign_subject: bytes
    expected_final_predecessor: bytes
    numerical: ProductionQualification
    rng_restart: ProductionQualification
    checkpoint: ProductionQualification
    config_measurement: ProductionQualification
    execution: FinalExecutionReceipt
    admission: AdmissionReceipt
    measurement: MeasurementReceipt
    operator: Optional[OperatorReceipt]
    historical: Optional[HistoricalFidelityReceipt]
    analysis: SealedAnalysisReceipt
    systematic: SystematicLedgerReceipt

class ClosureError(ValueError):
    pass

def h(label: str) -> bytes:
    return hashlib.sha256(label.encode("utf-8")).digest()

def _u16(n: int) -> bytes:
    if not 0 <= n <= 0xFFFF:
        raise ClosureError("u16 range")
    return n.to_bytes(2, "big")

def _u32(n: int) -> bytes:
    if not 0 <= n <= 0xFFFFFFFF:
        raise ClosureError("u32 range")
    return n.to_bytes(4, "big")

def _lp(text: str) -> bytes:
    raw = text.encode("utf-8")
    if not raw or len(raw) > 0xFFFF:
        raise ClosureError("invalid string")
    return _u16(len(raw)) + raw

def _digest32(value: bytes, name: str) -> bytes:
    if not isinstance(value, bytes) or len(value) != 32:
        raise ClosureError(f"{name} must be 32 bytes")
    return value

def _same_campaign(expected: bytes, actual: bytes, label: str) -> None:
    if actual != expected:
        raise ClosureError(f"CampaignSplice:{label}")

def _require_qualified(receipt: ProductionQualification, campaign: bytes) -> None:
    _same_campaign(campaign, receipt.campaign_subject, receipt.component)
    _digest32(receipt.subject_digest, f"{receipt.component}.subject_digest")
    if receipt.state != QUALIFIED:
        raise ClosureError(f"ImplementationNotQualified:{receipt.component}:{receipt.state}")

def evidence_closure(bundle: EvidenceBundle) -> tuple[bytes, str]:
    campaign = _digest32(bundle.final_campaign_subject, "final_campaign_subject")
    expected_predecessor = _digest32(bundle.expected_final_predecessor, "expected_final_predecessor")

    for receipt in (bundle.numerical, bundle.rng_restart, bundle.checkpoint, bundle.config_measurement):
        _require_qualified(receipt, campaign)

    _same_campaign(campaign, bundle.execution.campaign_subject, "execution")
    if bundle.execution.final_campaign_predecessor != expected_predecessor:
        raise ClosureError("FinalCampaignPredecessorMismatch")
    if bundle.execution.retained_configuration_count != EXPECTED_CONFIGS:
        raise ClosureError("RetainedConfigurationCountMismatch")
    config_set = _digest32(bundle.execution.configuration_set_commitment, "configuration_set_commitment")

    _same_campaign(campaign, bundle.admission.campaign_subject, "admission")
    if bundle.admission.configuration_set_commitment != config_set:
        raise ClosureError("AdmissionConfigurationSetMismatch")
    if bundle.admission.disposition != "Admissible":
        raise ClosureError(f"FinalSampleNotAdmissible:{bundle.admission.disposition}")
    _digest32(bundle.admission.receipt_digest, "admission.receipt_digest")

    _same_campaign(campaign, bundle.measurement.campaign_subject, "measurement")
    if bundle.measurement.configuration_set_commitment != config_set:
        raise ClosureError("MeasurementConfigurationSetMismatch")
    if bundle.measurement.measurement_key_count != EXPECTED_MEASUREMENT_KEYS:
        raise ClosureError("MeasurementSetIncomplete")
    if bundle.measurement.provenance_qualification != QUALIFIED:
        raise ClosureError("MeasurementProvenanceNotQualified")
    measurement_set = _digest32(bundle.measurement.measurement_set_commitment, "measurement_set_commitment")

    if bundle.operator is None:
        raise ClosureError("MissingOperatorRobustnessReceipt")
    _same_campaign(campaign, bundle.operator.campaign_subject, "operator")
    _digest32(bundle.operator.receipt_digest, "operator.receipt_digest")
    if bundle.operator.disposition not in {
        "OperatorRobustWithinDeclaredDomain",
        "OperatorSensitivityDetected",
        "Inconclusive",
    }:
        raise ClosureError("UnknownOperatorDisposition")

    if bundle.historical is None:
        raise ClosureError("MissingHistoricalFidelityLedger")
    _same_campaign(campaign, bundle.historical.campaign_subject, "historical")
    _digest32(bundle.historical.ledger_digest, "historical.ledger_digest")
    if bundle.historical.claim_ceiling != "ExactVolumeUnderDeclaredConventionsOnly":
        raise ClosureError("HistoricalClaimCeilingMismatch")

    _same_campaign(campaign, bundle.analysis.campaign_subject, "analysis")
    if bundle.analysis.configuration_set_commitment != config_set:
        raise ClosureError("AnalysisConfigurationSetMismatch")
    if bundle.analysis.measurement_set_commitment != measurement_set:
        raise ClosureError("AnalysisMeasurementSetMismatch")
    if not bundle.analysis.target_isolated:
        raise ClosureError("AnalysisNotTargetIsolated")
    if bundle.analysis.benchmark_digest is not None:
        raise ClosureError("BenchmarkLeakageIntoAnalysis")
    for value, name in (
        (bundle.analysis.analysis_subject_digest, "analysis_subject_digest"),
        (bundle.analysis.sealed_result_digest, "sealed_result_digest"),
    ):
        _digest32(value, name)

    _same_campaign(campaign, bundle.systematic.campaign_subject, "systematic")
    if bundle.systematic.sealed_result_digest != bundle.analysis.sealed_result_digest:
        raise ClosureError("SystematicLedgerWrongSealedAnalysis")
    if bundle.systematic.operator_receipt_digest != bundle.operator.receipt_digest:
        raise ClosureError("SystematicLedgerWrongOperatorReceipt")
    if bundle.systematic.historical_ledger_digest != bundle.historical.ledger_digest:
        raise ClosureError("SystematicLedgerWrongHistoricalLedger")
    _digest32(bundle.systematic.ledger_digest, "systematic.ledger_digest")

    if bundle.operator.disposition != "OperatorRobustWithinDeclaredDomain" or bundle.operator.unresolved_systematic:
        claim_ceiling = "InconclusiveOnly"
    else:
        claim_ceiling = "ExactVolumeUnderDeclaredConventionsOnly"

    out = bytearray(CLOSURE_TAG)
    out += campaign
    out += expected_predecessor
    for receipt in (bundle.numerical, bundle.rng_restart, bundle.checkpoint, bundle.config_measurement):
        out += _lp(receipt.component)
        out += _lp(receipt.state)
        out += receipt.subject_digest
    out += _u32(bundle.execution.retained_configuration_count)
    out += config_set
    out += bundle.admission.receipt_digest
    out += _u32(bundle.measurement.measurement_key_count)
    out += measurement_set
    out += bundle.operator.receipt_digest
    out += bundle.historical.ledger_digest
    out += bundle.analysis.analysis_subject_digest
    out += bundle.analysis.sealed_result_digest
    out += bundle.systematic.ledger_digest
    out += _lp(claim_ceiling)
    return hashlib.sha256(bytes(out)).digest(), claim_ceiling

def comparison_closure(
    evidence_digest: bytes,
    sealed_result_digest: bytes,
    systematic_ledger_digest: bytes,
    benchmark_digest: bytes,
    claim_ceiling: str,
) -> bytes:
    for value, name in (
        (evidence_digest, "evidence_digest"),
        (sealed_result_digest, "sealed_result_digest"),
        (systematic_ledger_digest, "systematic_ledger_digest"),
        (benchmark_digest, "benchmark_digest"),
    ):
        _digest32(value, name)
    out = (
        COMPARISON_TAG
        + evidence_digest
        + sealed_result_digest
        + systematic_ledger_digest
        + benchmark_digest
        + _lp(claim_ceiling)
    )
    return hashlib.sha256(out).digest()

def make_bundle(label: str, operator_disposition: str = "OperatorRobustWithinDeclaredDomain", unresolved: bool = False) -> EvidenceBundle:
    campaign = h(f"{label}:final-campaign")
    predecessor = h(f"{label}:pilot-completed")
    config_set = h(f"{label}:config-set-4000")
    measurement_set = h(f"{label}:measurement-set-768000")
    operator_receipt = h(f"{label}:operator-receipt:{operator_disposition}:{unresolved}")
    historical_ledger = h(f"{label}:historical-ledger")
    sealed_result = h(f"{label}:sealed-result")
    systematic_ledger = h(f"{label}:systematic-ledger:{operator_receipt.hex()}")
    return EvidenceBundle(
        final_campaign_subject=campaign,
        expected_final_predecessor=predecessor,
        numerical=ProductionQualification("numerical-profile", campaign, QUALIFIED, h(f"{label}:numerical")),
        rng_restart=ProductionQualification("rng-restart", campaign, QUALIFIED, h(f"{label}:rng")),
        checkpoint=ProductionQualification("checkpoint-authority", campaign, QUALIFIED, h(f"{label}:checkpoint")),
        config_measurement=ProductionQualification("config-measurement-provenance", campaign, QUALIFIED, h(f"{label}:config-measurement")),
        execution=FinalExecutionReceipt(campaign, predecessor, EXPECTED_CONFIGS, config_set),
        admission=AdmissionReceipt(campaign, config_set, "Admissible", h(f"{label}:admission")),
        measurement=MeasurementReceipt(
            campaign, config_set, EXPECTED_MEASUREMENT_KEYS, measurement_set, QUALIFIED
        ),
        operator=OperatorReceipt(campaign, operator_disposition, operator_receipt, unresolved),
        historical=HistoricalFidelityReceipt(
            campaign, historical_ledger, "ExactVolumeUnderDeclaredConventionsOnly"
        ),
        analysis=SealedAnalysisReceipt(
            campaign,
            config_set,
            measurement_set,
            h(f"{label}:analysis-subject"),
            sealed_result,
            True,
            None,
        ),
        systematic=SystematicLedgerReceipt(
            campaign,
            sealed_result,
            operator_receipt,
            historical_ledger,
            systematic_ledger,
        ),
    )

def expect_error(name: str, bundle: EvidenceBundle, prefix: str, results: dict[str, str]) -> None:
    try:
        evidence_closure(bundle)
    except ClosureError as exc:
        value = str(exc)
        if not value.startswith(prefix):
            raise AssertionError(f"{name}: expected {prefix}, got {value}")
        results[name] = value
    else:
        raise AssertionError(f"{name}: unexpectedly accepted")

def main() -> int:
    a = make_bundle("campaign-A")
    b = make_bundle("campaign-B")

    evidence_digest, ceiling = evidence_closure(a)
    assert ceiling == "ExactVolumeUnderDeclaredConventionsOnly"
    benchmark_a = h("EHK-beta6-benchmark-v1")
    comparison_a = comparison_closure(
        evidence_digest,
        a.analysis.sealed_result_digest,
        a.systematic.ledger_digest,
        benchmark_a,
        ceiling,
    )
    benchmark_b = h("EHK-beta6-benchmark-mutated")
    comparison_b = comparison_closure(
        evidence_digest,
        a.analysis.sealed_result_digest,
        a.systematic.ledger_digest,
        benchmark_b,
        ceiling,
    )
    assert comparison_a != comparison_b
    assert evidence_closure(a)[0] == evidence_digest

    negative = {}
    expect_error(
        "campaign_splice",
        replace(a, checkpoint=replace(b.checkpoint, component="checkpoint-authority")),
        "CampaignSplice:",
        negative,
    )
    expect_error(
        "source_reviewed_checkpoint",
        replace(a, checkpoint=replace(a.checkpoint, state=SOURCE_REVIEWED)),
        "ImplementationNotQualified:",
        negative,
    )
    expect_error(
        "inadmissible_sample",
        replace(a, admission=replace(a.admission, disposition="NotAdmissible")),
        "FinalSampleNotAdmissible:",
        negative,
    )
    expect_error(
        "retained_3999",
        replace(a, execution=replace(a.execution, retained_configuration_count=3999)),
        "RetainedConfigurationCountMismatch",
        negative,
    )
    expect_error(
        "measurement_767999",
        replace(a, measurement=replace(a.measurement, measurement_key_count=EXPECTED_MEASUREMENT_KEYS - 1)),
        "MeasurementSetIncomplete",
        negative,
    )
    expect_error(
        "benchmark_leakage",
        replace(a, analysis=replace(a.analysis, benchmark_digest=benchmark_a)),
        "BenchmarkLeakageIntoAnalysis",
        negative,
    )
    expect_error(
        "stale_final_predecessor",
        replace(a, execution=replace(a.execution, final_campaign_predecessor=h("stale"))),
        "FinalCampaignPredecessorMismatch",
        negative,
    )
    expect_error(
        "missing_historical_ledger",
        replace(a, historical=None),
        "MissingHistoricalFidelityLedger",
        negative,
    )
    expect_error(
        "missing_operator_receipt",
        replace(a, operator=None),
        "MissingOperatorRobustnessReceipt",
        negative,
    )
    expect_error(
        "systematic_wrong_analysis",
        replace(a, systematic=replace(a.systematic, sealed_result_digest=h("wrong-sealed-result"))),
        "SystematicLedgerWrongSealedAnalysis",
        negative,
    )

    sensitive = make_bundle("campaign-S", "OperatorSensitivityDetected", True)
    sensitive_digest, sensitive_ceiling = evidence_closure(sensitive)
    assert sensitive_ceiling == "InconclusiveOnly"
    sensitive_comparison = comparison_closure(
        sensitive_digest,
        sensitive.analysis.sealed_result_digest,
        sensitive.systematic.ledger_digest,
        benchmark_a,
        sensitive_ceiling,
    )

    unresolved = make_bundle("campaign-U", "Inconclusive", True)
    _, unresolved_ceiling = evidence_closure(unresolved)
    assert unresolved_ceiling == "InconclusiveOnly"

    # Model current real-program authority status conservatively. The actual
    # Rust checkpoint candidate #2952 is source-reviewed, not executed-qualified.
    real_beta6_comparison_capable = False
    current_blockers = {
        "checkpoint_authority": SOURCE_REVIEWED,
        "rust_execution_evidence": "QueuedOrAbsent",
        "real_final_sample_admission": "NotExecuted",
        "real_measurement_set": "NotProduced",
        "real_sealed_analysis": "NotProduced",
    }

    result = {
        "oracle_id": ORACLE_ID,
        "expected_final_configs": EXPECTED_CONFIGS,
        "measurement_keys_per_config": EXPECTED_KEYS_PER_CONFIG,
        "expected_measurement_keys": EXPECTED_MEASUREMENT_KEYS,
        "valid_fixture": {
            "evidence_closure_sha256": evidence_digest.hex(),
            "claim_ceiling": ceiling,
            "comparison_closure_sha256": comparison_a.hex(),
        },
        "benchmark_mutation": {
            "upstream_evidence_unchanged": evidence_closure(a)[0] == evidence_digest,
            "comparison_digest_changed": comparison_a != comparison_b,
            "mutated_comparison_sha256": comparison_b.hex(),
        },
        "operator_sensitivity_fixture": {
            "claim_ceiling": sensitive_ceiling,
            "comparison_closure_sha256": sensitive_comparison.hex(),
        },
        "operator_inconclusive_fixture": {
            "claim_ceiling": unresolved_ceiling,
        },
        "negative_controls": negative,
        "real_beta6_comparison_capable": real_beta6_comparison_capable,
        "current_real_program_blockers": current_blockers,
        "scientific_boundary": (
            "cross_receipt_composition_semantics_only_no_real_beta6_authorization_"
            "no_equilibration_no_ehk_agreement"
        ),
    }
    canonical = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result_sha = hashlib.sha256(canonical).hexdigest()
    print("ok")
    print(f"result_sha256={result_sha}")
    print(canonical.decode())
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
