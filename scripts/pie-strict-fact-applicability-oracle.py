#!/usr/bin/env python3
"""PIE-002J strict evidence-bound fact applicability oracle.

Composes the qualified PIE-002I per-evidence resolver rather than copying it.
V1 is deliberately conservative: every evidence item attached to one fact must
resolve Applicable under one exact study/profile before the fact is Applicable.
"""

from __future__ import annotations

import argparse
import importlib.util
import inspect
import math
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


def _load_pie_002i():
    path = Path(__file__).with_name("pie-evidence-applicability-oracle.py")
    spec = importlib.util.spec_from_file_location("pie_002i_qualified_oracle", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load qualified PIE-002I oracle")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PIE_I = _load_pie_002i()

ApplicabilityProfile = PIE_I.ApplicabilityProfile
ApplicabilityStatus = PIE_I.ApplicabilityStatus
EvidenceClass = PIE_I.EvidenceClass
EvidenceScope = PIE_I.EvidenceScope
ScopeClaim = PIE_I.ScopeClaim
ScopedEvidence = PIE_I.ScopedEvidence
UtilityStudyScope = PIE_I.UtilityStudyScope
resolve_evidence_applicability = PIE_I.resolve_evidence_applicability

SITE = PIE_I.SITE
STORAGE_CONFIGURATION = PIE_I.STORAGE_CONFIGURATION


class FactApplicabilityStatus(Enum):
    APPLICABLE = "applicable"
    INAPPLICABLE = "inapplicable"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class AttachedEvidence:
    evidence_id: str
    evidence_class: EvidenceClass

    def validate(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("attached evidence_id required")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("attached evidence_class invalid")


@dataclass(frozen=True)
class EvidenceBoundNumericalFact:
    fact_key: str
    minimum: float
    maximum: float
    evidence: tuple[AttachedEvidence, ...]

    def validate(self) -> None:
        if not isinstance(self.fact_key, str) or not self.fact_key:
            raise ValueError("fact_key required")
        if not math.isfinite(self.minimum) or not math.isfinite(self.maximum):
            raise ValueError("fact range must be finite")
        if self.minimum > self.maximum:
            raise ValueError("fact range must be ordered")
        if not self.evidence:
            raise ValueError("fact requires attached evidence")
        seen: set[str] = set()
        for item in self.evidence:
            item.validate()
            if item.evidence_id in seen:
                raise ValueError("duplicate attached evidence_id")
            seen.add(item.evidence_id)


@dataclass(frozen=True)
class EvidenceScopeBinding:
    evidence_id: str
    scope: EvidenceScope

    def validate(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("scope binding evidence_id required")
        self.scope.validate()


@dataclass(frozen=True)
class FactApplicabilityReceipt:
    fact_key: str
    minimum: float
    maximum: float
    evidence: tuple[AttachedEvidence, ...]
    evidence_scopes: tuple[EvidenceScopeBinding, ...]
    study_scope: UtilityStudyScope
    applicability_profile: ApplicabilityProfile
    per_evidence: tuple[object, ...]
    status: FactApplicabilityStatus


def _canonical_scope(scope: EvidenceScope) -> EvidenceScope:
    scope.validate()
    return EvidenceScope(tuple(sorted(scope.claims, key=lambda item: item[0])))


def _canonical_study(study: UtilityStudyScope) -> UtilityStudyScope:
    study.validate()
    return UtilityStudyScope(tuple(sorted(study.values, key=lambda item: item[0])))


def _canonical_profile(profile: ApplicabilityProfile) -> ApplicabilityProfile:
    profile.validate()
    return ApplicabilityProfile(tuple(sorted(profile.material_dimensions)))


def resolve_fact_applicability(
    fact: EvidenceBoundNumericalFact,
    evidence_scopes: tuple[EvidenceScopeBinding, ...],
    study: UtilityStudyScope,
    profile: ApplicabilityProfile,
) -> FactApplicabilityReceipt:
    """Resolve one exact fact by recomputing PIE-002I for all attached evidence."""

    fact.validate()
    canonical_study = _canonical_study(study)
    canonical_profile = _canonical_profile(profile)

    bindings: dict[str, EvidenceScope] = {}
    for binding in evidence_scopes:
        binding.validate()
        if binding.evidence_id in bindings:
            raise ValueError("duplicate evidence scope binding")
        bindings[binding.evidence_id] = _canonical_scope(binding.scope)

    attached_ids = {item.evidence_id for item in fact.evidence}
    bound_ids = set(bindings)
    if attached_ids != bound_ids:
        missing = sorted(attached_ids - bound_ids)
        extra = sorted(bound_ids - attached_ids)
        raise ValueError(f"evidence scope set mismatch: missing={missing}, extra={extra}")

    canonical_evidence = tuple(sorted(fact.evidence, key=lambda item: item.evidence_id))
    canonical_bindings = tuple(
        EvidenceScopeBinding(item.evidence_id, bindings[item.evidence_id])
        for item in canonical_evidence
    )

    per_evidence = tuple(
        resolve_evidence_applicability(
            canonical_study,
            ScopedEvidence(
                evidence_id=item.evidence_id,
                evidence_class=item.evidence_class,
                fact_key=fact.fact_key,
                scope=bindings[item.evidence_id],
            ),
            canonical_profile,
        )
        for item in canonical_evidence
    )

    statuses = {receipt.status for receipt in per_evidence}
    if ApplicabilityStatus.INAPPLICABLE in statuses:
        status = FactApplicabilityStatus.INAPPLICABLE
    elif ApplicabilityStatus.INDETERMINATE in statuses:
        status = FactApplicabilityStatus.INDETERMINATE
    else:
        status = FactApplicabilityStatus.APPLICABLE

    return FactApplicabilityReceipt(
        fact_key=fact.fact_key,
        minimum=fact.minimum,
        maximum=fact.maximum,
        evidence=canonical_evidence,
        evidence_scopes=canonical_bindings,
        study_scope=canonical_study,
        applicability_profile=canonical_profile,
        per_evidence=per_evidence,
        status=status,
    )


def _scope(**claims: ScopeClaim) -> EvidenceScope:
    return EvidenceScope(tuple(claims.items()))


def _study(**values: str) -> UtilityStudyScope:
    return UtilityStudyScope(tuple(values.items()))


def _must_fail(callable_, label: str) -> None:
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError(label)


def self_test() -> None:
    site_a = _study(site="site-a", storage_configuration="storage-a")
    profile = ApplicabilityProfile((SITE, STORAGE_CONFIGURATION))

    a = AttachedEvidence("a", EvidenceClass.LAB_MEASURED)
    b = AttachedEvidence("b", EvidenceClass.LITERATURE_MODEL)
    h = AttachedEvidence("h", EvidenceClass.HYPOTHESIS)

    fact_one = EvidenceBoundNumericalFact("available_peak_power_w", 30.0, 35.0, (a,))
    scopes_one = (
        EvidenceScopeBinding(
            "a",
            _scope(
                site=ScopeClaim.exact("site-a"),
                storage_configuration=ScopeClaim.exact("storage-a"),
            ),
        ),
    )
    receipt = resolve_fact_applicability(fact_one, scopes_one, site_a, profile)
    assert receipt.status is FactApplicabilityStatus.APPLICABLE
    assert receipt.minimum == 30.0 and receipt.maximum == 35.0

    fact_two = EvidenceBoundNumericalFact(
        "available_peak_power_w", 30.0, 35.0, (b, a)
    )
    scopes_two = (
        EvidenceScopeBinding(
            "b",
            _scope(
                site=ScopeClaim.exact("site-a"),
                storage_configuration=ScopeClaim.exact("storage-a"),
            ),
        ),
        scopes_one[0],
    )
    all_ok = resolve_fact_applicability(fact_two, scopes_two, site_a, profile)
    assert all_ok.status is FactApplicabilityStatus.APPLICABLE
    assert tuple(item.evidence_id for item in all_ok.evidence) == ("a", "b")

    indeterminate_scopes = (
        scopes_one[0],
        EvidenceScopeBinding("b", _scope(site=ScopeClaim.exact("site-a"))),
    )
    indeterminate = resolve_fact_applicability(
        fact_two, indeterminate_scopes, site_a, profile
    )
    assert indeterminate.status is FactApplicabilityStatus.INDETERMINATE

    mismatched_scopes = (
        scopes_one[0],
        EvidenceScopeBinding(
            "b",
            _scope(
                site=ScopeClaim.exact("site-a"),
                storage_configuration=ScopeClaim.exact("storage-b"),
            ),
        ),
    )
    inapplicable = resolve_fact_applicability(
        fact_two, mismatched_scopes, site_a, profile
    )
    assert inapplicable.status is FactApplicabilityStatus.INAPPLICABLE

    both_bad = (
        EvidenceScopeBinding("a", _scope(site=ScopeClaim.exact("site-a"))),
        EvidenceScopeBinding(
            "b",
            _scope(
                site=ScopeClaim.exact("site-a"),
                storage_configuration=ScopeClaim.exact("storage-b"),
            ),
        ),
    )
    assert (
        resolve_fact_applicability(fact_two, both_bad, site_a, profile).status
        is FactApplicabilityStatus.INAPPLICABLE
    )

    _must_fail(
        lambda: resolve_fact_applicability(fact_two, scopes_one, site_a, profile),
        "missing evidence scope must fail",
    )
    extra = scopes_two + (
        EvidenceScopeBinding("extra", _scope(site=ScopeClaim.general())),
    )
    _must_fail(
        lambda: resolve_fact_applicability(fact_two, extra, site_a, profile),
        "extra evidence scope must fail",
    )
    duplicated_fact = EvidenceBoundNumericalFact(
        "available_peak_power_w", 30.0, 35.0, (a, a)
    )
    _must_fail(
        lambda: resolve_fact_applicability(duplicated_fact, scopes_one, site_a, profile),
        "duplicate attached evidence must fail",
    )
    duplicate_binding = scopes_one + scopes_one
    _must_fail(
        lambda: resolve_fact_applicability(fact_one, duplicate_binding, site_a, profile),
        "duplicate scope binding must fail",
    )

    # Set-like input ordering cannot change the canonical receipt.
    reversed_fact = EvidenceBoundNumericalFact(
        "available_peak_power_w", 30.0, 35.0, (a, b)
    )
    reversed_scopes = tuple(reversed(scopes_two))
    reversed_study = UtilityStudyScope(tuple(reversed(site_a.values)))
    reversed_profile = ApplicabilityProfile(tuple(reversed(profile.material_dimensions)))
    assert resolve_fact_applicability(
        reversed_fact, reversed_scopes, reversed_study, reversed_profile
    ) == all_ok

    # Same fact/evidence is recomputed for a changed study scope.
    site_b = _study(site="site-b", storage_configuration="storage-a")
    assert (
        resolve_fact_applicability(fact_one, scopes_one, site_b, profile).status
        is FactApplicabilityStatus.INAPPLICABLE
    )

    # A weaker profile can change the result, so verdicts must remain profile-bound.
    mismatch_scope = (
        EvidenceScopeBinding(
            "a",
            _scope(
                site=ScopeClaim.exact("site-a"),
                storage_configuration=ScopeClaim.exact("storage-b"),
            ),
        ),
    )
    assert (
        resolve_fact_applicability(fact_one, mismatch_scope, site_a, profile).status
        is FactApplicabilityStatus.INAPPLICABLE
    )
    site_only = ApplicabilityProfile((SITE,))
    assert (
        resolve_fact_applicability(fact_one, mismatch_scope, site_a, site_only).status
        is FactApplicabilityStatus.APPLICABLE
    )

    # Evidence class is preserved and cannot override scope resolution.
    hypothesis_fact = EvidenceBoundNumericalFact(
        "available_energy_capacity_j", 100.0, 120.0, (h,)
    )
    hypothesis_scope = (
        EvidenceScopeBinding(
            "h",
            _scope(
                site=ScopeClaim.exact("site-a"),
                storage_configuration=ScopeClaim.exact("storage-a"),
            ),
        ),
    )
    hypothesis_receipt = resolve_fact_applicability(
        hypothesis_fact, hypothesis_scope, site_a, profile
    )
    assert hypothesis_receipt.status is FactApplicabilityStatus.APPLICABLE
    assert hypothesis_receipt.per_evidence[0].evidence_class is EvidenceClass.HYPOTHESIS

    # Changing the numerical fact necessarily changes the returned receipt.
    changed_fact = EvidenceBoundNumericalFact(
        "available_peak_power_w", 31.0, 36.0, (a,)
    )
    changed_receipt = resolve_fact_applicability(changed_fact, scopes_one, site_a, profile)
    assert changed_receipt != receipt
    assert (changed_receipt.minimum, changed_receipt.maximum) == (31.0, 36.0)

    # Preferred API has no detached applicability-receipt argument.
    params = tuple(inspect.signature(resolve_fact_applicability).parameters)
    assert params == ("fact", "evidence_scopes", "study", "profile")

    print("ok")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        parser.error("only --self-test is supported")


if __name__ == "__main__":
    main()
