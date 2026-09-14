#!/usr/bin/env python3
"""Independent PIE-002I evidence-to-study applicability oracle.

Resolves declared evidence scope against an exact study scope one evidence item
at a time. Applicability != truth, freshness/currentness, independence,
calibration validity, numerical validity, feasibility, dispatch, or authority.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from enum import Enum, IntEnum


PROCESS_CONTENT = "process_content_sha256"
BODY = "body"
SITE = "site"
PLANT_CONFIGURATION = "plant_configuration"
SUPPLY_CONFIGURATION = "supply_configuration"
STORAGE_CONFIGURATION = "storage_configuration"
RECOVERY_CONFIGURATION = "recovery_configuration"
OPERATING_MODE = "operating_mode"
ENVIRONMENT_PROFILE = "environment_profile"
TEMPORAL_BASIS = "temporal_basis"

KNOWN_DIMENSIONS = frozenset(
    {
        PROCESS_CONTENT,
        BODY,
        SITE,
        PLANT_CONFIGURATION,
        SUPPLY_CONFIGURATION,
        STORAGE_CONFIGURATION,
        RECOVERY_CONFIGURATION,
        OPERATING_MODE,
        ENVIRONMENT_PROFILE,
        TEMPORAL_BASIS,
    }
)


class EvidenceClass(IntEnum):
    HYPOTHESIS = 0
    LITERATURE_MODEL = 1
    VENDOR_PROJECTION = 2
    LAB_MEASURED = 3
    RELEVANT_ENVIRONMENT_MEASURED = 4
    INTEGRATED_DEMONSTRATION = 5
    QUALIFIED = 6


class ClaimKind(Enum):
    EXACT = "exact"
    GENERAL = "general"
    UNKNOWN = "unknown"


class ApplicabilityStatus(Enum):
    APPLICABLE = "applicable"
    INAPPLICABLE = "inapplicable"
    INDETERMINATE = "indeterminate"


@dataclass(frozen=True)
class ScopeClaim:
    kind: ClaimKind
    value: str | None = None

    def validate(self, dimension: str) -> None:
        if self.kind is ClaimKind.EXACT:
            if not isinstance(self.value, str) or not self.value:
                raise ValueError(f"{dimension}: exact scope requires a value")
            if dimension == PROCESS_CONTENT and not re.fullmatch(r"[0-9a-f]{64}", self.value):
                raise ValueError("process content scope must be a lowercase SHA-256 hex digest")
        elif self.value is not None:
            raise ValueError(f"{dimension}: {self.kind.value} scope cannot carry a value")

    @staticmethod
    def exact(value: str) -> "ScopeClaim":
        return ScopeClaim(ClaimKind.EXACT, value)

    @staticmethod
    def general() -> "ScopeClaim":
        return ScopeClaim(ClaimKind.GENERAL)

    @staticmethod
    def unknown() -> "ScopeClaim":
        return ScopeClaim(ClaimKind.UNKNOWN)


@dataclass(frozen=True)
class UtilityStudyScope:
    values: tuple[tuple[str, str], ...]

    def validate(self) -> None:
        seen: set[str] = set()
        for dimension, value in self.values:
            if dimension not in KNOWN_DIMENSIONS:
                raise ValueError(f"unknown study dimension: {dimension}")
            if dimension in seen:
                raise ValueError(f"duplicate study dimension: {dimension}")
            seen.add(dimension)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{dimension}: study value required")
            if dimension == PROCESS_CONTENT and not re.fullmatch(r"[0-9a-f]{64}", value):
                raise ValueError("study process content must be a lowercase SHA-256 hex digest")

    def value_for(self, dimension: str) -> str | None:
        for name, value in self.values:
            if name == dimension:
                return value
        return None


@dataclass(frozen=True)
class EvidenceScope:
    claims: tuple[tuple[str, ScopeClaim], ...]

    def validate(self) -> None:
        seen: set[str] = set()
        for dimension, claim in self.claims:
            if dimension not in KNOWN_DIMENSIONS:
                raise ValueError(f"unknown evidence dimension: {dimension}")
            if dimension in seen:
                raise ValueError(f"duplicate evidence dimension: {dimension}")
            seen.add(dimension)
            claim.validate(dimension)

    def claim_for(self, dimension: str) -> ScopeClaim:
        for name, claim in self.claims:
            if name == dimension:
                return claim
        # Omission is explicitly unknown, never a wildcard.
        return ScopeClaim.unknown()


@dataclass(frozen=True)
class ApplicabilityProfile:
    material_dimensions: tuple[str, ...]

    def validate(self) -> None:
        if not self.material_dimensions:
            raise ValueError("applicability profile requires at least one material dimension")
        if len(set(self.material_dimensions)) != len(self.material_dimensions):
            raise ValueError("duplicate material applicability dimension")
        unknown = set(self.material_dimensions) - KNOWN_DIMENSIONS
        if unknown:
            raise ValueError(f"unknown material dimensions: {sorted(unknown)}")


@dataclass(frozen=True)
class ScopedEvidence:
    evidence_id: str
    evidence_class: EvidenceClass
    fact_key: str
    scope: EvidenceScope

    def validate(self) -> None:
        if not isinstance(self.evidence_id, str) or not self.evidence_id:
            raise ValueError("evidence_id required")
        if not isinstance(self.evidence_class, EvidenceClass):
            raise ValueError("unknown evidence class")
        if not isinstance(self.fact_key, str) or not self.fact_key:
            raise ValueError("fact_key required")
        self.scope.validate()


@dataclass(frozen=True)
class EvidenceApplicabilityReceipt:
    evidence_id: str
    evidence_class: EvidenceClass
    fact_key: str
    status: ApplicabilityStatus
    exact_matches: tuple[str, ...]
    explicit_generalizations: tuple[str, ...]
    mismatches: tuple[str, ...]
    unresolved_dimensions: tuple[str, ...]


def resolve_evidence_applicability(
    study: UtilityStudyScope,
    evidence: ScopedEvidence,
    profile: ApplicabilityProfile,
) -> EvidenceApplicabilityReceipt:
    study.validate()
    evidence.validate()
    profile.validate()

    matched: list[str] = []
    generalized: list[str] = []
    mismatched: list[str] = []
    unresolved: list[str] = []

    for dimension in profile.material_dimensions:
        study_value = study.value_for(dimension)
        claim = evidence.scope.claim_for(dimension)

        if study_value is None:
            unresolved.append(dimension)
            continue
        if claim.kind is ClaimKind.UNKNOWN:
            unresolved.append(dimension)
            continue
        if claim.kind is ClaimKind.GENERAL:
            generalized.append(dimension)
            continue
        assert claim.value is not None
        if claim.value == study_value:
            matched.append(dimension)
        else:
            mismatched.append(dimension)

    if mismatched:
        status = ApplicabilityStatus.INAPPLICABLE
    elif unresolved:
        status = ApplicabilityStatus.INDETERMINATE
    else:
        status = ApplicabilityStatus.APPLICABLE

    return EvidenceApplicabilityReceipt(
        evidence_id=evidence.evidence_id,
        evidence_class=evidence.evidence_class,
        fact_key=evidence.fact_key,
        status=status,
        exact_matches=tuple(matched),
        explicit_generalizations=tuple(generalized),
        mismatches=tuple(mismatched),
        unresolved_dimensions=tuple(unresolved),
    )


def resolve_field_evidence(
    study: UtilityStudyScope,
    evidence_items: tuple[ScopedEvidence, ...],
    profile: ApplicabilityProfile,
) -> tuple[EvidenceApplicabilityReceipt, ...]:
    if not evidence_items:
        raise ValueError("field requires at least one evidence item")
    return tuple(resolve_evidence_applicability(study, item, profile) for item in evidence_items)


def _scope(**kwargs: ScopeClaim) -> EvidenceScope:
    return EvidenceScope(tuple(kwargs.items()))


def _study(**kwargs: str) -> UtilityStudyScope:
    return UtilityStudyScope(tuple(kwargs.items()))


def _must_fail(callable_, label: str) -> None:
    try:
        callable_()
    except ValueError:
        return
    raise AssertionError(label)


def self_test() -> None:
    process_a = "a" * 64
    process_b = "b" * 64

    study = _study(
        process_content_sha256=process_a,
        body="Moon",
        site="shackleton-rim-a",
        plant_configuration="plant-v3",
        supply_configuration="solar-nuclear-v2",
        storage_configuration="storage-v4",
        recovery_configuration="dc-recovery-v2",
        operating_mode="nominal-day",
        environment_profile="lunar-polar-v1",
        temporal_basis="study-epoch-2032",
    )
    process_profile = ApplicabilityProfile(
        (
            PROCESS_CONTENT,
            BODY,
            SITE,
            PLANT_CONFIGURATION,
            STORAGE_CONFIGURATION,
            OPERATING_MODE,
            ENVIRONMENT_PROFILE,
            TEMPORAL_BASIS,
        )
    )

    exact = ScopedEvidence(
        "ev-exact",
        EvidenceClass.LAB_MEASURED,
        "available_peak_power_w",
        _scope(
            process_content_sha256=ScopeClaim.exact(process_a),
            body=ScopeClaim.exact("Moon"),
            site=ScopeClaim.exact("shackleton-rim-a"),
            plant_configuration=ScopeClaim.exact("plant-v3"),
            storage_configuration=ScopeClaim.exact("storage-v4"),
            operating_mode=ScopeClaim.exact("nominal-day"),
            environment_profile=ScopeClaim.exact("lunar-polar-v1"),
            temporal_basis=ScopeClaim.exact("study-epoch-2032"),
        ),
    )
    receipt = resolve_evidence_applicability(study, exact, process_profile)
    assert receipt.status is ApplicabilityStatus.APPLICABLE
    assert set(receipt.exact_matches) == set(process_profile.material_dimensions)

    changed_process = ScopedEvidence(
        "ev-stale-process",
        EvidenceClass.QUALIFIED,
        "available_peak_power_w",
        _scope(
            process_content_sha256=ScopeClaim.exact(process_b),
            body=ScopeClaim.exact("Moon"),
            site=ScopeClaim.exact("shackleton-rim-a"),
            plant_configuration=ScopeClaim.exact("plant-v3"),
            storage_configuration=ScopeClaim.exact("storage-v4"),
            operating_mode=ScopeClaim.exact("nominal-day"),
            environment_profile=ScopeClaim.exact("lunar-polar-v1"),
            temporal_basis=ScopeClaim.exact("study-epoch-2032"),
        ),
    )
    stale = resolve_evidence_applicability(study, changed_process, process_profile)
    assert stale.status is ApplicabilityStatus.INAPPLICABLE
    assert stale.mismatches == (PROCESS_CONTENT,)

    different_site = ScopedEvidence(
        "ev-site",
        EvidenceClass.RELEVANT_ENVIRONMENT_MEASURED,
        "storage_acceptance_j",
        _scope(
            process_content_sha256=ScopeClaim.exact(process_a),
            body=ScopeClaim.exact("Moon"),
            site=ScopeClaim.exact("malapert-a"),
            plant_configuration=ScopeClaim.exact("plant-v3"),
            storage_configuration=ScopeClaim.exact("storage-v4"),
            operating_mode=ScopeClaim.exact("nominal-day"),
            environment_profile=ScopeClaim.exact("lunar-polar-v1"),
            temporal_basis=ScopeClaim.exact("study-epoch-2032"),
        ),
    )
    site_result = resolve_evidence_applicability(study, different_site, process_profile)
    assert site_result.status is ApplicabilityStatus.INAPPLICABLE
    assert SITE in site_result.mismatches

    different_storage = ScopedEvidence(
        "ev-storage",
        EvidenceClass.QUALIFIED,
        "available_energy_capacity_j",
        _scope(
            process_content_sha256=ScopeClaim.exact(process_a),
            body=ScopeClaim.exact("Moon"),
            site=ScopeClaim.exact("shackleton-rim-a"),
            plant_configuration=ScopeClaim.exact("plant-v3"),
            storage_configuration=ScopeClaim.exact("storage-v5"),
            operating_mode=ScopeClaim.exact("nominal-day"),
            environment_profile=ScopeClaim.exact("lunar-polar-v1"),
            temporal_basis=ScopeClaim.exact("study-epoch-2032"),
        ),
    )
    storage_result = resolve_evidence_applicability(study, different_storage, process_profile)
    assert storage_result.status is ApplicabilityStatus.INAPPLICABLE
    assert STORAGE_CONFIGURATION in storage_result.mismatches

    missing_site = ScopedEvidence(
        "ev-missing",
        EvidenceClass.QUALIFIED,
        "available_sustained_power_w",
        _scope(
            process_content_sha256=ScopeClaim.exact(process_a),
            body=ScopeClaim.exact("Moon"),
            plant_configuration=ScopeClaim.exact("plant-v3"),
            storage_configuration=ScopeClaim.exact("storage-v4"),
            operating_mode=ScopeClaim.exact("nominal-day"),
            environment_profile=ScopeClaim.exact("lunar-polar-v1"),
            temporal_basis=ScopeClaim.exact("study-epoch-2032"),
        ),
    )
    missing_result = resolve_evidence_applicability(study, missing_site, process_profile)
    assert missing_result.status is ApplicabilityStatus.INDETERMINATE
    assert missing_result.unresolved_dimensions == (SITE,)

    explicit_general = ScopedEvidence(
        "ev-general",
        EvidenceClass.LITERATURE_MODEL,
        "recovery_delivery_fraction",
        _scope(
            process_content_sha256=ScopeClaim.exact(process_a),
            body=ScopeClaim.general(),
            site=ScopeClaim.general(),
            plant_configuration=ScopeClaim.exact("plant-v3"),
            storage_configuration=ScopeClaim.exact("storage-v4"),
            operating_mode=ScopeClaim.exact("nominal-day"),
            environment_profile=ScopeClaim.exact("lunar-polar-v1"),
            temporal_basis=ScopeClaim.exact("study-epoch-2032"),
        ),
    )
    general_result = resolve_evidence_applicability(study, explicit_general, process_profile)
    assert general_result.status is ApplicabilityStatus.APPLICABLE
    assert general_result.explicit_generalizations == (BODY, SITE)

    omitted_not_wildcard = ScopedEvidence(
        "ev-omitted",
        EvidenceClass.LITERATURE_MODEL,
        "recovery_delivery_fraction",
        _scope(
            process_content_sha256=ScopeClaim.exact(process_a),
            plant_configuration=ScopeClaim.exact("plant-v3"),
            storage_configuration=ScopeClaim.exact("storage-v4"),
            operating_mode=ScopeClaim.exact("nominal-day"),
            environment_profile=ScopeClaim.exact("lunar-polar-v1"),
            temporal_basis=ScopeClaim.exact("study-epoch-2032"),
        ),
    )
    omitted_result = resolve_evidence_applicability(study, omitted_not_wildcard, process_profile)
    assert omitted_result.status is ApplicabilityStatus.INDETERMINATE
    assert set(omitted_result.unresolved_dimensions) == {BODY, SITE}

    qualified_mismatch = resolve_evidence_applicability(study, changed_process, process_profile)
    assert qualified_mismatch.evidence_class is EvidenceClass.QUALIFIED
    assert qualified_mismatch.status is ApplicabilityStatus.INAPPLICABLE

    hypothesis = ScopedEvidence(
        "ev-hyp",
        EvidenceClass.HYPOTHESIS,
        "available_peak_power_w",
        exact.scope,
    )
    hypothesis_result = resolve_evidence_applicability(study, hypothesis, process_profile)
    assert hypothesis_result.status is ApplicabilityStatus.APPLICABLE
    assert hypothesis_result.evidence_class is EvidenceClass.HYPOTHESIS

    # Numerical similarity cannot influence this theorem: there is intentionally no
    # numeric-value argument to the resolver, so scope mismatch remains decisive.
    assert resolve_evidence_applicability(study, different_site, process_profile).status is ApplicabilityStatus.INAPPLICABLE

    shared_good = ScopedEvidence(
        "shared-source",
        EvidenceClass.LITERATURE_MODEL,
        "recoverable_energy_j",
        exact.scope,
    )
    shared_bad = ScopedEvidence(
        "shared-source",
        EvidenceClass.LITERATURE_MODEL,
        "storage_acceptance_j",
        different_site.scope,
    )
    shared_results = resolve_field_evidence(study, (shared_good, shared_bad), process_profile)
    assert shared_results[0].status is ApplicabilityStatus.APPLICABLE
    assert shared_results[1].status is ApplicabilityStatus.INAPPLICABLE
    assert shared_results[0].evidence_id == shared_results[1].evidence_id

    historical = ScopedEvidence(
        "ev-historical",
        EvidenceClass.LAB_MEASURED,
        "storage_charge_power_w",
        _scope(
            process_content_sha256=ScopeClaim.exact(process_a),
            body=ScopeClaim.exact("Moon"),
            site=ScopeClaim.exact("shackleton-rim-a"),
            plant_configuration=ScopeClaim.exact("plant-v3"),
            storage_configuration=ScopeClaim.exact("storage-v4"),
            operating_mode=ScopeClaim.exact("nominal-day"),
            environment_profile=ScopeClaim.exact("lunar-polar-v1"),
            temporal_basis=ScopeClaim.exact("study-epoch-2028"),
        ),
    )
    historical_result = resolve_evidence_applicability(study, historical, process_profile)
    assert historical_result.status is ApplicabilityStatus.INAPPLICABLE
    assert TEMPORAL_BASIS in historical_result.mismatches

    site_profile = ApplicabilityProfile(
        (BODY, SITE, PLANT_CONFIGURATION, SUPPLY_CONFIGURATION, STORAGE_CONFIGURATION)
    )
    site_level = ScopedEvidence(
        "ev-site-infra",
        EvidenceClass.INTEGRATED_DEMONSTRATION,
        "available_peak_power_w",
        _scope(
            body=ScopeClaim.exact("Moon"),
            site=ScopeClaim.exact("shackleton-rim-a"),
            plant_configuration=ScopeClaim.exact("plant-v3"),
            supply_configuration=ScopeClaim.exact("solar-nuclear-v2"),
            storage_configuration=ScopeClaim.exact("storage-v4"),
        ),
    )
    site_result = resolve_evidence_applicability(study, site_level, site_profile)
    assert site_result.status is ApplicabilityStatus.APPLICABLE
    assert PROCESS_CONTENT not in site_profile.material_dimensions

    rebound = resolve_evidence_applicability(study, changed_process, process_profile)
    assert rebound.status is ApplicabilityStatus.INAPPLICABLE

    hard_mismatch_plus_unknown = ScopedEvidence(
        "ev-mismatch-unknown",
        EvidenceClass.QUALIFIED,
        "storage_discharge_power_w",
        _scope(
            process_content_sha256=ScopeClaim.exact(process_b),
            body=ScopeClaim.exact("Moon"),
        ),
    )
    mixed = resolve_evidence_applicability(study, hard_mismatch_plus_unknown, process_profile)
    assert mixed.status is ApplicabilityStatus.INAPPLICABLE
    assert PROCESS_CONTENT in mixed.mismatches
    assert SITE in mixed.unresolved_dimensions

    _must_fail(
        lambda: ApplicabilityProfile((SITE, SITE)).validate(),
        "duplicate material dimension",
    )
    _must_fail(
        lambda: ScopeClaim(ClaimKind.GENERAL, "hidden-value").validate(SITE),
        "general scope cannot carry a hidden value",
    )
    _must_fail(
        lambda: ScopeClaim.exact("not-a-digest").validate(PROCESS_CONTENT),
        "invalid process content digest",
    )

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
