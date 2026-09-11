#!/usr/bin/env python3
"""Independent LL-008 deterministic study-release admission oracle.

Consumes already-computed launcher/pod/corridor/catcher evidence and applies
caller-supplied policy thresholds. It does not compute trajectories or command
hardware. Passing means only "Phase-0 study snapshot satisfies declared gates".
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


ALLOWED_SAFE_MISS_CLASSES = {
    "RecoverableSurfaceMiss",
    "DeclaredDisposalRegion",
    "RecoverableOrbitalMiss",
}


@dataclass(frozen=True)
class Policy:
    policy_id: str
    max_launcher_age_s: float
    max_pod_age_s: float
    max_terrain_age_s: float
    max_target_age_s: float
    max_clock_disagreement_s: float
    max_mass_disagreement_kg: float
    min_lower_bound_terrain_clearance_m: float
    max_protected_zone_risk_upper: float
    max_endpoint_uncertainty_m: float
    require_target: bool
    require_remote_communications: bool

    def validate(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("policy_id must be nonempty")
        nonnegative = [
            self.max_launcher_age_s,
            self.max_pod_age_s,
            self.max_terrain_age_s,
            self.max_target_age_s,
            self.max_clock_disagreement_s,
            self.max_mass_disagreement_kg,
            self.min_lower_bound_terrain_clearance_m,
            self.max_endpoint_uncertainty_m,
        ]
        if not all(math.isfinite(v) and v >= 0.0 for v in nonnegative):
            raise ValueError("policy scalar thresholds must be finite and nonnegative")
        if (
            not math.isfinite(self.max_protected_zone_risk_upper)
            or not (0.0 <= self.max_protected_zone_risk_upper <= 1.0)
        ):
            raise ValueError("protected-zone risk threshold must be in [0,1]")


@dataclass(frozen=True)
class Snapshot:
    evidence_lineage: str
    launcher_state_valid: bool
    launcher_age_s: float
    pod_state_valid: bool
    pod_age_s: float
    declared_pod_mass_kg: float
    measured_pod_mass_kg: float
    cargo_envelope_valid: bool
    clock_disagreement_s: float
    frame_constants_match: bool
    terrain_evidence_present: bool
    terrain_age_s: float
    lower_bound_terrain_clearance_m: float
    protected_zone_risk_upper: float
    endpoint_uncertainty_m: float
    safe_miss_class: str
    safe_miss_permitted: bool
    contingency_branch_present: bool
    target_state_valid: bool
    target_age_s: float
    correction_nominal_demand_m_s: float
    correction_nominal_budget_m_s: float
    protected_correction_reserve_intact: bool
    remote_communications_available: bool

    def validate_shapes(self) -> list[str]:
        reasons: list[str] = []
        if not self.evidence_lineage.strip():
            reasons.append("MissingEvidenceLineage")
        scalars = {
            "launcher_age_s": self.launcher_age_s,
            "pod_age_s": self.pod_age_s,
            "declared_pod_mass_kg": self.declared_pod_mass_kg,
            "measured_pod_mass_kg": self.measured_pod_mass_kg,
            "clock_disagreement_s": self.clock_disagreement_s,
            "terrain_age_s": self.terrain_age_s,
            "lower_bound_terrain_clearance_m": self.lower_bound_terrain_clearance_m,
            "protected_zone_risk_upper": self.protected_zone_risk_upper,
            "endpoint_uncertainty_m": self.endpoint_uncertainty_m,
            "target_age_s": self.target_age_s,
            "correction_nominal_demand_m_s": self.correction_nominal_demand_m_s,
            "correction_nominal_budget_m_s": self.correction_nominal_budget_m_s,
        }
        if any(not math.isfinite(v) for v in scalars.values()):
            reasons.append("NonFiniteEvidence")
            return reasons
        if any(
            v < 0.0
            for k, v in scalars.items()
            if k not in {"lower_bound_terrain_clearance_m"}
        ):
            reasons.append("NegativeEvidenceValue")
        if not (0.0 <= self.protected_zone_risk_upper <= 1.0):
            reasons.append("InvalidProtectedZoneRisk")
        return reasons


@dataclass(frozen=True)
class Receipt:
    decision: str
    denial_reasons: tuple[str, ...]
    policy_id: str
    evidence_lineage: str


def evaluate(snapshot: Snapshot, policy: Policy) -> Receipt:
    policy.validate()
    reasons = snapshot.validate_shapes()

    if not snapshot.launcher_state_valid:
        reasons.append("LauncherStateInvalid")
    if math.isfinite(snapshot.launcher_age_s) and snapshot.launcher_age_s > policy.max_launcher_age_s:
        reasons.append("LauncherStateStale")

    if not snapshot.pod_state_valid:
        reasons.append("PodStateInvalid")
    if math.isfinite(snapshot.pod_age_s) and snapshot.pod_age_s > policy.max_pod_age_s:
        reasons.append("PodStateStale")
    if (
        math.isfinite(snapshot.declared_pod_mass_kg)
        and math.isfinite(snapshot.measured_pod_mass_kg)
        and abs(snapshot.declared_pod_mass_kg - snapshot.measured_pod_mass_kg)
        > policy.max_mass_disagreement_kg
    ):
        reasons.append("PodMassDisagreement")
    if not snapshot.cargo_envelope_valid:
        reasons.append("CargoEnvelopeViolation")

    if (
        math.isfinite(snapshot.clock_disagreement_s)
        and snapshot.clock_disagreement_s > policy.max_clock_disagreement_s
    ):
        reasons.append("ClockDisagreement")
    if not snapshot.frame_constants_match:
        reasons.append("FrameOrConstantsMismatch")

    if not snapshot.terrain_evidence_present:
        reasons.append("TerrainEvidenceMissing")
    if math.isfinite(snapshot.terrain_age_s) and snapshot.terrain_age_s > policy.max_terrain_age_s:
        reasons.append("TerrainEvidenceStale")
    if (
        math.isfinite(snapshot.lower_bound_terrain_clearance_m)
        and snapshot.lower_bound_terrain_clearance_m
        < policy.min_lower_bound_terrain_clearance_m
    ):
        reasons.append("TerrainClearanceInsufficient")

    if (
        math.isfinite(snapshot.protected_zone_risk_upper)
        and snapshot.protected_zone_risk_upper > policy.max_protected_zone_risk_upper
    ):
        reasons.append("ProtectedZoneRiskTooHigh")
    if (
        math.isfinite(snapshot.endpoint_uncertainty_m)
        and snapshot.endpoint_uncertainty_m > policy.max_endpoint_uncertainty_m
    ):
        reasons.append("EndpointUncertaintyTooHigh")

    if snapshot.safe_miss_class not in ALLOWED_SAFE_MISS_CLASSES:
        reasons.append("SafeMissClassUnknownOrUnsafe")
    if not snapshot.safe_miss_permitted:
        reasons.append("SafeMissNotPermitted")
    if not snapshot.contingency_branch_present:
        reasons.append("NoAdmissibleContingencyBranch")

    if policy.require_target:
        if not snapshot.target_state_valid:
            reasons.append("TargetStateInvalid")
        if math.isfinite(snapshot.target_age_s) and snapshot.target_age_s > policy.max_target_age_s:
            reasons.append("TargetStateStale")

    if (
        math.isfinite(snapshot.correction_nominal_demand_m_s)
        and math.isfinite(snapshot.correction_nominal_budget_m_s)
        and snapshot.correction_nominal_demand_m_s > snapshot.correction_nominal_budget_m_s
    ):
        reasons.append("CorrectionDemandExceedsNominalBudget")
    if not snapshot.protected_correction_reserve_intact:
        reasons.append("ProtectedCorrectionReserveCompromised")

    if policy.require_remote_communications and not snapshot.remote_communications_available:
        reasons.append("RequiredRemoteCommunicationsUnavailable")

    ordered_unique = tuple(dict.fromkeys(reasons))
    decision = "AdmitStudyRelease" if not ordered_unique else "DenyStudyRelease"
    return Receipt(decision, ordered_unique, policy.policy_id, snapshot.evidence_lineage)


def fixture_policy(**overrides) -> Policy:
    values = dict(
        policy_id="synthetic-policy-v1",
        max_launcher_age_s=2.0,
        max_pod_age_s=2.0,
        max_terrain_age_s=3600.0,
        max_target_age_s=2.0,
        max_clock_disagreement_s=0.01,
        max_mass_disagreement_kg=0.1,
        min_lower_bound_terrain_clearance_m=20.0,
        max_protected_zone_risk_upper=1.0e-6,
        max_endpoint_uncertainty_m=5.0,
        require_target=True,
        require_remote_communications=False,
    )
    values.update(overrides)
    return Policy(**values)


def fixture_snapshot(**overrides) -> Snapshot:
    values = dict(
        evidence_lineage="synthetic-lineage-v1",
        launcher_state_valid=True,
        launcher_age_s=0.2,
        pod_state_valid=True,
        pod_age_s=0.2,
        declared_pod_mass_kg=100.0,
        measured_pod_mass_kg=100.02,
        cargo_envelope_valid=True,
        clock_disagreement_s=0.001,
        frame_constants_match=True,
        terrain_evidence_present=True,
        terrain_age_s=100.0,
        lower_bound_terrain_clearance_m=50.0,
        protected_zone_risk_upper=1.0e-8,
        endpoint_uncertainty_m=2.0,
        safe_miss_class="DeclaredDisposalRegion",
        safe_miss_permitted=True,
        contingency_branch_present=True,
        target_state_valid=True,
        target_age_s=0.2,
        correction_nominal_demand_m_s=0.2,
        correction_nominal_budget_m_s=0.8,
        protected_correction_reserve_intact=True,
        remote_communications_available=False,
    )
    values.update(overrides)
    return Snapshot(**values)


def self_test() -> None:
    p = fixture_policy()
    s = fixture_snapshot()
    ok = evaluate(s, p)
    if ok.decision != "AdmitStudyRelease" or ok.denial_reasons:
        raise AssertionError(f"nominal fixture denied: {ok}")

    if evaluate(s, p) != evaluate(s, p):
        raise AssertionError("evaluation not deterministic")
    if evaluate(s, p).decision != "AdmitStudyRelease":
        raise AssertionError("remote communications incorrectly required")

    denied = evaluate(
        fixture_snapshot(
            launcher_age_s=3.0,
            lower_bound_terrain_clearance_m=10.0,
            protected_zone_risk_upper=1.0e-4,
            endpoint_uncertainty_m=8.0,
            safe_miss_class="UnknownConsequence",
            safe_miss_permitted=False,
            contingency_branch_present=False,
            protected_correction_reserve_intact=False,
        ),
        p,
    )
    expected = (
        "LauncherStateStale",
        "TerrainClearanceInsufficient",
        "ProtectedZoneRiskTooHigh",
        "EndpointUncertaintyTooHigh",
        "SafeMissClassUnknownOrUnsafe",
        "SafeMissNotPermitted",
        "NoAdmissibleContingencyBranch",
        "ProtectedCorrectionReserveCompromised",
    )
    if denied.denial_reasons != expected:
        raise AssertionError(f"denial ordering changed: {denied.denial_reasons}")

    for key, value in [
        ("lower_bound_terrain_clearance_m", 19.999),
        ("protected_zone_risk_upper", 2.0e-6),
        ("endpoint_uncertainty_m", 5.001),
        ("launcher_age_s", 2.001),
    ]:
        if evaluate(fixture_snapshot(**{key: value}), p).decision != "DenyStudyRelease":
            raise AssertionError(f"worsened {key} was admitted")

    bad = evaluate(fixture_snapshot(endpoint_uncertainty_m=float("nan")), p)
    if "NonFiniteEvidence" not in bad.denial_reasons or bad.decision != "DenyStudyRelease":
        raise AssertionError("nonfinite evidence did not fail closed")

    require_comms = evaluate(s, fixture_policy(require_remote_communications=True))
    if "RequiredRemoteCommunicationsUnavailable" not in require_comms.denial_reasons:
        raise AssertionError("required communications outage not detected")

    corr = evaluate(fixture_snapshot(correction_nominal_demand_m_s=0.81), p)
    if "CorrectionDemandExceedsNominalBudget" not in corr.denial_reasons:
        raise AssertionError("correction budget overrun not denied")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("LL-008 release admission oracle self-test: PASS")
        return

    if not args.input:
        parser.error("--input or --self-test is required")

    payload = json.loads(Path(args.input).read_text())
    receipt = evaluate(Snapshot(**payload["snapshot"]), Policy(**payload["policy"]))
    print(json.dumps(asdict(receipt), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
