# LL-008 Deterministic Study-Release Admission Evidence

## Purpose

This artifact defines a fail-closed **Phase-0 study admission** boundary for launcher/corridor evidence. It does not compute trajectories and it cannot command launcher hardware.

Passing produces only `AdmitStudyRelease`: the supplied snapshot satisfies the supplied policy gates. It is not physical launch authorization.

## Inputs and provenance

The model consumes already-computed evidence for launcher/pod freshness, mass agreement, cargo envelope, time/frame consistency, terrain evidence, lower-bound clearance, protected-zone risk, endpoint uncertainty, safe-miss consequence, target/catcher freshness, smart-pod correction reserve, and communications policy.

Every receipt preserves:

- policy id;
- evidence lineage;
- deterministic decision;
- ordered denial reasons.

No utility score or optimizer can override a failed hard gate.

## Executed self-test

The committed self-test was executed before commit and passed.

It demonstrates:

1. a nominal synthetic snapshot admits;
2. identical inputs replay exactly;
3. remote communications may be absent when policy permits fresh local evidence;
4. ordered simultaneous denial reasons remain deterministic;
5. worsening terrain clearance, protected-zone risk, endpoint uncertainty, or launcher freshness cannot create an admission;
6. non-finite evidence fails closed;
7. communications outage denies only when the declared policy requires remote communications;
8. nominal smart-pod correction demand cannot exceed the non-reserved correction budget.

## Fixed denial-order fixture

A synthetic snapshot with stale launcher state, insufficient terrain clearance, excessive protected-zone risk, excessive endpoint uncertainty, unknown/unpermitted safe-miss outcome, no contingency branch, and compromised correction reserve produces the ordered denial list:

1. `LauncherStateStale`
2. `TerrainClearanceInsufficient`
3. `ProtectedZoneRiskTooHigh`
4. `EndpointUncertaintyTooHigh`
5. `SafeMissClassUnknownOrUnsafe`
6. `SafeMissNotPermitted`
7. `NoAdmissibleContingencyBranch`
8. `ProtectedCorrectionReserveCompromised`

Reason ordering is part of the deterministic receipt contract so evidence regressions can be compared exactly.

## Architecture boundary

Symthaea may:

- propose candidate corridors;
- choose what uncertainty reduction to study;
- compare transport architectures;
- recommend maintenance or re-observation;
- explain which hard gate failed.

Symthaea may **not** convert a failed hard gate into admission because a route is economically attractive or because a probabilistic model assigns high nominal success.

Likewise, `AdmitStudyRelease` remains upstream of any future qualified local hardware interlock. A future physical release path requires separate hardware qualification, authority, human/organizational governance, and independently validated safety policy.

## Non-claims

The synthetic policy values in the oracle are test fixtures only. They do not establish real lunar values for:

- data freshness;
- mass tolerance;
- clock tolerance;
- terrain clearance;
- protected-zone risk;
- endpoint covariance;
- smart-pod correction reserve;
- acceptable miss/disposal classes.

Those must remain explicit mission- and governance-specific evidence inputs.
