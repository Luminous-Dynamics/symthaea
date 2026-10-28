# VIS-DVPC-033 — Compositional evidence bundle v1

**Status:** implementation draft; synthetic cross-tool conformance only.

Tracking issue: https://github.com/Luminous-Dynamics/symthaea/issues/8778

## Why this exists

The earlier VIS-DVPC steps answer separate questions. Candidate-universe completeness is not statistical identification; structural support is not finite-sample inference; exact synthetic estimator behavior is not real-trace calibration; and a region gate can only reason about the region supplied to it. A set of individually valid component reports can still be an invalid composite if they refer to different traces or if a downstream consumer silently upgrades a diagnostic into authorization.

VIS-DVPC-033 composes the existing CLI reports by explicit schema and fail-closed checks. It cross-binds the trace SHA-256 from VIS-DVPC-032 to the input digest emitted by VIS-DVPC-027, preserves the VIS-DVPC-028 zero-support result and its partial-identification region, then checks that VIS-DVPC-031 does not authorize that unresolved region.

## Frozen component snapshots

The qualification workflow checks out the exact source snapshots below rather than whatever branch head happens to exist when this workflow executes:

- VIS-DVPC-032 candidate-action universe: 9f19817cf429c53ddac2cab8812aa430130b3f4a
- VIS-DVPC-027 selective-feedback support audit: 02c3458fb75c5fdcdeecb5333f4d8fdd1f8bfbc5
- VIS-DVPC-028 exact OPE benchmark: 19a03ae8510b29157a0d0ff5d5f124e88e4e24a8
- VIS-DVPC-031 interval decision gate: 6778c4818bff34e1cfb48fd1899dfe50b755a8e0

These hashes make this conformance run reproducible. The pins must be intentionally refreshed when a component contract changes; successful output against an old pin is not evidence about a newer component head.

## Adversarial composition

The workflow verifies fixture SHA-256 sum files, builds each standalone Rust program, runs its own tests, and obtains real CLI reports from these snapshots. The context/action fixture has only two contexts, so it can prove exact universe agreement while intentionally failing the support audit's effective-sample-size floor. The OPE enumerator runs the frozen synthetic grid, including a zero-support case. The region gate receives the partial region [0.1, 0.6], value floor 0.5, and risk region [0.01, 0.03] with risk cap 0.05.

The bundle must remain DIAGNOSTIC_ONLY only when all of the following hold:

1. The universe report says COMPLETE_ACTION_UNIVERSE.
2. The universe trace digest equals the support report's input digest.
3. Support remains SUPPORTED_BUT_WEAK_OVERLAP, not approval.
4. The zero-support scenario remains NOT_IDENTIFIED, its estimator metrics are SUPPRESSED, and its region remains [0.1, 0.6].
5. OPE output explicitly says confidence intervals were not computed and production authorization is not claimed.
6. The decision gate returns INSUFFICIENT_EVIDENCE, with unresolved value and cleared risk states.
7. The report always records automatic_policy_authorization=true under PROHIBITION.

Any violated check yields BLOCKED. Missing, duplicated or malformed required fields are execution errors and exit nonzero. Unit tests mutate the bundle reports to test digest mismatch, missing evidence, an incomplete universe, reintroduced point-estimator output, relabeling an identification region, and an authorization-shaped gate response.

## Receipt

The workflow's artifact includes all component reports, the composite report, invocation output, checked-out component revisions, Rust version, source/fixture hashes, and report hashes. Hashes establish byte identity only; they do not by themselves authenticate a manifest's owner, prove a data custodian's authority, or attest that declared propensities match a real logger.

## Benchmark interpretation

This is an integration-contract benchmark, not yet the equal-budget closed-loop benchmark tracked by #8686. It does not estimate risk bounds, produce calibrated confidence intervals, address clustered/adaptive sampling, establish performance on real logged data, or prove improvement over a baseline. The useful success metric here is zero false authorizations under the declared mutation suite—not high authorization rate.

## Run locally

The component branches are frozen at the SHA values above. After placing their worktrees under component/dvpc032, component/dvpc027, component/dvpc028, and component/dvpc031, run:

rustc --edition=2024 --test tools/vision-dvpc-evidence-bundle.rs -o /tmp/vis-dvpc-033-tests
/tmp/vis-dvpc-033-tests
rustc --edition=2024 tools/vision-dvpc-evidence-bundle.rs -o /tmp/vis-dvpc-033
/tmp/vis-dvpc-033 --audit /tmp/universe.tsv /tmp/support.tsv /tmp/ope.tsv /tmp/gate.tsv
