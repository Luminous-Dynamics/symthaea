# Marine liveaboard mission-evidence join (V0)

A standalone, dependency-free Rust contract that joins the declared outcomes of independent engineering owners **only when every report refers to the same frozen mission, vessel configuration, operating case, condition set and duration**.

It is an evidence-alignment verifier, not a physics engine, optimizer, digital twin, safety case or approval system.

## Run

```sh
cargo test --locked --offline --manifest-path crates/tools/marine-mission-evidence-join/Cargo.toml
cargo run --locked --offline --manifest-path crates/tools/marine-mission-evidence-join/Cargo.toml --example thirty_day_case
```

The example uses synthetic evidence labels and IDs. It is not a real vessel assessment.

## Why a join is needed

Separate results do not compose merely because their labels mention the same vessel:

```text
energy result for hull revision A / calm sea
+ autonomy result for hull revision B / 30 days
+ habitability result for hull revision A / rough sea
!= one mission-case result
```

Every required owner row must match the exact profile manifest, vessel configuration, mission contract, operating-case ID, condition manifest and target duration frozen by the mission case.

## V0 contract

The caller freezes:
- profile ID and profile-manifest SHA-256;
- exact vessel configuration SHA-256;
- mission/service-contract SHA-256;
- operating-case ID;
- condition-manifest SHA-256;
- target duration in integer seconds;
- a non-empty, unique required-owner roster.

Each owner report supplies the same bindings plus a report reference, result digest, method revision, evidence kind, currentness state and owner-declared outcome.

Evidence kinds (synthetic, assumed, model estimate, external source, measured) are descriptive metadata, not a quality ranking. The joiner never upgrades them.

## Result meanings

- `AllRequiredOwnersSupportDeclaredScope`: one complete, exact-matching, current (or explicitly not-currentness-applicable) report exists for each required owner, and every owner declares support for its scope.
- `AtLeastOneRequiredOwnerDoesNotSupportScope`: one or more required owners explicitly report that their bound is not supported.
- `Indeterminate`: required evidence is missing, unknown, incomplete or no longer current.
- `Invalid`: malformed identity, duplicate or unexpected owner evidence, or exact-subject mismatch.

The report preserves each owner result and all findings. It has no weighted score, no averaged feasibility metric and no mission-autonomy scalar. Invalid takes precedence; an explicit contradiction takes precedence over mere missing evidence.

## Currentness

Every evidence row declares one of:
- `Current` — requires a non-empty currentness reference from the owning domain;
- `Expired` — indeterminate;
- `Unknown` — indeterminate;
- `NotRequired` — the owner declares currentness checking is not applicable to this report type.

The verifier does not infer expiry from timestamps or decide whether an owner’s currentness declaration is true.

## Explicit boundaries

This crate does not:
- calculate energy, water, food, waste, hydrodynamics, seakeeping, stability, structure, fire/flood behaviour or route feasibility;
- verify or recalculate report bytes/digests, authenticate publishers, calibrate instruments or validate solvers;
- establish that an owner's declared outcome is correct;
- grant construction, dispatch, actuation, statutory, class/flag, seaworthiness, passenger-operation or commercial approval.

It composes with the existing owners rather than replacing them:
- #8092 / #8102 — marine mission energy and voyage traces;
- #10956 — resource autonomy vector;
- #10966 — task/location habitability and seakeeping;
- #8130 — frozen per-operating-case coverage;
- exact structural, stability, route, serviceability and fire/flood owners.

## Frozen synthetic test campaign

The unit tests cover complete support, explicit contradiction, missing required owner, duplicate roster/rows, unexpected owner, stale configuration, mismatched conditions, wrong target duration, missing result metadata, expired/unknown currentness, assumption-label preservation and input permutation.

No runtime or throughput claim is made. Measure p50/p95 and allocations only after actual local execution on frozen 4 / 8 / 16 / 32 / 64-owner cases.

**Claim ceiling:** a complete result proves only that the declared owner reports are present and aligned to the exact same mission-case identity and carry their owners' declared dispositions. It does not establish that the reports or the physical vessel are correct, safe or approved.
