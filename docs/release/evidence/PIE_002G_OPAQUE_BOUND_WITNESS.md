# PIE-002G Opaque Subject-Bound Electrical Witness

Status: production Rust candidate; construction-integrity theorem only.

## Purpose

PIE-002D validates and binds a complete utility projection to explicit supply/recovery context. PIE-002E validates an exact invocation-supplied `ProcessDefinition`, derives its utility projection inside the call, and immediately binds that fresh projection.

The lower-level `BoundElectricalAccountingCase` intentionally remains a public Serialize+Deserialize DTO for compatibility and expert/testing use. Therefore possession of that DTO alone does not prove that PIE-002D/E actually ran.

PIE-002G adds a separate authoritative Rust type:

`SubjectBoundElectricalAccountingCase`

whose construction state is private and whose only public creation path is successful PIE-002E composition.

## Core theorem

```text
ProcessDefinition
  -> full validation
  -> fresh in-call utility projection
  -> explicit PIE-002D binding
  -> private construction
  -> SubjectBoundElectricalAccountingCase
```

Possession of the opaque witness proves only that this Rust value crossed that composition boundary.

## Type-level integrity

The witness:

- has private state;
- exposes no public `new`, `from_bound`, unchecked constructor, or mutable accessor;
- does not implement `Deserialize`;
- may implement `Serialize` one-way for evidence/reporting;
- exposes read-only getters for the exact bound demand/context;
- can materialize a lower-level `ElectricalUtilityCase` for explicit expert/testing evaluation;
- cannot be reconstructed from an arbitrary `ElectricalUtilityCase` or `BoundElectricalAccountingCase` through a public API.

The type is defined in the composition module itself so the private struct literal used by the validated composition function is not available to sibling modules merely through a crate-visible constructor.

## External compile-fail evidence

The Rust documentation contains external-crate `compile_fail` fixtures for two negative capabilities:

1. external code cannot access the private inner `bound` field;
2. `SubjectBoundElectricalAccountingCase` does not satisfy `serde::de::DeserializeOwned`.

Qualification must execute package doctests in addition to ordinary tests/check/Clippy. A green ordinary unit-test run alone is insufficient for PIE-002G because unit tests nested in the module are descendants and can observe more private implementation detail than a downstream crate.

## Compatibility

PIE-002G does not delete or privatize `BoundElectricalAccountingCase` and does not change PIE-002D's lower-level binding API. The distinction is explicit:

```text
bind_electrical_accounting_case(...)
    -> BoundElectricalAccountingCase
    -> lower-level/expert DTO

compose_subject_bound_electrical_accounting_case(...)
    -> SubjectBoundElectricalAccountingCase
    -> preferred construction-integrity witness
```

Existing callers that intentionally operate at PIE-002D may continue to do so. Callers that want the preferred subject-bound theorem use PIE-002E/G.

## One-way serialization

Serialization of the witness is permitted only for evidence/reporting. It does not create a trusted restoration format.

A serialized representation can still be deserialized into lower-level DTOs if those DTOs support it, but such a value does not regain witness authority. Any future trusted restoration theorem must independently bind and revalidate its receipt/state before constructing a witness.

## Getter semantics

Read-only getters preserve the already-bound values exactly:

- process ID;
- gross electrical energy;
- batch duration;
- peak power;
- recoverable energy;
- recovery duration;
- storage energy acceptance;
- storage charge/discharge power;
- recovery-delivery fraction;
- available energy capacity;
- available sustained/peak power;
- unresolved non-electrical utility semantics.

No getter evaluates feasibility or promotes unresolved thermal/cooling semantics.

## Required qualification

An exact immutable Rust 1.96 qualification must pin subject/parent/tree/file scope and run at minimum:

- `cargo fmt -p symthaea-planetary-industry -- --check`;
- locked all-target `cargo check`;
- locked all-target package tests;
- strict all-target Clippy with `-D warnings`;
- package doctests, including the external `compile_fail` opacity checks;
- diff hygiene;
- source-boundary checks that the preferred function still accepts `&ProcessDefinition` + `&ElectricalSupplyRecoveryContext`, performs fresh projection and binding, returns the opaque witness, contains no detached `ProcessUtilityProjection` parameter, and does not call the feasibility evaluator;
- source checks that the witness does not derive/implement `Deserialize` and does not expose its private state or a public unchecked constructor.

A machine-readable receipt must preserve all gate outcomes.

## Deliberate non-claims

PIE-002G does **not** prove:

- evidence provenance;
- source authenticity/truth;
- evidence applicability;
- applicability-profile sufficiency;
- persistence/authority currentness;
- electrical feasibility;
- storage dispatch;
- thermal/cooling closure;
- equipment qualification;
- economics;
- execution authority.

Those remain separate theorems. PIE-002K is the later composition layer that may combine provenance/applicability with this opaque numerical witness without redefining G.

Tracks #2826, #2782, #2764, #2785, #2935, #3114, #3131, #1610, #1647, and master #1604.
