# SCI-001 PR Notes

These notes freeze the intended review boundary for the Scientific Method Kernel audit.

## Exact scope

Relative to `main@2a8b8fd3ab38a9a7fd15dc8ebd98c5e74bbbdfd1`, SCI-001 is documentation-only.

It must not modify:

- Rust product source;
- Cargo manifests or `Cargo.lock`;
- workflows or qualification logic;
- existing scientific status enums;
- existing evidence/authority semantics;
- causal, ALife, neuroscience, economics, futures, matter, physical-agency, or RCA runtime behavior.

## Review question

The PR should be evaluated on one question only:

> Does the audit define sufficiently precise common and non-common scientific semantics to constrain future shared-kernel implementation without transferring authority or weakening a domain theorem?

## Explicit non-claims

SCI-001 does not establish that:

- a shared `symthaea-science-kernel` crate should definitely be created;
- any existing branch is qualified;
- existing domain types are mutually interchangeable;
- any scientific discovery is independently validated;
- any proposed experiment-planning, symbolic-discovery, causal, or Theory Atlas improvement has been implemented;
- any evidence object may authorize action.

## Follow-up rule

Every SCI-002+ implementation PR should cite the exact theorem from `SCIENTIFIC_METHOD_KERNEL_AUDIT_V1.md` it intends to materialize and should state which existing domain semantics it must preserve.
