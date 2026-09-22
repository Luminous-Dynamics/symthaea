# RES-OSINT-SEC-001 — secret-safe investigation diagnostics patch plan

This branch-local note intentionally contains no product authority. It records the exact intended source change before applying it to the full `symthaea-investigation` source:

- role-reference types retain explicit `as_str()` access for canonical binding;
- ordinary `Debug` redacts the inner reference value;
- `InvestigationError` ordinary `Display`/`Debug` stops echoing raw identifier strings;
- diagnostics preserve variant/role/reason without revealing investigation target identifiers;
- regression tests prove representative sensitive values do not appear in default diagnostics;
- no constructors, semantic state transitions, evidence logic, or authority ceilings change.

This note is not a qualification result and should be removed or superseded if the full source patch is materialized as the exact review subject.
