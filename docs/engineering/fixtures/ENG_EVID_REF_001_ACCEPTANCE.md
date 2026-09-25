# ENG-EVID-REF-001 acceptance semantics

Parent: #5683

The corpus is intentionally staged so domain repair can proceed without pretending unresolved external subjects are already globally resolvable.

## Stage A — local closure

Required before a domain PR leaves draft:

- local model/member/result references resolve exactly once;
- duplicate semantic IDs reject;
- `...Required` capability markers cannot produce prediction evidence;
- evidence-plane compatibility is checked;
- serialization does not bypass validation.

## Stage B — external composition resolution

Required by ENG-DEVICE-001 integration claims:

- external component/material/geometry/FIELD/solver refs resolve through an explicit resolver;
- unresolved refs remain representable but block claims requiring resolution;
- wrong class/revision/generation rejects the stronger claim.

## Stage C — numerical-result closure

Required before external solver results become admitted prediction evidence:

- exact solver input closure;
- explicit execution context;
- execution receipt;
- parser/result identity;
- domain-specific result-plane compatibility.

## Nonclaim

Passing Stage A does not imply Stage B or C, and none of these stages establishes physical validation.
