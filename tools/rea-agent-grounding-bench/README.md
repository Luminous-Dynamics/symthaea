# REA source-grounded reverse-engineering scorecard v0

This crate is the first executable scoring slice for issue #10783. It is deliberately not an agent runner and does not execute tools or inspect arbitrary binaries itself.

## What it measures

For every eligible task, it separates:
- raw answer correctness against the independently authored expected answer;
- evidence disposition after verifying every cited raw REA Evidence record;
- evidence-grounded correctness;
- parametric leakage: a correct raw answer that lacks the expected evidence-grounded disposition;
- false supported and false refuted claims;
- paired relevant-mutation sensitivity;
- paired irrelevant-mutation invariance.

The PLR denominator is **raw-correct answers**, so PLR is undefined when no answer is raw-correct. Raw accuracy and grounded accuracy use the full task denominator, including missing submissions as abstentions.

## Evidence rule

V0 only gives positive grounding credit for a provider-reported normalized observation:
- exact artifact digest;
- exact operation;
- exact represented platform/environment scope;
- exact source predicate type;
- JCS-equal normalized result and expected answer;
- verified REA semantic Evidence ID.

Duplicate and unreferenced Evidence records are rejected. A correct answer without valid source-bound evidence receives no grounded credit. The scorer reuses the REA Evidence identity/JCS primitives but independently implements its narrow benchmark disposition; it does not authenticate producers or infer provider independence.

## Paired controls

- Baseline vs relevant mutation: both answers must be raw-correct and different.
- Baseline vs irrelevant mutation: both answers must be raw-correct and equal.
- Unsupported and conflict variants remain in the full task denominator. V0 cannot positively ground unimplemented stronger predicates.

## Important scope limitation

REA Evidence v1 does not encode platform version or exact archive-member occurrence identity. This scorer therefore marks platform-version requests underdetermined and does not claim member-occurrence grounding from Evidence v1 alone. A member-specific benchmark requires the versioned archive-member verification receipt and exact occurrence binding.

## Claim ceiling

This scorecard measures source grounding on the declared task, raw Evidence records, expected labels and protocol. It does not establish universal RE competence, producer authenticity, artifact benignness, temporal freshness, complete acquisition, or performance on unseen artifact families.
