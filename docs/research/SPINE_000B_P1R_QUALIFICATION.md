# SPINE-000B-P1R Qualification Boundary

Status: measurement-only hardening of Issue #3036.

P1R exists because the first P1 development subject (`fa7b68645dd2109abb12c70f493e50e16c5740a4`) established useful partial Rust↔Python agreement but did not satisfy the full preregistered comparison surface.

P1R MUST NOT alter runtime cognition. It may change only the fixture generator, checked-in fixtures, the P1 Rust test, the independent contract verifier, and its dedicated workflow.

## Required theorem surface

For deterministic fixture classes covering 0, 1, 2, and N>=4 contributors:

- production Rust `OutputCollector::integrate()` is the Rust authority;
- Python `spine_000b_influence_oracle.py` remains an independent comparison implementation;
- every top-level integrated field is compared canonically;
- every leave-one-out `I_withoutS` field is compared canonically;
- derived `changed_channels`, `uniquely_contributed_flags`, and `integration_changed` agree;
- Python execution is mandatory, not best-effort;
- fixture generation is deterministic and regeneration produces no diff;
- checkout contents are unchanged by qualification execution.

Required coverage tags are enforced by `scripts/verify_spine_000b_p1r_contract.py` and include each preregistered single-channel case, required two-contributor interactions, and an N-contributor mixed-all-channels case.

## Claim boundary

P1R PASS establishes only cross-implementation agreement for the frozen proposal-integration measurement semantics on the qualified fixtures.

It does not establish:

- that any subsystem ran in a live cognitive cycle;
- runtime proposal admission;
- state-application consumption;
- behavioral influence;
- causal load;
- intelligence, correctness, consciousness, or epistemic authority.

Runtime `ProposalInfluenceReceipt` wiring remains blocked until P1R passes on an exact frozen head.
