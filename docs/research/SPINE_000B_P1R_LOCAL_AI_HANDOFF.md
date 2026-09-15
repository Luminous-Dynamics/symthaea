# SPINE-000B-P1R Local AI Handoff

Target branch: `research/spine-000b-p1r-audit-hardening`

Base subject: `fa7b68645dd2109abb12c70f493e50e16c5740a4`

Do not start runtime proposal-receipt wiring on this branch. This branch exists only to close Issue #3036 exactly.

## Required implementation changes

1. Expand `scripts/generate_spine_golden_fixtures.py` so every generated case contains explicit `coverage_tags` and the union covers every tag enforced by `scripts/verify_spine_000b_p1r_contract.py`.
2. Preserve the independent Python oracle unchanged unless a real semantic defect is discovered. If the oracle itself must change, create a new measurement-policy lineage rather than silently editing expectations.
3. Harden `test_spine_000b_p1_golden_equivalence` in `src/cognitive_loop/subsystem_trait.rs`:
   - Python must be required: failure to spawn `python3` is test failure.
   - Read each expected receipt's `integrated_without_subject`.
   - Compare all canonical `I_withoutS` fields bit-for-bit: confidence, LR, exploration, arousal, valence, flags, contributor count.
   - Continue comparing `changed_channels`, `uniquely_contributed_flags`, and `integration_changed`.
4. Regenerate and check in `tests/fixtures/spine_000b_golden_fixtures.json` only after the generator is final.
5. Run `python3 scripts/verify_spine_000b_p1r_contract.py` before the Rust test. The verifier is intentionally fail-closed.
6. Run the targeted Rust equivalence test and existing focused OutputCollector tests.
7. Push only the allowed P1R files. The dedicated workflow rejects unrelated changes.

## Qualification meaning

A green `SPINE-000B P1R Equivalence` workflow means only that production Rust collector semantics and the independent oracle agree over the frozen preregistered fixture surface. It does not establish runtime influence or causal load.
