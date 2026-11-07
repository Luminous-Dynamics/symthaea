# Guidance withdrawal / methodological autonomy

This research lane composes with, but does not duplicate, #10877 research taste.

- Research taste asks whether the researcher chooses good directions, mechanisms, experiments and interpretations.
- Methodological autonomy asks how independently the researcher can construct and validate the method when instructions progressively disappear.

## Pinned comparator

ASI-Bench 0.3.0, pinned to `apexin-ai/ASI-Bench@eb60952124b90e9f0886c1212bf7463e26d0bc19`, reports 60 project-level tasks across 11 scientific domains and four matched B1–B4 guidance conditions. The release pin, Git blob identities, scope descriptors, seed boundaries and leaderboard claim limit are in:

- `SYM_RSI_GUIDANCE_WITHDRAWAL_MANIFEST_V0_2026-11-07.json`
- `SYM_RSI_GUIDANCE_WITHDRAWAL_SCHEMA_V0_2026-11-07.json`

The experiment protocol is in `SYM_RSI_GUIDANCE_WITHDRAWAL_PROTOCOL_V0_2026-11-07.md`.

The preregistered B3 method-selection policy is supported by `SYM_RSI_METHOD_ADJUDICATION_FIXTURE_V0_2026-11-07.json` and its schema. The synthetic fixture contains two admissible method families to demonstrate that alternative valid methods can pass when assumptions and evidence fit; it is not human expert gold.

## Key design safeguards

1. B1–B4 are reproduced exactly before adding local extensions.
2. B3 accepts multiple admissible methods when their assumptions, applicability and evidence are defensible; exact-method matching is not the construct.
3. B3-neutral-length is a Symthaea-only control to separate context-length effects from semantic distraction in B4.
4. Raw scores, task-level results, valid artifacts, integrity, cost, human intervention and uncertainty remain a vector.
5. Public seed31415 local scoring is distinct from private-reference seed42 official submission.
6. The currently rendered leaderboard is not treated as a final official public baseline.
7. A guidance-autonomy result is not a scientific discovery or R4 result.

## Preregistration status

The machine-readable template is:

`SYM_RSI_GUIDANCE_WITHDRAWAL_PREREGISTRATION_V0_2026-11-07.json`

Its companion structural schema is:

`SYM_RSI_GUIDANCE_WITHDRAWAL_PREREGISTRATION_SCHEMA_V0_2026-11-07.json`

The template has 46 unresolved placeholders and remains `TEMPLATE_NOT_PREREGISTERED`. It is a design artifact, not permission to describe any run as confirmatory. Before confirmation, freeze task IDs/revisions, prompt/scorer bundles, model/harness/tool identities, budgets, split custody, adjudication criteria, statistical plan and human/data approvals.

No benchmark task has been run by this work. No official leaderboard position, human-level result, scientific discovery or R4 claim is made.
