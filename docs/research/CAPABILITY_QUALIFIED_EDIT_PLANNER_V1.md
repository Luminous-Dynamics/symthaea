# CQEP-001 — Capability-qualified audio edit plan compiler

**Date:** 10 November 2026  
**Status:** deterministic plan compiler and synthetic contract tests; no model execution or audio result claimed  
**Owner:** https://github.com/Luminous-Dynamics/symthaea/issues/11525

## Research hypothesis

A planner that compiles normalized requests into typed capability-qualified DAGs with independent observation obligations can reduce invalid tool selection, unnecessary global rerenders and preservation regressions relative to a naive language-model router, while maintaining or improving task success under matched compute budgets.

## Industry comparison

The ICASSP 2027 challenge defines an Agent Track in which systems can plan, invoke multiple local models or DSP tools, inspect intermediate results and revise outputs. The published reference baseline routes between DSP, SAM-Audio and AuK. Its new hidden test Track 2 card reports IFR 44.13%, CR 76.09%, EMR 8.00%; the public baseline repository separately reports a locally judged 2,000-example result of IFR 44.06%, CR 74.63%, EMR 7.45%. Do not compare these as one score series: test scope and evaluation package differ.

A useful competitive claim therefore requires matched evaluation. The CQEP study should compare the same systems on the same cases, with the same evaluator, source modalities and resource budgets. Internal plan-validity metrics are not a substitute for official Qwen3-Omni rubric scores.

## Paradigm

A normalized edit specification becomes a typed execution DAG before any audio mutation. The request declares target scope, required operation/capabilities, typed input/output artifacts, protected constraints, required observations, locality ceiling, determinism level and resource/risk budgets.

A versioned operator registry declares capabilities, artifact types, pre/postconditions, supported locality, determinism, resource estimates and exact artifact digests. A separate observer registry declares metric coverage and independent qualification. The selected observer artifact must differ from the editor artifact. This prevents simple self-evaluation by the selected generator, though a distinct digest alone is not proof of statistical independence.

The compiler refuses permissive fallback. An unsupported local edit remains unsupported; it cannot silently become a whole-stem rerender. A valid plan is still only a proposal: runtime gates must revalidate registry, input digests, preconditions, evidence and budget before every mutation.

## V1 contracts

The standalone crate tools/capability-qualified-edit-planner:
- validates content-addressed initial artifacts and registry snapshot;
- validates unique stage/output IDs and typed input/dependency references;
- requires every dependency output to be explicitly consumed;
- topologically sorts the stage DAG, rejecting cycles;
- selects operators by hard capability, I/O type, precondition, postcondition, locality and determinism constraints;
- selects independently declared observers for every required metric;
- deterministically ranks candidates by weighted resource/risk objective and stable ID;
- checks aggregate compute, latency and risk budgets;
- carries initial artifact digests, resource budget, objective weights and scope/determinism ceilings in the compiled plan receipt;
- distinguishes invalid, unsupported and budget-exceeded outcomes.

The compiler has no model, audio, filesystem or network dependency. Canonical plan serialization/digest is external and versioned rather than improvised through Debug formatting.

## Experiments

Compare four agents on the same MMAE public development examples and a separately licensed/consented Muse singing-edit set:
1. LLM-only router;
2. fixed hand-authored sequence;
3. static capability-qualified plan;
4. capability-qualified plan with observed uncertainty/response feedback.

Record:
- official/local IFR, CR and EMR only when the exact rubric evaluator and scope match;
- unsupported/abstention rate and false acceptance;
- stale artifact, missing precondition, type/capability, cycle and locality rejection counts;
- tool calls, render attempts, compute, latency and plan-repair count;
- preserved-property regression across turns;
- human preference/naturalness, melody, lyric correctness and identity outcomes as separate dimensions.

## Development targets

Targets, not observations:
- 100% rejection of malformed DAG, missing capability, invalid type, stale artifact and budget-control tests;
- zero execution of an unregistered or unsupported operator;
- zero global fallback for a local-only request;
- 100% required observer metric coverage before a dependent/final mutation;
- lower total tool/render cost than the router baseline at equal or higher matched EMR/IFR/CR;
- publish results by task/modality/complexity, including abstentions and worst strata.

## Claim boundary

Synthetic tests qualify static plan validation and deterministic selection only. They do not qualify operator profiles, observer validity, natural-language parsing, runtime execution or audio quality. Empirical operator capability and observer independence require held-out evidence.
