# External benchmark reproduction guide

**Registry:** [`manifest.json`](manifest.json)  
**Schema:** [`manifest.schema.json`](manifest.schema.json)  
**Status:** metadata-only registry; no external suite is currently qualified for execution from this checkout.  
**Last updated:** 2026-11-04

## Rules

1. The registry is the single metadata index for external benchmarks. Do not create another per-project registry or duplicate CEB's task/receipt schemas.
2. A registry record is not a runnable adapter. Status must say whether a suite is metadata-only, access-gated, adapter-in-progress, runnable, qualified, or retired.
3. Pin the exact task/data revision, runner revision, environment/image, toolchain, model/harness/configuration, network policy, time/resource budget, grader revision, and repeat policy before a result is called reproducible.
4. Preserve split boundaries and the exact task universe. Do not train or tune on a hidden test set or copy hidden task/grader material.
5. Distinguish benchmark-maintainer leaderboard results, independent third-party evaluations and local reproductions. Never compare different splits, versions, harnesses, model effort, resource budgets or score definitions as though they were one experiment.
6. Report per-task outcomes, all eligible/attempted/graded denominators, abstentions, refusals, timeouts, tool failures, ungraded cases and repeated-run variability. Missing or ungraded tasks do not silently disappear from a denominator.
7. Use the benchmark maintainer's official evaluator where available. Do not rewrite a grader and claim equivalent scoring without an independent equivalence study.
8. A schema-valid manifest does not authorize access, execution, security claims, engineering approval or performance claims.

## PLE — metadata-only procedure

The first registered suite is Physical-Engineering's Last Exam (PLE).

Authoritative public references:

- Official site / leaderboard: https://ple-bench.com/
- Official Apache-2.0 runner kit: https://github.com/google-research/ple
- Official CC-BY-4.0 development dataset (access-gated): https://huggingface.co/datasets/google/physical-engineerings-last-exam
- Paper, arXiv v2 revised 2026-10-30: https://arxiv.org/abs/2610.30537

### Current metadata

The benchmark contains 85 tasks: structural 10, civil 18, electrical 27 and mechanical 30. Its public development split contains 29 tasks: 4 structural, 9 civil, 9 electrical and 7 mechanical. The hidden test split contains 56 tasks and must remain in the official competition trust boundary.

The dataset card identifies 12 `checks`, 14 `rubric` and 3 `checks-judge` dev tasks. Seventeen tasks require the official evaluation runner to build task-specific verifier images; standalone Harbor is not a full substitute. The official runner requires Harbor 0.20.0, uses `eval/run.py`, and enforces a six-hour per-task limit. The official paper/site report three-run evaluation; each local repetition must be separately identified.

### Required gates before any run

- [ ] PLE dataset access is granted to the authorized account and its exact CC-BY-4.0 terms are accepted.
- [ ] Exact dataset commit is pinned and recorded. The official runner downloads the latest dataset revision by default; do not rely on a moving `latest` for reproducible comparison.
- [ ] Exact official `google-research/ple` source commit, Harbor 0.20.0, task/verifier image digests and all dependencies are pinned.
- [ ] CEB task/receipt mappings are reviewed against the exact CEB v0 schema; do not introduce a second task or receipt format.
- [ ] Agent-visible task files and evaluator/grader contents are separate trust domains under CEB #10436.
- [ ] The Luminous execution boundary is independently qualified under Symthaea #10064. A host-opt-in runner, Nix development shell or container label alone does not establish isolation.
- [ ] Model endpoint and judge endpoint are explicitly configured, secrets are held outside source control, and outbound networking follows the official allowlist.
- [ ] Run IDs, task IDs, exact inputs, harness/model identity, grader/image identity, outputs, per-task status, resource usage and all failure dispositions are retained.

### Official execution after qualification

Follow the official procedure in the upstream [evaluation README](https://github.com/google-research/ple/blob/main/eval/README.md). Do not use a locally invented replacement grader.

The upstream dev command is structurally:

```bash
uv run python eval/run.py <submission> --env-file <private-env-file>
```

This is documentation of the official interface, **not authorization to execute it in the current environment**. It invokes Python, Harbor, Docker images, task-provided software, local services and an agent harness; it must not run until all isolation/custody gates are satisfied. Keep `HF_TOKEN`, model keys and judge keys in a secret store / private environment file outside the repository.

Run all 29 dev tasks. For repeatability, execute three independent full-set runs with distinct run IDs rather than confusing Harbor's `--trials` parallelism or per-task retry count with repeated benchmark trials. Re-grade only through the official tool. Record official per-task scores and raw artifacts as the evaluator returns them. Preserve tasks that are unattempted, ungraded, interrupted or unresolved.

### Score interpretation

The official site reports Claude Opus 5.5 at 48.3% (±1.4%) and GPT-6 Astra at 45.8% (±0.6%) on the 56-task hidden test leaderboard. The paper abstract separately states that the strongest configuration reached a 50.6% mean score. The published page and paper describe different reported summaries; this registry stores both as context only and forbids treating them as a single matched baseline.

The official site also reports 91% average individual-constraint satisfaction but only 41% all-hard-check clearance, with 27M tokens and 63 minutes per task for the top configuration. These are different measures and must remain separate. The hard-check gate is non-compensatory: a good cost score cannot rescue a critical constraint failure.

For CEB comparisons, first reproduce the official public dev leaderboard using the exact 29-task dev split and same scoring protocol. Only then estimate matched improvements. Never compare a local dev-set score to a hidden-set leaderboard score.

### Current claim ceiling

The PLE registry entry is **access_gated / metadata-only**. No PLE task has been downloaded, run, independently regraded or scored by Luminous in this work. No runner adapter is qualified. No SOTA, engineering competence, licensure, code-compliance, safety or physical-validation claim follows from the registry record.
