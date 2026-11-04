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

## PLE-to-Luminous evidence mapping

**Do not force PLE deliverables into CEB's patch-centric v0 receipt.** The current CEB v0 schema requires a candidate patch, reference-patch hash and source/result tree subjects. PLE grades engineering deliverables such as calculations, technical reports, drawings, CAD/BIM artifacts and schedules; a source patch is not the semantic unit of result. Treating a PLE deliverable as a patch would misstate what was evaluated.

The adapter boundary must be resolved through the existing **benchmark-neutral run-accounting contract #10314** and **evidence-convergence issue #10370**. CEB #10115 / DSSE #10167/#10212 remains authoritative for Symthaea code-change tasks that actually have source patches. Do not add a third receipt schema or alter the meaning of the CEB patch subject.

| PLE evidence | Canonical Luminous mapping | Invariant |
|---|---|---|
| Official benchmark/split/task universe | Generic benchmark identity + complete task-universe commitment (#10314) | Every eligible PLE task has exactly one terminal disposition per run |
| Dataset revision + PLE task ID + task bundle bytes | Task-row input identity and digest | Record the Hugging Face dataset commit and exact task ID; compute a separate digest over the permitted materialized task bundle |
| Agent/harness/model/configuration | Candidate/agent/harness identity in the generic run record | Pin model, provider, effort, prompt, harness, tool policy and retry policy |
| Task/serve/verifier images and Harbor/runner | Environment/execution identity | Pin immutable image digests, official PLE runner commit, Harbor 0.20.0, platform and resource limits |
| Official `checks`, `rubric`, `checks-judge` grader | Grader identity, grading protocol and raw grader output digest | Use official verifier unchanged; preserve the grader type and judge model/revision |
| Deliverable files, trajectory, logs and resource data | Evidence-artifact inventory | Hash every artifact; do not invent a patch hash for a report, drawing or STEP file |
| Official per-task reward and hard-check outcomes | Official score plus separately derived coverage/yield/feasibility metrics | Never replace official score with a local approximation; never average away hard-gate failure |
| Three repeated runs | Distinct run IDs linked to one benchmark configuration | Do not confuse `--trials` parallelism or evaluator retries with independent benchmark repetitions |

The adapter may use a small **benchmark-profile mapping** to project PLE task metadata into generic accounting fields. That projection must not duplicate PLE task prompts, tests, graders or hidden labels, and it must not change official scoring semantics. A CEB task ID may identify the Luminous adapter/profile work, but the external PLE task ID remains a separate source identifier.

Before execution, #10370 must decide normative ownership for benchmark/task identity, accounting rows, artifact digests, official score provenance, and interaction with CEB's DSSE receipts. If those contracts are not yet reconciled, the PLE score remains an external source observation attached to its own official artifacts—not a qualified CEB run.

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
