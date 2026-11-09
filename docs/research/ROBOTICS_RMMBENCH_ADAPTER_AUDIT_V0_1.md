# RMMBench adapter source audit — v0.1

**Status:** source reconnaissance only; no benchmark execution or Luminous result  
**Audit date:** 2026-11-09  
**Parent work:** Symthaea #4781, #11231, #11275  
**Purpose:** freeze the facts and reproducibility risks that the first adapter must resolve before accepting a score.

## 1. Source identity inspected

| Item | Inspected identity | Interpretation |
|---|---|---|
| Upstream repository | https://github.com/mxxq-stack/RMMBench | Public project repository |
| Code revision | `11b0d16abe04e37ebe7cec9a59c3c1df7578657c` | Exact commit used for this source inspection; not yet a qualified benchmark release |
| Paper | https://arxiv.org/abs/2610.05414 | Abstract says 70 canonical scenarios |
| Project page | https://mxxq-stack.github.io/rmmbench-project/ | Current page says 90 canonical scenarios |
| Runner inspected | `scripts/vlm_eval.py` at the code revision above | Runner configuration and result semantics inspected |
| Requirements inspected | `requirements.txt` at the code revision above | Python package pins and one unpinned VCS dependency observed |

The 70-versus-90 discrepancy is unresolved. Do not silently pick the abstract or project-page number. The exact task/episode manifest, paper version, repository revision, and scene-asset revision must be reconciled to one released benchmark package before any aggregate is labelled comparable.

The runner and README refer to these files:

- `RMMBench/vlm_evaluation/configs/episodes/manipulation_episodes.json`
- `RMMBench/vlm_evaluation/configs/episodes/nav_episodes.json`

Those exact paths were not retrievable through the repository contents endpoint during this inspection. This is **not** proof that no manifest exists; the next source-pin step must establish whether they are generated, separately packaged, ignored, relocated, or missing from the inspected checkout. Until then, task-count and denominator qualification is blocked.

## 2. Runner facts observed

The inspected runner exposes materially distinct evaluation choices:

- VLM endpoint and backend name;
- history maximum length;
- maximum skills per episode;
- robot profile;
- GraspNet use;
- early stopping;
- visual navigation-ray assistance;
- collision feedback and whether reasoning is retained in history;
- seed, episode configuration, task subset and configuration index;
- serial or multiprocessing execution.

It has distinct navigation and manipulation episode-config paths. Without an episode config, the runner can also evaluate one task using random initialization. These are different protocol modes and must not be mixed.

The runner's resume identity list (`RUN_META_KEYS`) contains the episode-config **path**, backend label, task selection, seed, robot, GraspNet/early-stop/ray/collision switches, history length and skill limit. It does not include a digest of the episode-config contents, scene assets, the actual VLM endpoint/model revision, prompt/template contents, or the resolved dependency graph. A path or friendly backend label can remain unchanged while the underlying evaluation changes. A strict external receipt must bind immutable digests/versions for those inputs; credentials must never be included.

The runner snapshots run metadata into per-episode results, stores episode result JSON, records terminal `task_success`, and aggregates successful episodes over the evaluated work list. Exceptions can be downgraded into failed episode records. The adapter must preserve the full declared denominator and distinguish:
- terminal task success;
- partial condition/progress metrics;
- infrastructure/physics abort;
- task failure;
- missing or unexecuted episode;
- resumed result.

Do not infer terminal success from progress, a skill sequence, a controller receipt, or an API response.

## 3. Assistance and autonomy profile

The benchmark's claims are conditional on the actual observation/action and assistance stack. At minimum the adapter must capture:

| Channel | Evidence to bind |
|---|---|
| Vision input | Camera set, image/depth/point-cloud modalities, preprocessing and resolution |
| Navigation rays | Whether visual-ray assistance is enabled and the exact algorithm/configuration |
| Grasp proposals | Whether GraspNet is enabled, repository revision, checkpoint digest and proposal provenance |
| VLM | Provider, immutable model/revision where available, endpoint identity (not secret), inference parameters, prompt/template digest |
| Memory/context | History maximum length, retained fields, truncation policy and context representation |
| Skills | Skill library revision, action schema, max skill count, retry/early-stop policy |
| Human involvement | Human verification, correction, teleoperation, fallback or intervention events |
| Environment | Simulator version, robot/model, task manifest, scene assets, seed and randomized parameters |

The project page describes optional BEV input, GraspNet candidate generation, and a human-verified fallback. If any human-verified candidate or other oracle is used, the result must be reported as an assisted condition, not silently included in an unassisted autonomous score.

The upstream requirements pin dm-control and MuJoCo package versions, but include a VCS dependency for RRT algorithms without an immutable commit in the requirements line inspected. A reproduction lock must resolve and record that dependency commit and every downloaded model/asset checksum.

## 4. Adapter contract

The adapter belongs at the external-result import boundary; it must not put Python, MuJoCo, VLM or GraspNet dependencies into the Symthaea Rust production core.

Reuse existing owners:
- #4781 — external benchmark result import;
- #11231 — benchmark validity and transfer audit;
- #4800 — experiment/run identity and evidence envelope;
- #4866 — evidence-grade instrument/observation semantics where physical evidence is involved;
- #11275 — RMMBench-specific pinned adapter.

For each run, bind at least:
1. upstream code commit, paper version and benchmark release;
2. episode manifest and asset-manifest digests;
3. simulator, package and VCS dependency identities;
4. task ID, task family, config index, seed, randomized parameters and declared split;
5. model/checkpoint/provider revision, endpoint identity, prompt/configuration digest and resource budget;
6. all observation, assistance, skill and human-intervention channels;
7. per-episode terminal result, native metrics, failure/abort category and raw artifact reference;
8. the complete expected episode set, including missing, extra, duplicate and unexecuted episodes.

A source or protocol change creates a new evaluation identity. Never repair a mismatched result by editing metadata after execution.

## 5. Required hostile fixtures

The first Rust parser/validator should reject or explicitly downgrade:

- unreconciled 70/90 task-manifest disagreement;
- absent or changed task-manifest / asset digest;
- duplicate, missing, extra or out-of-order episode records;
- changed episode-config contents at the same path;
- changed VLM endpoint/model, prompt/template or skill library under an unchanged friendly label;
- GraspNet, BEV, visual-ray, privileged-state or human-fallback use omitted from metadata;
- human-verified fallback classified as fully autonomous;
- navigation progress presented as terminal manipulation success;
- partial task progress presented as full terminal completion;
- physics abort, infrastructure failure, task failure and unexecuted task collapsed into one undocumented class;
- resume results mixed across different protocol identities;
- credentials or secrets embedded in public receipts.

## 6. Benchmark validity and statistical reporting

Compose #11231's benchmark-entry diagnostics:
- shortcut-solvability probes;
- powered statistical comparison and confidence intervals;
- held-out scene/layout/object/configuration tests for creeping overfit;
- training-data and pretraining-source overlap disclosure.

Report navigation, manipulation, composite task completion, terminal success, progress, efficiency, intervention burden and resource cost as separate metrics. Use benchmark-native metrics first. Do not collapse RMMBench, RoboCasa365, RoboTwin 2.0 or physical NIST methods into a universal score.

For matched comparisons, freeze model budget, prompts, context/history, action skill library, observation modalities, number of episodes, seeds, task configuration, simulator/assets, and assistance. Pair seeds/configurations when possible and report per-task distributions, not just means.

## 7. Execution ladder

1. Resolve the 70/90 discrepancy and establish where the complete navigation and manipulation manifests live.
2. Pin upstream code, paper version, task manifests, assets, all dependencies and VLM/model configuration.
3. Export a minimal untouched upstream result corpus and preserve raw per-episode result files.
4. Implement a Rust-first parser and strict synthetic positive/negative fixtures; run tests on an exact commit.
5. Reproduce a published baseline under the exact pinned release and disclose every assistance channel.
6. Add controlled memory, observation, GraspNet, visual-ray and skill-budget ablations.
7. Add held-out layout/scene perturbations and paired statistical comparisons.
8. Only after the simulation adapter is qualified, define a separate transfer study; simulator success is not physical mobile-robot qualification.

## 8. Nonclaims

This audit does not claim that Luminous has run RMMBench, reproduced a paper result, resolved the task-count discrepancy, measured an improvement, established autonomous generality, or qualified a physical robot. It is a source-backed protocol audit to prevent those claims from being made prematurely.

## Sources inspected

- Paper: https://arxiv.org/abs/2610.05414
- Project page: https://mxxq-stack.github.io/rmmbench-project/
- README at inspected revision: https://github.com/mxxq-stack/RMMBench/blob/11b0d16abe04e37ebe7cec9a59c3c1df7578657c/README.md
- Runner at inspected revision: https://github.com/mxxq-stack/RMMBench/blob/11b0d16abe04e37ebe7cec9a59c3c1df7578657c/scripts/vlm_eval.py
- Requirements at inspected revision: https://github.com/mxxq-stack/RMMBench/blob/11b0d16abe04e37ebe7cec9a59c3c1df7578657c/requirements.txt
