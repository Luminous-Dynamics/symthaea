# Prediction-Advantage Tournament: Upstream Pins and Reproduction Contract (v0.1)

**Tracking:** #13389 — BENCH-PREDICTION-ADVANTAGE-001  
**Causal pairing / information filtration:** #13375 / PR #13377  
**Decision-value interaction:** #13279 / PR #13281  
**Tournament protocol:** #13216 / PR #13282  
**Statistical inference:** #8578  
**Status:** source revisions and key protocol files pinned; no benchmark code, assets, model checkpoints or evaluation episodes have been downloaded or executed by this contribution.

## 1. Why two benchmarks

RoboDyna and LIBERO-Dynamic are complementary testbeds, not one leaderboard.

- **RoboDyna** tests broader physically grounded dynamics: 30 tasks, five capability categories, compositional difficulty, household tasks and paired synchronous/asynchronous evaluation on the same seeds.
- **LIBERO-Dynamic** supplies a more controlled latency/motion experiment across all 40 LIBERO tasks, three motion families, three target speeds and a fixed 1,800-episode population per configuration.

A result from one must not be reported as a score on the other. Their action spaces, task sets, checkpoints, data, training and runtime environments differ.

## 2. Pinned upstream source revisions

The following are the exact repository revisions retrieved during this source audit. These pins make the audit reproducible; they do not imply that the runtime assets or model checkpoints have been acquired.

### RoboDyna

- Repository: https://github.com/RoboDyna/robodyna
- Pinned repository commit: `066e11e01e983ffa9238ee295e6d50935d321b12`
- Commit: https://github.com/RoboDyna/robodyna/commit/066e11e01e983ffa9238ee295e6d50935d321b12
- README blob: `801feecb677702e08c96dcdd318f7dae358577c9`
- Fixed seed manifest: `task_config/eval_seeds.yml`, blob `0a43c299d26cfa6e35dede73cd62759087448128`
- Per-task rollout limits: `task_config/eval_step_limits.yml`, blob `80558be25826945d48328e7b06f397de83b19645`
- Official launchers:
  - π0.5: `scripts/eval/run_pi05_eval.sh`, blob `724122893e82181e8d9e36c38cfb9c7a46c77b73`
  - X-VLA: `scripts/eval/run_xvla_eval.sh`, blob `2cf7a83ae4fa5839c7c96050d0bd85cb768f4703`
  - FastWAM: `scripts/eval/run_fastwam_eval.sh`, blob `5a74b9a4e57876e4a5398df6e17a0216f057b623`

Pinned protocol details read from those exact files:

1. The seed manifest declares version 1 and ten fixed seeds per task/scenario slot. The README states that the same fixed seeds are used for human experiments and policy evaluation.
2. Base conceptual tasks use `default`, `opt1`, `opt2`, `opt1+2`; household tasks have one scenario each.
3. The rollout step limit is 1.2 times the longest published expert episode for that task. A policy that has not completed within the limit is a failed episode.
4. One policy step is one task action at 16.667 Hz.
5. `frozen` holds the world still during inference; `async` charges measured inference wall time to simulation time before the action is applied. The two modes must remain paired on the same seed and reported separately.
6. The launchers set `PYTHONHASHSEED=0`, use the fixed seed manifest and share the same step-limit file. The π0.5 launcher defaults to checkpoint ID 25000; X-VLA to 30000; FastWAM to 13170. These are launcher defaults, not proof that every paper table uses those exact model artifacts; the checkpoint and configuration identities must be verified against the published result row before reproduction.
7. The FastWAM launcher defaults to 24 executed actions per inference. At 16.667 Hz this commits about 1.44 seconds of open-loop actions. This is a meaningful confounder for a prediction-vs-reactive comparison and must remain an explicit action-horizon stratum.
8. The FastWAM launcher notes that in-process inference needs roughly 28 GiB of VRAM; split inference is supported through a separate model server. Do not silently substitute a different model or checkpoint because of hardware constraints.
9. The README lists Linux, an NVIDIA CUDA-capable driver, Python 3.10, Vulkan and FFmpeg; the asset downloader retrieves about 1.2 GiB of runtime assets.

### LIBERO-Dynamic paper/evaluation repository

- Repository: https://github.com/renshaojie233/LIBERO-Dynamic
- Pinned repository commit: `fd45961603cc50debaa959be52116626bc0fda1f`
- Commit: https://github.com/renshaojie233/LIBERO-Dynamic/commit/fd45961603cc50debaa959be52116626bc0fda1f
- README blob: `53c4243f98f79a6d4de000dd870bf5efac4a6993`
- Formal evaluation protocol: `docs/formal_dynamic_eval.md`, blob `4feb7f5155850e1221e20c968974f845cad29d30`
- Reproducibility notes: `docs/reproducibility_notes.md`, blob `daec93b026aab442dee26ed48cf25bd2106184b9`
- Async π0.5 launcher: `examples/evaluate_pi05_pytorch_async.py`, blob `2ca74da9f5fbf10507eb9443a8cf787063b43a76`

Pinned protocol details:

1. Forty LIBERO tasks × three motion families × five reviewed trajectories × three target speeds gives 1,800 episodes per fixed configuration.
2. Target speeds are 0.03, 0.06 and 0.09 m/s. The object is moved through contact by a carrier; it is not directly teleported.
3. The paper README reports model-specific calibrated inference latency while the scene advances. Its policy results and execution strategy results must be retained as distinct comparisons.
4. The formal-evaluation document describes a 20 Hz action clock, with continuous MuJoCo time advanced for synchronous inference latency while the robot joints are held. Its latency source supports a fixed calibrated value, measured wall-clock diagnosis, or a declared empirical sample distribution. Formal comparisons must use the premeasured fixed value from the official calibration procedure.
5. The reproducibility notes declare Python 3.10, robosuite 1.4.0, MuJoCo 2.3.7, PyTorch 2.13.0+cu130, CUDA user-space packages 13.0.3 and EGL rendering. They describe a validation host with RTX 5090 and 580.126.18 driver. Those are reproduction environment details, not a guarantee that the same environment is available here.
6. The notes declare a separate baseline source repository, `https://github.com/renshaojie233/libero_dynamic_base.git`, and source commit `f1a86422d738b1f5759ad871f7e341d99c88f7c9`. The exact source tree could not be retrieved through the connected GitHub source API during this audit (404 / no repository search result). It is therefore recorded as a publisher-declared pin, **not independently verified**. The formal baseline must remain blocked until that exact source revision is obtained from an authoritative accessible copy and its tree/commit is verified.
7. The async launcher requires explicit checkpoint and tokenizer paths, supports an action-chunk size, and starts a policy-server / evaluator split. The official checkpoint and tokenizer files still need publisher checksums and byte-level identity recorded.

## 3. Official raw-output contracts

### RoboDyna

Additional implementation pins at the RoboDyna commit above:

- `script/eval_metrics.py`: blob `473385e6574de4f6fcc716954559e9111dfd130c`
- `script/eval_policy.py`: blob `41c781dbbe23445d5147249661f05a2d8708f3fa`
- `script/evaluation_protocol.py`: blob `f6618cf84ececc6edf83e611dc6c05cf8eac65c3`
- `envs/_base_task.py`: blob `45890d0e8adc180d0e89f3f32d049525db1bb817`

The per-episode `EpisodeMetrics` record contains:
- `episode_id`, fixed `seed`, binary `success`;
- `steps_taken` and `max_steps`;
- `route_completion`, `manipulation_score`, task-defined `partial_score`;
- `total_penalty_factor`, typed `penalty_events`, `fail_reason`;
- `metric_detail`, including simulation time, wall time, total steps, option label and optional task/partial-score detail.

The aggregate summary separately reports total episodes, success count/rate, manipulation-score mean/std, route-completion mean/std, partial-score count/mean/std, simulation and wall-time mean/std, total-step mean/std and penalty-event totals. Preserve the underlying episode records: aggregate means alone cannot reconstruct per-seed sync/async flips or tail failures.

The evaluator journals each completed seed to an append-only JSONL cache and flushes/fsyncs the record before starting the next episode. On resume it reuses only cached seeds that still occur in the frozen seed manifest; stale seed rows are dropped. Keep the cache root, cache file digest and seed manifest together with the final outputs. A run should use a fresh cache or an explicitly named resume lineage; never merge results from different source commits, checkpoints or timing modes.

The official launcher defaults `EVAL_EXPERT_GATE=0`. The expert pass still creates the dynamic-motion trajectory that the policy episode replays. With the gate disabled, a first replayable expert scene can be used even when the expert's task-success verdict fails; enabling the gate can retry up to three times and select a different expert trajectory. This is not a harmless evaluation toggle: it changes the exogenous scene history. Reproduce the launcher default for the official baseline and record the effective environment value. `EVAL_SEEDS_OVERRIDE` is diagnostic-only and must not be present in confirmatory runs.

The async evaluator wraps `model.get_action()` in the environment inference window. In `async` mode, measured inference duration advances simulation time before the action lands; in `frozen` mode the world pauses. The evaluator's protocol summary reports inference call count, episodes, mean charged milliseconds, advanced simulation steps and simulation seconds. Preserve both those values and each episode's `total_time_sim_s` / `total_time_wall_s`; do not use wall time as a substitute for charged simulation time.

### LIBERO-Dynamic

Additional raw-output pins at the LIBERO-Dynamic commit above:

- `scripts/run_pi05_sync_eval.py`: blob `4fe2611481ba1a54d0d1d60a40451a37c940a345`
- `scripts/eval_dynamic_formal.py`: blob `76fc3c485fdb62ee3b826922dfc47ac96e133dbf`
- `scripts/run_pi05_async_eval.py`: blob `ccdfd6620e83a4d19a804bfd8ab95632b3a14a6b`
- `examples/_common.py`: blob `66902337d99e3f3b23113dba9c8f24b9d8c3af9f`
- `results/README.md`: blob `cab10da651a939807a73304bf8e8b7c240768e67`
- `results/cross_model_async.csv`: blob `a6b918176f731494b9a07d2a669781738ee167b3`
- `results/execution_strategy_best_points.csv`: blob `d4bf6185f583b34a4389959412dae938e35fc5fe`
- `results/dynamic_finetuning_ksweep.csv`: blob `3cee2350c2fdfa3a49d7def1151470d8f1ffcb9b`
- `results/real_robot_results_by_k.csv`: blob `65086a79f7ab6d2a6e45f7d2830f4a367d2c2440`

The formal-evaluation document says each run emits a per-item manifest and aggregate `summary.json`. It lists raw task success, platform-path and object-follow error, tilt, relative drop, visibility and diagnostic flags, plus queue underflow, chunk count, policy requests, calibrated simulation latency, wall-clock policy latency and RTC requested/supported/used state. It also records `action_l2` and `action_delta_l2`; the optional step trace retains per-control-step values, chunk boundaries, poses and contacts. Treat task success, infrastructure health and strict physics/visibility QA as separate outcomes.

The async π0.5 wrapper fixes `control-mode=async_queue`, `prefetch-threshold=2`, `chunk-size-threshold=0.5` and `empty-action=hold`. The synchronous formal wrapper fixes `control-mode=sync_hold`, `prefetch-threshold=0` and `empty-action=zero`. These are different execution strategies, not interchangeable implementation details. Formal comparisons use the fixed calibrated latency source; measured-wallclock mode is documented as diagnosis only.

The pinned machine-readable result files contain 1,800 rows per model/configuration (600 at each speed). They must be ingested with their counts, horizon (K), speed strata and success counts, not copied as rounded percentages only. The strategy table contains five distinct references:
- Async Queue: 871/1,800 = 48.39%, (K=6), latency 147.2 ms.
- Measured-latency Sync Hold: 775/1,800 = 43.06%, (K=10), latency 147.2 ms.
- RTC: 681/1,800 = 37.83%, (K=16), latency 169.8 ms.
- Sync Zero Latency: 907/1,800 = 50.39%, (K=8), zero latency. This is an optimistic diagnostic upper-control, not a deployable real-time comparison.
- VLASH continuous-world: 998/1,800 = 55.44%, (K=4), latency 104.0 ms. VLASH is separately trained and is an end-to-end system reference, not a same-checkpoint controller ablation.

## 4. Reported results to reproduce (not Symthaea results)

RoboDyna's pinned README reports overall success of π0.5 18%, X-VLA 11% and FastWAM 6%, averaging synchronous and asynchronous protocols. It reports 5–12% paired episode outcome flips between sync and async even though the average success change is at most three percentage points.

LIBERO-Dynamic's pinned cross-model CSV reports async best-measured-horizon success of π0.5 (JAX) 57.83% (1,041/1,800), FastWAM-Joint 40.61% (731/1,800), VLA-JEPA 35.72% (643/1,800), FastWAM 32.06% (577/1,800), π0 (JAX) 31.50% (567/1,800) and OpenVLA-OFT 27.28% (491/1,800). Its π0.5 strategy CSV includes Async Queue 48.39%, measured-latency Sync Hold 43.06%, RTC 37.83% and Sync Zero Latency 50.39%; the last is an optimistic diagnostic upper-control, not deployable. VLASH is 55.44% but is separately trained and must be labelled an end-to-end system reference, not a controller-only ablation.

These values are source-reported result-table values, not reproduced results from this Symthaea campaign. Do not pool them or compare their percentages across benchmarks.

## 5. Execution-readiness checklist

Before adding a Symthaea adapter:

- [ ] Verify exact repository commit and clean working tree for each benchmark.
- [ ] Download runtime assets and model checkpoints; record official archive/checkpoint checksums and licences.
- [ ] Reproduce one policy-free physics/infrastructure smoke test on the pinned environment.
- [ ] Reproduce one official baseline configuration and compare its raw per-episode results to the published aggregate.
- [ ] Preserve official fixed seeds, task/scenario splits and rollout limits.
- [ ] Record per-model inference latency calibration, device, driver, GPU clocks/thermal state and runtime mode.
- [ ] Capture raw per-episode action, delivery, inference and environment-time traces.
- [ ] Verify paired sync/async episode IDs and transition counts, not just aggregate success.
- [ ] Reproduce the official action horizon and queue-underflow semantics for each model.
- [ ] Record model server and simulator environments separately where official launchers use split execution.
- [ ] Keep all failures, expert-gate anomalies, timeouts, missed deadlines and incomplete episodes in the denominator.

For LIBERO-Dynamic, also verify the declared `libero_dynamic_base` source tree and commit before claiming formal baseline reproduction.

## 6. Symthaea comparison design

Within each compatible benchmark and task stratum, compare frozen direct VLA/reactive policies and frozen predictive world/world-action policies. Add the same Symthaea decision layer to each as a separately labelled arm. Within each frozen policy family, reuse the four #13281 treatments (baseline, compute-only, observation-only, joint). Do not change the official baseline checkpoint, scoring, task population, latency calibration, action domain or queue semantics.

The #13377 causal pairing certificate must bind each arm's action-conditioned trajectory and decision-time event filtration. Its combined `validate_causal_pairing_with_interaction` entrypoint must join the exact scored action/time or terminal non-application disposition to the #13281 regret outcome.

Use #8578 for independent-cluster inference, #13176 for the existing resource/exposure decomposition, and #9437/#8715 for generic evaluator isolation and canonical evidence receipts. No additional factorial engine, world simulator or signing scheme is introduced here.

## 7. Claim boundary

This document pins source code and protocol artifacts only. It does not mean that either benchmark was installed, its assets downloaded, its model checkpoints acquired, or its results reproduced. No Symthaea performance, physical, latency, energy or SOTA improvement is claimed. The inaccessible LIBERO base-source revision is an explicit unresolved reproduction dependency, not an assumed success.
