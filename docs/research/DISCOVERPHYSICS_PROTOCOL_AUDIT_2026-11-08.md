# DiscoverPhysics protocol audit — 2026-11-08

Status: **source-audited, reproduction blocked on protocol parity**  
Upstream repository: [SampsonML/DiscoverPhysics](https://github.com/SampsonML/DiscoverPhysics)  
Pinned upstream main commit: 33b7fa9df96de9c35744efd181ca7e5a8dd60ad5  
GitHub signature status: verified; repository licence: MIT.

This document records what the pinned public source actually implements. It does not claim an executed reproduction, access to private benchmark tasks, or a Symthaea result.

Machine-readable companion receipt: [DISCOVERPHYSICS_PROTOCOL_AUDIT_V0_2026-11-08.json](DISCOVERPHYSICS_PROTOCOL_AUDIT_V0_2026-11-08.json) and [schema](DISCOVERPHYSICS_PROTOCOL_AUDIT_V0_2026-11-08.schema.json). The receipt stores exact Git blob SHAs for each audited source file and sets `benchmark_claim_allowed` to false until parity is resolved.

## 1. Protocol entry point and output contract

The documented benchmark path is **scripts/run_benchmark.py**, configured by YAML. Each trial starts a DiscoverPhysics science agent in an altered simulated world. The agent iteratively designs experiments, observes particle trajectories, then submits both:
- an executable Python law / force implementation, evaluated on held-out trajectories;
- a natural-language explanation, scored independently by an LLM judge against a world-specific ground-truth explanation/rubric.

The primary runner defaults to a maximum of 16 experimental rounds in the README's example configuration. The explanation judge defaults to claude-opus-4-6 in the public config and evaluator constructor. The README describes Python 3.9+ and JAX prerequisites.

Symthaea's adapter must preserve this dual-output construct. A typed PhysicsIR hypothesis may be canonical internally, but the official boundary still needs a safe deterministic projection to the benchmark's Python-law interface. Never execute arbitrary agent-generated Python in a privileged evaluator; use a separately restricted process/container and allowlisted AST/code generation.

## 2. Three distinct scoring protocols exist in the pinned repository

| Source | Normalized trajectory error | Explanation score | Interpretation |
|---|---:|---:|---|
| scripts/run_benchmark.py | mean_pos_error / Var(GT_world) < 0.1 | >= 0.75 | Primary YAML runner documented by README |
| scripts/run_stats.py | mean_pos_error / Var(GT_world) < 0.1 | >= 0.75 | Main-path analysis and pass@k aggregation |
| scripts/yml_benchmark.py | geometric mean of raw mean_pos_error < 0.5 | >= 0.7 | Legacy runner; not the current README's primary command |
| Public leaderboard page | normalized MSE < 0.1 | >= 0.9 | Public leaderboard description; not present in the pinned public primary runner |

The README's batch-benchmarking section explicitly directs users to scripts/run_benchmark.py and repeats the 0.1 / 0.75 criteria. The pinned public source contains no 0.9 pass threshold. The leaderboard's 0.9 threshold may reflect a different scorer revision or evaluator configuration, but the pinned source does not establish that explanation. Treat the leaderboard and public code as **not parity-equivalent** until the benchmark owners supply the scorer revision/judge contract that generated the leaderboard.

Do not pick the more favourable threshold and label it official. The public-source reproduction should be named and reported separately from a leaderboard comparison.

## 3. World inventory is internally inconsistent

| Source artifact | Inventory exposed at pinned commit | Observation |
|---|---:|---|
| ScienceAgent/scienceagent/worlds.py WORLDS dictionary | 12 registered keys | gravity, yukawa, fractional, wave, oscillator, extra_dimensions, dark_matter, three_species, ether, hubble, circle, coulomb_easy |
| README “Predefined Worlds” table | 11 rows | Lists 11; omits wave |
| README ready-to-run YAML snippet | 11 world names | A smaller 11-world example using two seeds [0, 1] |
| Checked-in configs/bench.yml | 13 names | Includes running_coupling and force_geography, which are not keys in the current WORLDS mapping; get_world() rejects unregistered names |
| Primary runner scripts/run_benchmark.py WORLD_VARS | 21 variance keys | Contains more names than the current 12-key world registry |
| Analysis scripts/run_stats.py | 22 denominator | Hard-pins the production denominator to 22 and comments “11 public + 11 private” |

The pinned public files do not provide an authoritative mapping of exactly 11 private worlds, the corresponding variance constants and the seed ledger. The public registry/config/variance table/leaderboard denominator therefore cannot be reconciled from the public tree alone. Do not infer private-world names from variance-table entries or silently count all registry keys as public.

The README's 11-world sample appears to consist of names that are actually registered. The checked-in 13-name config is not a reliable public task manifest as written because two names are absent from the current registry. A clean-run dry test of the official config should be part of the protocol audit, but should not be run with live model credentials or private data until the exact allowed resource/cost policy is pinned.

## 4. Pass@k and E@k aggregation semantics

The pinned scripts/run_stats.py implementation specifies:

- Seed-pool size is read from the archived run config.yml; missing results count as failures for their intended seed positions.
- @k: a world is counted as passed if at least one of seed indices 0..k-1 passes the per-trial thresholds. Missing seed slots count as failures.
- E@k: draw K distinct seed positions without replacement from the configured seed pool; estimate expected worlds passed over 1,000 Monte Carlo draws using fixed RNG seed 0.
- E@k counts are converted to percentages using the hard-pinned denominator of 22 worlds, not the number of worlds present in the current results directory.

A public sample config with seeds [0, 1] cannot by itself reproduce an @5 result. A public subset must not be divided by the 22-world denominator and represented as the leaderboard's full-suite score.

## 5. Required protocol-parity receipt

Before comparing Symthaea with the published leaderboard, capture and pin:

1. Exact runner/scorer commit and the explanation threshold actually used for the leaderboard.
2. Exact world manifest with public/private flags, per-world task revisions and all 22 variance constants.
3. Exact intended seed pool per world and the @k / E@k aggregation configuration.
4. Judge model, prompt, per-world rubrics, parsing rules, missing/failed judge policy and raw judge outputs.
5. Normalized-MSE definition, per-world variance constants and any trajectory filtering/held-out-split details.
6. Experiment round cap, noise configuration, critic/supervisor treatment, model access, tool permissions, token/compute/wall-clock budgets.
7. Public/private access rights and private-ledger custody.
8. The official baseline's exact code, prompt, model version and complete attempted-run denominator.

A protocol audit receipt must be content-addressed to the exact upstream source files and config files. A public-only run must be labelled DiscoverPhysics-public-source@33b7fa9 (or a later exact source SHA) and list each included world/seed. It is a partial reproduction, not a full leaderboard replication.

## 6. Symthaea integration constraints

- Reuse #6879/#6880 ScientificModel / PhysicsIR for typed canonical hypotheses.
- Reuse the blind AgentView / AgentAction process contract in PR #11060 for bounded experiment actions and evidence history.
- Reuse #8286 for active intervention semantics, #8669 for equivalence/support scoring, #8705 for transfer and #8721/#11015 for tool allocation / prospective research action.
- Generate benchmark-specific Python only from an allowlisted AST/typed expression projection. Keep the generated source digest, model digest and projection-loss report together.
- Run the official JAX simulator/evaluator in a separate restricted process/container. The Rust microcosm is not a replacement for the official simulator.
- Keep trajectory score, explanation score, mechanism recovery, experiment efficiency, cost, failures and selective abstention separate.

## 7. Stretch target — only after parity

Only after the task and scorer contracts are reconciled and the official baseline is reproduced should Symthaea set a comparative target such as:
- at least +5 absolute percentage points on Pass@1 and Pass@5 under the same world pool, seed pool, scorer and baseline;
- no worsening of per-world normalized trajectory MSE or explanation score in protected structural families;
- lower inclusive cost to first independently valid law;
- no increase in false-unique mechanism claims, unsupported Python execution or evidence-integrity failures.

These are prospective hypotheses, not current results or industry standards. Beating a subset, using more rounds, changing the judge threshold or comparing against a legacy runner is not beating SOTA.

## 8. Pinned source links

- Repository and MIT licence: https://github.com/SampsonML/DiscoverPhysics/tree/33b7fa9df96de9c35744efd181ca7e5a8dd60ad5
- Primary runner and thresholds: https://github.com/SampsonML/DiscoverPhysics/blob/33b7fa9df96de9c35744efd181ca7e5a8dd60ad5/scripts/run_benchmark.py
- Pass@k / E@k: https://github.com/SampsonML/DiscoverPhysics/blob/33b7fa9df96de9c35744efd181ca7e5a8dd60ad5/scripts/run_stats.py
- Legacy runner: https://github.com/SampsonML/DiscoverPhysics/blob/33b7fa9df96de9c35744efd181ca7e5a8dd60ad5/scripts/yml_benchmark.py
- Public world registry: https://github.com/SampsonML/DiscoverPhysics/blob/33b7fa9df96de9c35744efd181ca7e5a8dd60ad5/ScienceAgent/scienceagent/worlds.py
- Checked-in config: https://github.com/SampsonML/DiscoverPhysics/blob/33b7fa9df96de9c35744efd181ca7e5a8dd60ad5/configs/bench.yml
- README's documented primary protocol: https://github.com/SampsonML/DiscoverPhysics/blob/33b7fa9df96de9c35744efd181ca7e5a8dd60ad5/README.md
- Public leaderboard: https://sampsonml.github.io/DiscoverPhysicsLeaderboard/
