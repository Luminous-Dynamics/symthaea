# SYM-ARCH-001 Result — 2026-08-25

**Status:** frozen result record  
**Preregistered verdict:** **NEGATIVE**  
**Experiment PR:** #53  
**Merged to `main`:** `6a25ea6d5faccabff2351d577a808d2cb36eb35b`

## Executive result

SYM-ARCH-001 produced its first valid confirmatory observation on 2026-08-25.

Under the decision rule frozen before behavioral results were observed, the benchmark-local `hdc_ltc_hebbian` candidate received a **NEGATIVE** verdict.

The candidate did not beat the strongest control by the preregistered `0.05` margin on any of the three target metrics. It regressed beyond the `0.05` tolerance on all three:

- retention delta vs strongest control: `-0.296875`;
- held-out compositional delta vs strongest control: `-0.25390625`;
- reversal-final delta vs strongest control: `-0.486328125`;
- target wins at margin: `0`;
- target regressions beyond tolerance: `3`.

Mean forgetting for the candidate was `0.04296875`, compared with best-control forgetting `0.02604167`. This difference remained inside the frozen forgetting tolerance and was **not** the condition that caused the negative verdict.

The correct conclusion is narrow:

> The tested 512-dimensional benchmark-local HDC-LTC-Hebbian bundle did not outperform the preregistered controls on the joint retention/composition/reversal criterion under the frozen SYM-ARCH-001 protocol.

This is not evidence that HDC, LTC/CfC, Hebbian plasticity, or Symthaea as a whole are ineffective.

## Evidence provenance

The first hosted attempt failed before behavioral execution because an unrelated psych-bench test fixture still supplied the removed `LoopTrialResult::cycle_reward` field. No behavioral result from that attempt was observed or used to alter the experiment.

The only repair before the valid retry removed that stale synthetic fixture field/comment:

- frozen scientific head before repair: `e7406955fa3797cc8293df6af6fbd5113978b7f9`;
- retry PR head: `eb931fbaed960f066406caf969c5aaad350c9d64`;
- repair delta: one unrelated file, `0` additions / `5` deletions;
- no changes to the benchmark implementation, preregistration, thresholds, seeds, task splits, timing, agents, or decision rule.

The valid run was GitHub Actions run `32825418156`, attempt/run number 3. Every experiment step completed successfully:

1. experiment unit tests;
2. release campaign execution;
3. report-schema/evidence validation;
4. SHA-256 generation;
5. artifact upload.

GitHub Actions evaluated synthetic pull-request merge commit:

`60165a666eb6e5a358445acf3f087a7775e38f0b`

whose commit message is:

`Merge eb931fbaed960f066406caf969c5aaad350c9d64 into 8de0ca10e69c2da42844fd7a202e639bf21e32bc`

A commit comparison from PR head `eb931fba...` to tested merge commit `60165a...` contains **zero file differences**. Thus the tested working tree is equivalent to the retry PR head.

### Artifact integrity

Uploaded artifact:

- artifact id: `9557446155`;
- artifact name: `sym-arch-001-60165a666eb6e5a358445acf3f087a7775e38f0b`;
- artifact ZIP SHA-256 reported by GitHub: `7f8424e2285f0f020b2bed07f7b716caafa10e547e3a3bf1837e307c7fdc4684`;
- `report.json` SHA-256: `841b78ffdd8e63b801d54c667f12ff6ab4db00abe88ec1403c74544500901985`.

The downloaded `SHA256SUMS` entry matched an independent recomputation of `report.json`.

## Frozen configuration

| Parameter | Value |
|---|---:|
| representation dimension | 512 |
| independent RNG seeds in v1 | 16 |
| train epochs per world | 16 |
| reversal epochs | 12 |
| prototype alpha | 0.15 |
| Hebbian learning rate | 0.002 |
| target win margin | 0.05 |
| regression tolerance | 0.05 |
| worlds | 4 |
| categorical values per factor | 4 |
| held-out combinations per world | 4 |
| observations per agent | 960 |

The world identity is explicitly encoded in the input. HDC-LTC evolution uses fixed `dt = 0.05`. Reversal is a contingency inversion, not a causal-intervention benchmark.

## Agent results

The intervals below are the normal-approximation 95% intervals emitted by the frozen v1 implementation. They are retained exactly as part of the result record; they are not upgraded post hoc to the stronger paired/hierarchical statistics designed for SYM-ARCH-002.

| Agent | Retention | Composition | Forgetting ↓ | Reversal final | Reversal latency ↓ |
|---|---:|---:|---:|---:|---:|
| `linear_sgd` | 0.9284 | 0.8008 | 0.0260 | 0.8086 | 175.8 |
| `vanilla_hdc` | 0.6120 | 0.6445 | 0.3307 | **0.9961** | **48.1** |
| `fixed_diagonal_ssm` | 0.5990 | 0.5938 | 0.2305 | 0.7617 | 54.3 |
| `hdc_ltc_frozen` | 0.6445 | 0.6133 | 0.0443 | 0.4219 | 184.1 |
| `hdc_ltc_hebbian` | 0.6315 | 0.5469 | 0.0430 | 0.5098 | 170.4 |

### Emitted 95% intervals

| Agent | Retention 95% CI | Composition 95% CI | Forgetting 95% CI | Reversal 95% CI | Latency 95% CI |
|---|---|---|---|---|---|
| `linear_sgd` | [0.9179, 0.9389] | [0.7707, 0.8308] | [0.0158, 0.0363] | [0.7206, 0.8966] | [168.2, 183.4] |
| `vanilla_hdc` | [0.5778, 0.6461] | [0.5714, 0.7176] | [0.2937, 0.3678] | [0.9909, 1.0013] | [46.2, 50.0] |
| `fixed_diagonal_ssm` | [0.5829, 0.6150] | [0.5621, 0.6254] | [0.2059, 0.2550] | [0.7302, 0.7933] | [46.7, 61.8] |
| `hdc_ltc_frozen` | [0.5910, 0.6981] | [0.5498, 0.6768] | [0.0272, 0.0613] | [0.3369, 0.5069] | [166.5, 201.6] |
| `hdc_ltc_hebbian` | [0.5784, 0.6846] | [0.4904, 0.6033] | [0.0269, 0.0590] | [0.4209, 0.5987] | [145.6, 195.3] |

The `vanilla_hdc` normal interval extending slightly above `1.0` is a useful limitation of v1's unconstrained normal approximation. The result is left unchanged. SYM-ARCH-002 uses stronger paired/hierarchical methods rather than retroactively changing v1 statistics.

## What the negative result actually teaches us

### 1. The bundled HDC-LTC-Hebbian candidate is not justified by this task

The candidate lost badly to the strongest observed controls on all three target metrics. There is no basis for carrying the entire 001 bundle forward as a favored architecture and tuning it until it wins.

Future work should factorize the mechanisms and make each one earn its complexity independently.

### 2. `linear_sgd` is a serious control, not a ceremonial baseline

`linear_sgd` dominated final retention and held-out composition in this task family:

- retention `0.9284`;
- composition `0.8008`;
- forgetting `0.0260`.

That is a warning against assuming a cognitively elaborate architecture must beat a simple online learner on a small explicitly contextualized relational task.

SYM-ARCH-002B therefore begins with stronger simple controls before adding more sophisticated neural/SSM models.

### 3. Vanilla HDC shows a striking stability/plasticity tradeoff

`vanilla_hdc` had weak retention and severe forgetting but almost perfect final reversal performance:

- retention `0.6120`;
- forgetting `0.3307`;
- reversal final `0.9961`;
- reversal latency `48.1`.

This does not prove a general HDC property because the result includes the benchmark-local prototype update/readout. It does show that v1 contains qualitatively different behavioral regimes that a single aggregate score would hide.

### 4. Low forgetting is not sufficient evidence of good continual learning

Both liquid variants showed low mean forgetting (`~0.043–0.044`) while their absolute retention/composition remained modest.

That exposes an important ambiguity:

> A system can appear stable because it retains knowledge, or because it never acquired the task strongly enough for a large peak-to-final decline to occur.

This is precisely why SYM-ARCH-002 adds the full task-by-time performance matrix, acquisition speed, average incremental accuracy, forward/backward transfer, and learning-curve measures. The need for those metrics is now empirically motivated by 001 rather than merely methodological preference.

## Strictly post-hoc mechanism observation

The following comparison was **not** the preregistered primary decision and must not be reported as confirmatory evidence.

Comparing `hdc_ltc_hebbian` against its closest frozen-liquid ablation, `hdc_ltc_frozen`, gives mean directional differences across the same 16 seeds:

| Metric | Hebbian − frozen |
|---|---:|
| retention | -0.0130 |
| composition | -0.0664 |
| forgetting | -0.0013 |
| reversal final | +0.0879 |
| reversal latency | -13.6 trials |

Directionally, Hebbian adaptation improved reversal behavior while composition worsened. No paired preregistered inference was defined for this secondary contrast, so this is an **exploratory hypothesis generator only**.

A valid follow-up is not to tune the existing candidate. It is to preregister a mechanism-factorized comparison where representation, temporal dynamics, plasticity, and readout are controlled separately.

## Limitations frozen with the result

SYM-ARCH-001 is a mechanism-level synthetic benchmark, not a full Symthaea or frontier-model evaluation.

Important limitations include:

- only four hand-authored relation rules;
- 16 RNG seeds do not represent 16 independent task programs;
- explicit world/context identity in every encoded input;
- fixed `dt = 0.05`, so the central irregular-time motivation of CfC/LTC is not exercised;
- only four held-out combinations per world;
- deterministic held-out selection;
- prototype readout is not matched against every representation/control;
- strongest control is selected separately by metric;
- normal-approximation intervals rather than paired hierarchical inference;
- no prospective power calculation;
- no task-free drift;
- no resource-matched neural baseline;
- reversal is contingency inversion rather than causal intervention;
- the old live reward-consumption path is intentionally bypassed.

These are reasons to design a stronger next experiment, not reasons to reinterpret the frozen result.

## Consequence for SYM-ARCH-002

The result strengthens the rationale for the 002 program already tracked in issue #55:

1. preserve DEV / CONFIRM / REPL separation;
2. generate genuinely different environments rather than treating RNG seeds as independent cognitive tasks;
3. measure full acquisition/retention dynamics;
4. use paired environment-level inference and prospective power;
5. validate generated benchmarks before scoring architectures;
6. attack benchmarks with shortcut controls;
7. compare fixed random features, HDC, and stronger conventional learners using matched readouts/resources;
8. isolate HDC representation, liquid dynamics, adaptive timescales, Hebbian plasticity, and associative memory separately;
9. test latent/task-free context and irregular physical time where liquid dynamics should have a theoretically specific advantage;
10. reserve causal claims for intervention-based mechanism tests and independent replication.

The purpose of 002 is therefore not to rescue the 001 candidate. It is to discover which, if any, Symthaea mechanisms earn their complexity under harder controls.

## Claim ceiling

This document supports the statement:

> SYM-ARCH-001 produced a valid preregistered **NEGATIVE** result for the tested benchmark-local HDC-LTC-Hebbian candidate under the frozen 2026-08-24 protocol.

It does not support claims that:

- Symthaea as a whole failed;
- HDC is inferior in general;
- LTC/CfC is ineffective;
- Hebbian plasticity is harmful in general;
- linear SGD is a superior general cognitive architecture;
- vanilla HDC is generally optimal for reversal learning.

Those questions require the mechanism-specific, task-diverse, resource-aware follow-up program.