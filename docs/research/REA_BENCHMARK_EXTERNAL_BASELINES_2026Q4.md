# Reverse-Engineering Benchmark Baselines and Comparability Matrix

**Snapshot date:** 2026-11-03  
**Purpose:** external reference points for Symthaea + REA evaluation  
**Status:** source-dated public baselines; no Luminous benchmark result is implied

## Executive conclusion

There is no single meaningful “SOTA reverse-engineering score.” The strongest public benchmarks ask different questions:

- **SRE-Bench:** can an agent reverse engineer realistic, held-out binaries across several domains?
- **AgentRE-Bench:** can an agent recover structured behavior and hidden details from unknown ELF/PE binaries under a fixed tool-call budget?
- **Decompile-Bench:** how closely does a decompiler reconstruct source-level function text, and can it be re-executed?
- **ReproBench:** can an agent start from a CVE identifier, identify and acquire the correct firmware, unpack it, localize the relevant binary, rehost the actual service and validate behavior?
- **OWASP MASVS/MASTG/MASWE:** does an application meet selected mobile security controls, and is there an appropriate test/evidence item for each claim?

These are complementary evaluation dimensions, not interchangeable leaderboards. The project should publish a dashboard of separate scorecards, not one weighted “reverse engineering intelligence” number.

## 1. Public external baselines

### SRE-Bench — realistic held-out agentic reverse engineering

**Official snapshot:** DAPLab page updated October 2026.

| Dataset/methodology | Publicly reported scope |
|---|---:|
| Private programs | 19 |
| Binary instances | 262 |
| Graded tasks | 1,572 |
| Tasks per binary | 6 |
| Domains | firmware, malware, game, file format, network protocol |
| Source languages | C, C++, Rust, Go |
| Anti-analysis mechanisms | 44 |
| Expert effort reported by authors | 5,000+ hours |

The public-facing results are scored over **graded instances**, not necessarily all eligible instances:

| Model | Graded | Mean score / 6 | Fully solved | Zero-credit |
|---|---:|---:|---:|---:|
| GPT-6-Astra | 181/262 (69.1%) | 5.55 | 149/181 (82.3%) | 7/181 (3.9%) |
| GPT-5.6-Sol | 254/262 (96.9%) | 3.69 | 80/254 (31.5%) | 41/254 (16.1%) |
| Claude-Fable-5.1 | 223/262 (85.1%) | 2.58 | 60/223 (26.9%) | 96/223 (43.0%) |

**Interpretation:** GPT-6-Astra's high conditional solve rate is impressive, but its graded coverage is materially lower than GPT-5.6-Sol's. Coverage, refusal and abstention must be reported beside the official conditional score. Never infer ungraded instances as passes or failures.

**Luminous evaluation plan:** obtain authorized access/official grading. Never rehost private SRE binaries, graders or reference answers. Report official score, graded coverage, eligible/attempted/refused/unsupported/timeout counts, costs and safety setting separately.

Sources:
- https://daplab.cs.columbia.edu/benchmarks/sre-bench/
- https://daplab.cs.columbia.edu/benchmarks

### AgentRE-Bench — long-horizon binary behavior reconstruction

**Official snapshot:** Version 3.1, October 2026.

The public benchmark describes 33 distinct binary challenges across Linux and Windows. Its V3.1 Windows PE evaluation contains 10 programs, each with symbols and stripped variants; five models, five attempts per artifact, and 500 total episodes. Each attempt uses a fixed static-analysis toolset and a 25-call budget.

| Model | Overall / 100 | Main / 100 | Bonus / 100 | Valid answers | Reported 95% interval |
|---|---:|---:|---:|---:|---:|
| Muse Wonder | 87.37 | 87.37 | 87.37 | 100/100 | 85.18–89.56 |
| GPT-6 Astra | 87.04 | 86.40 | 92.82 | 100/100 | 85.80–88.29 |
| Grok 4.7 | 79.62 | 81.02 | 67.04 | 96/100 | 76.59–82.65 |
| Claude Opus 5.5 | 69.66 | 68.84 | 77.03 | 78/100 | 62.57–76.74 |

The benchmark's overall score weights Main at 90% and Bonus at 10%; averages are over five attempts per binary, and unanswered attempts count as zero.

**Interpretation:** AgentRE evaluates detailed static tool use, multi-step reconstruction and structured answer correctness. It does not establish that an agent can acquire arbitrary firmware or reconstruct a live service environment from a CVE ID.

**Luminous evaluation plan:** use its fixed call budget and structured output as a separate binary analysis cohort. Preserve its version, artifact set, symbols/stripped split, attempt count, tool-call budget, and deterministic grader. Do not pool scores with SRE-Bench.

Sources:
- https://www.agentre-bench.ai/
- https://github.com/agentrebench/AgentRE-Bench

### Decompile-Bench — function-level reconstruction

**Published:** NeurIPS 2025 Datasets and Benchmarks Track.

The paper reports:
- two million binary-source function pairs distilled from 100 million collected pairs;
- 450 GB of compiled binaries from permissively licensed projects;
- an evaluation set including HumanEval/MBPP and repositories released after 2025 to reduce leakage;
- a reported 20% improvement in re-executability rate from fine-tuning with the dataset, relative to prior benchmarks.

**Interpretation:** source-text similarity and function-pair matching are useful but do not prove semantic equivalence, correct control/data flow or safe behavior. The Luminous suite must first reproduce the official published metric exactly, then add separate execution and semantic-oracle tracks.

**Luminous evaluation plan:** pin the dataset snapshot, item selection, metric implementation and source licenses. Report textual similarity, compile success, test equivalence, control/data-flow recovery and behavior-preserving edit cost separately.

Source:
- https://proceedings.neurips.cc/paper_files/paper/2025/hash/079cf13ae174c31f148207d94d213bdc-Abstract-Datasets_and_Benchmarks_Track.html
- https://github.com/anonepo/LLM4Decompile

### ReproBench — pre-environment IoT firmware reproduction

**Published:** 28 September 2026.

This benchmark starts from a CVE identifier rather than a prepared binary or container. It evaluates six sequential phases:
1. information gathering;
2. firmware acquisition;
3. firmware extraction;
4. binary identification and root-cause localization;
5. service rehosting;
6. vulnerability triggering.

The study used 30 IoT firmware CVEs, five models and three trials per CVE (450 runs). It reports 45.3% simulation substitution and 5.3% successful real-world reproductions among CVE-model pairs. It explicitly gates rehosting and triggering on the real firmware target; mock services, host-native harnesses and static-only analyses cannot earn real-target phase credit.

**Interpretation:** this measures a capability that binary-first benchmarks do not: reconstructing the target environment from scratch. The main failure mode is not merely inaccurate reverse engineering; it is claiming progress by replacing the actual target environment with a simulation.

**Luminous evaluation plan:** implement #10298 with authorized public/vendor firmware and safe local validation. Keep mock/simulated experiments in a separate category. No third-party live-device testing, remote exploitation or destructive payloads.

Source:
- https://arxiv.org/abs/2609.34450

### OWASP mobile application security

Use the OWASP MASVS control baseline, MASWE weakness taxonomy and MASTG test catalogue. This is a control-verification programme, not a binary RE leaderboard. A static scanner finding cannot satisfy a control that requires runtime, device, manual or business-logic evidence.

Source:
- https://mas.owasp.org/MASVS/
- https://mas.owasp.org/MASTG/
- https://mas.owasp.org/MASWE/

## 2. What the combined evidence says about the field

The current public frontier is already strong at some binary-first tasks. AgentRE V3.1 shows near-87/100 scores for leading systems on its Windows PE cohort. SRE-Bench shows GPT-6-Astra at 5.55/6 on graded binaries, while coverage and model-side safety restrictions materially affect denominators. But ReproBench exposes a much harder pre-environment task where complete real-world reproduction is rare and mock substitution is common.

Therefore, the opportunity is not simply to produce more decompilation or increase a static benchmark score. It is to improve **end-to-end evidence continuity**:

```
source identity → artifact acquisition → transformation lineage
→ semantic recovery → target reconstruction → independent behavior validation
→ reproducible decision
```

The same evidence object must explain which phase succeeded, what artifact was observed, what remains unknown and which independent oracle validated the conclusion.

## 3. Luminous scorecard design

Do not aggregate these axes into one score:

| Axis | Required measures |
|---|---|
| Binary understanding | task pass rate, function localization, MRR, semantic claim precision/recall |
| Decompilation | exact official text metric, compile rate, executable equivalence |
| Firmware reconstruction | phase completion, source/hash binding, lineage, real-target fidelity |
| Mobile application security | MASTG test coverage, MASVS control evidence coverage, unsupported-oracle rate |
| Evidence integrity | digest parity, replay/tamper rejection, lineage completeness, raw/derived separation |
| Causal diagnosis | top-1/top-3 localization, false-cause rate, calibration, selective risk |
| Decision quality | EVSI prediction error, p90 realized regret, wrong-action rate, abstention correctness |
| Efficiency | cost per verified task, p50/p95 latency, tool calls, CPU/RAM, energy and target impact |
| Integrity/accounting | eligible/attempted/graded/refused/abstained/unsupported/timeouts; budget and grader provenance |

## 4. How to improve beyond SOTA

The proposed research advantage is a layered system rather than a monolithic model:

- REA supplies mature analyzers and observations.
- Rust-native adapters preserve deterministic evidence, artifact lineage, tool effects and process lifecycle.
- Symthaea builds cross-layer hypotheses and chooses the next experiment using expected value of information and regret-aware planning.
- HARDSEC and external oracles qualify observations, target binding and claim ceilings.
- Benchmark receipts bind every public result to task denominator, exact tool/model revisions, budgets, environment, grader and raw outputs.

A superiority claim requires matched inputs, budgets, safety settings, attempts and denominators. Report confidence intervals and per-domain/per-platform results. An internal derived score must never be renamed as an external benchmark's official score.

## 5. Work items

- #9911 — A/B/C causal diagnosis and next-experiment selection.
- #9969 — SRE-Bench and OWASP MAS Crackmes external programme.
- #10026 — Decompile-Bench and OWASP control-verification programme.
- #10029 — benchmark run receipt and denominator/accounting validator.
- #10298 — from-scratch firmware reconstruction.
