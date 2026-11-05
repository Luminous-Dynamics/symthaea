# Reverse-Engineering Benchmark Baselines and Comparability Matrix

**Snapshot date:** 2026-11-04  
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
- #10368 — bounded per-occurrence ZIP content verification, incremental SHA-256 and independent CRC-32, with a test-only zlib-rs Deflate differential oracle.
- #10451 — content-addressed ZIP corpus and bounded coverage-guided fuzz qualification.
- #10452 — platform-qualified held-file identity; non-Unix content verification remains fail-closed until this is resolved.
- #10455 — archive-session performance research without weakening per-member decompression limits.
- #10477 — exact-head receipt validator qualification: 15/15 schema/semantic/JCS tests, formatting and Clippy passed at head `667e26121c371eadbfab6a8cd68729f5efb839c5`; replay remains a separate gate.
- #10507 — independent zip-rs/zlib-rs receipt replay (draft implementation; exact-head compile/test qualification pending).
- #10508 — Android APK v1–v4 and Apple code-signing/provisioning-profile evidence, separated by scheme and platform policy.
- #10368 — exact-head ZIP verifier qualification passed 41/41 tests, formatting and Clippy at head `241b42feae94937bb1b0716841abcf77def2e1d1`.
- #10451 — bounded 181-second fuzz smoke at head `d8d54f6e918838eabf9cb110db79d4f23986fede`: 3,371,635 executions, 18,627 exec/s, 283 MiB peak RSS, zero crashes/timeouts in that run; not sustained fuzz qualification.


## Verification refresh — 2026-11-04

This section records a fresh check of the official benchmark/catalogue pages. It supersedes earlier page snapshots only where explicitly stated; the existing task protocols and historical observations remain unchanged.

### SRE-Bench: current official result and denominator interpretation

The DAPLab page reports 262 eligible binary instances and 1,572 deterministic tasks, six tasks per instance, across five domains. Its public-facing table currently lists:

| Model | Graded / 262 | Mean tasks / 6 | Fully solved / graded | Zero-credit / graded |
|---|---:|---:|---:|---:|
| GPT-6-Astra | 181 (69.1%) | 5.55 | 149 (82.3%) | 7 (3.9%) |
| GPT-5.6-Sol | 254 (96.9%) | 3.69 | 80 (31.5%) | 41 (16.1%) |
| Claude-Fable-5.1 | 223 (85.1%) | 2.58 | 60 (26.9%) | 96 (43.0%) |
| Claude-Opus-5 | 256 (97.7%) | 1.91 | 32 (12.5%) | 118 (46.1%) |
| GPT-5.5 | 262 (100%) | 1.02 | 10 (3.8%) | 148 (56.5%) |
| DeepSeek-V4.1-Flash | 262 (100%) | 0.85 | 2 (0.8%) | 166 (63.4%) |
| Grok-4.5 | 218 (83.2%) | 0.45 | 2 (0.9%) | 160 (73.4%) |
| GLM-5.2 | 262 (100%) | 0.21 | 0 (0%) | 217 (82.8%) |

The GPT-6-Astra public-facing full-solve proportion is conditional on 181 graded instances: 149/181 = 82.3%. Against all 262 eligible instances, those 149 full solves are 56.9%. Both values are useful, but they answer different questions. The ungraded 81 instances are not silently passes or failures.

The same official page reports a distinct **unconstrained** setting in which GPT-6-Astra achieved pass@1 = 88.0% and pass@4 = 99.2%. Do not combine those numbers with the public-facing safety-enabled table or describe pass@4 as one-shot reliability. The 11.2 percentage-point gap makes repeated-attempt selection a separate capability from reliable first-attempt diagnosis.

### AgentRE-Bench v3.1: latest official Windows PE scorecard

The current v3.1 results page reports 500 attempts: five models × 20 stripped/unstripped artifacts × five attempts. The 20 artifacts represent 10 Windows programs with and without symbols. Each attempt has a 25-tool-call budget.

| Model | Overall / 100 | Main / 100 | Bonus / 100 | Valid answers |
|---|---:|---:|---:|---:|
| Muse Wonder | 87.37 | 87.37 | 87.37 | 100/100 |
| GPT-6 Astra | 87.04 | 86.40 | 92.82 | 100/100 |
| Grok 4.7 | 79.62 | 81.02 | 67.04 | 96/100 |
| Claude Opus 5.5 | 69.66 | 68.84 | 77.03 | 78/100 |

The benchmark reports 95% intervals and preserves original answers/transcripts plus checksum-verified scoring. Its methodology states that valid-answer coverage and answer quality are separate, and that the winner label does not establish repeat-run superiority or practical equivalence. This remains a Windows PE cohort and must not be pooled or ranked numerically against SRE-Bench.

### Decompile-Bench: item-level release measurement

The Stanford AIMS measurement database now exposes 72,059 released function-level decompiler outputs for HumanEval, MBPP and GitHub2025. Its released edit-similarity metric is explicitly **textual similarity, not executable correctness**. The current response matrix lists:

- IDA Pro: 71,496 observations, mean 0.206.
- Ghidra: 69,635 observations, mean 0.202.

The two counts sum to 141,131 observations across the subjects; they are not 141,131 unique benchmark items. Preserve repeated/item-level structure and reproduce the official metric before adding separate compile, held-out execution-equivalence, control-flow/data-flow and side-effect oracles.

### OWASP mobile verification: MASTG v2 and MASWE v1

MASTG v2.0.0 is the first stable release of the refactored, machine-readable mobile testing guide. Its tests carry stable IDs and structured metadata, and explicitly separate **Observation** (raw test output) from **Evaluation** (the pass/fail rule). MASWE v1.0.0 is the first stable weakness catalogue.

The architecture consequence is unchanged but now grounded in stable released versions:

`MASVS control → MASWE weakness → MASTG test → technique/tool → raw observation → versioned evaluation → REA Evidence → independent claim receipt`

A test result is not automatically a MASVS control pass. Keep static, dynamic, hook, manual and business-logic evidence distinct; platform and target profiles must remain explicit. A missing test/oracle is an oracle gap, not a pass.

### MobSF current tool baseline

The MobSF GitHub releases page lists v4.6.4, released 2026-10-31, as the latest release at this refresh. Its notes add direct standalone DEX static analysis, a built-in authenticated live Web Traffic viewer using mitmproxy, and other static/dynamic fixes. Record the exact MobSF release, configuration, Android/iOS target profile and instrumentation environment for matched comparisons; do not assume a release tag means that every dynamic target configuration is supported.

### Updated competitive interpretation

The latest evidence does not support a claim that a new orchestration layer alone beats specialist tools. It supports a more precise opportunity:

1. Match or exceed specialist baselines on **selected** parser/inventory/decompiler operations with exact parity and measured resource cost.
2. Beat single-provider workflows on **claim reliability**, measured by held-out correctness, calibration, abstention and false confident claims.
3. Beat static-only workflows on **end-to-end investigation quality**, measured by causal localization, experiment value, independent postcondition validation and time-to-correct-diagnosis.
4. Improve **evidence continuity** across acquisition, transformations, analysis, runtime observations and adjudication.
5. Show the benefit under identical inputs, versions, target/environment, access permissions, model/tool-call budgets, time/compute and complete denominators.

No Luminous result is reported by this refresh. All values above are external, publisher-reported reference data.

### Official sources checked 2026-11-04

- SRE-Bench current leaderboard and methodology: https://daplab.cs.columbia.edu/benchmarks/sre-bench/
- AgentRE-Bench v3.1 results (October 25, 2026): https://www.agentre-bench.ai/results/v3-1/
- Stanford AIMS Decompile-Bench item-level database: https://aims.stanford.edu/measurement-db/decompile_bench
- MASTG v2.0.0 release announcement: https://mas.owasp.org/news/2026/07/04/mastg-v200-release/
- MASTG v2.0.0 signed release: https://github.com/OWASP/mastg/releases/tag/v2.0.0
- MASWE stable catalogue: https://mas.owasp.org/MASWE/
- MobSF official releases: https://github.com/MobSF/Mobile-Security-Framework-MobSF/releases

## 2026-11-05 refresh — agentic RE, mobile security and what the industry actually measures

### SRE-Bench (Columbia DAPLab / Vals AI)

The current official benchmark page reports a released benchmark with 19 private programs, 262 binary instances and 1,572 deterministically graded tasks. The programs span five RE domains (firmware, malware, game, file format and network protocol), four source languages (C, C++, Rust and Go), and 44 in-house anti-analysis mechanisms. Each task is evaluated in an isolated environment with a task-specific verifier. See [official SRE-Bench page](https://daplab.cs.columbia.edu/benchmarks/sre-bench/).

Its public-facing model evaluation and its unconstrained evaluation must remain distinct:
- Public-facing GPT-6-Astra: mean 5.55/6 on 181 graded binaries; 149 full solves (82.3% of the 181 graded) and 7 zeros. The full eligible corpus has 262 instances, so the 149 solved correspond to 56.9% of all eligible instances; do not report 82.3% as full-corpus coverage.
- Unconstrained GPT-6-Astra: 88.0% pass@1 and 99.2% pass@4. This is a different setting without production safeguards or budget limits; it must not be pooled with public-facing data.
- Other public results also need their graded denominators and zero rates. A high conditional score with incomplete coverage is not the same as full-suite completion.

The strategic lesson is that RE is increasingly measured through contamination-resistant, domain-specific environments and deterministic task verifiers, not by a language-model judge scoring plausible explanations. Our REA artifact-reader tests are a separate layer: they validate the parser and content-verification substrate that those workflows rely on. We should not claim to beat SRE-Bench until Symthaea has actually run its agent on the benchmark under the benchmark's rules.

### AgentRE-Bench v3.1 (Windows PE, October 25 2026)

The official v3.1 results report 500 attempts: five models × 20 artifacts × five attempts. The 20 artifacts are ten Windows PE programs with stripped and unstripped variants. Overall weights are 90% main and 10% bonus; refusals and budget exhaustion count as zero. Results:
- Muse Wonder: 87.37/100, 100/100 valid answers.
- GPT-6 Astra: 87.04/100, 100/100 valid answers.
- Grok 4.7: 79.62/100, 96/100 valid answers.
- Claude Opus 5.5: 69.66/100, 78/100 valid answers.
- Gemini 4 Argon: 31.20/100, 36/100 valid answers.

The Muse/Astra difference is only 0.3243 points and its simultaneous 95% interval spans -3.4940 to +4.1426; this run does not establish repeat-run superiority. The report also shows Astra's stronger encryption/key recovery and Muse's higher technique-identification score. See [AgentRE-Bench v3.1 results](https://www.agentre-bench.ai/results/v3-1/) and [benchmark methodology](https://www.agentre-bench.ai/).

Do not compare AgentRE's weighted 0–100 score to SRE's mean tasks-per-binary or pass@k. They have different artifacts, task definitions, budgets and scoring. They are complementary agent-capability probes, not substitutes for exact-head parser, fuzz, differential, mobile-signature or evidence-reliability tests.

### OWASP Mobile Application Security Testing Guide

OWASP MASTG v2.0.0 is the first stable, non-beta release of its v2 refactor (June 30, 2026). The release makes the guide's components individually referenceable and machine-readable with cross-linking among MASVS, MASWE, tests, techniques, tools and best practices. See [official v2.0.0 release](https://github.com/OWASP/MASTG/releases/tag/v2.0.0).

For REA, the trace should remain explicit:
`MASVS control → MASWE weakness → MASTG test → tool/platform observation → raw evidence → versioned evaluation → qualified claim`.
A static test finding is not automatically a passed control; a single tool result is not a complete platform evaluation.

### Platform-signature reference contracts

Android's official documentation confirms why APK signatures require scheme-specific interpretation:
- v2 signs protected APK content sections and the signing block's signed-data records; the digest is chunked and covers the ZIP content region, central directory and EOCD. [Android v2](https://source.android.com/docs/security/features/apksigning/v2)
- v3 adds supported SDK ranges and proof-of-rotation information in the signing block. [Android v3](https://source.android.com/docs/security/features/apksigning/v3)
- v4 uses an fs-verity-shaped Merkle tree over the APK bytes, stores the signature in a separate `.apk.idsig`, and requires a complementary v2 or v3 signature. [Android v4](https://source.android.com/docs/security/features/apksigning/v4)

Apple's reference material makes a different distinction:
- Provisioning profiles are CMS-wrapped authorization documents containing an entitlement allowlist; code-signature entitlements must be authorized by the profile. Apple documents the profile structure as implementation-defined, so supported platform tools/APIs—not assumed plist serialization—must anchor platform-trust claims. [TN3125](https://developer.apple.com/documentation/technotes/tn3125-inside-code-signing-provisioning-profiles)
- CodeDirectory contains per-page and special-slot hashes that protect executable pages, resources and metadata. [TN3126](https://developer.apple.com/documentation/technotes/tn3126-inside-code-signing-hashes)

The same core rule applies to both platforms: content integrity, signature mathematics, signer key/certificate identity, chain/trust policy, application authorization and runtime behavior are distinct propositions. The benchmark must include valid-but-unexpected signers and authorization mismatches, not only corrupted-signature negatives.

