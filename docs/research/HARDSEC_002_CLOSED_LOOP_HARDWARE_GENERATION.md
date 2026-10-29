# HARDSEC-002 — Closed-loop hardware security generation and assurance

**Status:** research protocol draft; no model-comparison results claimed  
**Tracking issue:** [symthaea#9038](https://github.com/Luminous-Dynamics/symthaea/issues/9038)  
**Protocol date:** 2026-10-29  
**Reference benchmark:** HardSecBench, IJCAI 2026, DOI [10.24963/ijcai.2026/57](https://doi.org/10.24963/ijcai.2026/57)  
**Official benchmark repository:** [chenqirui2002/HardSecBench](https://github.com/chenqirui2002/HardSecBench)

## 1. Research question

Can a Symthaea-directed investigation loop increase the rate of **jointly functionally correct and security-correct RTL / firmware artifacts** at a fixed compute, token, tool, and wall-clock budget—without causing functional regressions, evaluator coupling, or unsupported assurance claims?

This is a distinct track from [HARDSEC-001](https://github.com/Luminous-Dynamics/symthaea/issues/8990), which evaluates runtime RISC-V telemetry. The long-term program links design-time requirements, executable tests, runtime observations, and physical evidence, but the scores for these domains must stay separate.

## 2. External benchmark and the opportunity

The HardSecBench IJCAI 2026 paper describes 924 tasks across Verilog RTL and firmware C, covering 76 hardware-related CWE entries. Each task is described as including a structured specification, secure reference implementation, and executable tests. The authors report that functional success can coexist with security defects and that results vary by prompting. These reported counts and findings motivate this study; Phase A must independently validate the exact downloaded corpus and tool behavior before treating them as our own inventory.

References:
- Paper / proceedings: https://ijcai.org/proceedings/2026/57
- DOI: https://doi.org/10.24963/ijcai.2026/57
- Official source: https://github.com/chenqirui2002/HardSecBench

The upstream repository documents Icarus Verilog and GCC as relevant toolchain dependencies, single-shot/pass@k evaluation, optional iterative refinement, and security-hint conditions (none, generic, CWE-specific). It keeps generated data and outputs under ignored paths and provides a compressed benchmark archive. We must pin the actual source commit and data-archive digest at evaluation time; a moving `main` URL is not a reproducibility commitment.

**License caveat:** the upstream repository has a license file, but Phase A must inspect the actual data archive, task artifacts, embedded reference code, tests, and any third-party source separately. A repository-level license must not be assumed to cover every embedded artifact or all derived redistribution.

## 3. Principle: investigate, do not self-certify

The proposed loop is:

1. **Read the specification:** extract functional requirements, security invariants, expected threat model, environmental assumptions, and what remains unknown.
2. **Propose a candidate or a discriminating test:** Symthaea may prioritize requirements and tests by expected decision value per cost; it is not the security oracle.
3. **Run authoritative tools:** upstream task harness and isolated Icarus Verilog / GCC execution, plus a task-appropriate formal/property checker where one exists and is separately qualified.
4. **Classify evidence:** record functional tests, security tests, mutation controls, formal results, compile outcomes, and missing evidence separately.
5. **Diagnose and repair:** expose the smallest failed requirement and propose a bounded candidate repair.
6. **Rerun the complete gate:** every earlier functional and security regression suite is rerun after repair. A repair cannot erase the prior failure.
7. **Issue a scoped result:** “acceptable for this task and tool configuration” only if the required checks pass. Missing checker output or inadequate mutation sensitivity remains inconclusive.

The independent verifier must not accept a code generator's self-report, explanation, confidence, or claimed test result. Simulation pass does not establish side-channel resistance, fault-injection resistance, or silicon security.

## 4. Phase A — baseline / data integrity audit

Before running models:

- Pin official repository commit, archive SHA-256, license files, and tool versions.
- Enumerate actual task files, CWE folders, task IDs, language, specification digest, reference source digests, testbench digests, and task-lineage metadata.
- Reconcile observed counts against the paper's reported 924 tasks / 76 CWEs; explain any differences rather than hardcoding expected counts as if they were observed.
- Store RTL and C as distinct evaluation strata. Never pool compile outcomes from different toolchains without recording them.
- Detect exact duplicates and near-duplicates across task descriptions, code, testbenches, and generated lineage before splitting.
- Keep reference code and testbench generation separate from the evaluated candidate. Verify that the required positive and negative controls behave as expected.
- Run controlled insecure mutants (only within disposable, local test artifacts) to measure whether each task's security oracle catches the defect class it claims to test. If a task's harness cannot discriminate secure from insecure behavior, label that task oracle-inadequate; do not count its candidate as secure merely because the harness passes.
- Do not expose secret holdout manifests or private canary tasks to a generator prompt.
- Do not treat a public benchmark's test split as contamination-proof: state that the code and task specifications may already be in model pretraining.

The benchmark archive contains untrusted generated code. Use isolated, disposable environments with bounded resource usage and no access to secrets or production hardware. Do not deploy generated code or synthesized designs.

## 5. Frozen comparison matrix

Within each task, all arms receive the same functional specification and the same authorized data. Freeze model/provider revision, prompt template, sampling parameters, attempt budget, iteration budget, and tool budget before confirmatory runs.

| Arm | Purpose |
|---|---|
| A. Single-shot, no security hint | Reference baseline |
| B. Generic security hint | Separates prompting effect from added architecture |
| C. CWE-specific hint | Matches the upstream benchmark's documented hint axis |
| D. Iterative repair with functional feedback only | Measures whether functional feedback alone leaves security defects |
| E. Iterative repair with independently generated security feedback | Tests value of security-specific verification |
| F. Symthaea-directed requirement and test selection | Measures whether the investigation policy adds value beyond prompting |
| G. Symthaea + bounded Symtropy scenario bookkeeping (later phase only) | Tests whether explicit scenario/state organization improves coverage or repair efficiency |

The official evaluator should remain the result authority for benchmark-compatible measures until any alternative implementation has passed parity testing. Our Rust-first code should manage manifests, orchestration, evidence binding, and aggregation; do not silently rewrite the upstream evaluator or change its oracle semantics.

Do not force HDC/CfC to be the code generator. The hypothesis is that Symthaea may be a better investigation / experiment-selection layer when paired with a separate code-generation provider. Any claim about Symthaea itself generating code must be evaluated as a distinct arm.

## 6. Metrics — separate correctness, oracle quality, and cost

Report all outcomes separately by language and CWE:

- functional requirement pass rate;
- security requirement pass rate;
- **joint functional + security pass rate** (primary endpoint);
- compilation and simulation success;
- formal/property-check pass, fail, unsupported, or not available;
- mutation-detection rate and oracle-inadequate task count;
- repair-induced functional regression rate and security regression rate;
- pass@1 and pass@k under fixed independent attempts, with deduplication policy reported;
- worst-CWE and macro-CWE performance rather than only task-weighted averages;
- time, tokens / API cost, simulator time, refinement iterations, and cost per jointly acceptable artifact;
- per-task lineage, failed attempts, verifier outputs, raw logs and exact digests.

Functional and security pass counts must not be merged so that a candidate with correct behavior but a missing security invariant appears successful. “Compile failed”, “verification unavailable”, and “testbench did not exercise the property” are distinct outcomes.

## 7. Proposed challenge gates (not industry standards)

A Symthaea-directed loop earns further investment if it satisfies at least one preregistered improvement and no safety-critical regression gate fails:

1. Improves joint functional+security pass rate over the strongest matched baseline at the same total attempt/tool budget, with uncertainty estimated at task/CWE-lineage level.
2. Alternatively, maintains joint pass rate and reduces cost per jointly acceptable artifact by at least 20%.
3. Adding security feedback must not reduce functional correctness beyond a preregistered non-inferiority margin; set that margin before observing confirmatory results.
4. Security-oracle mutation adequacy is published and used to bound claims. Tasks that cannot detect their seeded negatives are not credited as successful security assessments.
5. Every claimed success is reproducible from exact task, candidate, testbench, toolchain and output digests. Missing or inconsistent receipts are inconclusive.
6. Results must survive an independent rerun and the outcome must not depend on access to the benchmark answer or testbench during generation.

The 20% cost gate is a proposed engineering target, not a result. If task counts per CWE are too small for useful uncertainty estimates, publish the per-task descriptive results and defer broad claims.

## 8. Where this might beat the state of the art

The intended contribution is an **assurance-efficiency frontier**, not a larger one-shot pass@k:

- jointly acceptable artifacts per fixed budget;
- repair success without regressions;
- performance on low-frequency CWE families;
- test selection value per unit cost;
- oracle adequacy / mutation coverage;
- reproduction cost and completeness of evidence;
- robustness to prompts and model/provider changes.

A genuine advantage might be fewer security defects for the same functional pass rate, or fewer tool calls per artifact that passes both gates. A simple baseline wins if it delivers the better frontier. We must not claim state-of-the-art performance from a protocol, a model rationale, or a few successful tasks.

## 9. Symtropy boundary

Symtropy may coordinate explicit scenario IDs, deterministic replays, expected-versus-observed mismatches, and a declared validity envelope. It must not replace Icarus Verilog, GCC, a qualified formal engine, physical test equipment, or the benchmark's authoritative harness. A simulation result is tagged with tool/model version and evidence class; no physical/silicon claim follows from simulation alone.

## 10. Data and reporting contract

Every run should bind:

- benchmark repository commit and archive digest;
- task ID, CWE, language, specification digest, and task-lineage / split manifest;
- model/provider ID, prompt/template digest, temperature, attempt and token budget;
- generated artifact digest and complete patch/repair history;
- compiler/simulator/formal-tool version and exact invocation;
- reference/testbench digest, mutation-control outcomes and test coverage;
- raw result digests, metric implementation revision, uncertainty method, and claim level.

Receipts establish byte identity only. Unless separately signed and anchored, they do not prove who generated an artifact or that the underlying labels represent real-world truth.

## 11. Relationship to HARDSEC-001

- **HARDSEC-001 (#8990):** telemetry detection, capture observer effects, and generalization across RISC-V programs/configurations.
- **HARDSEC-002 (#9038):** design-time RTL/C security generation and independent functional/security verification.
- Long-term synthesis: a common evidence graph links a design requirement and exact artifact to executable tests, runtime observations, and (where available) physical measurements. Each domain retains separate ground truth, tooling, and claim limits.

## Immediate next actions

1. Audit the upstream benchmark commit, archive and task/license metadata.
2. Verify the local compiler/simulator prerequisites and reproduce the upstream evaluator on a small non-secret smoke subset.
3. Verify secure reference and insecure-mutant test behavior before running any model.
4. Freeze the evaluation matrix and artifact/split manifests.
5. Run the fixed baseline comparison, then add Symthaea-directed investigation as one controlled arm.
