# Mesh Security Benchmark Contract v1

**Status:** proposed preregistered evaluation contract; not executed benchmark evidence  
**Scope:** PR #8355, #8376, #8412, #8365, #8471, #8491, #8508 and #8519  
**Machine-readable suite:** `mesh-security-scenarios-v1.json`  
**Suite schema:** `mesh-security-scenarios-v1.schema.json`  
**Evidence receipt schema:** `mesh-security-evidence-v1.schema.json`  
**Contract-audit receipt schema:** `mesh-security-contract-audit-v1.schema.json`

## Research proposition

The competitive unit should be an **independently accepted, reproducible security claim**, not a model-generated finding, a passing test count, or the number of protocol defects narrated by an agent.

The benchmark is designed to test a discovery-and-assurance system as a whole:

1. The proposer declares a security property, threat assumptions, and expected falsifiable observation before running an attack.
2. A separate executor runs a deterministic witness, model, fuzzer, differential test, or proof checker against the frozen source revision.
3. An evaluator validates the outcome against the pre-registered oracle. A nonzero process exit, timeout, malformed model, or unavailable tool is not automatically a discovered vulnerability.
4. Positive controls preserve valid behavior for the same task family.
5. A separate novelty reviewer checks prior art before a result is labelled novel.
6. The evidence receipt binds the exact source, artifact, tool, configuration, raw output, controls, and claim ceiling.

The generator must not author the only oracle that accepts its own output. HDC or other learned Symthaea representations may prioritize, retrieve, or cluster candidate weaknesses; they cannot decide that a cryptographic property is satisfied.

## Why a claim-level contract is the useful paradigm shift

Industry and academic systems are increasingly combining models with executable attack tasks, expert-graded cryptographic benchmarks, and proof assistants. For example:

- **CryptanalysisBench (NeurIPS 2026)** reports 191 tasks across six cryptographic primitive families. The authors report frontier-model attacks on 65–86% of known-practical-break tasks, attacks on 6–12 full-strength Tier-2 schemes across the tested models, and attacks on 24–61 scaled-down variants. The benchmark also surfaced reported research results including an attack on SpoC AEAD and an issue in a published KINDI CCA-security proof. These are the authors' claims and benchmark outcomes, not Symthaea results; reproduce task-by-task and do not use a single percentage as a universal security score. https://avital.shafran.co.il/publication/cryptanalysis-bench/
- **AICrypto (ICML 2026)** contains 135 knowledge questions, 150 CTF tasks and 30 proof problems, with expert-reviewed scoring and human baselines. The authors report frontier models match or exceed experts on some memorization, common exploit and routine-proof tasks while continuing to struggle with abstract mathematics and multi-step reasoning. https://proceedings.mlr.press/v306/wang26em.html
- **Anthropic's July 2026 report** describes an improved key-recovery attack against HAWK and a faster attack on seven-round AES, explicitly noting that neither reported result affected production systems. A later portion of the report describes further work on reduced-round LEA and Serpent. Treat these as reported cryptanalytic research, not evidence that full-strength AES or production systems were broken. https://www.anthropic.com/research/discovering-cryptographic-weaknesses
- **Microsoft's September 2026 SymCrypt/Aeneas/Lean work** reports machine-checked safety, panic-freedom and functional-correctness results for 16.7 KLOC of Rust using a 237 KLOC Lean development. This provides a more relevant benchmark for implementation assurance: verified lines, theorem coverage, assumptions, proof maintenance, runtime performance and independent proof checking. It is not an unrestricted proof of cryptographic security. https://arxiv.org/abs/2609.15648

The useful competitive distinction is that CryptanalysisBench measures attack discovery, AICrypto spans knowledge/exploitation/proof tasks, and SymCrypt's verified-code work measures implementation correctness. None alone covers end-to-end assurance of a stateful embedded mesh protocol. Symthaea's differentiator should be **closed-loop assurance**: discover a candidate, preregister its oracle, reproduce it on frozen code, repair it, run positive controls, verify the repaired implementation and record bounded residual risk. The corpus must include both attack-oriented tasks and correctness/availability qualification tasks; optimizing only for finding more attacks would reward false positives and destabilizing repairs.

Headline scores from these efforts are not directly comparable: their tasks, model versions, resource budgets, and acceptance rules differ. Symthaea should benchmark on a frozen, licence-respecting subset of public tasks plus a held-out mesh-specific corpus, with exactly matched compute and tool budgets.

## Evidence classes and claims

| Evidence class | What can be claimed | What must not be inferred |
|---|---|---|
| Model check | Named properties hold in the bounded state space and declared configuration. | Rust implementation conforms; unbounded proof; cryptographic security. |
| Source regression | Named witness reproduced against exact source revision. | Novel discovery; real deployment exposure. |
| Implementation conformance | Named runtime/API behavior matched a frozen reference corpus in the tested configuration. | Primitive security or untested configurations. |
| Differential test | Implementations agreed or disagreed on named vectors and versions. | Correctness merely because two implementations agree. |
| Independent review | Reviewer assessed the specified artifact and scope. | Other scopes or changed heads are covered automatically. |

A result has a single explicit `claim_ceiling`; it may not be promoted to a stronger class by a summary, benchmark score, CI badge, or AI-generated report.

## Preregistered scenarios

The current versioned registry defines **23 security properties and 23 scenarios**. Both the scenario registry and the eventual run receipt have separate Draft 2020-12 schema contracts. All scenarios remain proposed and unexecuted until each has an exact implementation/model oracle and a machine-readable receipt.

The current registry separates nonce uniqueness, sender namespace, restart/rollback, counter wrap, snapshot cloning, key-rotation boundaries, fragmentation/MTU, transport-authenticated source binding, cleartext fallback, replay-window poisoning, spoofed-source rate-limit exhaustion, replay-table capacity exhaustion/recovery, timestamp freshness, identity/transcript binding, mutual confirmation, real backend availability, downgrade, replay, resource limits and interoperability. Each scenario declares both a legacy expectation and a required replacement observation.

The known-failure cases are calibration, never novel discoveries. An attack task counts as correctly solved only when its specified witness is reproduced and its positive controls remain green. A repair only scores when it rejects the same attack and preserves valid controls on the exact repair revision.

## Recommended held-out experiment

Freeze corpus version `v1` and hash it before any prompt, rule, or model tuning. Partition the tasks by protocol family so near-duplicate mutations do not leak from development into holdout. Keep the hidden labels and scoring oracle isolated from the proposing agent. Have two reviewers adjudicate ambiguous outcomes; record agreement and arbitration.

Compare with equal caps on wall-clock time, model tokens/API cost, CPU/memory, tool invocations, and human attention:

- conventional unit + property testing;
- conventional model checking without an AI proposer;
- one tool-using model;
- separate AI attack generator and evaluator;
- Symthaea HDC-assisted retrieval/prioritization plus independent generator/evaluator;
- a cryptographer-led baseline on a stratified sample.

Keep the source commit, dependencies, tool identities, model IDs, prompts, seeds, and exact resource caps fixed. Public benchmark tasks used during development cannot be treated as held-out evidence.

## Primary and secondary metrics

| Metric | Required reporting |
|---|---|
| Critical-seeded-fault recall | Detected / seeded and reachable faults, per property family. |
| False accepted-finding rate | Claims accepted by the evaluator without the intended property violation. Target: 0. |
| Mutant kill rate | Intended mutation rejected for the expected invariant, not merely any process failure. Target: 100% for the preregistered critical mutants. |
| Positive-control retention | Valid controls that remain accepted after each repair. Target: 100% on the frozen reference corpus. |
| Novelty precision | Independently accepted novel findings / all findings labelled novel. Report confidence intervals. |
| Reproduction rate | Independent reruns with the same outcome from the frozen artifact. Target: 100% for accepted critical findings. |
| Time to accepted finding | Wall-clock from task assignment to independent acceptance. |
| Cost per accepted finding | Compute/API + measured human adjudication cost. |
| Counterexample quality | Fraction automatically minimized while preserving the same violation. |
| Operational cost | p50/p95/p99 handshake/record latency, bytes, fragment count, retries, energy/airtime, peak memory, reconnect and recovery. |
| Reviewer agreement | Agreement before arbitration for finding validity, severity, and novelty. |
| Coverage integrity | Number of declared property families tested versus the frozen required denominator; missing properties are not silently dropped. |

The hard security gate is lexicographic: first satisfy every mandatory property and preserve valid controls; only then compare speed, energy, memory, or discovery cost. A fast insecure result loses.

## Release and reporting rules

- Keep known breaks, held-out seeded faults, novel research findings, and performance data on separate scoreboards.
- Every reported result links to a machine-readable evidence receipt conforming to the v1 JSON Schema.
- An unavailable runner or missing artifact is `unknown` / infrastructure error, never `pass`.
- A counterexample expected from a mutant is a benchmark-control success only if the named invariant violation is observed. A parser error does not count.
- Exact-source results expire when source, model, configuration, tool, dependency lock, or the property statement changes.
- Do not call a bounded result "formally secure"; say which property held, under which bounds and assumptions.
- Do not claim that Symthaea exceeds SOTA until a published, held-out, matched-budget evaluation demonstrates a statistically credible advantage in independently accepted findings, false positives, cost, or review latency.

## Run the contract audit

The isolated Rust checker is intentionally outside the monorepo workspace:

```sh
cargo fmt --manifest-path tools/mesh-security-contract-audit/Cargo.toml -- --check
cargo test --manifest-path tools/mesh-security-contract-audit/Cargo.toml
cargo run --manifest-path tools/mesh-security-contract-audit/Cargo.toml -- \
  --root . \
  --out /tmp/mesh-security-contract-audit.json
```

The emitted receipt includes the exact Git revision and SHA-256 digests for the suite, all three schemas, validator source/manifest and resolved Cargo lockfile. Receipt-persistence failure returns a nonzero exit instead of making a printed-but-unwritten receipt look successful. It labels the result `schema_and_registry_integrity_only`; it is not a security qualification. The checker explicitly disables external reference resolution and includes negative tests for remote schema references, invalid schema keywords, duplicate IDs, and undeclared property references.

## Next executable phase

1. Confirm the exact-head workflow runs the Rust checker and all Draft 2020-12 meta-schema/instance checks successfully.
2. Emit the evidence receipt from each model/runner invocation and hash raw logs, configs and counterexample traces.
3. Map all registry entries to actual test functions or mark them explicitly `not_implemented`; no unimplemented scenario may count in the denominator as passed.
4. Run the nonce and handshake model suites separately, then add trace-conformance tests at the real Rust boundary.
5. Publish the frozen benchmark bundle and baseline protocol after external review of the scoring oracle.

