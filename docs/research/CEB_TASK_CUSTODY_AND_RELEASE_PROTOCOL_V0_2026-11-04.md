# CEB task custody and release protocol V0 — 2026-11-04

**Status:** design-only release contract; no task package or execution path is qualified.  
**Parent:** [CEB-TASK-CUSTODY-001 #10436](https://github.com/Luminous-Dynamics/symthaea/issues/10436)  
**Related:** #10419 engineering-intelligence task families; #10173 repository twin; #10115 CEB task/receipt V0; #10314 general run receipt; #10370 evidence convergence; #10064 execution isolation.  
**Claim ceiling:** this document defines a packaging, custody, rights and oracle-qualification boundary. It does not create a new CEB task schema or receipt format, authorize task execution, or claim a benchmark result.

## 1. The actual blocker

The current task bank consists of blueprints and design-only records. The repository-twin source fixture and its design/evaluator artifacts are co-located. There is no materialized source-only agent view. Therefore, those fixtures are not eligible for agent scoring.

A prompt saying "do not inspect the gold file", a hidden directory name, a .gitignore entry, or a container label is not a security boundary. The agent must not have the oracle in its filesystem, environment, credentials, inherited file descriptors, mount namespace or network reachability.

The release unit must have two physically separated views:

- **Agent view:** only the task statement, exact allowlisted source snapshot, declared public context and writable paths.
- **Evaluator view:** private labels, root-cause hypotheses, hidden tests, reference patches, allowed alternatives, severity, mutation seeds, adjudication notes and grader configuration.

The existing CEB task schema remains the semantic task contract. This document specifies how to materialize and custody the two views; it is not a proposal for another task schema.

## 2. Separation of trust domains

Use four distinct planes:

| Plane | May access | Must not access |
|---|---|---|
| Authoring | Upstream source, task design, private oracle, license inventory | Production agent credentials |
| Release | Pinned source tree, public manifest, approved agent-visible task statement | Hidden labels, solution patch, private test oracle |
| Agent execution | Released agent view, declared tools, bounded writable area | Evaluator storage, signing keys, private labels, unsealed task history |
| Evaluation / regrade | Agent patch and trajectory, private oracle, evaluator revision | Ability to alter the agent-visible input after task start |

The agent and evaluator should use separate identities and separate storage/ACL domains. The agent process must not inherit evaluator credentials, access tokens, environment variables, file descriptors, IPC endpoints or mounts. The evaluator receives the submitted patch and trajectory through a one-way, size-bounded handoff and returns a grade/receipt reference.

If separate hosts are not available, the future runner design must still establish independently tested OS-level process, mount, user, network and credential isolation. A single host with differently named folders is insufficient. The #10064 isolation prerequisite remains a hard stop: no repository-controlled build scripts, procedural macros, tests, graders or candidate programs may run until that prerequisite is independently qualified.

## 3. Agent-view materialization

### 3.1 Inputs

Materialization must start from immutable inputs:

1. Exact source repository identity and commit/tree object.
2. An explicit path allowlist approved by the task author and independently reviewed.
3. Intended file modes and symlink policy.
4. The task's existing CEB record and revision.
5. Configuration-world identity: target, features, toolchain contract and other task-relevant settings.
6. Declared agent-readable and agent-writable path sets.

Do not copy a developer working tree. Do not create the agent view by copying everything and deleting known secret folders afterward. Start from the pinned tree and materialize only approved objects. A denylist is a useful secondary check, not the primary selection mechanism.

### 3.2 Path and object rules

- Do not include repository history, refs, remotes, commit messages, tags, reflogs, untracked files or author scratch material.
- Do not include private gold labels, reference patches, hidden tests, expected output, answer-bearing filenames, grader source or evaluator logs.
- Preserve path spelling, file modes and intended executable bits.
- Reject absolute paths, parent traversal, duplicate paths after normalization, case-colliding paths on case-insensitive filesystems and malformed path encodings.
- Reject symlinks that resolve outside the task root. For V0, the safest default is to reject symlinks entirely unless the fixture explicitly requires them and a reviewer verifies the resolved target is included.
- Reject gitlinks/submodules unless each referenced commit and its materialized contents are pinned and included in the allowlist.
- Reject unpinned Git LFS pointers or other external object references. The exact object bytes must be available and digest-bound before release.
- Keep package caches, build outputs and generated artifacts out unless a task explicitly requires them; do not let stale caches expose known fixes or hidden test names.
- A source-only Rust task may need Cargo manifests, lockfiles, build scripts, feature declarations and public tests. Include these only when the task contract needs them; they remain untrusted code and may not be executed before #10064.

### 3.3 Public manifest

The agent-facing manifest may contain:

- task ID and task revision;
- exact source revision and configuration-world identity;
- a complete inventory of agent-visible paths, modes and content digests;
- declared read-only and writable paths;
- toolchain and environment contract;
- allowed patch scope and expected output shape;
- public task-family and metric metadata;
- release-manifest revision.

It must not contain oracle-only file paths, hidden test counts, root-cause labels, private grader versions that reveal answers, known-bad mutation identifiers or other side-channel metadata. The manifest inventory is an explicit description of what the agent may see, not a listing of the full authoring repository.

Use the existing artifact-digest and receipt path. The causal-ledger Digest is only an opaque caller-supplied commitment; it does not calculate or authenticate content hashes. If current receipt interfaces cannot bind the agent-view manifest and evaluator artifact references without a new envelope, resolve the compatibility question under #10370 before adding fields. Do not introduce a competing CEB receipt or DSSE format.

### 3.4 Determinism

For frozen inputs, materialization should produce byte-identical agent views and manifests. Canonicalization must specify path ordering, encoding, file mode treatment, symlink treatment and digest inputs. Repackaging the same source commit with the same allowlist must not silently change the task.

The packager should emit a machine-readable inventory plus a human-readable audit. An independent verifier should reconstruct the inventory from the released artifact rather than trusting the packager's summary.

## 4. Evaluator custody and grading

The evaluator-only bundle should contain, as applicable:

- hidden root-cause labels and competing/null hypotheses;
- required, supporting, contradicting and unresolved evidence spans;
- hidden functional and security tests;
- reference patch and approved alternative patches;
- plausible known-bad and wrong-scope patches;
- mutation operators and mutation seeds;
- severity, impact, exploit preconditions and expected trigger conditions;
- admissible edit regions and patch-quality constraints;
- evaluator revision, scoring code identity and adjudication provenance;
- source/dataset license and attribution records.

Evaluator artifacts should live in a private repository or service with a different access-control boundary from the public source and agent workspaces. Private answers must not be committed to a public branch, hidden behind a prompt, encrypted with a key available to the agent, or left in an artifact cache shared with the runner.

The grading process should be independently identified and versioned. The agent may not select its own evaluator revision, write grader state, or observe hidden assertions. A primary grade and a pristine independent regrade must have distinct execution identities and must refer to the exact same candidate patch digest.

### 4.1 Leakage audit

Before a task is accepted, search the agent view and task statement for answer leakage through:

- directory and file names, test names, module names and generated symbol names;
- Git metadata, branch names, commit IDs, remotes and patch context;
- build logs, package metadata, debug symbols and cached artifacts;
- public issues, pull requests, release notes and web-searchable fix descriptions;
- error messages, environment variables, process arguments and grader output;
- hidden test counts, mutation identifiers, oracle filenames or deterministic random seeds.

Use a unique canary secret in evaluator storage. A controlled negative test must prove that the agent process and all descendants cannot read it through path traversal, mount discovery, environment inspection, process inspection, IPC, network or inherited handles. This canary test is not a substitute for #10064; it is one acceptance test within that isolation qualification.

## 5. Oracle qualification

A task must not be called executable or scoreable until it passes all of the following:

1. **Reference-patch control:** the accepted reference patch passes the full declared acceptance gate.
2. **Empty-patch control:** the empty patch fails the gate it is intended to test.
3. **Plausible known-bad control:** an attractive but incorrect or insecure patch is rejected.
4. **Wrong-scope control:** an unrelated or overly broad patch cannot pass by changing the grader surface.
5. **Mutation control:** intentionally mutate or remove a hidden assertion and verify the oracle-integrity procedure detects it.
6. **No-answer control:** the evaluator reports missing, not-run, censored, invalidated and unresolved outcomes explicitly rather than converting them to numeric zero or success.
7. **Pristine regrade:** grade the exact submitted patch in a clean, independent environment and compare the two grades.
8. **Leakage control:** the agent-visible artifact contains no evaluator-only data or answer-bearing side channel.

The reference patch and empty patch provide the two-sided baseline. They are necessary, not sufficient: a broken oracle can accept the reference patch and reject the empty patch while still omitting the target defect. Independent review, known-bad variants and mutation sensitivity are required.

For security tasks, functional and security correctness are separate gates. Report at least: functional pass rate; secure pass rate over all tasks; secure pass rate conditional on functional success; and joint functional-and-secure pass rate. Do not quote a conditional security score as if it were the overall secure solve rate.

## 6. Benchmark and data-rights audit

Public leaderboard results are useful calibration references. They are not automatically reusable task fixtures or sealed holdouts. Keep three questions separate for each candidate source:

1. Is the benchmark harness/code licensed for reuse?
2. Are the benchmark annotations and dataset records licensed for reuse and redistribution?
3. Are the underlying repository files, issue text, tests and patches licensed for the intended use?

A permissive license on the harness does not relicense the underlying projects or automatically settle rights to expert annotations and dataset packaging. Track the exact upstream license and attribution per task; preserve required notices and record modifications. When the dataset-specific terms are unclear, keep it in a comparison-only lane until reviewed.

### 6.1 ContextBench (EuniAI issue-resolution context retrieval)

The intended reference is the EuniAI ContextBench project and its context-retrieval leaderboard, not the separate Facebook Research ContextBench for context-window retention.

The EuniAI repository states Apache-2.0 for the benchmark project code and documents 1,136 tasks across 66 repositories and eight languages, including 63 Rust tasks. Its public Hugging Face dataset exposes fields including gold_context, reference patch, test_patch, problem statement, fail-to-pass and pass-to-pass test information. The leaderboard reports Recall, Context F1 and Pass@1 separately; the current Recall leader and Context F1 leader are different systems. The official site also notes that run-specific task coverage and dataset revisions vary by submission.

**CEB use:** external retrieval calibration only. Do not treat the published train dataset as a secret holdout: gold context and reference artifacts are already public. The rendered dataset card did not expose a separate dataset license field in this audit. The Apache-2.0 repository license must not be interpreted as a blanket license for all annotations or upstream repository content. Reconcile the published count/configuration (the project describes 1,136 full tasks plus a 500-instance verified subset; the dataset viewer displayed 1,636 rows across its configurations) and freeze an exact dataset revision before quoting a local reproduction.

Sources:
- [EuniAI ContextBench repository and Apache-2.0 code license](https://github.com/EuniAI/ContextBench)
- [ContextBench leaderboard and methodology](https://contextbench.github.io/)
- [ContextBench Hugging Face dataset and visible fields](https://huggingface.co/datasets/Contextbench/ContextBench)
- [ContextBench paper](https://arxiv.org/abs/2602.05892)

### 6.2 Agent Security League / SusVibes

Endor Labs describes Agent Security League as an evolving leaderboard built around 200 security-relevant tasks from 108 open-source Python projects and 77 CWE classes. It distinguishes functional correctness from security correctness and describes anti-cheating controls, including workspace sanitization and detection of agents using repository history or retrieving known fixes. Its security result is conditional on functional success in the published definition.

**CEB use:** use the public leaderboard to calibrate functional/security divergence and anti-cheating methodology. Do not pool rows across versions or treat the public top score as directly matched to CEB. Before task import or redistribution, verify the exact SusVibes artifact version, license, per-repository terms and security-test custody. Keep the evaluator test set hidden from agent identities.

Sources:
- [Endor Labs Agent Security League leaderboard and methodology](https://www.endorlabs.com/research/ai-code-security-benchmark)
- [Agent Security League methodology and SusVibes background](https://www.endorlabs.com/learn/agent-security-league-evaluating-the-security-of-ai-coded-software)
- [SusVibes paper](https://arxiv.org/abs/2512.03262)

### 6.3 CRBugBench

Microsoft Research describes CRBugBench as 1,000 human-verified review tasks from 98 open-source repositories plus 800 additional tasks from one company's proprietary repositories. Its October 2026 study reports a best top-five bug-detection result of 71.9%, with weaker performance on complex/non-local bugs and private repositories. The study also reports an exploration-prompt improvement of up to 5.8% at 10% higher cost.

**CEB use:** comparative research reference for pre-merge review, especially non-local reasoning and realistic project context. The proprietary 800-task portion cannot be assumed available. No public redistribution license or complete task package was established in this audit; do not copy CRBugBench tasks or hidden annotations into CEB. Reproduce the evaluation concept with independently sourced, rights-cleared tasks.

Source:
- [Microsoft Research CRBugBench publication, October 2026](https://www.microsoft.com/en-us/research/publication/when-should-a-software-factory-be-lights-off-benchmarking-code-review-agents-on-bug-detection/)

### 6.4 Rust-SWE-bench / RustForger

Rust-SWE-bench and the RustForger study provide Rust-specific repository-level evaluation, with a historical report of 500 tasks across 34 Rust repositories and a 28.6% result for the studied RustForger + Claude Sonnet 3.7 configuration. This is a historical comparison, not a current frontier leaderboard result.

**CEB use:** use for Rust task-family design and matched Rust-specific baselines. Freeze the exact benchmark revision, task set, agent implementation, toolchain and scoring policy before comparing results. Audit task and repository licenses individually; do not assume that the existence of a public GitHub repository gives permission to repackage all source, task statements, patches and tests.

Sources:
- [Rust-SWE-bench / RustForger repository](https://github.com/GhabiX/Rust-SWE-Bench)
- [Rust-SWE-bench paper](https://arxiv.org/abs/2602.????)

The paper URL above is intentionally not guessed. Resolve the exact citation and benchmark revision from the repository/paper metadata before publication.

### 6.5 SWE-Bench Pro V2

Scale's September 22, 2026 V2 release describes 642 tasks across 11 repositories after removing 89 invalid tasks. It reports correction of 69 instruction/test contradictions, disables web tools in the agent phase, requires reference-patch pass and empty-patch fail before release, and regrades submitted diffs on a pristine image. It also documents residual protocol risks and examples caught by the pristine regrade.

**CEB use:** adopt its two-sided release qualification and pristine regrade as baseline controls. Do not treat any leaderboard percentage as a directly matched CEB comparison. Track each upstream repository license and the benchmark artifact's actual terms separately.

Source:
- [SWE-Bench Pro V2 methodology and leaderboard](https://labs.scale.com/leaderboard/swe_bench_pro_public_v2)

## 7. The CEB benchmark ladder

Use different lanes for different scientific purposes:

| Lane | What belongs here | Permitted claim |
|---|---|---|
| External calibration | Public benchmarks with published labels/results | Reproduction or comparison under matched protocol |
| Public development | Rights-cleared public synthetic and historical tasks, with known-answer controls | Engineering iteration; not a sealed generalization result |
| Private confirmatory holdout | Newly authored or licensed tasks with evaluator-only labels and access control | Confirmatory performance on the declared held-out distribution |
| Longitudinal regression | New tasks and controlled mutations added after release, with versioned source snapshots | Drift, retention, robustness and regression over time |
| Operational deployment | Authorized customer/production incidents with consent and privacy controls | External validity only within declared domain and governance constraints |

Stratify splits by repository, project family, root-cause family, time window and near-duplicate cluster. Do not randomly split examples from one upstream issue, fork, patch family or repository into both train and holdout. Use post-cutoff tasks where possible, and maintain a contamination register. Public benchmarks whose labels and patches are already released are never a hidden holdout.

Start with a manually inspectable seed set: 12–24 rights-cleared tasks, balanced across context acquisition, contract discovery, non-local review, security, Rust semantics and operational/debugging. Each task should have at least one adversarial or negative-control case. Expand only after oracle quality and leakage audits pass.

## 8. Research paradigm shifts and measurable hypotheses

The aim is not to produce another agent that writes more patches. The candidate paradigm is **evidence-qualified engineering intelligence**:

1. **From patch completion to proof-carrying repairs.** Every repair carries a minimal claim/evidence graph, exact changed-scope justification, tests, independent review and known limitations. Measure evidence coverage and unsupported-change rate, not patch size.
2. **From context volume to minimal sufficient context.** Retrieval should optimize useful evidence per token and causal coverage, not Recall@k alone. Measure Context F1, sufficiency, irrelevant-token fraction and downstream claim accuracy at equal budget.
3. **From one-agent confidence to structured disagreement.** Separate proposer, skeptical verifier, security reviewer and independent grader roles. Preserve disagreement and provenance; do not average conflicting answers into false certainty.
4. **From passive debugging to controlled discrimination.** The agent predicts before acting, chooses a low-risk experiment, compares intact/lesion/sham/rescue or equivalent matched arms, and reports non-identifiability when evidence cannot distinguish causes.
5. **From unconditional autonomy to calibrated act/abstain/escalate.** Reward correct abstention when the oracle is unavailable, causal identification is impossible, or risk exceeds authority. Track false action, missed action, escalation utility and calibration separately.
6. **From benchmark pass rate to operational value.** Measure expert-normalized time saved, confirmed defect recall, severity-weighted false negatives, human review burden, cost, latency and regression risk. A system that solves more tasks by creating hidden security defects is not better.

A distinctive scientific result would be a measured improvement in the joint chain:

**minimal sufficient context → accurate specification → discriminating investigation → admissible minimal repair → independent functional/security verification → reproducible receipt.**

No current evidence establishes that Symthaea achieves this chain. The chain is the falsifiable research target.

## 9. Benchmark metrics and target gates

Keep metrics disaggregated by task family, severity, language, repository, task age and evidence authority. Targets are prospective gates, not current Symthaea results.

| Measure | Pilot target | Sealed holdout target |
|---|---:|---:|
| Context Recall@10 | >=0.85 | >=0.90 |
| Context F1 | >=0.45 | >=0.50 |
| Evidence-backed authorization claims | 100% | 100% |
| Review defect Recall@5 | >=0.80 | >=0.85 |
| Actionable review precision | >=0.70 | >=0.80 |
| High-severity false-negative ceiling | 0 in pilot | 0 in holdout, or explicit non-qualification |
| Secure task pass, with functional noninferiority | >=0.55 | >=0.65 |
| Rust-SWE resolution | >30% | >35%, only in a matched, frozen replication |
| Paired act/abstain accuracy | >=0.75 | >=0.85 |
| Human-agent review complementarity | +5 pp defect recall at equal effort, or noninferior recall with >=20% less human review time | Confirmed with preregistered paired design |
| Expert-normalized speedup | >=0.50x | >=0.80x |

These are stretch targets to test, not numbers derived from current model capability. For rare high-severity failures, zero observed misses on a tiny sample is not a statistical guarantee; report confidence intervals and sample sizes. Do not hide critical failures inside a weighted average. Report pass@k separately from pass@1 and per-task reliability.

## 10. No-runner sequence

Work that can proceed statically:

1. Review the source-only allowlist and path-lifecycle rules on the #10173 fixture.
2. Prepare a synthetic, rights-cleared task specification and independent evaluator bundle outside the agent tree.
3. Build a deterministic manifest verifier that only reads packaged bytes; do not make it execute source, build scripts or candidate code.
4. Audit the artifact and dataset rights table per task before copying third-party content.
5. Add leakage canaries and negative-control design to the task qualification checklist.
6. Reconcile agent-view and evaluator artifact references with #10370 and existing receipts.
7. Obtain independent review of custody, rights and the validator semantics.
8. Only after #10064 qualifies isolation, test the canary boundary and run the reference/empty/known-bad oracle controls.

Do not mark a task runnable because its JSON parses, its manifest hashes match, or its tests are present. Packaging validity, isolation, oracle validity and candidate correctness are separate claims.

## 11. Immediate decision

The next implementation target is not a new autonomous-agent subsystem. It is the smallest deterministic, source-only task materializer plus independent manifest verifier, designed around the existing task and receipt interfaces. Begin with one synthetic Rust fixture and no third-party repository data. Keep its evaluator bundle in a separate private trust domain. Promote it only after independent custody review and the #10064 execution gate.

