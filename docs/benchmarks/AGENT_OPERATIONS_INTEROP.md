# Agent-Operations Benchmark Interoperability Plan

**Status:** proposal; no AOBench run has been executed by this document.

## Scope and naming

This document describes how Luminous systems could be evaluated against the public AOBench project as an external benchmark. It does not claim ownership of that project, fork its corpus, or redefine its name.

The public AOBench benchmark is maintained at https://github.com/MSKazemi/aobench. Its benchmark card currently documents version 0.4.1. It is an HPC operations benchmark, not a general agent benchmark and not a human-AI collaboration benchmark.

Do not label a Luminous evaluation as an AOBench result unless it was executed using the exact upstream release, task corpus, scoring implementation, and reporting requirements.

## Upstream capability baseline

The current public benchmark card describes 88 tasks, 29 environment bundles, five requester roles, ten QCAT categories, and seven weighted dimensions. The default profile covers outcome, governance/RBAC, tool use, grounding, robustness, workflow, and efficiency. An RBAC violation is a hard fail. Environments are deterministic frozen snapshots with mock HPC tools; six environments are grounded in real Marconi100 ExaData.

The upstream roadmap documents REST and MCP access, A2A evaluation, terminal evaluation, trace export, replay cassettes, in-toto attestation, cost and carbon accounting, pass^k, bootstrap confidence intervals, and planned injection-resistance and elicitation scorers. Version-specific status must be checked before any run.

AOBench itself states important limits: its mock environments do not expose real production consequences, its authorization policies are synthetic, role granularity is coarse, and a benchmark score is not a production safety certification or a substitute for security review.

## Proposed Luminous evaluation path

1. Pin the upstream AOBench release and verify its repository commit and corpus digest.
2. Use only the upstream development split unless the maintainers explicitly authorize another protocol.
3. Expose Symthaea through a documented adapter. Do not change the scenario, score, role, or environment semantics to suit the evaluated system.
4. Run the official baseline and the candidate adapter under matched model snapshots, sampling settings, timeouts, tool budgets, and retry policies.
5. Repeat runs according to the upstream submission requirements; report every run, not only the best one.
6. Publish hard failures, per-dimension scores, cost, latency, judge configuration, and uncertainty.
7. Treat a successful snapshot evaluation as pre-deployment evidence only. Follow with separate sandbox, concurrency, fault-injection, and site-specific authorization testing before any real operational use.

## Mapping to Luminous evidence architecture

| External benchmark dimension | Luminous evidence to retain |
|---|---|
| Outcome | Verified final state, not merely a plausible answer |
| Governance / RBAC | Principal, role, allowed action set, denied-action attempts, and authorization receipt |
| Tool use | Tool name, arguments, call order, result, and error path |
| Grounding | Source references and a claim-to-evidence map |
| Robustness | Repeated-run outcomes, pass^k, and confidence interval |
| Workflow | Event graph, dependency order, and verified postconditions |
| Efficiency | Wall-clock latency, model/tool cost, tokens, and resource use |

The Luminous execution model should preserve the sequence: intent → authority → execution → observation → independent verification → reproducible evidence. An unknown execution outcome must not be interpreted as success; reconcile state before any subsequent mutation.

## Gaps worth researching

The upstream AOBench benchmark card identifies a gap between deterministic mock evaluation and real operational consequences. A Luminous extension should not silently patch this gap inside the upstream score. Instead, develop a separate, clearly named follow-on evaluation for:

- State-changing tools with explicit preconditions and independently observed postconditions.
- Concurrency, stale observations, duplicate commands, retries, and partial failure.
- Unknown outcomes and mandatory reconciliation before a second mutation.
- Prompt-injection and compromised-data attacks across tool and document boundaries.
- Least-authority delegation, time-bounded grants, revocation, and audit completeness.
- Cross-domain operational tasks beyond HPC, with domain experts and site-specific policy packs.

These are candidate research directions, not completed capabilities. Keep any extended score separate from the official AOBench score.

## Reproducibility record

A public result must include the upstream version and commit, split, scoring profile, exact model snapshot, adapter revision, judge snapshot where applicable, run count, per-run artifacts, hard-fail count, cost, latency, and a reproducible command or endpoint. Do not compare scores across versions or profiles without a matched re-run.
