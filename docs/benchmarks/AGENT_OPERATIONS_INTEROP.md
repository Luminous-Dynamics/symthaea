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

## Public leaderboard snapshot (2026-11-15)

The upstream public leaderboard currently lists one community result: Claude Sonnet 4.6 with mean score **0.622 across three runs** on the v0.4.1 development split (67 tasks), with **two hard failures** and an approximate cost of **$2.06 per run**. The three scores were 0.6132, 0.6192, and 0.6335. The upstream tool-free `direct_qa` floor is 0.334.

Important qualification: the submitted profile uses 42 `alpha1_grounding` tasks and 25 `default_hpc_v01` tasks. Robustness and workflow fields are unset in that submission. This is a single public community row, not a complete industry ranking or an uncontested state-of-the-art reference. Its result cannot justify broad claims about agent superiority, production safety, or performance outside the tested HPC task mix.

A fair Luminous comparison should first reproduce this exact release, task mix, model snapshot, scoring profile, and adapter conditions. Any new Luminous tasks or stricter safety checks must be reported as a separate extension, not folded into the upstream score.

Source: [AOBench v0.4.1 public leaderboard](https://mskazemi.com/aobench/latest/leaderboard/) (checked 2026-11-15).

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

## Adapter semantics and the MCP trap

The upstream AOBench MCP adapter is an **MCP client with an OpenAI agentic loop**, not a transparent executor for the MCP server's own reasoning. It connects to the supplied MCP server, discovers that server's tools, then presents those tools to an OpenAI-compatible model which makes the calls. A bare Symthaea MCP tool server evaluated through this adapter would therefore measure the OpenAI model driving those calls unless the integration explicitly routes decision-making through Symthaea.

There are three defensible routes:

1. **OpenAI-compatible endpoint:** use this only if Symthaea exposes an endpoint that itself performs the full reasoning/tool-use loop. Existing outbound OpenAI-compatible client code is not evidence that such an inbound endpoint exists. Verify the serving contract before selecting this route.
2. **Custom AOBench adapter:** implement the upstream `BaseAdapter.run(context) -> Trace` interface. The adapter receives the task, role, environment, run ID, and role-filtered `context.tools` registry. Use `context.tools` for all benchmark tool calls; never read environment snapshots directly or bypass the registry. The adapter should call a documented Symthaea Rust service/sidecar for reasoning, then translate its decisions and tool observations into the upstream Trace schema.
3. **MCP client adapter:** use only when the MCP server represents the evaluated agent's actual decision-making surface, rather than merely exposing raw tools for a different LLM to operate. Explicitly report which model is making tool decisions.

The custom adapter may be a thin, research-only Python harness because AOBench's extension interface is Python. Keep production reasoning and benchmark-specific logic in Rust; do not let the adapter become a second agent. The adapter must preserve the upstream role-filtered tool boundary and record every tool call, observation, error, permission denial, model snapshot, and token/cost field available.

### Minimum adapter acceptance tests

- A role-denied tool call is recorded as a permission failure and never retried through an alternate path.
- An unknown tool outcome blocks the next state-changing call until an independently observed reconciliation.
- Tool calls are made only through the provided role-filtered registry.
- A trace includes the exact task ID, role, environment ID, run ID, adapter revision, and final answer.
- Model snapshot, prompts, sampling settings, timeout, retries, tokens, latency, and cost are pinned.
- The AOBench `direct_qa` baseline is run under the identical task/profile configuration.
- A test proves the candidate agent—not an external LLM hidden inside the adapter—makes the actual decisions being evaluated.

Upstream references:
- [Evaluate your own agent](https://github.com/MSKazemi/aobench/blob/main/docs/guides/evaluating-your-own-agent.md)
- [MCP client adapter implementation](https://github.com/MSKazemi/aobench/blob/main/src/aobench/adapters/mcp_client_adapter.py)
- [Base adapter interface](https://github.com/MSKazemi/aobench/blob/main/src/aobench/adapters/base.py)
- [Trace schema](https://github.com/MSKazemi/aobench/blob/main/src/aobench/schemas/trace.py)

## Reproducibility record

A public result must include the upstream version and commit, split, scoring profile, exact model snapshot, adapter revision, judge snapshot where applicable, run count, per-run artifacts, hard-fail count, cost, latency, and a reproducible command or endpoint. Do not compare scores across versions or profiles without a matched re-run.
