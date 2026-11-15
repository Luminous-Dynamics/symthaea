# Symthaea Agent Decision Protocol v0.1

**Status:** design contract; not implemented by the current service daemon  
**Purpose:** expose Symthaea's decisions to external benchmark harnesses without granting the reasoning service ambient tool authority.

## 1. Why this protocol exists

An operations benchmark must evaluate the system that actually makes the decisions. A tool server is not automatically an agent, and an MCP adapter can accidentally put another model in charge of the tool loop.

Symthaea currently has a JSON service gateway at `POST /v1/service`. Its `query` request calls `Symthaea::process` and returns a text response. The optional `context` field is currently ignored. The `symthaea-soma` package has a distinct LiteRT tool-use loop with a small built-in tool set, and the REPL has a separate sandboxed executor. None of these is yet a generic AOBench role-scoped agent protocol.

This proposal defines the missing seam:

**Symthaea proposes → benchmark harness authorizes and executes → harness returns observation → Symthaea updates its decision.**

The reasoning process does not receive credentials, direct environment access, or an execution callback.

## 2. Security and authority invariants

1. **No ambient execution authority.** The Symthaea decision service can return a proposal; it cannot invoke benchmark tools directly.
2. **Harness-owned authorization.** The harness supplies only tools allowed for the current principal and role. The harness re-validates the selected tool name, JSON arguments, schema, budget, and policy before execution.
3. **No authority inheritance.** A delegated specialist or replacement agent does not inherit another principal's role or permission grants.
4. **No hidden environment reads.** The decision service sees only the task brief, explicit constraints, approved evidence, and observations sent by the harness. It must not inspect benchmark snapshots, environment files, or external systems.
5. **Unknown means unknown.** A tool result with unknown execution state blocks every subsequent external side effect until independently reconciled.
6. **Untrusted observations remain data.** Tool output, documents, terminal text, and user-provided artifacts are untrusted content. They cannot override the harness's system instructions or authority policy.
7. **Human control remains explicit.** The service can ask, defer, suggest, or return a final answer. A benchmark action is not a human approval or an authorization receipt.
8. **Fail closed.** Malformed decisions, unavailable tools, invalid arguments, expired budgets, or missing authorization are rejected. The harness records the rejection and does not silently repair the proposal into a different action.

## 3. Request

The harness sends an `agent_turn` request containing immutable protocol and run identifiers; task ID, environment ID, role ID, goal and task brief; explicit constraints and decision boundaries; the role-filtered tool catalogue and each tool's input JSON Schema; a bounded event history with actor and trust classification; the current authority digest; and remaining decision, tool, and time budgets.

Do not send authentication secrets, raw access tokens, or unfiltered environment state. Tool schemas are descriptions of available capabilities; they do not authorize execution.

See `agent-bridge/request.schema.json`.

## 4. Decision response

The service returns exactly one decision:

| Decision | Meaning |
|---|---|
| `tool_call` | Propose one named tool invocation and arguments |
| `ask` | Ask a clarifying question that materially changes the next step |
| `defer` | State why the task cannot safely or correctly proceed |
| `final_answer` | Return the answer with evidence references and limitations |

A tool call proposal includes a unique call ID, selected tool name, JSON arguments, a concise purpose statement, and an idempotency key. The harness checks the tool name against the current allowed-tool list and validates arguments against the exact schema. It then calls the harness-provided registry, not a tool client supplied by Symthaea.

The purpose field is a concise, externally inspectable reason for the proposed action. It is not a request for or disclosure of hidden chain-of-thought.

See `agent-bridge/decision.schema.json`.

## 5. Observation and reconciliation

The harness returns a `tool_observation` event with the original call ID and tool name; status (success, failure, permission denied, or unknown); structured result and/or bounded text observation; precondition and postcondition digests where available; whether an external side effect occurred and whether it is reversible; and an observation digest.

A failure response does not prove that no side effect occurred. Unknown outcomes are handled as follows:

1. Mark the call unresolved.
2. Block new tool calls and all other external side effects.
3. Reconcile the state through an independent observation path.
4. Append an `execution_reconciled` event that references the original call ID and includes the observed state digest and final status.
5. Only then resume the agent loop.

A tool observation can contain prompt injection. The adapter must delimit it as untrusted data and preserve source provenance.

See `agent-bridge/observation.schema.json`.

## 6. AOBench adapter design

The adapter is a thin compatibility layer, not a second reasoning agent:

1. AOBench supplies `ExecutionContext`, task, role, environment ID, and role-filtered `context.tools`.
2. The adapter converts this into the versioned request schema.
3. The Rust Symthaea service emits one decision.
4. For `tool_call`, the adapter validates the current task's permission boundary and calls only `context.tools.call(tool_name, method, **kwargs)`.
5. It records the raw tool call, tool result, policy decision, and relevant digests in the AOBench trace.
6. It sends the observation back to Symthaea for the next decision.
7. It stops on final answer, defer, timeout, exhausted budget, or unresolved unknown outcome.

A thin Python adapter may be required because AOBench's `BaseAdapter` is Python. Keep that file to protocol translation and AOBench trace construction; keep inference, prompt composition, decision validation, state management, and tool policy in Rust.

## 7. Experimental comparison

Evaluate these distinct systems rather than combining them:

- Existing Symthaea service query endpoint (text-only reference).
- Symthaea agent-turn protocol with role-filtered AOBench tool execution.
- AOBench `direct_qa` baseline.
- AOBench's documented OpenAI model adapters under the same task profile.
- The upstream MCP client adapter, but label the model that actually performs the agentic loop.

Pin the AOBench release, commit, benchmark split, profile, model snapshots, service build, prompt digest, tool registry, timeout, retries, and resource budget. Report hard failures separately.

## 8. Protocol versioning and compatibility

- Unknown fields are rejected in v0.1.
- A major version changes semantics or the authority boundary.
- A minor version may add optional fields and new decision types without changing existing meanings.
- Every decision and observation carries the protocol version and turn ID.
- The service and adapter must reject mismatched run, turn, call, task, or principal identifiers.
- A schema-valid response is not necessarily an authorized action. The harness's runtime policy remains authoritative.

## 9. Explicit non-claims

This document is a proposed integration contract. It does not claim that the current Symthaea daemon implements `agent_turn`, that the current AOBench adapter has been connected, or that any AOBench score has been achieved. It is a prerequisite for a fair benchmark of Symthaea as the decision-making agent.
