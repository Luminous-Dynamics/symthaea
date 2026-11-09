# Symthaea Gateway Process Model
**Status:** research/design specification v0.1  
**Scope:** cognitive awareness of safety screening, authorization, provider dispatch, and execution evidence  
**Non-goal:** giving the cognitive model authority to bypass deterministic policy or execute providers directly

## 1. Terminology: two gateways, separate contracts

The repository currently contains two distinct concepts that must not be collapsed.

1. **In-process SafetyGateway** ([`src/safety/gateway.rs`](../../src/safety/gateway.rs)) is a facade over the AmygdalaActor fast pattern veto and SafetyGuardrails hypervector checks. Its `SafetyCheck` inputs are query text, command text, or an `ActionIR`; its `SafetyDecision` is a screening result. The module documentation says it does not execute actions.
2. **Provider execution gateway** ([REA-RUST-020 / #10398](https://github.com/Luminous-Dynamics/symthaea/issues/10398)) is the authority-enforcing boundary immediately before API, CLI, or MCP provider dispatch. It re-derives policy, binds caller/session/consent/provider/arguments, atomically claims a one-shot invocation, invokes a private executor, and emits a terminal receipt. [MOBILE-REA-002 / #9838](https://github.com/Luminous-Dynamics/symthaea/issues/9838) separately defines a read-only mobile-analysis profile.

A positive SafetyGateway result is **not** authorization. Authorization is **not** proof of dispatch. A provider response is **not** proof that the external world reached the claimed state. Those facts require distinct records and verifiers.

## 2. Why Symthaea should know the process

A cognitive system that can plan tool use should have an accurate, current, machine-readable model of:
- which capabilities exist and their declared effect class;
- which operations are allowed in the current profile and why;
- which consent, session, target, provider revision, and policy bindings are required;
- what phase an invocation reached;
- why an invocation was denied, expired, replayed, cancelled, unavailable, or timed out;
- what evidence was actually observed, what remains unverified, and what reconciliation is required.

Without this model, Symthaea may repeatedly request impossible actions, misreport a denial as an execution, assume a timeout means failure, or treat software-reported observations as independently verified facts.

The model exposed to cognition must be **read-only and non-authoritative**. It informs planning and explanation; it does not make or override authorization decisions.

## 3. Typed process model

Represent these as separate types or equivalent closed enums, not one `allowed: bool`.

### 3.1 Screening decision
- `NotEvaluated`
- `ScreenedAllowed`
- `ScreenedBlocked { category, rule_id }`
- `ScreeningUnavailable`

This is the output of pre-cognitive or action screening. A screening pass only means that this layer did not veto the request.

### 3.2 Authorization decision
- `NotEvaluated`
- `Authorized { policy_digest, effect_catalog_digest, consent_digest, expires_at }`
- `Denied { reason_code }`
- `Expired`
- `Revoked`
- `Indeterminate`

Unknown policy, missing effect metadata, missing caller/target binding, unavailable policy storage, or an unverifiable consent record must resolve to denial/indeterminate-with-no-dispatch, never implicit permission.

### 3.3 Invocation lifecycle

Use a closed, append-only lifecycle with explicit terminal outcomes:

`Requested → CatalogBound → ContextBound → PolicyRevalidated → ClaimCommitted → Dispatched → Executed`

Alternative terminal outcomes include:
- `Denied`
- `Expired`
- `ReplayRejected`
- `ProviderUnavailable`
- `CancelledBeforeDispatch`
- `CancelledAfterDispatch`
- `TimedOut`
- `ExecutorFailed`
- `ReconciliationRequired`

Do not collapse `TimedOut`, `ExecutorFailed`, and `ProviderUnavailable` into “not executed.” Once dispatch may have occurred, an unknown outcome blocks another mutation until an explicit reconciliation process determines whether a new invocation is safe.

### 3.4 Observation and verification
Each observation carries an epistemic grade:
- `Unobserved`
- `AgentReported`
- `ProviderReported`
- `HostObserved`
- `IndependentlyVerified`
- `HardwareAttested`

These grades are not interchangeable. ADB/provider output remains software-reported observation unless a separate trusted mechanism verifies it.

## 4. Minimum read-only cognitive projection

Expose a narrow `GatewayContext` to planning and explanation. Suggested fields:

- opaque `invocation_id` and `session_alias` (never stable device identifiers);
- canonical operation ID and declared effect class;
- capability/profile identifier;
- screening, authorization, lifecycle, and verification states as separate enums;
- policy, effect-catalog, provider-revision, and argument digests;
- issue, expiry, dispatch, completion, and reconciliation timestamps;
- stable reason codes and bounded human-readable explanations;
- result digest, limitations, and references to immutable receipts;
- explicit flags for `mutation_may_have_occurred` and `reconciliation_required`.

Never expose secrets, raw credentials, stable hardware identifiers, unrestricted filesystem paths, or unbounded provider output to the cognitive planner by default. The projection must be derived from verified receipts rather than natural-language tool descriptions.

## 5. Architecture and authority boundary

```text
Human intent / trusted caller
  → cognitive planner (may propose an operation)
  → canonical operation catalog + effect metadata
  → SafetyGateway screening (veto / risk signal only)
  → deterministic policy + consent + session re-derivation
  → atomic one-shot claim store
  → private provider executor
  → bounded observation + immutable receipt
  → independent receipt verifier / reconciliation
  → read-only GatewayContext returned to Symthaea
```

- The planner may select only from a closed operation catalog.
- A language model must never manufacture a consent receipt, policy digest, session alias, claim-store result, provider identity, or execution receipt.
- The provider executor must not be reachable directly by adapters or the cognitive agent.
- A failed or unavailable safety/policy/claim-store component must fail closed.
- The safety screening layer must not silently become a substitute for capability policy or OS-level sandboxing.
- Host-side permissions, process isolation, bounded I/O, and target-level safeguards remain independent controls.

## 6. Current implementation caveats to preserve in the model

The present SafetyGateway is useful as a common screening facade, but its current implementation should not be described as a complete security boundary:
- It checks a limited set of textual patterns and forbidden path prefixes.
- Its text-to-HV path hashes words and bundles deterministic pseudo-random vectors; this is not a demonstrated natural-language security classifier.
- Lexical path normalization is not equivalent to race-safe filesystem containment or symlink-safe execution.
- Command-string screening does not prove the semantics of every executable, argument, environment variable, working directory, shell expansion, or downstream process.
- A structured `ActionIR` decision does not prove that every actual execution path invokes the gateway.
- The SafetyGateway module explicitly does not execute actions.

These are reasons to layer controls and add bypass tests, not reasons to discard the facade.

## 7. Benchmark and negative-control programme

### Dispatch integrity
- 100% of supported provider adapters (direct API, CLI, MCP) call the same execution boundary.
- Zero provider side effects across all denied, stale, replayed, unknown-effect, or unavailable-store controls.
- Exactly one dispatch for each valid one-shot claim under parallel races.
- Zero successful fallback to a more privileged provider or shell path.
- Zero authorization reuse after session, provider, argument, policy, or consent change.

### Lifecycle truthfulness
- 100% of attempts produce one terminal disposition or an explicit unresolved state.
- Zero claims of success on timeout, cancellation after dispatch, or uncertain provider outcome.
- Every uncertain mutation blocks further mutation until reconciliation.
- Every receipt binds operation, actor, scope, target alias, provider revision, argument digest, policy/effect digests, time, result digest, and limitations.

### Screening quality
Report false-positive and false-negative rates by attack family, not only aggregate accuracy. Include command aliases, whitespace/Unicode substitutions, shell metacharacters, path traversal, symlink and TOCTOU cases, nested ActionIR, unknown operations, prompt injection in tool output, malformed receipts, and adversarially phrased intent. Do not treat a high screening score as proof of dispatch security.

### Performance and reproducibility
Report p50/p95/p99 for screening, policy re-derivation, claim-store transaction, provider dispatch, and total latency separately. Pin binaries, provider revisions, catalogs, schemas, database migrations, policies, fixture digests, workload seeds, and hardware. Publish model/in-memory results separately from durable multi-process PostgreSQL results and from physical-device qualification.

## 8. Immediate engineering sequence

1. Keep #10398 as the source of truth for the execution boundary; do not fork its transactional-claim contract.
2. Add a typed, read-only cognitive projection over gateway state and immutable receipts.
3. Add stable machine-readable reason codes; keep explanatory prose separately.
4. Add a mock provider with side-effect tracing and direct-API/CLI/MCP bypass controls.
5. Prove replay and concurrency behavior with a durable shared claim store and restart tests.
6. Add an evidence reconciler for post-dispatch uncertainty.
7. Add prompt-injection and malformed-provider-output tests at the cognitive boundary.
8. Only then evaluate measured latency and usability trade-offs.

## 9. Acceptance and claim ceiling

The planner can accurately describe what it may request, what the gateway allowed, what actually ran, and what remains unknown. It cannot authorize itself or upgrade an observation's evidence grade.

Passing synthetic tests qualifies only the tested software contracts. It does not prove genuine human consent, provider binary integrity, physical-device state, or hardware-backed attestation. Those require separate evidence.

## References
- [Symthaea SafetyGateway](https://github.com/Luminous-Dynamics/symthaea/blob/main/src/safety/gateway.rs)
- [REA-RUST-020 shared provider gateway](https://github.com/Luminous-Dynamics/symthaea/issues/10398)
- [MOBILE-REA-002 read-only REA gateway](https://github.com/Luminous-Dynamics/symthaea/issues/9838)
- [CONSCIOUSNESS-ADV-001 theory-neutral evidence discipline](https://github.com/Luminous-Dynamics/symthaea/issues/8971)
