# Symthaea Inference Fabric — IF-8 Deterministic Capability Router v1

Status: pure route-planning child of IF-7.

## Purpose

IF-8 turns the inference fabric from a provider priority list into an inspectable,
deterministic capability planner. It does not execute inference and does not mint
resource or execution authority.

The planner produces an ordered fallback plan. Runtime integration must still
acquire the selected provider's IF-7 resource reservation and later IF-2/IF-4
execution authority immediately before use.

## Two-stage theorem

Routing is strictly separated into:

1. **hard eligibility**;
2. **soft deterministic ranking**.

No score, quality observation, latency estimate, or learned success statistic can
override a hard rejection.

### Hard eligibility

Each candidate must first pass the existing IF-0 `InferencePolicy::admit()` checks,
including:

- semantic purpose;
- context window;
- streaming/tool/structured-output capability;
- raw information-flow restrictions;
- remote-execution policy;
- provider training/retention/routing policy;
- request monetary budget.

Remote-provider and community-peer candidates additionally require an IF-7 resource
guard snapshot. Candidates are rejected when the current credential epoch is
blocked, a local cooldown is active, request authority is exhausted, or token
authority cannot cover the planned conservative exposure.

Local-process/device/network routes do not require an IF-7 guard unless a caller
chooses to attach one.

## Planned token exposure

The v1 planner uses:

`estimated_input_tokens + max_output_tokens`

with checked arithmetic. Overflow fails explicitly.

This value is a conservative planning/reservation input, not a tokenizer proof.
Callers remain responsible for ensuring the input estimate covers all wire-relevant
context. A later tokenizer-aware estimator can strengthen this boundary without
changing router authority semantics.

## Deterministic scoring

Only candidates that survive hard eligibility are scored.

All v1 score components and weights are integers. There is no floating-point route
ordering, randomness, or input-order tie break.

Inspectable components are:

- quality observation;
- reliability observation;
- locality preference;
- resource headroom;
- cost efficiency within the already-authorized monetary envelope;
- latency observation.

Unknown quality/reliability/latency observations receive a neutral fixed-point
prior rather than being treated as proven good or bad.

Quality, reliability, and latency are **observations**, not authority.

## Sovereign preference

The default weights intentionally favor local execution, but locality remains a
preference rather than an authorization rule. `InferencePolicy::sovereign_default()`
is the actual no-remote authority boundary.

Therefore a high-quality remote model can never outvote sovereign policy. Under a
policy that explicitly admits remote execution, a remote route may rank first if
its soft score is better.

## Quota conservation

Among otherwise similar remote candidates, lower remaining resource headroom is
penalized. This helps preserve scarce free-tier capacity for requests where it is
more useful.

The headroom score does not replenish or mutate quota. Only IF-7 owns that state.

## Stable tie breaking

Equal scores are resolved by deterministic semantic fields:

1. quality;
2. reliability;
3. lower admitted monetary charge;
4. more local execution class;
5. stable provider/model identity key.

Input array order is never used as a tie break.

## Ordered fallback plan

The planner returns all eligible routes in preference order rather than a single
backend handle. This supports graceful fallback without allowing the fallback path
to weaken privacy or capability requirements: every route in the plan has already
passed the same hard request policy.

## Planning is not authority

Resource state can change after planning. The preferred route must still call the
actual IF-7 `reserve()` operation. A stale plan cannot spend exhausted quota.

Likewise the routing plan cannot create an IF-2 permit or bypass the IF-4 executor.

## Relationship to IntelligentDispatcher

The existing code-generation `IntelligentDispatcher` remains unchanged. Its useful
ideas—success observations, energy-aware preference, and fallback—can later feed
IF-8 metrics, but adaptive observations must stay on the soft side of the hard
policy/resource boundary.

## Explicit non-claims

- no execution;
- no resource reservation inside the planner;
- no credential ownership;
- no provider registry/presets;
- no adaptive learning update loop;
- no tokenizer-exact exposure estimator;
- no current `IntelligentDispatcher` replacement;
- no current `LLMBackend` factory/routing change;
- no claim that quality/reliability/latency observations are trustworthy facts.
