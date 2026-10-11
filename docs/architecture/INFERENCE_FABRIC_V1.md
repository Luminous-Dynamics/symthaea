# Symthaea Inference Fabric v1 — IF-0 Semantic Contract

Status: contract-only foundation.

## Invariant

Symthaea owns cognition, policy, and execution authority. Inference backends only
provide bounded computational capabilities.

A configured credential is not permission to export data. A provider/model must
pass explicit request-purpose, information-flow, capability, privacy, routing,
and cost admission before it can become an execution candidate.

## Information-flow boundary

Raw episodic memory, raw cognitive state, identity secrets, and authentication
secrets are non-exportable by construction. Policy flags cannot override that
boundary. Future remote minimization must create a distinct `RemoteSafeDerived`
payload through a separately qualified declassification step.

## Fail-closed provider policy

When a user/deployment disallows provider training, retention, third-party
routing, or monetary cost, `Unknown` provider claims do not count as compliance.
The candidate is rejected.

## Model identity

The contract distinguishes:

- content-verified local model identity;
- provider-attested remote model identity;
- opaque endpoint identity.

Those evidence strengths must not be treated as equivalent in scientific or
reproducibility claims.

## Authority boundary

`AdmittedInferenceRoute` is policy-checked metadata only. It cannot execute a
request and is intentionally not an inference permit. A later IF tranche will
bind exact request/policy/provider state into short-lived one-use permits close
to execution, following the existing Symthaea pattern of separating admission
from authority.

External model output is untrusted evidence/proposal data. It must never mint
motor, filesystem, network, governance, financial, or other action authority.

## IF-0 explicit non-claims

- no runtime module export yet;
- no provider discovery;
- no provider HTTP changes;
- no credentials or secret storage;
- no quota telemetry;
- no retry/circuit-breaker logic;
- no permit minting;
- no remote declassifier implementation;
- no tool-execution authority;
- no claim that any current provider satisfies a policy profile.

The integration test compiles the contract directly so the semantic surface is
qualified independently before it is wired into the runtime module graph.
