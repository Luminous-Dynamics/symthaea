# EPI-AUTH-FLOW-001 R5R5-R2 — broad discovery + exact production witnesses

## Purpose

R5R5-R2 is a fresh direct-child repair after R5R5 was superseded before qualification.

R5R5 materially improved discovery by adding both ordinal-bypass statuses (`Certain` and `Probable`), Broca's 1D and 4D gates, Liquid-Mamba controls, strict-code behavior, and LLM prompt authority. Further preflight found two limitations that should be explicit before a frozen inventory is built:

1. the generic coding LLM path derives an epistemic status from coding prediction confidence, packages that status together with Phi in `ConsciousnessContext`, and injects the resulting context into the LLM system prompt independently of `llm_organ`;
2. file-level lexical discovery excludes standalone `tests/`/`examples/`/`benches/`, but a production file can still be selected because of a lexical match inside an embedded `#[cfg(test)]` module. Therefore file-set membership is a conservative lexical surface, not itself proof of runtime reachability.

R5R5-R2 retains the broad R5R5 dynamic discovery unchanged and adds a second exact production-witness audit.

## Two-layer measurement model

### Layer A — broad conservative lexical discovery

`scripts/discover_epistemic_authority_flow.py`

Emits sorted file sets and path-set digests for the broad authority vocabulary/transport/consumer surface.

Interpretation:

```text
file appears in lexical surface
!= every matching occurrence is production/runtime
!= authority is semantically valid
```

The surface is intentionally conservative so new files cannot hide merely by changing call topology.

### Layer B — exact production witness audit

`scripts/audit_epistemic_expression_consumers.py`

Requires exact source witnesses for current production chains that matter to the repair program.

It explicitly freezes:

```text
coding prediction confidence
 -> confidence_to_epistemic
 -> ConsciousnessContext { epistemic_status, phi, ... }
 -> GenerationParams.consciousness_context
 -> LLM backend to_system_supplement()
 -> system prompt
```

and separates the verified-generation properties:

```text
compiled under execution profile
passed exact test suite
formal contract verified (optional)
attestation present (optional)
```

from the broader current vocabulary:

```text
compiled && tests_passed -> "guaranteed correct"
```

The audit records that vocabulary as debt; it does not assert that compile/test/formal verification is weak evidence. Those are strong but proposition-specific properties.

## Core theorems

```text
internal confidence / Phi / familiarity / resonance
!= source-admitted proposition authority
```

```text
Certain OR Probable
!= permission to bypass epistemic expression safeguards
```

```text
deterministic computation
!= generic correctness proof
```

```text
compiled + tests passed
!= semantic correctness outside the exact tested contract
```

```text
formal proof
establishes only the exact formalized proposition/profile
```

## Relationship to MEL-EPI

R5R5-R2 does not add another claim system.

The intended positive path remains:

```text
domain-native evidence
 -> source-specific verification/admission
 -> admitted MEL-EPI claim scope
 -> Symthaea expression policy
 -> Broca / external LLM translator
```

Internal model state may still influence search, routing, resource allocation, caution, or style. It cannot independently manufacture source-admitted factual authority.

## Qualification contract

An exact-head qualifier may establish only both measurement results on the exact frozen product:

```text
# broad discovery
schema=epi-auth-flow-001-r5r5-discovery-v1
authority_scope=measurement-only-producer-bypass-status-ordinal-cube-prompt-gates-and-controls
result=PASS_DISCOVERY

# exact production consumer witness layer
schema=epi-auth-flow-001-r5r5-r2-production-witness-v1
authority_scope=measurement-only-exact-production-consumer-witnesses
lexical_surface_semantics=conservative-file-level-membership-may-include-embedded-cfg-test-code
result=PASS_CONSUMER_WITNESS
```

Both complete outputs must be preserved.

Neither success state is a frozen inventory or semantic PASS.

## R6 handoff

Only after both executable measurements succeed should R6 be built mechanically as a fresh direct child of unchanged `main`.

R6 should freeze:

- every broad surface's exact ordered path list;
- path-set digest;
- exact Git blob identity (or exact byte hash) for every inventoried file;
- the production witness groups that establish current runtime/consumer chains;
- the lexical-surface interpretation above;
- exact R5R5-R2 product + qualifier run identities used to derive the inventory.

R6 may claim only `PASS_INVENTORY`.

## Exact base

`adb69f11fa8068b019cc5bb598d0c7726a197fc9`

Base tree:

`35a6c5fdba319556af9bb487838734f67c8ac0d6`

## Nonclaims

This tranche does not establish valid cryptographic verification, reproducibility, replication, axiomatic truth, foundational status, causal validity, generic computation correctness, generic code correctness, scientific truth, action authority, source authenticity, direct-expression authority, hedging-suppression authority, or gate-bypass authority.
