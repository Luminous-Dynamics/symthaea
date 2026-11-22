# MACE-CoS Guard v0.1 — protocol

**Status:** deterministic structured-input research reference, dated 2026-11-11.

## Research question

Can a composition gate reject information-flow, context-boundary, target-binding and authority-laundering defects that are invisible when skills are evaluated individually?

## Request contract

A request contains four inputs:

1. **Context:** request ID, principal, tenant/scope, purpose, audience, expected target, current capabilities, evaluation time, exposure budget and maximum action label.
2. **Skill registry:** content digest, status/expiry, scope/purpose/audience applicability, readable data ceiling, approved exposure modes, required capabilities, declared conflicts, capability grants and per-skill token budget.
3. **Source manifest:** artifact ID/digest, data label, scope, purpose, audience, target and currentness.
4. **Execution plan:** ordered steps, skill ID, input artifact IDs, output ID/label/target, exposure mode/audience/token cost and final sinks.

A step can consume only source artifacts or outputs from earlier steps. Forward references fail closed.

## Core invariants

- **Local applicability:** selected skills must be active, unexpired and applicable to the request. Scope/purpose agnosticism requires explicit flags; audience is never an implicit wildcard.
- **Monotone classification:** the prototype uses public < internal < confidential < restricted. Each derived output gets the maximum of its declared label and all input labels. No declassification path exists.
- **Context continuity:** purpose is preserved; non-public artifacts must remain within the request scope and audience.
- **Target continuity:** targets are inherited from inputs. Multi-target inputs and rebinding fail.
- **No capability minting:** a skill may require existing principal capabilities but cannot grant them.
- **Separate effect authority:** an external action requires a separate receipt structurally bound to the exact principal, scope, action, target, argument digest, required capability, expiry and one-use nonce.
- **Composition-wide conflicts:** conflicts are checked over selected skills, including one-sided declarations.
- **Exposure constraints:** step exposure mode and audience must be admitted; per-skill and whole-request token budgets apply.
- **Canonical evidence:** the report binds canonical digests for context, skill registry and source manifest plus the ordered plan and findings.

## Decision semantics

The validator returns accepted only when every declared step and sink passes. A rejection means the structured request violated the declared contract; it does not prove malicious intent.

## Trust boundary

This prototype checks schema, digest shape, content identity, labels, bindings and structural receipt fields. It does not authenticate issuers, raw source bytes, cryptographic signatures, the truth of an upstream verifier-status flag or the execution result of an external action. A digest is not proof of provenance or authority. Production enforcement requires an independently trusted broker and immediate pre-dispatch revalidation.
