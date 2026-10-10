# Creditism Semantic Frontier and Versioning Contract

## Purpose

Freeze which Common Planet Creditism semantics a Mycelix/Symthaea experiment is actually testing.

Creditism is an evolving architecture. Current documentation and older working documents can differ materially. A scenario must therefore bind an exact semantic frontier rather than treating every document mentioning Creditism as simultaneously authoritative.

## 1. Current frontier

For `creditism-research-v0`, use the current 2026 Common Planet web architecture as the primary candidate mechanism source:

- https://common-planet.org/creditism
- https://common-planet.org/creditism/architecture
- https://common-planet.org/creditism/transition

The current architecture states, among other things:

- Personal Credit is allocated to people and deletes at defined use;
- buyer spending and contribution recognition are separate;
- Personal Credit is non-transferable, non-interest-bearing, non-collateralizable, non-investable and non-inheritable as concentrated power;
- Community Credit flows to community accounts by population plus equal personal directing shares, with only group accounts spending CC;
- Bonus flows as Bonus Personal Credit to members;
- Exchange deletes buyer Credit and separately credits the seller up to their acquisition amount;
- The Balance is an explicit price/credit-volume balancing mechanism;
- Housing is a stewardship/access system rather than a conventional market purchase;
- contribution qualification, verification and appeals remain open design questions.

These are profile inputs, not empirical conclusions.

## 2. Historical divergence must remain visible

An older 2025 Common Planet PDF describes Community Credit differently, including personal CC being directed to any group or person in a global crowdfunding-like model.

Source:
- https://common-planet.org/wp-content/uploads/2025/12/Creditism_-an-economic-evolution.pdf

This differs materially from the current architecture's statement that every person directs an equal share while **only group accounts spend CC**.

Do not silently reconcile the two documents.

Represent the relationship as:

`HistoricalProfile != CurrentProfile`

and preserve the older semantics as a named superseded candidate if comparative research requires it.

## 3. Semantic frontier object

Every Creditism scenario should bind:

- `profile_id`;
- `source_urls`;
- `source_retrieval_date`;
- `semantic_version`;
- `source_content_commitment` where practicable;
- `known_divergences`;
- `supersedes` / `superseded_by` relations;
- `unresolved_semantics`;
- `model_mapping_profile`.

Do not infer currentness from URL existence alone.

## 4. Supersession is not deletion

A revised Creditism document does not erase the scientific relevance of an older proposal.

An experiment should be able to compare:

- historical proposal;
- current proposal;
- hybrid proposal;
- mechanically isolated components.

This allows the research to distinguish:

specification evolution
vs
model failure
vs
implementation failure.

## 5. Source binding rule

Claims about what Creditism currently specifies must cite the exact frontier.

Claims about older Creditism proposals must identify the historical source explicitly.

Claims that combine multiple source versions require an explicit merge/mapping profile.

Never use:

`latest=true`

as a semantic authority.

## 6. Divergence-aware qualification

If a scenario is valid under Profile A but invalid under Profile B, report both dispositions.

Example:

`CC personal -> arbitrary recipient`

may be:

- valid under a historical profile;
- invalid under the current 2026 profile.

That is evidence about specification evolution, not an implementation contradiction.

## 7. Machine-readable example

`creditism-research-v0.json` should include:

- current source frontier;
- historical source references when relevant;
- explicit profile version;
- unresolved fields.

Any downstream result must inherit that frontier identity.

## 8. Claim ceiling

A semantic-frontier PASS establishes only that a scenario is bound to the declared source/profile version and does not silently mix superseded and current semantics.

It does not establish which version is correct, desirable, politically adopted, or economically viable.