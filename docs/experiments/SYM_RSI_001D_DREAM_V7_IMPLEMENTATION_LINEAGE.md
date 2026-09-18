# SYM-RSI-001D — Dream v7 Implementation Lineage

**Status:** implementation qualification target only. This document records code/evidence identity before any SYM-RSI-001D sealed measurement is consumed.

## Active stack

| Layer | Identity |
| --- | --- |
| Parent fresh-token / claim-sealing base | `feat/sym-rsi-c-fresh-token-v1` at `5c05fe028069e091b8c65da772fc88ba2ce9eaa6` |
| D-v7 integration branch | `feat/sym-rsi-dream-v7-structural-support-integration` |
| Corrected D-v6 source blob | `723e45ed90ea61d212dc37a66d7176dfe8fd413f` |
| Archived D-v6 path | `docs/experiments/archive/SYM_RSI_001_GROUNDED_DREAM_V6.rs.txt` |
| D-v7 source blob | `795e186bcf70d1b64d9a16f92c9475e9f097ac5f` |
| Structural-support source blob | `6440648feb0e357c9585dbc09e2efc8e8ee61c55` |
| Active facade blob before this document | `3ccc96c97bc36a425fd5dc9640668d9a627e9dd9` |
| V7 preregistration document blob before claim-boundary clarification | `563b1be64f91093f0523b6b13de62b96f6711d3c` |
| V7 preregistration after transition-predictor clarification | `a3d3108a057addba5b1896d9cf8a2052f370b9f4` |

The corrected v6 source is preserved byte-for-byte as documentation/evidence rather than as an orphan live Rust module. The active module facade resolves only the v7 implementation.

## Frozen v7 code identities

The active implementation declares:

- grounded dream schema: `symthaea.sym-rsi-001.grounded-dream-model.v2`;
- policy: `sym-rsi-grounded-dream-policy-v7`;
- model version: `symthaea-dream-transition-memory-v4-structural-support`;
- prediction provenance namespace: `symthaea.sym-rsi-001.dream-prediction.v5`;
- structural support schema: `symthaea.sym-rsi.structural-support-index.v1`.

The model evidence digest commits to the frozen TrainingReplay corpus identity, transition-memory contents, action-support census, structural-support evidence digest, model schema/version, action-fingerprint semantics, and v7 scoring rule.

The prediction provenance digest additionally commits to the model evidence identity, structural-support evidence identity, state digest, action, raw predicted quality, structurally grounded quality, structural support, failure probability, model confidence, training action-support count, actionability, and exact model-call count.

## Qualification dependencies

D-v7 may not be admitted to the canonical 301–304 measurement lineage unless all earlier public-lineage gates remain satisfied:

1. canonical 1–8 TrainingReplay corpus and candidate family;
2. exact canonical C training-selection receipt;
3. independently recomputed 101–104 parent holdout with `FreshExecutionEligible`;
4. `ParentCQualification` minted from those frozen sources;
5. D-v7 implementation and structural-support qualification cleanly passing format/compile/test/Clippy and relevant governance checks.

The canonical lineage must not consume 401–404 until the 301–304 dream verification gate passes. It must not consume 1201–1204 until the parent-qualified fresh D-vs-C receipt is frozen.

## What “sealed” means in this open-source experiment

The fixture code, seed numbers, and deterministic transition functions are open source. A developer with source access can always call lower-level fixture transitions or write a separate program that evaluates a published seed. Local API visibility therefore cannot provide cryptographic outcome secrecy.

For SYM-RSI-001/001D, **sealed** means evidence-lineage discipline:

- the canonical experiment implementation, tests, qualification jobs, and evidence-producing workflows must not evaluate a sealed partition before its preregistered authorization gate;
- canonical receipts must be produced only through the qualified measurement path and bind the required predecessor evidence;
- generated or manually inspected off-protocol outcomes are not admissible as canonical evidence;
- if a sealed outcome is manually evaluated, inspected, or used to tune the subject before authorization, the affected preregistered lineage is contaminated and must not be reported as untouched fresh evidence;
- token-gated APIs enforce the intended canonical evidence path, not secrecy against a developer who deliberately bypasses the protocol.

A future stronger experiment may use a commit–reveal or externally held blind-evaluation service so subject developers cannot know evaluation seeds/outcomes until after code and preregistration are frozen. That would be a new protocol version rather than a retroactive reinterpretation of SYM-RSI-001D.

## Epistemic boundary

D-v7 separates **support/trust** from the transition predictor:

- v7 structural support is the preregistered quality-independent 7-D same-domain/same-action support metric;
- `symthaea-dream` transition retrieval still operates in its existing quality-bearing 16-D model state;
- transition-retrieval similarity is not structural support, confidence, or support distance;
- generated predictions remain non-empirical and cannot populate TrainingReplay support or `ExperienceTree`.

A future version may align structural support and transition retrieval to the same observed training transition, but v7 may not be changed that way after inspecting sealed v7 outcomes.

## Unconsumed partitions

At the time this lineage document is created, no code or test in this stack intentionally consumes:

- C-vs-A fresh: 201–204;
- D verification: 301–304;
- D-vs-C fresh: 401–404;
- D OOD: 1201–1204.

Ordinary unit tests may use TrainingReplay and the already-designated 101-series parent held-out replay substrate only.

## Claim boundary

A green implementation/CI result qualifies code execution only. It is not scientific evidence that D improves C.

A later positive 301–304 verification permits canonical fresh execution but is not the primary result. A later positive 401–404 result is a bounded D-vs-C result under the frozen fixture protocol. A two-stage recursive-improvement claim additionally requires the separately qualified C-vs-A fresh stage and the dual-token synthesis gate.
