# MACE Claim-Relative Invalidation and Revalidation V1

**Snapshot:** 2026-11-26  
**Status:** Rust policy/reference module, schema and deterministic tests authored. Compilation, test execution, external benchmark reproduction and production deployment are not claimed.

## 1. Why claim-relative invalidation is a paradigm shift

A source revision is not the same thing as a claim becoming false.

A document, repository file, conversation event or upstream record may change while a specific proposition derived from it remains true. Conversely, a small revision can invalidate a critical claim even when a broad diff classifier calls the change behavior-preserving.

The September 2026 paper *Impact Is Not Invalidation* makes this distinction concrete. Its study reports 10,369 candidate claims mined from 23 Python libraries and checks truth changes across 184 changed values and 187 commits. The registry classifies the work as paper-only because it did not identify an official downloadable evaluation artifact in its 2026-10-11 review.

The MACE opportunity is to move from:

`source changed → invalidate every dependent memory`

toward:

`source changed → identify affected claims → independently revalidate each claim → rebuild derived artifacts only when justified`

This complements the dependency-closure invalidation already implemented in the temporal projection reference. It does not weaken the initial invalidation: projections become non-current first; claim revalidation determines whether the proposition itself remains supportable and whether a fresh projection may be constructed.

## 2. Implemented reference contract

Module:

`research/broca-mace-transfer-bench-v0-1/aml-adapter/src/claim_relative_invalidation.rs`

Schema:

`data/schemas/mace_claim_relative_revalidation_v1.schema.json`

The module accepts:
- the exact streaming ledger;
- a source transition receipt retained by that ledger;
- a claim-specific binding;
- optional claim-specific execution evidence;
- an injected independent evidence verifier.

The binding commits to:
- stable claim ID and canonical claim digest;
- the source event ID;
- exact source transition receipt digest;
- source revision before and after;
- test/assertion artifact digest;
- verifier policy digest.

The claim ID alone is not enough: the canonical claim representation and the exact source transition must also match.

### Dispositions

| Disposition | Meaning | May build a new projection? |
|---|---|---:|
| `RevalidationRequired` | Evidence absent, skipped, inconclusive, untrusted or unavailable | No |
| `Invalidated` | Independently verified claim check failed | No |
| `PreservedNeedsRebuild` | Independently verified claim check passed after correction | Yes, under a new projection ID |
| `Quarantined` | Source remains quarantined | No |
| `Revoked` | Source has been revoked | No |
| `ErasureRequired` | Rights erasure requested | No |

A passing check does **not** set an old projection to current. It returns `PreservedNeedsRebuild`; the projection layer must construct a new artifact with a new ID, fresh content digest, current source dependencies and a new derivation receipt.

Quarantine, revocation and rights erasure are hard overrides. No functional test, including a passing one, may turn those source states into permission to reuse the source.

## 3. Independent verification boundary

The classifier does not trust an evidence record merely because it says Pass or contains a hash. It accepts an injected `ClaimRevalidationEvidenceVerifierV1`.

That verifier is responsible for independently checking:
1. the exact claim and source-transition binding;
2. the exact before/after source revision commitments;
3. the test/assertion artifact and verifier policy;
4. the evaluator identity and execution environment;
5. the execution receipt and result digest;
6. that the result is not self-reported or derived from an untrusted cache.

If the independent verifier is absent or returns no attestation, the disposition remains `RevalidationRequired`. This module's assessment digest is an integrity commitment, not a digital signature, trusted timestamp or independent proof. The verifier receipt must be validated by a separate trust boundary in production.

Skipped tests, infrastructure failures, timeouts, missing tools and inconclusive outcomes never count as Pass. A missing test is not a successful test.

## 4. Lifecycle override semantics

The source transition is first verified against the exact ledger instance. The claim binding must name an event in that transition's affected-event set, including sources that were already hidden by an earlier quarantine.

The classifier then follows the source lifecycle cause:
- Correction: claim-specific independent revalidation is permitted.
- Quarantine: return Quarantined regardless of any test result.
- Revocation: return Revoked regardless of any test result.
- Rights erasure: return ErasureRequired regardless of any test result.

This preserves the authority boundary between functional truth and rights/permission state. A statement may remain semantically true while still being prohibited from use or retention.

## 5. Deterministic test coverage

Nine test functions are authored for:
1. independently verified pass returns PreservedNeedsRebuild and never reactivates the old projection;
2. independently verified failure invalidates the bound claim;
3. missing or unverified evidence requires revalidation;
4. NotRun, Skipped, InfrastructureFailure and Inconclusive never preserve the claim;
5. quarantine, revocation and erasure override a passing functional check;
6. mismatched source-event and transition-receipt bindings are rejected;
7. evidence bound to another claim cannot be replayed;
8. a self-consistent transition from another ledger instance is rejected;
9. non-invalidating transitions cannot revalidate a claim.

These are authored regression tests, not executed test results.

## 6. Benchmark design: measure both over-invalidation and stale-claim escape

A strong system must not simply invalidate everything. That trivially suppresses stale-use while destroying memory utility. Conversely, it must not preserve everything because the source diff appears harmless.

Measure these dimensions separately:

| Metric | Definition |
|---|---|
| Claim invalidation precision | Invalidated claims that were truly false / all claims classified invalidated |
| Claim invalidation recall | Truly false claims correctly invalidated / all truly false claims |
| Preservation precision | Claims preserved after correction that remain true / all claims classified preserved |
| Stale-claim escape rate | False claims still available to downstream decision paths / all false claims |
| False invalidation rate | True claims made unavailable / all true claims exposed to a source change |
| Revalidation coverage | Claims with a valid independently verified result / claims requiring a decision |
| Abstention/revalidation rate | Claims left unresolved because evidence or verifier was unavailable |
| Rebuild correctness | New projections bound to the correct claim, current source revision, dependencies and derivation receipt |
| Time to safe projection | Time from source transition to a correctly rebuilt or safely withheld projection |
| Rebuild cost | Test execution, re-extraction, model, tokens, index writes, bytes, p50/p95 latency and monetary cost |

### Proposed initial qualification gates

- 100% correct classification on a frozen, exhaustive synthetic corpus covering all six dispositions.
- Zero old projection reactivations after claim preservation.
- Zero promotion on self-reported, skipped, timed-out, unavailable or inconclusive evidence.
- 100% quarantine/revocation/erasure override enforcement in the deterministic corpus.
- Zero cross-claim or cross-source evidence replay acceptance.
- Report both stale-claim escape and false invalidation; do not optimize either in isolation.

For external paper-comparable work, reproduce the exact Impact Is Not Invalidation tasks and grader only if the official data and evaluation artifact become available. Until then, any synthetic or reconstructed study must be explicitly labelled a MACE diagnostic, not a reproduction of the paper.

## 7. Adversarial extensions

The next benchmark increment should include:
- claims whose truth is unchanged by irrelevant source revisions;
- a small change that invalidates one critical claim but not neighboring claims;
- source changes that preserve a function's outputs but alter a separate security or rights assertion;
- a passing test run against the wrong source revision;
- a stale or replayed execution receipt;
- a test that is skipped because its dependency is unavailable;
- test artifacts with changed contents but reused IDs;
- multiple claims sharing one source and one claim depending on multiple sources;
- quarantine followed by revalidation, then revocation;
- erasure requested after prior quarantine or revocation;
- rebuild attempts from a historical source revision;
- two claims with identical natural-language phrasing but different scope and evidence roots.

Each case should have a frozen expected classification and a negative control. The scorer must keep gold claim truth and expected dispositions outside the agent-facing input.

## 8. External comparators and relationship to MACE

This is not a substitute for TimlMemBench, StreamMemBench or COLLABMEM:
- TimlMemBench tests lifecycle, scope, provenance, activation and restraint in dialogue memory.
- StreamMemBench tests evidence use, feedback incorporation and future reuse.
- COLLABMEM tests role-aware sharing, routing, authority and conflict resolution.
- Impact Is Not Invalidation tests whether a claim remains valid after a source change.
- Knowing the Store tests whether systems understand store lifecycle metadata.

Claim-relative revalidation bridges the MACE derived-memory invalidation work with the adaptive decision layer: before a later policy relies on a preserved claim, a current qualified projection must exist and the consumer's DecisionNeed must still authorize its use.

## 9. Production boundary

The current implementation is a policy/reference classifier plus a guarded in-memory projection-registration path. Production still requires:
1. a trusted independent verifier with pinned trust roots and key rotation;
2. a sandboxed, reproducible test execution service;
3. a durable claim/projection registry;
4. concurrency-safe source revision binding and replay protection;
5. a revalidation queue with retry and unknown-outcome reconciliation;
6. downstream projection rebuild and lifecycle propagation;
7. external store/cache/replica deletion receipts for rights erasure;
8. an independent audit of every claim transition and later decision use.

No production capability or physical erasure is implied by the reference module.


## 10. Integrated rebuild admission

The assessment now carries and commits to the claim digest, original source event ID and exact source-transition receipt digest as explicit fields. Its integrity check validates those fields in addition to the claim binding and verifier evidence.

The function `register_rebuilt_projection_after_claim_revalidation_v1` connects this assessment to `DerivedProjectionIndexV1`. It admits registration only when:
1. the assessment integrity commitment is valid;
2. disposition is `PreservedNeedsRebuild`, with new-build permission true and old-projection reactivation false;
3. claim ID, claim digest, original source event, source-transition digest and binding digest match;
4. the transition is an exact receipt retained by the same streaming ledger;
5. the transition is a correction affecting the bound source event;
6. the ledger and projection spec remain at that exact transition sequence;
7. the new projection depends on the correction event (the new active source revision), not the superseded source event;
8. the projection registry itself accepts the new ID and current dependencies.

The registry must already have consumed the same transition. If not, its sequence check rejects the rebuild. An old projection ID cannot be silently replaced because the registry treats projection IDs as immutable and unique.

Four integration-focused tests exercise successful fresh rebuild, rejection of the old source ID, rejection before invalidation-index catch-up, and refusal after a failed claim check. A further test commits claim/source/transition identity into the assessment digest. The module therefore has fourteen authored deterministic tests in this snapshot; none have been compiled or executed here.

This closes the in-memory reference path from source correction to projection invalidation, claim-specific revalidation and fresh projection admission. It does not yet run the test itself, independently verify the injected verifier's signature, rebuild embeddings or remote caches, or provide durable transactional storage.
