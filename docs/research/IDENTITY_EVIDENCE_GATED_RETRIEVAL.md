# Identity-Evidence-Gated Retrieval

**Status:** reference API and falsifiable experiment added; exact-head Rust validation pending  
**Branch:** `research/identity-evidence-gated-retrieval`  
**Implementation:** `crates/core/symthaea-core/src/hdc/identity_evidence.rs`  
**Executable:** `crates/core/symthaea-core/examples/identity_evidence_boundary.rs`

## The paradigm shift

Separate three questions that prototype classifiers often collapse:

1. **Resemblance:** which registered prototype is geometrically closest to this vector?
2. **Identifiability:** do the supplied observations distinguish the claimed identity from plausible alternatives?
3. **Identity evidence:** is there a trusted registry/enrollment fact that licenses the semantic identity claim?

Hypervector similarity addresses the first question. Thresholds and prediction sets can help with the second. Neither can manufacture the independent evidence needed for the third.

This is an epistemic contract, not a claim that adding a receipt improves the HDC algebra. It aims to make an invalid inference—“nearest vector therefore verified identity”—unrepresentable in the ordinary API.

## Falsifiable identifiability control

Let a vector-only resolver be (f(x)), with acceptance decision, candidate label, or prediction set determined solely by observed vector (x). Construct a known/noisy observation (x_k) and an unregistered-identity counterfactual (x_u) such that (x_k = x_u), while the semantic registration labels differ.

Then for every deterministic vector-only resolver:

[
f(x_k) = f(x_u).
]

For a randomized resolver whose randomization is independent of the hidden semantic label, the output distributions are also identical conditional on that same observed vector. The resolver cannot simultaneously accept the known identity and reject the unregistered identity reliably from the vector alone. This is an input-information limit, not an HDC-specific weakness and not something another similarity threshold can repair.

The current selective-retrieval experiment already provides the empirical motivation: in the completed earlier run at source SHA `932f237ace2159c90b756633282a38245213acb4`, all tested score/margin strategies accepted every query in the separate near-prototype unknown challenge, treating all as incorrect registered-identity claims. Its explicit paired identity controls also retain identical vectors and score features while changing the registry label. See [workflow run 39648161350](https://github.com/Luminous-Dynamics/symthaea/actions/runs/39648161350) and its [research artifact](https://github.com/Luminous-Dynamics/symthaea/actions/runs/39648161350/artifacts/12370175382). These observations are specific to the synthetic protocol—not deployment error-rate estimates.

That same run is a useful warning about set-valued prediction. The raw split-conformal baseline achieved 100% empirical marginal coverage on the known-query evaluation mixture but returned about 32.83 prototypes per query on average (out of 64), with no singleton predictions. The DCO-inspired tuned-score variant averaged 95.16% known-query marginal coverage, a mean set size of 5.78, and an 82.19% singleton fraction. Yet on the separate unregistered near-prototype challenge, its mean prediction-set size was exactly 1.0 and all sets were singletons across the five seeds. The set method therefore produced compact, apparently decisive candidates for observations that were semantically unregistered. This is not a violation of its scoped known-query marginal-coverage claim: that guarantee does not say the query belongs to the registered identity set. It is precisely why identity evidence needs a separate interface.

## In-process capability binding

A registration receipt is bound both to the registry instance and to the exact enrollment event that issued it. The registry-instance token prevents reuse in another newly constructed registry even when the public registry name matches. The per-enrollment token prevents replay across independently diverging clones that register the same identity name at the same revision. Cloning an unchanged registry preserves capabilities for the enrollment events already present in that clone.

These tokens are process-local capabilities only. They are not signatures, do not authenticate a caller or enrollment authority, and must not be serialized as if they were portable credentials. Any cross-process design needs authenticated enrollment provenance, a canonical registry snapshot/version, freshness and revocation semantics, and an explicit wire-verifiable proof format.

## Reference contract

The new `hdc::identity_evidence` module distinguishes:

- **CandidateOnly:** vector similarity suggests a registered prototype, but no identity proof has been supplied.
- **Ambiguous:** the best two registered candidates are too close under the declared margin policy.
- **NoCandidate:** no nearest prototype clears the declared similarity threshold; the nearest score remains visible for diagnostics.
- **EvidenceRequired:** the enrollment receipt is valid and the vector is geometrically consistent, but enrollment alone does not bind this observation to that identity.
- **Verified:** a separate observation-identity assertion from an upstream trusted source is bound to the exact query vector and matching enrollment event, and the vector clears the similarity and margin checks while ranking that identity first.
- **EvidenceConflict:** enrollment evidence exists, but the vector contradicts the claimed identity or violates score/margin requirements. Neither receipt overrides contradictory vector evidence.

The reference registry is append-only and records registry and enrollment revisions. It rejects blank or duplicate identity IDs, cross-registry receipts, dimensional mismatch, invalid thresholds, low similarity and ambiguous nearest matches.

**Trust boundary:** a `RegistrationReceipt` proves only that a prototype was enrolled; it is not evidence that an observation belongs to the registered entity. `ObservationIdentityEvidence::from_authenticated_source` binds an upstream claim to the exact vector digest, unique capture ID, and enrollment event, but it does not authenticate that source itself. The caller must authenticate the observation-to-identity assertion (for example, verify a signed device/authority response) before constructing this in-process evidence. Neither type is a signature, remote credential, biometric proof, or tamper-resistant attestation. Persisted evidence still needs a defined signature, revocation, freshness, and registry-snapshot/version protocol.

## Executable protocol

The runner uses five deterministic seeds, 64 randomly generated registered prototypes, and 4,096-dimensional bipolar vectors. It samples 256 noisy observations per seed at each of 5%, 10%, and 20% bit-flip corruption levels.

Every same-vector observation is tested in two logically distinct ways:

- **Vector-only:** rank registered candidates and report CandidateOnly / Ambiguous / NoCandidate. It must never return a Verified identity because no identity receipt was provided.
- **Registration-only control:** supply the enrollment receipt without observation-bound evidence. Even for an exact prototype match, the result must be EvidenceRequired, never Verified.
- **Evidence-gated:** attach both the enrollment receipt and an observation-bound assertion from the synthetic known-query oracle. Return Verified only when registry/enrollment scope, the exact observation digest, nearest-candidate identity, similarity, and top-two margin all agree. Contradictions return EvidenceConflict.
- **Counterfactual twin:** the exact same vector is assigned an unregistered semantic identity. Because its vector has not changed, vector-only geometry has exactly the same result; the unregistered identity simply lacks an identity receipt.

The executable is intended to report per-seed and per-noise counts for candidate-only results, ambiguity, no candidate, nearest-label correctness, registration-only EvidenceRequired outcomes, registration-only conflicts, observation-evidence-backed verification, evidence conflicts, and exact-vector counterfactual pairs. No threshold is tuned on the measured outcomes.

## Acceptance criteria

This work is a correctness/control benchmark, not a leaderboard claim.

1. **Type/API safety:** 100% of similarity-only calls return a similarity outcome, and 100% of registration-only checks return EvidenceRequired or EvidenceConflict, never Verified identity.
2. **Exact-vector twin invariant:** 100% of paired cases retain identical vectors and therefore identical vector-only outcomes, while registry evidence is kept separate.
3. **Receipt binding:** wrong-registry or wrong-enrollment receipts fail closed.
4. **Observation binding:** changing even one bit of a query invalidates its observation evidence.
5. **Contradiction handling:** a valid enrollment receipt paired with a query whose nearest identity contradicts the receipt returns EvidenceConflict, not Verified.
6. **Ambiguity handling:** equal or near-equal top candidates cannot silently become a unique identity.
7. **Reproducibility:** fixed seeds and pinned toolchain, no hidden tuning, raw output included in the workflow artifact with SHA-256 manifest.
8. **Production guardrail:** documentation never implies that the in-process receipt authenticates a real-world person/object. External authentication, revocation, privacy, spoofing and sensor-compromise tests are separate follow-up requirements.

## How this should be compared with the field

HDC/VSA should still be compared on shared classification tasks against the methods used by practitioners. The 2025 comparative review evaluates seven commonly used HDC/VSA datasets and 121 UCI tasks; it also reports that SVM/RVFL can outperform many prototype-based HDC classifiers in some settings. That is a reason to use conventional baselines, not evidence that this identity API is better at general classification. [Vergés et al. (2025)](https://doi.org/10.1007/s10462-025-11181-2).

For the adjacent selective-prediction problem, compare the existing similarity/margin and split-conformal approaches against published methods such as [Selective Classification via One-Sided Prediction](https://proceedings.mlr.press/v130/gangrade21a.html) and recent [selective conformal risk-control research](https://arxiv.org/abs/2606.08517). These methods can improve risk/coverage under stated assumptions; they still cannot infer hidden semantic identity from exactly identical observable features.

The useful scorecard is multi-dimensional:

| Dimension | Target |
|---|---|
| Vector retrieval | Standard task accuracy, retrieval recall and K95 at fixed byte budgets |
| Selective prediction | Risk–coverage curves, accepted-query risk bounds, singleton rate and prediction-set size |
| Semantic novelty | False-accept rate for held-out and near-prototype unregistered queries, reported separately by shift type |
| Identity correctness | Zero Verified outputs without authenticated identity evidence; all conflicts explicitly surfaced |
| Cost | Bytes including metadata/receipt indexes, p50/p95 query latency, throughput and registry update cost |
| Robustness | Correlated/near-duplicate prototypes, stale/wrong-registry receipts, registry revision changes, damaged vectors and adversarial claims |

No single scalar should combine identity correctness and vector retrieval quality. A high retrieval score cannot compensate for unsupported identity claims.

## Next research steps

1. Replace the synthetic observation oracle with authenticated issuer verification: a documented signature scheme, canonical observation digest, registry snapshot, expiry/revocation, and freshness semantics. Do not serialize either in-process capability as if it were self-authenticating.
2. Add property-based tests over exact-vector identity twins, near-duplicate prototypes, dimension mismatches, stale registry views, and conflicting receipts.
3. Compare nearest-neighbour, score/margin, split conformal, and DCO-style prediction sets while keeping identity resolution outside their outputs.
4. Test against an open-set dataset with genuine semantic novelty and published splits. The current synthetic bit-flip challenge is an identifiability control, not sufficient evidence of practical open-set performance.
5. Report worst-stratum metrics and paired confidence intervals, not only averages. Publish all negative and inconclusive cases.

## Current status

The module, its unit tests, the registered executable, workflow steps, and this protocol have been committed on the separate research branch. The branch's dedicated workflow must still compile and run them before the result is treated as validated. A green future run would validate the listed contract tests, not cryptographic identity attestation or production readiness.
