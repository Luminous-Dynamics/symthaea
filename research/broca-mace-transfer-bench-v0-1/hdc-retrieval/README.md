# MACE HDC/HDC-LTC retrieval adapter v0.1

This package implements the first executable slice specified by [#13127](https://github.com/Luminous-Dynamics/symthaea/issues/13127) and [PR #13128](https://github.com/Luminous-Dynamics/symthaea/pull/13128).

The adapter reuses the repository's existing `symthaea-hdc-ltc` implementation. It does not reimplement ContinuousHV or LTC dynamics.

## Contract

- Build one bipolar role/value associative bucket per exact `(regime, objective, action)` key.
- Bind each stable event-ID code to the role key, then bundle using `ContinuousHV::bundle`.
- Retrieve by unbinding the requested role key and ranking the stored event-ID codebook by cosine similarity.
- Optional `HdcLtcUnifiedNeuron` evolution applies a declared elapsed horizon to the same bundled state using zero input.
- Return candidate event IDs, scores, ranks, abstention, profile digest, latency and a vector/ID payload-byte lower bound.
- Never return an `Experience`, source receipt, causal proof, trust status or action authorization.

A consumer must resolve every returned ID against the canonical evidence registry and run the unchanged MACE source, upstream-root, measurement, intervention, estimand, objective, outcome, trust and retraction predicates. Unknown, duplicate or unresolved identifiers fail closed.

## Arms included

- Exact-scan candidate ceiling.
- Static ContinuousHV key/value associative bundle.
- HDC-LTC zero-input temporal evolution.
- Typed-content HDC, now included as a separate arm in the MACE paired decision runner. The runner builds a query from the current candidate-visible regime/objective/action and the MACE policy's categorical qualification requirements (RCT method, intervention, pre-registration, reconciliation, trust, non-retraction, interference, confounder control and delayed-horizon completion).

The typed runner profile intentionally does not encode propensity/balance thresholds as numeric point targets; those are inequality predicates enforced by the unchanged MACE policy after exact record resolution. A richer target outcome/horizon/estimand profile still requires a candidate-visible DecisionNeed from #13222 and must never be fabricated from evaluator labels.

Distractor, semantically confusable interference, irregular-gap and delayed-correction schedules remain follow-on arms from #13099. This package does not claim that HDC-LTC or typed-content HDC improves retrieval or decision utility.

## Run

```bash
cargo test --manifest-path research/broca-mace-transfer-bench-v0-1/Cargo.toml -p broca-mace-hdc-retrieval
```

Tests are authored but have not been executed in the current environment.

Candidate hits also carry a SHA-256 fingerprint of the candidate-visible `Experience` view at index-build time. The consumer compares it to the current canonical record before resolving the ID. A correction or retraction that changes the view invalidates the old hit and requires re-indexing. This is a consistency check only: it does not bind raw source bytes or authenticate source provenance.


## Typed-content, query-conditioned HDC v0.1

The crate also exposes `typed_content::TypedContentIndexV1`, a separate experimental index that encodes candidate-visible MACE fields rather than event IDs:

- scope regime/objective and action;
- observed utility and measurement-error bound;
- identification method, estimand digests, propensity and balance descriptors;
- intervention, pre-registration, reconciliation, trust, retraction, confounder and delayed-horizon flags.

Categorical features use deterministic role/value binding. Continuous values use deterministic random Fourier features for a one-dimensional Gaussian RBF kernel; utility, measurement-error, propensity and balance bandwidths are frozen in `TypedContentConfigV1` and its profile digest. `SemanticExperienceQueryV1` supplies the same typed fields, including optional numeric targets and evidence-quality cues.

**Identity and authority are deliberately not approximated.** Event IDs, source-family/root identity, source signatures, tenant/authority information and canonical receipt bytes remain outside the vector. Retrieval returns the exact event ID and candidate-view digest; the caller must resolve the canonical record and re-run MACE qualification.

The current policy `Query` only exposes regime/objective. The paired runner now integrates a restricted typed-content arm using that scope/action plus the MACE policy's fixed categorical qualification requirements. Scope/action are exact prefilters—not approximate vector cues—so wrong-scope candidates cannot crowd out applicable evidence. The profile digest binds this filter semantics (typed profile v1.1). The optional numeric target and estimand fields are still not populated from a task-level source. Do not fabricate them from evaluator truth. A broader agent-level comparison needs a candidate-visible `DecisionNeed` derived from the actual observation and task.

Regime/objective and action are retained as exact partition keys and applied as hard prefilters before approximate HDC ranking. A high-similarity wrong-scope or wrong-action item cannot consume a top-k slot for the current query. This behavior is bound into typed profile v1.1, and the payload estimate includes the exact partition keys.

The typed index is structured semantic retrieval, **not natural-language understanding** and not proof that HDC-LTC has learned temporal dynamics. It is static typed HDC v1.1; event-driven LTC and irregular-time streams remain separate follow-on work.

