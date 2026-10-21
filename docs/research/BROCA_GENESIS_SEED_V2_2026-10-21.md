# Broca Genesis-Seeded Baseline: V1 Compatibility and V2 Experiment

Date: 2026-10-21
Status: opt-in implementation and smoke benchmark; Rust runtime validation pending

## Recommendation

Yes, improve the genesis-seeded baseline—but do it as a versioned, measured improvement, not by silently replacing historical vectors. Preserve V1 as the default and keep V2 opt-in until exact-head tests and paired evidence justify a migration.

The baseline is not semantically informed by itself. It is a deterministic, approximately isotropic token-vector initializer. Any quality improvement must be demonstrated at the downstream task level; geometric cleanliness is necessary evidence about initialization, not evidence of understanding.

## Finding: the legacy domain tuple is ambiguous

V1 feeds SHAKE-256 the concatenation:

`seed_phrase || "::" || label`

Because neither component is length-prefixed, different tuples can encode the same byte sequence. For example:

- phrase `alpha::beta`, label `gamma`
- phrase `alpha`, label `beta::gamma`

Both become `alpha::beta::gamma` before hashing. Existing outputs are deterministically reproducible, but the tuple-to-stream mapping is not injective across seed-phrase/label pairs.

This is not an observed production token collision for the current fixed Broca phrase and token-label pattern. It is an API-level ambiguity that undermines the stronger promise of unambiguous domain separation.

## Implementation in this change

- Adds `GenesisSeed::domain_v2` with a versioned namespace and fixed-width, little-endian 64-bit byte lengths for the UTF-8 seed phrase and label.
- Adds `GenesisSeed::hv_v2`, reading one SHAKE word per coordinate and mapping its high 24 bits to a representable grid in `[-1, 1)`. This avoids the legacy full-`u32` to-`f32` precision collapse and excludes the exact upper endpoint.
- Adds `TokenEmbeddingSeedScheme::{LegacyV1, DomainSeparatedV2}` to Broca. V1 remains the default; the selected scheme affects token rows only, leaving the recurrent network and all other seed domains unchanged for a controlled experiment.
- Keeps `LanguageControllerConfig`'s serialized shape unchanged for checkpoint compatibility. Generator constructors track the scheme at runtime; V2 is recorded in the checkpoint's existing `metadata.feature_set` marker (`broca.token-seed-scheme.v2`), preserved during save, and restored on load. Older checkpoints without the marker default to V1; vocabulary extensions keep the active scheme.
- Tests cover V1 compatibility, V2 delimiter-boundary separation, repeatability, dimension-prefix stability, finite/range/distribution/near-orthogonality properties, scheme-consistent vocabulary extension, and checkpoint persistence through the existing metadata field.
- Adds `broca_seed_baseline_bench`, a paired smoke benchmark across legacy and V2 token initialization. It supports multiple genesis replicates, identical fixtures and optimizer settings, separate held-out phrases, geometry diagnostics, and a JSON receipt with source and fixture digests.

No production default changes. Pretrained checkpoint embedding tables are restored from the checkpoint and are not regenerated.

## Relevant external methods

- NIST SP 800-185 describes SHAKE-derived customizable functions and TupleHash, whose purpose includes unambiguous processing of tuple elements: https://csrc.nist.gov/pubs/sp/800/185/final
- RFC 5869's `info` input is used to bind derived key material to application- and context-specific information: https://www.rfc-editor.org/rfc/rfc5869/
- A 2026 reproducible HDC experiment compares HRR and MAP against an encoder-defined similarity reference across dimensions and random seeds. Its methodological lesson is to preserve the reference task and vary one representational factor at a time—not that its measured outcome transfers to Broca: https://hyperdimensionalcomputing.ai/blog/hrr-map-similarity/

V2 is an internal reproducibility scheme inspired by these principles, not an implementation of TupleHash/cSHAKE or a cryptographic protocol. Seed phrases are model initialization inputs, not secrets; this work makes no security claim.

## Benchmark interpretation

The built-in corpus is intentionally a smoke set (8 training targets, repeated three times; 8 distinct held-out phrases). The JSON schema labels it `smoke` even when multiple initial seeds are used. It can verify wiring, reveal gross regressions, and compare initial geometry; it cannot establish a meaningful SOTA claim or general language quality.

The benchmark reports:
- coordinate mean and RMS;
- non-finite coordinate count;
- mean and maximum absolute pairwise cosine among up to 32 token rows;
- held-out perplexity, English-word ratio, coherence, and the full evaluator result;
- exact initializer scheme, number of seed replicates, fixture digests, optimizer profile, and optional source revision.

The report retains each seed replicate independently. Do not pool the 8 held-out phrases as if they were an industry-scale confirmatory set.

## Proposed evaluation gates

These are targets to preregister, not achieved results:

1. **Compatibility:** default constructors produce bit-identical V1 token rows for the same genesis phrase and config; old configs lacking the new trailing field load as V1.
2. **Determinism:** repeat initialization and repeat benchmark arms produce exactly equal pre-training token rows for equal phrase, label, and scheme.
3. **Numerical health:** zero non-finite values; no coordinate equals +1 in V2; sampled coordinate mean absolute value below 0.02 and RMS between 0.56 and 0.60 for 16,384 dimensions; mean absolute cosine below 0.02 across the selected random-row sample, and no pathological duplicate rows.
4. **Seed robustness:** at least 5 matched genesis replicates and a larger fixture before interpreting downstream performance.
5. **Quality non-inferiority:** on a sealed, larger dataset, no more than a 2 percentage-point regression on core language controls; report paired uncertainty and all strata.
6. **Migration threshold:** make V2 a default only after backward-compatibility evidence and reproducible downstream improvement (or a justified correctness migration where quality neutrality has been demonstrated). If V2 only improves domain identity and geometric diagnostics without task-level gains, describe it as hardening—not a language-quality advance.

## Next research step

Expand the same paired design to an independently scored semantic/compositional corpus and held-out root families. Compare V1, V2, and semantic initialization as distinct arms, use multiple seeds, and freeze the scoring procedure before confirmation. Keep three outcomes separate:

- stream/domain correctness;
- embedding geometry;
- task-level language performance.

A clean initializer can prevent reproducibility errors; it cannot fix the lexical-prime aliasing reported in issue #8259 or create semantic knowledge absent from training and representation.

## Validation boundary

- GitHub read-back confirms source, config, tests, constructors, benchmark, and this note are present on the feature branch.
- The local environment used for this task does not have Cargo available; no Rust tests, formatting, Clippy, benchmark execution, or CI pass is claimed.
- The observed exact-head GitHub workflows are skipped. Keep PR in draft until validation is performed on the current head.

Related work:
- Baseline hardening PR: pending
- Lexical-prime aliasing issue: https://github.com/Luminous-Dynamics/symthaea/issues/8259
