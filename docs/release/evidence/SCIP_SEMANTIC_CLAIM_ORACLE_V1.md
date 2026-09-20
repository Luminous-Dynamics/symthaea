# SCIP Semantic Claim Oracle V1 — Evidence Note

## Scope

This note records the locally executed pre-commit evidence for the independent
structured-claim comparison oracle that follows the V16 semantic-realization
contract.

The oracle is deliberately narrower than a natural-language semantic verifier:

```text
grounded-source claim inventory
    + extracted-surface claim inventory
        -> structured claim comparison
        != qualified text-to-claim extraction
        != surface semantic fidelity
        != grounded truth
```

Its authority field is permanently `measurement-only` in V1.

## Local execution environment

- Python: `3.13.5`
- third-party Python dependencies: none

Executed locally before commit:

```text
python3 -m py_compile scripts/scip_semantic_claim_oracle.py \
  scripts/test_scip_semantic_claim_oracle.py
PASS

python3 scripts/scip_semantic_claim_oracle.py --self-test
PASS_SELF_TEST

python3 scripts/test_scip_semantic_claim_oracle.py
PASS_ADVERSARIAL comparison_sha256=e0d42e1a1ead213f4261405c22d1f80cb6f1d106fd5621481eb0e95a6cca4530
```

## Exact checked bytes

```text
oracle
0727bf1ef810fcaef57ea6290596fbfbb1b835074d995bc74d9b6e1da1a745aa

external adversarial harness
4855112ebc7a3178ddf12d31d42f25fbbdfe175dc4f8660702ac0a6608f2d1b0

positive fixture
b51a178dee532a99e46a11dc65779337af63e4749c67d744a809a5b3fe0c016b
```

## Semantic dimensions

The oracle reports the same eleven dimensions frozen by V16:

1. entity/reference;
2. relation direction;
3. numeric value and unit;
4. polarity and negation;
5. quantifier and cardinality;
6. temporal scope;
7. epistemic modality;
8. attribution and source;
9. causal strength;
10. unsupported additions;
11. required-detail coverage.

The external harness injects one targeted mutation for each dimension and
requires the corresponding dimension to become `contradicted`.

It additionally verifies:

- canonical input-order invariance;
- duplicate source/surface identity rejection;
- ambiguous duplicate mapping rejection;
- unknown/shadow top-level field rejection;
- duplicate JSON key rejection;
- canonical binary64 spelling;
- rejection of negative zero as a non-canonical binary64 value.

## Positive fixture boundary

The positive fixture produces:

```text
claim_inventory_equivalent=true
surface_fidelity_established=false
text_to_claim_extraction_qualified=false
authority=measurement-only
comparison_sha256=e0d42e1a1ead213f4261405c22d1f80cb6f1d106fd5621481eb0e95a6cca4530
```

This distinction is load-bearing. The oracle can establish equivalence between
two normalized structured claim inventories under its V1 comparison semantics.
It cannot establish that a natural-language surface was correctly converted
into that inventory.

## Next scientific boundary

The next tranche should create a preregistered text-to-claim extraction corpus
with human labels, held-out cases, paraphrase controls, and adversarial
mutations. Candidate extractors (deterministic parser, NLI pipeline,
model-assisted extractor, or ensemble) should be measured against that corpus.

No positive `SurfaceSemanticFidelity` capability should exist until the
extraction step and this structured comparison step are both separately
qualified and bound to the exact V15/V16 identities.
