# HDC Sequence / Order Retrieval Task Harness

## Purpose

This is the fourth independent task family in the HDC dimension evidence chain. It tests whether position information survives permutation-marked superposition and can be recovered from a deterministic item codebook.

The experiment varies:

`dimension × sequence length × query corruption → position-retrieval accuracy + margin + order discrimination`

Permutation is a standard HDC mechanism for encoding order, while bundling/superposition creates the distributed memory that must be decoded.

## Protocol

For a sequence of deterministic bipolar item vectors:

`sequence = bundle(permute(item_0, 0), ..., permute(item_n, n))`

A position query inverse-permutes the sequence at that position, then compares the noisy probe against the complete item codebook.

Each cell also creates a reversed-order control sequence. `order_discrimination` measures the similarity advantage of the correct-order probe over the corresponding reversed-order probe.

## Default matrix

- Dimensions: 1K through 256K.
- Sequence lengths: 2, 4, 8, 16, 32.
- Query corruption: 0%, 10%, 20%, 35%.
- Queries: 2 per position.
- Deterministic fixtures and canonical experiment identity.

## Why this is separate

This family should not be collapsed into associative cleanup. Associative cleanup asks whether a key can recover a value from a key/value superposition. Sequence retrieval asks whether **position itself** is encoded as structure and whether order survives superposition.

That distinction matters because binding is commutative while permutation is the standard mechanism for adding order information. citeturn0search1turn0search2

## Interpretation boundary

The experiment does not claim that a particular dimension is universally optimal. It produces a controlled response surface showing how dimension interacts with sequence length and perturbation.

It also does not treat order discrimination as a replacement for position retrieval accuracy. Both remain visible as independent measurements.

## Reproducibility

The fixture seed, dimension ladder, sequence-length ladder, noise ladder, query count, schema version, and scenario revision are committed into a canonical SHA-256 experiment identity. Evidence artifacts also receive a separate SHA-256 digest.

## Research basis

Recent HDC work continues to treat permutation as a core operation for sequence/order representation, alongside binding and bundling.
