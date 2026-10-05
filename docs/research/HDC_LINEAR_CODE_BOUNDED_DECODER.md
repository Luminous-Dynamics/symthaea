# Bounded-Distance Syndrome Decoder for the Linear-Code HDC Comparator

This document defines the next noisy-recovery layer after the exact Raviv-style
linear-code comparator. It is deliberately a separate research surface and does
not change production HDC behavior.

## Research boundary

Raviv's random-linear-code construction provides an algebraic recovery route for
clean bound representations by exploiting the subspace structure of the code.
The present branch does **not** reinterpret that clean solver as a noise decoder.
Instead it introduces an independent coding-theoretic decoder built around a
parity-check matrix and syndrome search.

For a binary linear code \(C\subseteq\mathbb F_2^n\) with minimum distance \(d\),
unique correction is guaranteed only through
\[
t=\left\lfloor\frac{d-1}{2}\right\rfloor.
\]
At the boundary \(2t=d\), a received word can have multiple equally-near
codewords; the decoder therefore reports ambiguity rather than selecting one by
an arbitrary tie-break. This is the semantics established by the finite
qualification fixture, not an asymptotic claim about the random-linear-code
family.

## Decoder construction

Given an independent generator basis \(G\), the decoder first derives a full-rank
parity-check matrix \(H\) by GF(2) row reduction. If the generator is reduced to
RREF, each non-pivot column produces one nullspace vector; those vectors are the
rows of \(H\), giving
\[
GH^T=0.
\]

For an observation \(y=c+e\), linearity gives
\[
Hy^T=He^T,
\]
so the decoder need not enumerate codewords. It computes the observed syndrome,
then searches error patterns in increasing Hamming weight until one or more
patterns have the same syndrome.

The implementation returns exactly one of:

- **Unique** — one minimum-weight error pattern was found within the explicit
  bound; the corresponding corrected codeword is returned.
- **Ambiguous** — multiple minimum-weight error patterns have the same syndrome,
  so unique decoding is unavailable at that distance.
- **NoMatchWithinBound** — no error pattern at or below the declared bound has
  the observed syndrome.
- **InvalidBound** — the requested error bound exceeds the block length.

No candidate codeword enumeration occurs inside the decoder.

## Why this is an independent algorithm

The existing finite oracle enumerates codewords, computes Hamming distances, and
constructs the complete nearest-codeword set. The new decoder instead:

1. constructs H from the generator basis;
2. computes a syndrome using the columns of H;
3. enumerates error supports by increasing weight;
4. compares error syndromes;
5. reconstructs a codeword only after a unique minimum-weight error is found.

The two paths therefore meet only at the test boundary. Agreement is evidence
that two materially different implementations describe the same finite geometry.

## Complexity boundary

The decoder's search space through weight t is
\[
\sum_{i=0}^{t}\binom{n}{i}.
\]
This is the standard bounded-weight syndrome-search surface. For fixed small
t, the number of tested error supports is polynomial in n; for growing
t, the search becomes combinatorial and is not claimed to solve general
syndrome decoding efficiently.

This distinction matters because minimum-weight syndrome decoding for arbitrary
binary linear codes is computationally hard. The branch therefore qualifies a
**bounded** decoder for controlled finite experiments, not a general efficient
decoder.

The implementation records deterministic work units:

- "weights_examined";
- "error_patterns_examined";
- "syndrome_column_xors";
- "matching_error_patterns".

These are algorithmic operation counts, not wall-clock performance measurements.

## Canonical finite boundary fixture

The decoder is cross-checked on the existing [8,2,4] Boolean fixture generated
by the two basis words

- "11110000";
- "00001111".

Its codewords have minimum distance 4, so the guaranteed unique radius is 1.

The independent qualification checks:

- all 36 observations formed from the four codewords and error weights 0 or 1;
  every observation decodes uniquely to the same nearest codeword;
- all 112 weight-2 boundary observations formed from the four codewords;
  each oracle nearest-codeword set is reproduced as either a unique result or
  an ambiguity with the same multiplicity;
- every clean target has exactly 16 unique and 12 ambiguous weight-2 cases;
- the ambiguous boundary cases have exactly two nearest codewords / minimum-weight
  syndrome matches;
- the observation "00110011" is at distance 4 from every codeword and therefore
  yields "NoMatchWithinBound" at bound 2;
- the decoder work ledger for the boundary case examines exactly
  \(1+8+\binom82=37\) error patterns.

The cross-target check is important: the result is not tied to the arbitrary choice
of one clean target.

## Relationship to the HDC literature

Raviv's 2024 construction is a clean-recovery result: random linear codes expose
subspace structure that permits algebraic factor recovery. It explicitly leaves
noise robustness as a separate issue.

Deng and Raviv's later work addresses that gap using a **different representation
family** based on Reed--Solomon/Hadamard concatenation and histogram-recovery
machinery related to list decoding. That construction should remain a separate
comparator rather than being folded into this Boolean random-linear-code branch.

Recent 2026 random-linear-code list-size results sharpen the theoretical boundary
around list decoding near capacity, but they do not establish noise performance
for this small deterministic Boolean fixture. The present branch therefore keeps
the finite observed distance, search radius, and ambiguity counts explicit.

## Qualification gates

A future stronger decoder may be added only after this bounded syndrome path
passes its independent finite qualification. The following remain separate
evidence layers:

**Representation layer** — random-linear-code construction and exact clean
factor recovery.

**Channel layer** — corruption radius, distribution, and observation alphabet.

**Codeword layer** — syndrome-decoder uniqueness, ambiguity, or no-match outcome.

**Factor layer** — affine factorization multiplicity after a candidate codeword is
identified.

A codeword-level unique decode does not imply a unique factorization: the existing
affine-fiber certificate may still contain multiple factor tuples. Conversely,
decoder ambiguity must not be credited or blamed on factorization geometry.

No production integration, timing claim, or superiority claim is made here.

## References

- Netanel Raviv, *Linear Codes for Hyperdimensional Computing*, 2024:
  https://arxiv.org/abs/2403.03278
- Zirui Deng and Netanel Raviv, *Efficient Vector Symbolic Architectures from
  Histogram Recovery*, 2025/ISIT 2026:
  https://arxiv.org/abs/2511.01838
- Shashwat Silas, *The list size of random linear codes at capacity*, 2026:
  https://arxiv.org/abs/2609.06570
- Error Correction Zoo, *Linear binary code*:
  https://errorcorrectionzoo.org/c/binary_linear
- Error Correction Zoo, *Binary code*:
  https://errorcorrectionzoo.org/c/bits_into_bits
- MathWorld, *Syndrome Decoding Problem*:
  https://mathworld.wolfram.com/SyndromeDecodingProblem.html
