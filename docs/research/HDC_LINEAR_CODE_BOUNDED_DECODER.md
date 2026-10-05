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
At the first radius beyond guaranteed unique decoding, \(t+1=\lceil d/2\rceil\),
a received word can have multiple equally-near codewords when the code geometry
permits it; the decoder therefore reports ambiguity rather than selecting one by
an arbitrary tie-break. For even \(d\), this is the half-distance radius
\(d/2\). This is the semantics established by the finite qualification fixture,
not an asymptotic claim about the random-linear-code family.

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

## Kernel and syndrome/coset invariant

For the canonical \([8,2,4]\) fixture, the qualification suite exhaustively
enumerates all \(2^8=256\) ambient words and partitions them by their six-bit
syndrome. It records 64 non-empty syndrome fibers, each of cardinality 4.

The zero-syndrome fiber is checked element-for-element against the four codewords,
establishing
\[
\ker(H)=C
\]
on the complete fixture. Every other fiber is checked in both directions:
members of the same syndrome fiber differ by a codeword, and translating a
representative by every codeword stays in that fiber. Thus, on the finite fixture,
\[
Hy^T=Hy'^T \iff y+y'\in C,
\]
so the syndrome classes are exactly the additive cosets of the code.

This invariant matters for decoder semantics. A bounded syndrome search can only
interpret a matched error pattern as a correction of a codeword because the
syndrome kernel is exactly the code. The exhaustive finite proof therefore closes
the representation-to-decoder boundary explicitly rather than inferring it from
annihilation alone.

The same fixture now carries a stronger exhaustive identity. All 256 ambient
observations are decoded with bound 4 (the fixture's covering radius), and the
number of minimum-weight error patterns sharing the observed syndrome is required
to equal the number of nearest codewords from the independent distance oracle.
The observed aggregate is 100 uniquely nearest observations, 156 ambiguous
observations, and 484 nearest-codeword incidences. This checks the multiplicity
correspondence across the entire ambient Boolean space rather than only on the
weight-2 boundary shell.
    
The qualification suite also explicitly separates this decoder ambiguity from
factor-presentation ambiguity. Four one-dimensional factor presentations of the
same [8,2,4] code have a two-dimensional factorization kernel, giving four
representations of the zero target. The same code, treated purely as a code
rather than as a factor presentation, has an observation whose nearest-codeword
and minimum-syndrome multiplicities are both two. The certificate records both
quantities and requires them to remain distinct rather than interpreting either
count as the other kind of ambiguity.


## Deterministic bounded minimum-syndrome lists

The decoder exposes a list-valued evidence surface in addition to the scalar
unique/ambiguous outcome. Callers provide both an error-weight bound and a
maximum capture size. The search still determines the exact minimum-syndrome
multiplicity; list_complete is false whenever the concrete list was truncated.

For the [8,2,4] fixture, the exhaustive 256-observation qualification first
computes the covering radius independently and requires it to equal 4. It then
uses a capture cap of 8 and requires every concrete minimum-syndrome list to be
complete and to match the nearest-codeword set from the independent distance
oracle. Separate tests exercise caps of 1 and 0 and require truncation to remain
explicit.

This is the finite coset-leader perspective: minimum-weight errors are
representatives of syndrome classes, while their multiplicity and captured list
are distinct evidence fields. Work on computing coset leaders likewise treats
complete leader sets and covering-radius structure as explicit finite-code
properties.

## Random-code list qualification and independent parity-check oracle

The randomized qualification surface now probes deterministic observations from the
existing moderate-rate and low-rate regimes, using an independent test-side
parity-check construction based on u64 nullspace elimination. It does not call the
decoder's parity-check constructor. The resulting syndrome is compared directly
with the production parity-check syndrome for every probe.

For each generated code, the probe bound is one above the guaranteed unique-decoding
radius. When the independent nearest-codeword oracle places the observation within
that bound, every minimum-syndrome error and corresponding nearest codeword must be
returned completely with capture cap 32. Their syndromes, weights, distances, and
set membership are independently checked. Observations outside the bound must
remain explicit NoMatchWithinBound results.

The entire probe is repeated with identical seeds and observations and the
complete multiplicity histogram (1 through 16) must be identical. The emitted
ledger therefore exposes the full observed list-size distribution rather than
only a selected prefix. This makes the empirical random-code result deterministic
without pretending it is an asymptotic theorem.

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
- Berlekamp, McEliece and van Tilborg (1978), *On the inherent intractability of certain coding problems*:
  https://doi.org/10.1109/TIT.1978.1055873
- MathWorld, *Syndrome Decoding Problem*:
  https://mathworld.wolfram.com/SyndromeDecodingProblem.html

## Deterministic random-code cross-check

The boundary fixture is not sufficient by itself to establish that the syndrome implementation is faithful to the general binary linear-code semantics. The qualification suite therefore adds deterministic sweeps over two independently generated Boolean random-linear-code regimes: a moderate-rate [12,4] family and a lower-rate [20,3] family that is closer to the low-rate regime of the paper-scale comparator.

For each usable seed whose exhaustive code geometry has unique-decoding radius between one and four, every codeword and every error pattern inside that guaranteed radius are passed through the syndrome decoder. The independent nearest-codeword oracle is required to report the same unique codeword, distance, and exact error pattern.

This remains a finite implementation cross-check. It is not a probabilistic claim about the entire random-code ensemble.

## Beyond-radius semantics

A separate fixture tests corruption beyond the unique-decoding radius without asking the decoder to recover an arbitrarily designated clean target.

For the [8,2,4] fixture, all 56 weight-3 corruptions around one clean codeword are classified against the exhaustive nearest-codeword oracle. With decoder bound 2:

- 8 observations have a unique nearest codeword at distance 1; the bounded decoder therefore returns that other codeword.
- 48 observations have nearest-codeword distance 3, so the bounded decoder correctly returns NoMatchWithinBound.
- no result is credited as intended-target recovery merely because the decoder returns some codeword.

This is an important evidence boundary: bounded-distance decoding identifies a nearest codeword within its declared radius; it does not establish recovery of the original semantic factor tuple once the channel exceeds the code's unique-decoding guarantee.

The factorization layer remains separate. Even a unique codeword result can still carry an affine factorization fiber of cardinality 2^d, and the decoder does not inspect that fiber when making its codeword-level decision.

## 2026 coding-theory context

Recent results sharpen, rather than collapse, this distinction. Silas determines the sharp typical worst-case list-size behavior of random linear codes at capacity for every finite field, while Yuan and Zhu obtain asymptotically optimal list-size scaling for fixed finite fields. These are asymptotic list-decoding results and do not certify the finite Boolean fixtures used here.

Deng and Raviv explicitly frame noisy VSA recovery as a separate difficulty for random-linear-code representations and use a Reed--Solomon/Hadamard representation with histogram-recovery algorithms. That remains a distinct future representation family rather than an implementation detail of this bounded Boolean decoder.

References:

- Silas (2026), The list size of random linear codes at capacity: https://arxiv.org/abs/2609.06570
- Yuan & Zhu (2026), Asymptotically Optimal List Size of Random Linear Codes: https://arxiv.org/abs/2609.01070
- Deng & Raviv (2025/ISIT 2026), Efficient Vector Symbolic Architectures from Histogram Recovery: https://arxiv.org/abs/2511.01838
