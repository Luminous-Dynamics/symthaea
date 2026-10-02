# HDC Linear-Code Recovery Comparator — Design Note

Status: research design only. No production integration is proposed by this document.

## Purpose

The resonator cleanup-rule sweep establishes that cleanup nonlinearity is an explicit algorithmic axis. The next useful separation is therefore a recovery family that changes the representation/recovery algebra rather than adding another resonator hyperparameter.

Raviv (2024), *Linear Codes for Hyperdimensional Computing*, proposes random linear codes as Boolean-field subspaces and derives recovery procedures for bundled and bound compositional representations. The paper reports that the bound-recovery method uses subspace structure and provides provable factorization under its stated framework; its bundled method constructs Boolean linear systems that reduce the search space. These are materially different from Symthaea's current bipolar random-code resonator path.

References:
- DOI: 10.1162/neco_a_01665
- arXiv: 2403.03278

## Research boundary

Do not implement a "linear-code-inspired" approximation by reusing the existing random bipolar codebook and calling a different search routine. That would not test the claimed representation-level distinction.

The comparator should have:

1. A deterministic Boolean linear-code generator.
2. An explicit GF(2) representation with a documented mapping to the HDC-facing vector representation.
3. A bound-composition fixture using factor domains/subcodes whose relationship is justified by the paper's construction; a generic single-subspace codebook is not sufficient.
4. A recovery implementation that uses the paper's exact code/subspace structure, not exhaustive n² search.
5. A matched exhaustive oracle used only for ground truth.
6. The existing resonator solver run on a separate, representation-matched ordinary bipolar fixture.
7. Separate measurements for exact factor recovery, spurious recovery, non-recovery, query corruption tolerance, and computational work.

## Fair-comparison rules

The experiment must not compare unlike problem instances.

For every trial:

- dimension is fixed;
- codebook cardinality is fixed;
- factor identities are fixed;
- the clean bound vector is generated once;
- query corruption is generated once and reused by both recovery families where the algebra permits it;
- the exhaustive oracle evaluates the same factor set;
- deterministic seeds and a complete fixture identity are recorded;
- no composite score is emitted;
- no universal winner or dimension ranking is emitted.

Because the linear-code representation has algebraic structure that ordinary random bipolar codebooks do not, representation storage/geometry should be measured separately from recovery performance. A faster solver on a structurally different representation is evidence about an architecture trade-off, not a direct replacement claim.

## Proposed staged implementation

### Stage A — algebraic kernel

Research-only module, preferably isolated from production resonator APIs.

Required primitives:

- GF(2) packed bit-vector representation;
- XOR/addition over GF(2);
- deterministic random full-rank generator matrix;
- row-reduction / rank;
- deterministic basis construction;
- Boolean subspace membership;
- conversion between packed GF(2) words and the experiment's bipolar observation vector, using Raviv's convention GF(2) 0 -> +1 and GF(2) 1 -> -1 so XOR is exactly bipolar Hadamard binding.

Tests should cover:

- closure under XOR;
- zero-vector membership;
- rank preservation;
- deterministic regeneration from seed;
- round-trip conversion;
- XOR/Hadamard equivalence under the bipolar mapping;
- rejection of malformed dimensions.

### Stage B — bound fixture

Do not assume that choosing both factors from one arbitrary linear subspace makes the pair identifiable. Since a linear code is closed under XOR, a composite can have multiple decompositions into members of that same code. The fixture must therefore reproduce the paper's actual factor-domain/subcode/key-value structure, or explicitly characterize the ambiguity rather than label an arbitrary pair as uniquely recoverable. Confirm the exact bound construction from the paper before fixing fixture semantics. The primary text now provides the required structural definition: for a parent linear code C, the notation C = K × V means K and V are subcodes whose XOR/direct-sum decomposition of every parent codeword is unique, with trivial intersection. The executable fixture should therefore derive K and V by partitioning one parent generator basis rather than merely sampling two unrelated codes.

The fixture must record:

- dimension;
- generator rank;
- codebook size;
- factor indices;
- clean composite;
- corruption model;
- seed;
- fixture revision hash.

The first benchmark should remain deliberately small enough that the exhaustive oracle is cheap.

### Stage C — recovery comparator

Implement the paper's bound-recovery construction faithfully after verifying the exact derivation against the paper.

Do not substitute a heuristic.

Output:

- recovered factor;
- exact correctness;
- candidate-set size / search-space reduction where defined;
- operation counts or equivalent deterministic work proxy;
- failure classification.

### Stage D — matched resonator control

Run the existing two-factor resonator harness against an ordinary bipolar random-code fixture with the same dimensions and codebook cardinality.

Keep its existing cleanup-rule matrix separate from this comparator. The primary control should use the production-default Softmax configuration, with additional cleanup rules only as a secondary analysis if justified.

### Stage E — decoder oracle and noise geometry

First establish the coding-theoretic boundary with an exhaustive, research-only nearest-codeword oracle. This oracle is ground truth for small fixtures; it is not a production decoder and must not be described as an implementation of error correction.

For a code with minimum Hamming distance d_min, enumerate all corruption patterns with weight at most floor((d_min - 1) / 2) and verify that the clean codeword is the unique nearest codeword. In the same cases, the exact GF(2) span solver should generally reject the corrupted observation because a nonzero error below d_min is outside the code.

This separates three contracts that must not be conflated:

- **membership detection**: is the observed vector itself a codeword?
- **exact algebraic recovery**: can the observation be represented in the code span?
- **error correction**: can a decoder infer the intended nearby codeword when the observation is outside the span?

Only after that oracle boundary is qualified:

- query corruption: 0.0, 0.10, 0.20 and explicit weight-based profiles;
- dimensions chosen from the smallest validated algebraic kernel through the existing HDC research scale;
- codebook sizes chosen so exhaustive ground truth remains practical;
- multiple deterministic seeds;
- an actual decoder objective and algorithm must be named before any production-quality noise claim is made.

Do not expand the benchmark matrix until the algebraic invariants and oracle semantics pass.

## Acceptance gates

The branch is ready for an evidence-producing benchmark only when:

- every generated code is full-rank according to its declared rank;
- every fixture is reproducible byte-for-byte from its seed and revision;
- the clean fixture's identifiability conditions are explicit and the recovery result matches exhaustive truth (including ambiguity) on all validation fixtures;
- intentionally corrupted queries produce classified outcomes rather than silent false positives;
- the resonator control uses the same ground-truth factor set and independently verified fixture;
- the evidence schema records the representation family explicitly.

## Why this is the next step

The current research chain has already isolated dimension, noise, associative cleanup, sequence order, resonator retrieval, representation separation, temperature, geometry×temperature, and cleanup nonlinearity. Another resonator knob would increasingly risk local optimization of one solver family.

A linear-code comparator tests a different hypothesis: whether algebraic representation structure can replace iterative attractor search for a class of factorization problems.

The literature supports treating this as a distinct recovery family. Raviv reports that random linear codes retain favorable storage properties while exposing subspace structure that enables specialized recovery; the paper reports substantial speed advantages in its own benchmark, but those results must not be transplanted to Symthaea without a matched implementation and experiment.

## Non-goals

- no production default change;
- no claim that linear codes are universally superior;
- no benchmark result in this design note;
- no composite "best HDC" score;
- no replacement of the resonator architecture;
- no production decoder or noise-tolerance claim from the exact span solver alone.


## Stage-C algebraic kernel status

The research-only GF(2) kernel now has both a general representative-recovery path and a stricter independent-factor path.

Implemented:

- solve_linear_combination performs packed-u64 Gaussian elimination over GF(2).
- recover_linear_bound follows Raviv's Theorem 2 construction: it forms a maximal linearly independent subset of the union of factor generator bases, solves the resulting system, and deterministically assigns retained generators to their first owning factor.
- recover_direct_sum_bound is the two-factor specialization for structurally disjoint subcodes.
- recover_independent_bound generalizes direct-sum recovery to F factor codebooks when the concatenated generator bases are jointly independent.
- The general API preserves factor order and returns a valid representative factorization without labeling it unique.
- The independent API remains fail-closed on overlapping/dependent factors when a uniqueness-guaranteed decomposition is required.
- The existing exhaustive two-factor oracle remains the ground truth.
- A three-factor parent-basis-partition fixture validates exact recovery against exhaustive truth.
- An overlap fixture now validates that representative recovery works while exhaustive enumeration demonstrates multiple valid decompositions.

This now matches the scope of Raviv's Theorem 2 more closely: the theorem constructs a maximal linearly independent subset of the union of the participating generator bases and then solves a GF(2) system. The theorem also explicitly notes that factorization need not be unique for overlapping subcodes. The comparator therefore distinguishes **existence/representative recovery** from **global uniqueness**, rather than treating overlap as either impossible or uniquely solvable.

### Primary-text verification

Raviv's 2024 paper defines the parent-code product/direct-sum construction using subcodes whose XOR decomposition is unique, and Section IV-A Theorem 2 gives the generator-basis GF(2) recovery construction for bound representations. The current two-factor and F-factor independent-basis implementations are faithful specializations of that algebraic path; they do not yet reproduce the paper's full benchmark harness or parameterization.

The implementation deliberately keeps the representation research-only. No Symthaea production default, resonator path, or cleanup rule is changed.

### Next research axis: noisy recovery

A 2026 Deng–Raviv result, Efficient Vector Symbolic Architectures from Histogram Recovery, identifies noisy decoding as a central limitation of random linear-code VSA and develops a different coding-theoretic construction based on Reed-Solomon/Hadamard concatenation and histogram recovery. The work provides a useful warning for this branch: clean GF(2) exact recovery is not sufficient evidence for noise robustness, and a noisy-recovery study should not be presented as a trivial extension of the current random-linear-code comparator.

For Symthaea, the next controlled experiment should therefore remain representation-explicit:

1. finish deterministic clean recovery qualification for the independent F-factor path;
2. qualify the exhaustive corruption taxonomy and bounded-distance nearest-codeword oracle;
3. implement a genuine decoder only after the noise objective and representation are explicit;
4. compare that decoder against the existing resonator control only after the representations and corruption models are matched.

The 2026 histogram-recovery construction should be treated as a separate comparator family unless its encoding and observation model are intentionally adopted; it should not be silently substituted into this experiment.


### Noise phase diagram: exact-recovery boundary

The first executable noise study intentionally uses a small exact Boolean space so every corruption pattern can be enumerated. Each error vector is classified by whether the error itself belongs to the generated linear-code subspace.

This distinguishes:

- **clean**: zero error; recovery returns the original codeword/message;
- **nonzero in-span corruption**: the observation remains an exact codeword, so algebraic recovery succeeds but generally identifies a different codeword/message;
- **out-of-span corruption**: the observation leaves the code, so the clean GF(2) solver rejects it.

For a rank-`r` binary linear code embedded in `n` Boolean coordinates, exactly `2^r` of the `2^n` possible error vectors are in the code. The exhaustive fixture verifies this invariant and verifies the semantic distinction between algebraic solvability and recovery correctness.

This is deliberately **not** presented as an error-correction result. It establishes the exact-recovery boundary that a decoder must cross.

### Bounded-distance decoder oracle

The deterministic fixture now also defines a research-only nearest-codeword oracle. It exhaustively computes the minimum Hamming distance and checks the standard unique-decoding radius floor((d_min - 1) / 2). For every corruption pattern inside that radius, the oracle requires the original clean codeword to be the unique nearest codeword.

The oracle is intentionally kept separate from solve_linear_combination: below the unique-decoding radius, a nonzero error can produce an observation outside the linear-code subspace even though the intended codeword is uniquely recoverable by Hamming distance. The test therefore asserts both facts simultaneously. This is an executable distinction between **detection/membership** and **correction**, not a heuristic claim about a production decoder.

The fixture also pins the minimum-distance ambiguity boundary. A corruption equal to a minimum-weight nonzero codeword leaves the observation inside the code, but changes it to another valid codeword. Exact span recovery therefore returns a different valid message; the original message is not identifiable from that observation alone. This is the concrete reason the clean algebraic solver cannot be promoted into an error-correcting decoder without an explicit noise model and decoding objective.

The recent Deng–Raviv noisy-VSA construction is a separate comparator family: it changes the code construction to a Reed–Solomon/Hadamard concatenation and uses histogram recovery/list-decoding machinery to obtain formal noise resilience. It should remain separate from the current random-linear-code exact-recovery implementation unless that representation is intentionally adopted.

A September 2026 result by Silas further sharpens the random-linear-code decoding boundary: at rates approaching Hamming list-decoding capacity, the worst-case list size is asymptotically determined for every finite field, with the binary case recovering the previously tight constant-order behavior. This is useful context for future Stage-E experiments, but it is a coding-theoretic asymptotic/list-decoding statement, not a claim about the small deterministic fixtures used here.

### Acceptance gates

The branch is ready for a benchmark-producing Stage-C/Stage-D experiment only when:

- the focused Rust qualification workflow is green on the exact PR head;
- every generated code is full-rank and reproducible;
- two-factor and three-factor clean recovery match exhaustive truth;
- overlap/dependence is classified as non-unique or rejected rather than mislabeled as exact recovery;
- corrupted targets have explicit in-span/out-of-span classification;
- the resonator control uses a separately validated, representation-matched fixture;
- evidence records representation family, factor count, code ranks, seed, corruption model, and revision.

No production integration is proposed by this document.

## References

- Raviv, N. (2024), Linear Codes for Hyperdimensional Computing, Neural Computation 36(6), 1084–1120. DOI: 10.1162/neco_a_01665; arXiv:2403.03278.
- Deng, Z. K., & Raviv, N. (2026), Efficient Vector Symbolic Architectures from Histogram Recovery, ISIT 2026, DOI: 10.1109/ISIT62367.2026.11654060.
- Silas, S. (2026), The list size of random linear codes at capacity, arXiv:2609.06570.
