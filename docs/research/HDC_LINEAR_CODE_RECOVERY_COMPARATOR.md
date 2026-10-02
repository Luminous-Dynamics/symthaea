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
- conversion between packed GF(2) words and the experiment's bipolar observation vector.

Tests should cover:

- closure under XOR;
- zero-vector membership;
- rank preservation;
- deterministic regeneration from seed;
- round-trip conversion;
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

### Stage E — noise and geometry

Only after exact clean recovery is validated:

- query corruption: 0.0, 0.10, 0.20;
- dimensions chosen from the smallest validated algebraic kernel through the existing HDC research scale;
- codebook sizes chosen so exhaustive ground truth remains practical;
- multiple deterministic seeds.

Do not expand the matrix until the algebraic invariants pass.

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
- no 3-factor expansion before the two-factor comparator is validated.


## Stage-C algebraic kernel status

The research-only GF(2) kernel now has a clean two-factor and generalized independent-F recovery path.

Implemented:

- solve_linear_combination performs packed-u64 Gaussian elimination over GF(2).
- recover_direct_sum_bound is the two-factor specialization.
- recover_independent_bound generalizes the same construction to F factor codebooks when the concatenated generator bases are jointly independent.
- The API preserves factor order and rejects dimension mismatch or overlapping/dependent factor bases rather than implying uniqueness where the algebra does not provide it.
- The existing exhaustive two-factor oracle remains the ground truth.
- A new three-factor parent-basis-partition fixture validates exact recovery against exhaustive truth.
- An overlap fixture explicitly verifies the conservative rejection path.

This is intentionally narrower than the fully general overlapping-code version of Raviv's Theorem 2. The paper's theorem constructs a maximal linearly independent subset of the union of the participating generator bases and solves the resulting GF(2) system. When the union is already independent, the maximal subset is simply the full union, which is the case implemented here. The paper also notes that overlapping factor subcodes can make recovery non-unique, so the current API does not convert that case into a false uniqueness guarantee.

### Primary-text verification

Raviv's 2024 paper defines the parent-code product/direct-sum construction using subcodes whose XOR decomposition is unique, and Section IV-A Theorem 2 gives the generator-basis GF(2) recovery construction for bound representations. The current two-factor and F-factor independent-basis implementations are faithful specializations of that algebraic path; they do not yet reproduce the paper's full benchmark harness or parameterization.

The implementation deliberately keeps the representation research-only. No Symthaea production default, resonator path, or cleanup rule is changed.

### Next research axis: noisy recovery

A 2026 Deng–Raviv result, Efficient Vector Symbolic Architectures from Histogram Recovery, identifies noisy decoding as a central limitation of random linear-code VSA and develops a different coding-theoretic construction based on Reed-Solomon/Hadamard concatenation and histogram recovery. The work provides a useful warning for this branch: clean GF(2) exact recovery is not sufficient evidence for noise robustness, and a noisy-recovery study should not be presented as a trivial extension of the current random-linear-code comparator.

For Symthaea, the next controlled experiment should therefore remain representation-explicit:

1. finish deterministic clean recovery qualification for the independent F-factor path;
2. add a research-only corruption taxonomy for the Boolean code representation;
3. investigate whether a genuine decoder for the selected linear-code family can correct bounded corruption without changing the clean-recovery semantics;
4. compare that decoder against the existing resonator control only after the representations and corruption models are matched.

The 2026 histogram-recovery construction should be treated as a separate comparator family unless its encoding and observation model are intentionally adopted; it should not be silently substituted into this experiment.


### Noise phase diagram: exact-recovery boundary

The first executable noise study intentionally uses a small exact Boolean space so every corruption pattern can be enumerated. Each error vector is classified by whether the error itself belongs to the generated linear-code subspace.

This distinguishes:

- **clean**: zero error; recovery returns the original codeword/message;
- **nonzero in-span corruption**: the observation remains an exact codeword, so algebraic recovery succeeds but generally identifies a different codeword/message;
- **out-of-span corruption**: the observation leaves the code, so the clean GF(2) solver rejects it.

For a rank-`r` binary linear code embedded in `n` Boolean coordinates, exactly `2^r` of the `2^n` possible error vectors are in the code. The exhaustive fixture verifies this invariant and verifies the semantic distinction between algebraic solvability and recovery correctness.

This is deliberately **not** presented as an error-correction result. It establishes the identifiability boundary that a genuine decoder must cross. The next decoder experiment should therefore introduce an explicit noise model and decoder objective, rather than adding a nearest-codeword heuristic to the exact solver.

The recent Deng–Raviv noisy-VSA construction is a separate comparator family: it changes the code construction to a Reed–Solomon/Hadamard concatenation and uses histogram recovery/list-decoding machinery to obtain formal noise resilience. It should remain separate from the current random-linear-code exact-recovery implementation unless that representation is intentionally adopted.

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
