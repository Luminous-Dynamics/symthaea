# HDC Linear-Code Recovery Comparator — Design Note

Status: research design only. No production integration is proposed by this document.

## Purpose

The resonator cleanup-rule sweep establishes that cleanup nonlinearity is an explicit algorithmic axis. The next useful separation is therefore a recovery family that changes the representation/recovery algebra rather than adding another resonator hyperparameter.

Raviv (2024), *Linear Codes for Hyperdimensional Computing*, proposes random linear codes as Boolean-field subspaces and derives recovery procedures for bundled and bound compositional representations. The paper reports that the bound-recovery method uses subspace structure and provides provable factorization under its stated framework; its bundled method constructs Boolean linear systems that reduce the search space. These are materially different from Symthaea's current bipolar random-code resonator path.

References:
- DOI: 10.1162/neco_a_01665
- arXiv: 2403.03278

## Literature boundary update

The 2024 Raviv result is the representation/recovery basis for this comparator: random linear codes are Boolean subspaces, and bound recovery is obtained through their subspace structure. Raviv explicitly distinguishes exact computation from approximate/noisy recovery. The published article is *Neural Computation* 36(6), 1084–1120 (2024).

A 2026 peer-reviewed follow-up by Deng and Raviv, *Efficient Vector Symbolic Architectures from Histogram Recovery* (ISIT 2026), addresses a different problem: noisy recovery. Its construction uses Reed–Solomon/Hadamard concatenation and histogram recovery with list-decoding methods, rather than treating the exact GF(2) span solver as an error-correcting decoder. This reinforces the comparator boundary: exact factorization and noise-resilient decoding must be benchmarked as separate capabilities.

Current coding-theory work in 2026 is also advancing time/space-efficient list decoding near capacity. Those results are useful context for a future decoder study, but they do not by themselves establish performance for the small deterministic linear-code fixtures in this repository.

Sources:
- Raviv (2024): https://arxiv.org/abs/2403.03278
- Deng & Raviv (2026 / preprint): https://arxiv.org/abs/2511.01838
- Deng & Raviv (ISIT 2026): https://doi.org/10.1109/ISIT62367.2026.11654060
- Fathollahi, Ron-Zewi & Wootters (2026): https://arxiv.org/abs/2608.15937

### Recent noise-decoding context

Deng and Raviv, *Efficient Vector Symbolic Architectures from Histogram Recovery* (arXiv:2511.01838v2, revised 2026-04-16), extends the linear-code VSA direction specifically because random binary linear codes are difficult to decode under noise. Their construction uses a Reed–Solomon outer code concatenated with a Hadamard inner code and a dedicated histogram-recovery/list-decoding layer. They retain linear-code binding-recovery via subcode structure, but treat noisy superposition recovery as a separate decoding problem.

This is a useful boundary for this comparator: the exact GF(2) span solver here establishes clean binding recovery and representation semantics; it is not evidence for noisy recovery. Any future noise experiment must name the decoder and code construction explicitly rather than interpreting out-of-span detection as correction.

## Reproducibility contract

Fixture generation uses the explicitly named `rand_chacha` `ChaCha12Rng` algorithm with a recorded `u64` seed, rather than `StdRng`, whose algorithm is intentionally non-portable. The qualification harness pins Rust to `1.96.1`, pins `rand_chacha` to `0.3.1`, records the resolved `Cargo.lock` hash, and records the exact source/test hashes and PR head. This makes fixture identity explicit as **seed + RNG algorithm + dependency lock + source/test revision**, rather than treating a seed alone as a permanent byte-level identity.

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
- deterministic random full-rank generator matrix using the pinned `rand_chacha` `ChaCha12Rng` algorithm;
- row-reduction / rank;
- deterministic basis construction;
- Boolean subspace membership;
- conversion between packed GF(2) words and the experiment's bipolar observation vector, using Raviv's convention GF(2) 0 -> +1 and GF(2) 1 -> -1 so XOR is exactly bipolar Hadamard binding.

Tests should cover:

- closure under XOR;
- zero-vector membership;
- rank preservation;
- deterministic regeneration from seed, with the RNG algorithm and dependency lock recorded;
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

#### Deterministic work ledger

The comparator now exposes a research-only `LinearCodeWork` ledger rather than relying on wall-clock timing. It records:

- span-membership checks used while constructing the maximal independent subset;
- rank pivots and packed-word XOR work during rank tests;
- solve pivots and packed-word XOR work during the GF(2) solve;
- retained generators in the resulting independent basis;
- input-word copies performed by rank construction;
- basis-bit probes and augmented-matrix word cells initialized by the GF(2) solver;
- packed-word XOR operations used to project the recovered coefficients back into factor codewords.

Repeated execution of the same deterministic fixture must produce both the same recovered result and the same ledger. These counters are algorithmic work units, not a universal hardware-independent runtime estimate, and must not be collapsed into a composite performance score.

Each emitted work/scaling record is now self-identifying with its dimension, factor ranks, and seed; the receipt therefore carries fixture identity alongside algorithmic work units.


### Deterministic code-geometry ledger

The comparator now records minimum and maximum Hamming distance, signed and absolute bipolar inner products, and the exact finite-code ε-balanced numerator/denominator for a bounded exhaustive fixture. A four-seed sweep at the same \(n=16,k=5\) scale also records each realization separately, reducing the chance that a single deterministic seed becomes an accidental geometry proxy. For a linear code, the minimum distance equals the minimum weight of a nonzero codeword, and every pairwise difference is again a codeword; therefore the exhaustive pairwise check cross-validates the geometry calculation. This is a representation property, not a quality score and not a substitute for the paper's asymptotic/random-code guarantees. The canonical \(n=96,k=8\) fixture is now geometry-checked as part of the fingerprint capture, while the separate seed sweep remains an explicit sensitivity analysis.

This ledger is intended to test the structure of the recovery algorithm before timing or energy measurements are introduced. In particular, the measurement should preserve the distinction between the maximal-independent-subset construction and the final linear solve described by the paper.

#### Small-fixture scaling qualification

The qualification harness now executes a deterministic structural sweep at dimensions 32, 64, and 96 with two equal factor ranks. For each fixture it records the recovery ledger and verifies the implementation-level invariant that an independent parent basis of total rank Δ produces exactly Δ retained generators, Δ span-membership checks, and Δ² rank pivots across the before/after rank tests used by the maximal-independent-subset construction. This is an implementation invariant, not a new theorem; it exists to make the measured ledger auditable before introducing larger matrices or wall-clock measurements.

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
- every fixture is reproducible byte-for-byte from its seed, pinned RNG algorithm, dependency lock, and revision;
- the clean fixture's identifiability conditions are explicit and the recovery result matches exhaustive truth (including ambiguity) on all validation fixtures;
- the packed GF(2) solver is exhaustively cross-checked against code-subspace membership on a bounded Boolean space;
- intentionally corrupted queries produce classified outcomes rather than silent false positives;
- the resonator control uses the same ground-truth factor set and independently verified fixture;
- the evidence schema records the representation family explicitly;
- deterministic work ledgers are reproducible on repeated qualification fixtures and distinguish construction work from solve work;
- bounded code-geometry ledgers cross-check minimum distance against exhaustive pairwise Hamming distance and bipolar inner product.
- rank/nullity identities are emitted and mechanically checked on clean fixtures, including the exact 2^d fiber cardinality;
- explicit two-way overlap and three-way dependency fixtures distinguish pairwise independence from global uniqueness;
- the paper-scale result digest commits to the algebraic geometry fields and representative-selection boundary.
- dependent fixtures expose and verify a complete kernel basis with exactly `kernel_dimension` witnesses.
- bounded fiber enumeration is explicitly capped and its exhaustiveness is qualified on the d=4 ambiguity fixture.

## Why this is the next step

The current research chain has already isolated dimension, noise, associative cleanup, sequence order, resonator retrieval, representation separation, temperature, geometry×temperature, and cleanup nonlinearity. Another resonator knob would increasingly risk local optimization of one solver family.

A linear-code comparator tests a different hypothesis: whether algebraic representation structure can replace iterative attractor search for a class of factorization problems.

The literature supports treating this as a distinct recovery family. Raviv reports that random linear codes retain favorable storage properties while exposing subspace structure that enables specialized recovery; the paper reports substantial speed advantages in its own benchmark, but those results must not be transplanted to Symthaea without a matched implementation and experiment.

### Paper-scale binding-recovery smoke matrix

The qualification harness now covers the paper's published \(n\in\{500,1000,2000\}\), \(k\in\{3,5,7\}\), and \(F\in\{3,4,5\}\) binding-recovery parameter grid, using ten deterministic repeats per setting, matching the published experiment's repeat count. The matrix deliberately records validity and joint-basis independence rather than timing or superiority. It also emits a deterministic BLAKE3 result digest over the case parameters, targets, original factors, and recovered factors, so a change that preserves aggregate success counts is still detectable.

### Published-grid search-space ledger

For bound recovery with factor ranks \(k_1,\ldots,k_F\), exhaustive factor enumeration contains exactly \(2^{\sum_i k_i}\) candidate tuples. The algebraic method instead solves for \(\sum_i k_i\) Boolean coefficients over the retained independent basis. The qualification harness records both quantities across the same published parameter grid. This is a structural search-space comparison only; it is not a measured runtime ratio, and it does not assume that an exhaustive implementation and the solver have identical constant factors.

### Exact rank/nullity and fiber-multiplicity ledger

The comparator now treats identifiability as an algebraic property of the factor-to-bound map rather than an absence-of-counterexample result from sampling.

For factor code dimensions k_1, ..., k_F, define Delta = sum_i k_i and concatenate independent generator bases into one GF(2) matrix. Let r be the rank of that concatenated basis and d = Delta - r. The factor coefficient domain has 2^Delta tuples, the image contains exactly 2^r reachable targets, and every representable target has exactly 2^d preimages. This follows because every non-empty fiber of a linear map is a coset of its kernel.

The research API records these quantities as first-class fields:

- factor_dimension_sum = Delta;
- union_generator_rank = r;
- kernel_dimension = d;
- raw_factor_tuple_count = 2^Delta;
- reachable_target_count = 2^r;
- factorization_count_per_target = 2^d;
- unique_factorization = (d = 0);
- dependency_order = the minimum number of factor groups supporting a non-zero dependency, or absent when the factors are jointly independent.

Cardinalities are stored symbolically as exact powers of two, so large search spaces are never truncated by machine integer overflow. The target-specific multiplicity API returns the same fiber cardinality for every representable target and returns no factorization for a target outside the union span.

### Kernel-basis ambiguity certificate
### Affine-fiber certificate
Factor ordering is deliberately separated from algebraic identity. Reordering the factor list leaves Delta, union rank, kernel dimension, reachable-target count, and fiber cardinality unchanged. It may change the canonical kernel-basis presentation and the deterministic representative, because those are defined relative to the ordered generator presentation. Qualification tests now verify both halves of this statement.


For a representable target, `factorization_affine_fiber` now packages one deterministic coefficient representative, the complete kernel basis, and the exact fiber cardinality in one object. Any alternate factorization is obtained by XORing the representative with a GF(2) mask over the kernel basis.

The affine-fiber certificate (schema v3) also carries private BLAKE3 provenance and integrity fingerprints. The source fingerprint commits to the target and the complete ordered factor presentation; the integrity fingerprint commits to that source binding plus the representative, kernel witnesses, factor supports, pivots, and exact cardinality. These bindings are versioned separately as `linear-code-affine-source-v1` and `linear-code-affine-integrity-v1` so evidence consumers can distinguish provenance-bearing certificates from earlier shape-only certificates. Because those anchors are not externally writable, mutation of the public certificate fields cannot silently create a fresh-looking valid certificate: bounded operations fail closed unless the current fields still match the originally constructed certificate.

### Bounded affine-fiber enumeration
`LinearCodeFactorizationFiber::iter_bounded(max_fibers)` now provides an explicit exhaustive validation path without weakening the symbolic default. The iterator is created only when the exact `2^d` fiber cardinality fits in `usize` and is no larger than the caller's stated bound; otherwise it returns no iterator. This prevents a paper-scale or high-nullity fixture from accidentally expanding an exponential search space merely because a certificate exists.

The iterator covers every kernel mask exactly once, so bounded qualification can independently verify three properties at once:

- the declared fiber cardinality is attained exactly;
- each enumerated coefficient tuple maps back to the same target;
- kernel-basis masks do not collapse to duplicate coefficient tuples.

`coefficients_for_mask` also rejects malformed certificate dimensions before applying a kernel direction. This keeps the affine-fiber object fail-closed even when its public fields are modified in a test fixture.

The fiber API additionally validates the kernel certificate structurally before enumeration: every declared dependent-generator coordinate must be in range and asserted in its witness, those coordinates must be distinct, and the witness coefficient vectors must have full rank equal to the declared kernel dimension. Thus the bounded iterator is backed by the same linear-independence invariant that makes the `2^d` mask construction bijective.This makes the theorem boundary explicit in the API itself:

`target in image` -> `affine fiber exists` -> `fiber cardinality = 2^d` -> `kernel basis gives the d ambiguity directions`.

The certificate does not enumerate the exponentially large fiber by default. Bounded qualification tests may enumerate it explicitly; paper-scale qualification remains symbolic while still committing the certificate geometry into the deterministic digest.

### Noisy factorization boundary
A new bounded oracle tests the noisy extension without pretending that an exact algebraic solver is a decoder. For a fixed ordered factor presentation with kernel dimension `d`, every reachable target has exactly `2^d` factor tuples. Consequently, for any received observation `y` and Hamming radius `t`, the number of factor tuples whose bound lies within that ball is exactly

`(# reachable targets within radius t of y) * 2^d`.

The qualification test exhausts this identity on a small repeated-factor construction across every ambient observation and several radii. This is stronger than checking one decoder output: it proves that the combinatorial candidate-list size factors into image-neighborhood ambiguity and intrinsic affine-fiber multiplicity. In particular, improved target decoding cannot make a non-unique factorization unique; the `2^d` multiplicity survives every zero-noise or noisy observation that lands on the same reachable target.


The comparator now exposes the complete kernel basis of the factor-to-bound map, not only its dimension or a single dependency witness.

During the ordered maximal-independent-subset construction, every generator that fails span extension yields one dependency witness by solving that generator against the already-retained independent basis. Each such witness has a distinct dependent-generator coordinate, so these witnesses are linearly independent. Their count is exactly Delta-r and therefore they form a basis of the full kernel. The public `factorization_kernel_basis` constructor now also self-audits the returned witness family: it checks the complete coefficient-space rank and re-verifies every witness against the supplied factor presentation before returning. This turns the claimed basis invariant into an executable postcondition rather than only a property inferred from the construction argument.

This gives a constructive interpretation of the multiplicity theorem: once one valid factor tuple is known, adding any GF(2) combination of the kernel-basis witnesses produces another valid coefficient tuple for the same target. The evidence therefore distinguishes:

- kernel dimension: how many independent ambiguity directions exist;
- kernel basis: the concrete ambiguity directions in the ordered generator presentation;
- factorization multiplicity: the exact 2^d number of tuples in every non-empty fiber.

The witness basis is canonical only relative to the declared factor order and generator-basis presentation. It is not a minimum-weight dependency basis, and no permutation-invariance of the representative is claimed.

The kernel-basis constructor is itself fail-closed: before returning, it requires distinct and in-range dependent-generator pivots, an asserted pivot bit for every witness, full coefficient-space rank equal to Delta-r, and semantic re-verification of every witness. The affine-fiber verifier reuses the complete structural certificate invariant rather than relying on a weaker duplicate subset of those checks.

Qualification also tests certificate non-transplantability. A certificate constructed for one target, ordered factor presentation, or generator-basis ordering must not verify against another of those contexts. The private source and integrity anchors make those rebinding attempts observable even when the resulting factor geometry has the same Delta, union rank, kernel dimension, and reachable-target cardinality.

### Adversarial dependency qualification

Pairwise subcode-intersection checks are not sufficient when three or more factors participate. The qualification suite therefore includes both a two-factor overlap fixture and a three-factor fixture

    C1 = span(e1), C2 = span(e2), C3 = span(e1 + e2),

where every pair is independent but the triple has Delta = 3, r = 2, d = 1 and therefore exactly two valid factorizations for every reachable target.

A deterministic n = 8, k = 3, F = 3 stress sweep over 20,000 realizations additionally searches for pairwise-trivial but globally dependent cases. This is an adversarial structural check, not a statistical claim about the paper-scale random ensemble.

### Digest and representative boundary

The paper-scale BLAKE3 result digest now commits to the algebraic geometry fields in addition to the fixture parameters, targets, original factors, and selected recovered representatives. Thus a change that preserves aggregate recovery counts but changes uniqueness or multiplicity is detectable.

The evidence boundary remains explicit: algebraic rank/nullity establishes existence, uniqueness, and multiplicity; the general recovery API may choose one deterministic representative under the declared ownership policy; representative choice is not promoted to a uniqueness claim for overlapping subcodes.
### Published-grid storage ledger

Raviv's Remark 6 distinguishes representation size from recovery work: an arbitrary codebook supplied explicitly requires \(|C_i|n\) bits per factor, while a linear code can be represented by its generator matrix using \(k_i n\) bits. The qualification harness now records those theoretical payload sizes separately from the actual packed-`u64` generator storage used by this implementation, along with target storage. These are raw representation quantities, not a composite efficiency score; struct/allocator overhead is intentionally excluded.
 When the sampled factor bases are jointly independent, the unique GF(2) representation makes exact original-factor recovery testable; when dependence occurs, the fixture is classified rather than treated as a benchmark failure. The published success contract remains valid factor membership plus exact rebinding, while exact retrieval of the originally sampled factors is only asserted in the identifiable cases.


## Non-goals

- no production default change;
- no claim that linear codes are universally superior;
- no benchmark result in this design note;
- no composite "best HDC" score;
- no replacement of the resonator architecture;
- no production decoder or noise-tolerance claim from the exact span solver alone;
- arithmetic aggregation paths fail closed on `usize` overflow rather than wrapping into a fabricated coefficient-space dimension.


## Stage-C algebraic kernel status

The research-only GF(2) kernel now has both a general representative-recovery path and a stricter independent-factor path.

Implemented:

- solve_linear_combination performs packed-u64 Gaussian elimination over GF(2).
- `recover_linear_bound` follows Raviv's Theorem 2 for the maximal-independent-subset and GF(2)-solve stages. In the product/direct-sum regime, exhaustive tests validate exact factor recovery. For overlapping factors, the final owner assignment is a deterministic Symthaea extension used to return one valid representative without claiming theorem-specified uniqueness.
- recover_direct_sum_bound is the two-factor specialization for structurally disjoint subcodes.
- recover_independent_bound generalizes direct-sum recovery to F factor codebooks when the concatenated generator bases are jointly independent.
- The general API preserves factor order and returns a valid representative factorization without labeling it unique.
- The independent API remains fail-closed on overlapping/dependent factors when a uniqueness-guaranteed decomposition is required.
- For overlapping factors, representative recovery is deterministic for a fixed factor ordering, but the representative may change when factor order changes; permutation-invariance is not claimed.
- Generator-basis ordering is likewise presentation-level: bounded exhaustive tests require every reordered basis presentation to preserve factor membership and the recovered XOR target, without claiming the representative coefficients are invariant.
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

The coding-theory boundary has also widened beyond ordinary worst-case list decoding. Guruswami, Li, and Singhal (2026) prove an average-radius list-decoding guarantee for random linear codes with the same \(O(1/\varepsilon)\)) list-size scaling near capacity, while Doron, Mosheiff, Resch, and Ribeiro's small-field list-recovery work distinguishes error-list and erasure-list models from ordinary Hamming-ball decoding. These are useful future comparator dimensions, but neither should be projected onto this Boolean HDC fixture without an explicit observation model. In particular, an HDC system receiving uncertain symbol values or erasures may be a list-recovery problem rather than a plain Hamming-ball problem.


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

### Current coding-theory literature boundary (September–October 2026)

Recent work reinforces the need to keep finite-fixture geometry and noisy decoding explicit. Doron et al. (2026) prove broad discrepancy results for random linear codes and show that, in the relevant asymptotic regime, random linear codes can behave like unstructured random codes for list-decoding above capacity. Kumar and Mon (2026) give improved near-capacity list-decoding bounds for random linear codes over finite fields. Neither result is a guarantee for a particular finite HDC fixture, so this comparator records actual code geometry instead of inferring robustness from the random-linear-code label.

References: https://arxiv.org/abs/2606.24471 ; https://eccc.weizmann.ac.il/report/2026/222/ ; https://arxiv.org/abs/2609.17020


A September 2026 result by Silas further sharpens the random-linear-code decoding boundary: at rates approaching Hamming list-decoding capacity, the worst-case list size is asymptotically determined for every finite field, with the binary case recovering the previously tight constant-order behavior. This is useful context for future Stage-E experiments, but it is a coding-theoretic asymptotic/list-decoding statement, not a claim about the small deterministic fixtures used here.

### Stage-E decoder boundary update

Recent September 2026 coding-theory results make the next experiment more precise. Silas determines the typical worst-case list size of random linear codes at capacity up to the sharp binary constant-order boundary, while Yuan and Zhu obtain asymptotically optimal list-size scaling for every fixed finite field. Doron et al. independently show that random linear codes exhibit near-random-code discrepancy behavior for list-decoding above capacity.

For a future noisy HDC comparator, this suggests recording at least three separate observables:

- whether the intended codeword remains in the candidate list;
- candidate-list cardinality at the observed corruption radius;
- whether factorization itself is uniquely identifiable after mapping candidate codewords back through the affine factorization fiber.

These quantities must remain distinct. A decoder may produce a short list while the factorization fiber still contains multiple valid tuples, and a large factorization fiber can persist even when the target codeword itself is exactly identified. No Stage-E decoder is introduced by this design update.

The Stage-E evidence schema should therefore preserve a three-layer decomposition:

- **channel layer** — corruption radius, corruption distribution, erasures/soft uncertainty, and the exact observation alphabet;
- **codeword layer** — candidate-list inclusion and size for the chosen decoder;
- **factor layer** — exact affine-fiber multiplicity and whether the intended factor tuple remains identifiable after candidate generation.

This prevents a decoder improvement from being credited for resolving algebraic ambiguity that the factorization map itself cannot remove. It also leaves room to compare ordinary list decoding, average-radius guarantees, and future list-recovery experiments without changing the meaning of the existing clean-recovery receipt.

Sources:
- Silas (2026), *The list size of random linear codes at capacity*: https://arxiv.org/abs/2609.06570
- Yuan & Zhu (2026), *Asymptotically Optimal List Size of Random Linear Codes*: https://arxiv.org/abs/2609.01070
- Doron et al. (2026), *Discrepancy for Random Linear Codes*: https://eccc.weizmann.ac.il/report/2026/222/

### Acceptance gates

The branch is ready for a benchmark-producing Stage-C/Stage-D experiment only when:

- the focused Rust qualification workflow is green on the exact PR head;
- every generated code is full-rank and reproducible from the recorded seed, `ChaCha12Rng` algorithm, dependency lock, and revision;
- two-factor and three-factor clean recovery match exhaustive truth;
- overlap/dependence is classified as non-unique or rejected rather than mislabeled as exact recovery;
- bounded exhaustive overlap fixtures verify recovery existence against the independent factor-pair truth oracle, not merely successful reconstruction of selected targets;
- corrupted targets have explicit in-span/out-of-span classification;
- the resonator control uses a separately validated, representation-matched fixture;
- evidence records representation family, factor count, code ranks, seed, corruption model, and revision.

No production integration is proposed by this document.

## References

- Raviv, N. (2024), Linear Codes for Hyperdimensional Computing, Neural Computation 36(6), 1084–1120. DOI: 10.1162/neco_a_01665; arXiv:2403.03278.
- Deng, Z. K., & Raviv, N. (2026), Efficient Vector Symbolic Architectures from Histogram Recovery, ISIT 2026, DOI: 10.1109/ISIT62367.2026.11654060.
- Silas, S. (2026), The list size of random linear codes at capacity, arXiv:2609.06570.

### Implementation hardening note

The research kernel now uses checked factor-rank aggregation in the independent recovery and dependency-verification paths. The affine-fiber and algebra constructors already carried checked aggregation; the remaining public aggregation sites were aligned so arithmetic overflow cannot silently produce a malformed coefficient-space size. This is an implementation invariant, not a practical claim that ordinary HDC fixtures approach machine-size limits.

Additional coding-theory references:
- Guruswami, Li, and Singhal (2026), *Average-Radius List-Decodability of Random Linear Codes*: https://arxiv.org/abs/2608.22663
- Doron, Mosheiff, Resch, and Ribeiro (2025/2026), *List-Recovery of Random Linear Codes over Small Fields*: https://arxiv.org/abs/2505.05935
