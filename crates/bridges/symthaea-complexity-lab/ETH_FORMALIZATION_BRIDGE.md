# ETH-022: theorem bridge and formalization-scope audit

**Status:** Primary-source theorem ledger and research plan. No new asymptotic theorem is claimed by this document.

## 1. Executive finding

There are three separate mathematical claims in the current research discussion. They have different verification status and must remain separate.

### Claim A — formalized exact counting result

The public OpenAI math repository's family-378 Lean scope document states that Lean proves a deterministic machine exactly counts satisfying assignments of canonically encoded fixed-width k-CNFs in time

\[
(L+1)^{O_k(1)}2^{O_k(n(\log\log n)^2/\sqrt{\log n})},
\]

where L is the encoded input length, with a polynomial upper bound on visited tape cells at every execution step.

The solution declaration is **OAI.Algorithms.SubexponentialSAT.SqrtLogSpace.exact_count_and_sat** in:

- **openai/math/lean/OAI/Algorithms/SubexponentialSAT/SqrtLogSpace/Final.lean**
- Scope summary: **openai/math/lean/docs/378.md**
- Comparator configuration: **openai/math/lean/ComparatorChallenges/SubexponentialSAT.json**

The Lean declaration is stronger than a formal identity: it specifies a fixed compiled primitive machine, exact count output tape, SAT answer tape, bit-time bound, and all-prefix visited-space bound. The comparator configuration lists **propext**, **Quot.sound**, and **Classical.choice** as permitted axioms. The solution file contains an explicit #print axioms audit for the main declarations.

### Claim B — sharper October 25 manuscript theorem

The October 25 manuscript *The Exponential Time Hypothesis is False* states an exact deterministic algorithm with bit-time

\[
(n+m)^{O_k(1)}
2^{O_k(n\log\log n/\sqrt{\log n})}
\]

for arbitrary fixed-width CNFs. For each fixed density constant D, it states the stronger bound

\[
m\le Dn \implies
(n+m)^{O_{k,D}(1)}
2^{O_{k,D}(n/\sqrt{\log n})}.
\]

Primary source:

**openai/math/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026/build/main_full_version.tex**

The family-378 Lean scope document explicitly lists these as the manuscript's sharper bounds, distinct from the weaker bound currently attributed to its Lean formalization.

### Claim C — October 29 weighted extension

*Subexponential Weighted Counting and Exact Optimisation* extends the construction to sparse weighted counting and exact optimization over commutative Q-algebras, using subtraction, rational interpolation and truncated polynomial algebras.

Primary source: https://hexagonmath.org/2610.00594v1

This is an unrefereed AI-assisted preprint. Its public reproduction uses the squared-log-log exponent and explicitly does not demonstrate the sharper dense counting-preserving construction. Its weighted extension must not be described as covered by the family-378 Lean theorem.

## 2. What Symthaea should build

The purpose of this line is not to copy the claimed asymptotic theorem. It is to build a verifiable bridge between:

1. a paper's mathematical statement;
2. a machine-checked formal theorem;
3. an executable algorithm;
4. exact result certificates;
5. observed resource measurements.

Every step needs a precise contract. A theorem about canonical bit encodings, an algorithm stated in terms of n and m, and a weighted extension over a Q-algebra are not interchangeable without a formal input-size and semantics bridge.

## 3. Theorem ledger format

Each theorem or algorithm claim must record:

| Field | Required evidence |
|---|---|
| Source | Immutable repository commit, file path and line/section range |
| Input model | Canonical encoding, k, n, m, density and declared-variable conventions |
| Output contract | #SAT, weighted partition function, optimum/multiplicity or complete profile |
| Algebra | Exact coefficient domain and proven laws |
| Main result | Exact theorem statement, with quantifier scope |
| Formal artifact | Lean declaration and imported proof dependency graph, or explicitly absent |
| Complexity | Bit operations, arithmetic operations, bit time, peak space and coefficient growth kept separate |
| Executable reference | Frozen fixtures, direct oracle and negative controls |
| Status | Manuscript-only, Lean-formalized, independently rederived, finite executable evidence, or benchmarked |

A weaker formal theorem never inherits a stronger manuscript exponent. An executable finite test never becomes an asymptotic proof. A correct algebraic identity never proves that a machine implements it within a resource bound.

## 4. Proof decomposition to audit

The October 25 manuscript's construction contains at least these reusable proof boundaries:

1. **Bounded-occurrence reduction:** transform a general fixed-width CNF into the bounded-occurrence setting, preserving exact assignment counts through the intended copy/sparsification semantics.
2. **Clause polynomial encoding:** translate clauses to exact multilinear Boolean indicator polynomials.
3. **Low-degree deformation:** construct replacement linear forms P(T) that restore original inputs at T=0 and maintain an explicit degree bound.
4. **Measurement separation:** prove that after fixing shared measurements, every transformed clause contribution is local to one variable group and the global sum factors exactly.
5. **Auxiliary matrix construction:** prove the identity/correction decomposition, support restrictions and common measurement-row bound.
6. **Schedule/copy construction:** balance variable groups and occurrence labels without changing the represented assignment count.
7. **Interpolation recovery:** recover the value at zero from exact Taylor data at prescribed nonzero points, including invertibility and coefficient bounds.
8. **Bit/space complexity:** bound integer and rational numerator/denominator growth, matrix/table construction, parsing and all arithmetic in bit operations and bits of space.
9. **Machine realization:** compile the abstract loops and arithmetic into a concrete deterministic machine and account for all visited tape cells.

The Lean formalization's weaker exponent should be treated as a known result. ETH-022 must locate exactly which sharper manuscript lemmas or parameter choices are outside its current declaration.

## 5. Weighted algebra compatibility

The October 29 extension is not valid over every semiring used by ETH-015. Its interpolation and signed/rational combinations require a commutative Q-algebra. The runtime algebra registry must distinguish at least:

- commutative semiring: addition, multiplication, zero, one and distributivity;
- commutative ring: additionally additive inverses;
- commutative Q-algebra: rational scalar action compatible with ring operations;
- truncated polynomial quotient: a concrete finite representation with exact reduction and coefficient extraction.

A caller-supplied supports_subtraction=true or supports_rational_scalars=true is not a proof. The capability must be tied to an implementation and formal theorem.

The initial reference domains should include Q and Q(i), then small finite truncated polynomial quotients. Floating point must not appear in exact equality, interpolation invertibility, cancellation or output comparison.

## 6. Independent executable experiments

The first independent experiments should be deliberately small:

- compare the clause-polynomial sum against direct #SAT enumeration for all small CNFs;
- verify measurement-separated sums against direct polynomial evaluation;
- compare interpolation recovery against exact evaluation at T=0;
- test exact cancellation and signed coefficients in Q and Q(i);
- verify truncation at each ideal generator and extract boundary coefficients;
- mutation-test wrong measurement maps, wrong group assignment, duplicate/missing variable copies, noninvertible interpolation data and coefficient-height overflow.

These are finite differential tests. They are valuable for catching implementation and transcription errors, but they do not establish the asymptotic schedule or theorem.

## 7. Competitive evaluation

Do not start the competitive comparison until the same exact input and output contracts exist.

- #SAT: compare against Ganak/D4/other exact model counters on identical instances.
- WMC including negative weights: compare exact rational output against matching exact WMC support.
- Complex AMC: Ganak MPQ is a required baseline; exact coefficient arithmetic is established prior art.
- PMC/PWMC: separate projections and weights from full-assignment counting. Hidden witnesses must be existentially collapsed before projected weights are applied.

Report producer time, certificate generation, independent checker time/RSS, peak memory, coefficient bit lengths and exact solved instances separately. Include parser failures and malformed-certificate rejection counts.

## 8. Acceptance gates

- Every claim has a source-backed theorem statement and explicit status.
- The family-378 Lean result is never described with the sharper October 25 exponent.
- The weighted Q-algebra extension is never described as Lean-formalized unless a specific theorem and dependency path are found.
- Every supported local transformation has an algebra-qualified certificate and independent checker.
- 100% exact agreement with independent finite oracles on the frozen tractable corpus.
- Zero accepted wrong-domain or malformed certificates in adversarial mutation tests.
- Any performance advantage is measured end-to-end, including verification, against the strongest compatible external baseline.

## 9. Sources and provenance

- OpenAI math repository README: https://github.com/openai/math
- October 25 manuscript: https://github.com/openai/math/tree/main/preprints/The-Exponential-Time-Hypothesis-is-False-October-25-2026
- Lean family-378 scope: https://github.com/openai/math/blob/main/lean/docs/378.md
- Lean exact-count implementation: https://github.com/openai/math/blob/main/lean/OAI/Algorithms/SubexponentialSAT/SqrtLogSpace/Final.lean
- Lean comparator config: https://github.com/openai/math/blob/main/lean/ComparatorChallenges/SubexponentialSAT.json
- October 29 weighted extension: https://hexagonmath.org/2610.00594v1

## 10. Claim ceiling

This document maps claim boundaries and defines a verification programme. It does not reproduce the paper's asymptotic proof, close the gap between manuscript and Lean exponents, establish a weighted subexponential theorem, or claim SOTA performance.
