# ETH-018: proof-carrying algebraic transformation discovery

**Status:** Finite exact Python reference prototype committed on a research branch. Rust search engine, Lean proofs and performance qualification remain open.

## 1. Why this follows ETH-015/016/017

The current research line has three distinct optimizers:

- ETH-015 selects the output algebra: feasibility, count, MaxCount, ParetoCount or a polynomial profile.
- ETH-016 certifies a particular exact factor-elimination execution.
- ETH-017 searches elimination permutations and prices algebra-specific intermediate work plus certificate cost.

ETH-018 adds a different search dimension: equivalent representations of the computation itself. It asks whether we can transform and factor the computation graph, rather than only permuting a fixed set of elimination operations.

Proposed pipeline:

    factor/constraint instance
       -> typed computation DAG
       -> algebra-qualified rewrite search
       -> nondominated implementation plans
       -> proof-carrying execution trace
       -> independent exact checker

This is a compiler and optimizer for exact mathematical computations, not another general-purpose SAT solver.

## 2. Prior-art boundary

Equality saturation and rewrite discovery are established fields. Relevant work includes:

- Willsey et al., *egg: Fast and Extensible Equality Saturation*, POPL 2021: https://doi.org/10.1145/3434304
- Nandi et al., *Rewrite Rule Inference Using Equality Saturation*, OOPSLA 2021: https://doi.org/10.1145/3485496
- The existing Symthaea EML e-graph spike is isolated under crates/core/symthaea-eml-egraph. ETH-018 does not reuse its EML rewrite rules as exact factor-algebra rules; the semantic domains and proof obligations differ.
- opt_einsum's dynamic-programming path planner already exploits traces and disconnected network components: https://optimized-einsum.readthedocs.io/en/stable/dp_path.html
- Certified propositional model counting is established prior art: Fichte, Hecher & Roland (SAT 2022), https://doi.org/10.4230/LIPIcs.SAT.2022.30; Bryant et al. (JAIR 2025), https://arxiv.org/abs/2501.12906.

The hypothesis to test is the combined system: output-contract-directed algebra selection, domain-qualified rewrite rules, replayable local proof objects, and plan extraction that includes the cost of independent verification. No novelty claim is justified until this combination is compared carefully with existing rewriting, tensor contraction and proof-carrying computation systems.

The October 29, 2026 weighted-counting paper is another reason to enforce algebra types strictly. Its shared-measurement/Taylor-Hermite computation is stated over a commutative Q-algebra and uses rational linear combinations/subtraction; those operations cannot be silently lifted to MaxCount or ParetoCount. Source: https://hexagonmath.org/2610.00594v1

## 3. Rewrite rule catalogue

The first prototype contains four semiring-valid transformations.

### R1 — extract an independent factor

    sum_x (A(x,y) * B(y,z)) = (sum_x A(x,y)) * B(y,z)

Guard: x must occur in A and must not occur in B. The factor order is preserved. This is a distributivity rewrite.

### R2 — exchange finite summations

    sum_x sum_y F(x,y,z) = sum_y sum_x F(x,y,z)

Guard: x and y are distinct finite bound variables. This uses the commutative finite-addition semantics of the qualified algebras.

### R3 — factor disconnected components

    sum_x sum_y (A(x,z) * B(y,z))
      = (sum_x A(x,z)) * (sum_y B(y,z))

Guards: x is absent from B; y is absent from A. Shared untouched boundary variables z are allowed. The rule is useful for connected-component decomposition and parallel evaluation.

### R4 — exchange multiplicands

    A * B = B * A

Guard: multiplication must be explicitly qualified as commutative. The prototype's closed algebra catalogue contains only commutative examples; a future noncommutative domain must not inherit this rule.

### R5 — additive-inverse cancellation (negative-domain demonstration)

    (-A) + A = 0

Guard: a qualified ring with additive inverses. The rule is rejected for Boolean feasibility, natural counts, MaxCount and ParetoCount. The current prototype tests the ring gate using truncated integer and rational polynomial rings.

### Explicitly excluded: shared-measurement interpolation

The weighted subexponential routine is not a generic semiring rewrite. It requires a commutative Q-algebra and polynomial/rational interpolation machinery. The algebra registry must grant that capability only to a formally justified domain. A user-supplied boolean such as supports_subtraction=true is not proof.

## 4. Proof-carrying rewrite certificate

Prototype format: symthaea.algebraic-rewrite-certificate/v1.

A certificate binds:
- the closed algebra identifier;
- finite variable domains;
- canonical source and target term trees;
- rewrite-rule identifier;
- SHA-256 digests of the canonical source and target encodings.

The independent checker does not trust the producer's selected rule or digest. It:
1. validates the format, algebra identifier and variable domains;
2. parses and validates factor scopes and exact table sizes;
3. rechecks the algebra-specific rule permission;
4. matches the source term against the rule schema;
5. recomputes the required target structure and checks exact equality.

The separate-process checker is implemented in a second file and imports no producer module. The Python prototype is not yet a Lean-refined verifier, and this v1 certificate proves only one local rewrite step. Hashes bind bytes; the rule replay, not the digest, establishes the local equivalence.

## 5. Executed finite reference experiment

Files:
- reference/algebraic_rewrite_oracle.py
- reference/algebraic_rewrite_checker.py
- reference/algebraic_rewrite_results.json

Closed algebra tags:
1. Boolean feasibility.
2. Natural-number counting.
3. MaxCount (optimum and multiplicity).
4. ParetoCount.
5. Truncated integer polynomial ring.
6. Truncated rational polynomial Q-algebra.

Results from deterministic seed 0xE7018:
- Four semiring-valid rewrite families x 24 trials x 6 algebras = 576 positive instances.
- Two additional ring-only cancellation instances.
- A separately implemented checker, executed as a fresh process, accepted 578/578 positive certificates.
- The checker rejected 9/9 negative controls: stale target digest; wrong rule identifier; an independent-factor scope violation; ring-only cancellation relabelled as MaxCount; non-boolean Boolean serialization; duplicate sparse polynomial monomial; unreduced/negative-denominator rational coefficient; duplicate factor scope; and a Pareto encoding containing a dominated earlier point.
- The checker now enforces exact node/certificate shapes, canonical sparse coefficient order, nonzero coefficients, positive multiplicities, reduced rational encodings, canonical scopes, and pairwise Pareto nondominance.
- The polynomial reference generator was corrected to never emit zero sparse coefficients.
- Every positive semantic fixture was exhaustively evaluated over all assignments in its finite domains.

This is meaningful reference evidence, not a proof of the universal laws. The checker and evaluator are Python prototypes; the Rust implementation has not been written or compiled, and there is no hardware performance result.

## 6. Architecture proposed for Rust and Lean

### Typed computation DAG

Start with a small closed language:
- Factor(id, scope, exact table)
- Multiply(left, right)
- Add(left, right)
- SumOut(variable, body)
- Zero and One
- later, explicit polynomial/rational transform and tensor contraction nodes

Every node carries inferred free-variable scope, coefficient-domain identity and canonical content digest. The rule matcher must use inferred scope, not caller-provided metadata.

### Algebra law registry

Represent capabilities as a closed, versioned catalogue linked to Lean theorem identifiers:
- CommutativeSemiring
- CommutativeRing
- CommutativeQAlgebra
- bounded polynomial quotient and coefficient-domain parameters

No arbitrary external plugin can self-assert a law. A domain is eligible only after its implementation and law theorem are qualified.

### Rewrite search

Begin with bounded rule-directed candidate enumeration. Then compare a typed e-graph/equality-saturation implementation against simpler rewrite search. Do not merge e-classes across domains or guards that are not proved.

The existing EML e-graph crate remains isolated. A future adapter may reuse a generic e-graph library, but ETH-018's algebra guards, exact semantics and proof extractor must remain independent of EML-specific rules.

### Multiobjective extraction

Do not optimize only operation count or induced width. Return a Pareto set over:
- semiring additions and multiplications;
- algebra-specific support-pair work and polynomial convolution;
- peak live factor-table entries and peak bytes;
- coefficient bit-height and arbitrary-precision operation cost;
- Pareto frontier size or polynomial support size;
- proof-trace bytes and independent checker work/RSS;
- planning and rewrite-search overhead.

The caller may then choose a plan under a memory cap, solve-time limit or verification budget. Do not hide the trade-off in an unvalidated scalar score.

## 7. How this could outperform current systems

Potential advantages to test—not claims already achieved:

1. **Search beyond orderings.** Current min-fill and weighted min-fill choose an elimination order. ETH-018 also considers decomposition, factor extraction, reassociation, component separation and representation changes.
2. **Optimize verified cost.** A faster producer with a huge or expensive-to-check proof is not necessarily the best implementation.
3. **Exploit output algebra.** MaxCount should not pay for a full polynomial profile; polynomial profiles should not be collapsed to one optimum when the full distribution is requested.
4. **Make transformations reusable.** One certified rewrite catalogue can serve finite CSPs, weighted model counting, exact optimization and compatible tensor contractions.
5. **Parallelize only after proving independence.** Component splitting creates explicit independently evaluable subproblems while preserving exact recombination semantics.

These ideas must be compared with existing knowledge-compilation, tensor-network and optimization systems on matched output contracts.

## 8. Benchmark programme

| Workload | Compatible baselines | Required output |
|---|---|---|
| Feasibility / exact count | CaDiCaL, Ganak, D4, MICE / POG-CPOG where semantics match | Boolean result / exact count |
| Weighted model count | Ganak exact rational track and exact WMI systems | Exact rational value |
| Max-Cut / QUBO | Rehfeldt-Koch-Shinano exact branch-and-cut | Optimum; add multiplicity only where baseline supports it |
| Elimination / factor inference | min-degree, min-fill, weighted min-fill, bounded exhaustive ordering | Same exact algebra result |
| Tensor contraction | opt_einsum, cotengra and compatible exact contraction paths | Same exact contraction and output indices |
| Full profiles / Pareto | direct enumeration, treewidth DP and exact profile implementations | Identical coefficient/frontier semantics |

Freeze public instances and exact output contracts. Split by structural family, not random rows from the same family. Include easy, hard, sparse, dense, bounded-width, high-cardinality and cancellation-heavy cases.

Report per instance:
- producer wall time and CPU;
- rewrite search time and number of candidate plans;
- independent checker time and CPU;
- peak RSS and peak live table entries;
- coefficient bit-height and support size;
- certificate bytes;
- arithmetic operation counts;
- exact output and oracle agreement.

Acceptance targets (aspirational, not achieved):
- 100% exact agreement on supported tractable instances;
- zero accepted invalid or wrong-domain rewrite certificates;
- Lean theorem coverage for every enabled rewrite;
- less than 5% planning overhead on a preregistered medium-instance slice;
- at least 10% lower measured verified end-to-end cost on a held-out structured slice, against the strongest compatible baseline, with memory and proof size reported.

## 9. Next implementation sequence

1. Freeze v1 encoding and expand the mutation corpus.
2. Port the closed algebra registry, typed term DAG and local rewrite checker to Rust.
3. Prove R1–R5 and their guards in Lean; connect algebra law theorems to the domain registry.
4. Add bounded rewrite search and Pareto cost extraction.
5. Integrate selected plans with ETH-017 ordering and ETH-016 elimination certificates.
6. Implement full profile and tensor-network cases only after the core path is qualified.
7. Benchmark against compatible industry systems with exact output and hardware budgets.

## 10. Claim ceiling

At this stage ETH-018 has a finite exact Python reference experiment and a separate-process structural checker. It does not have a Rust engine, Lean proof, general e-graph implementation, arbitrary-precision production backend, weighted subexponential implementation, scalable certificate format, or SOTA performance result.
