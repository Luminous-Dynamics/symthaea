## ETH-018 — dual-count MaxCount algebra

### Contract

For each feasible assignment, return three separate quantities: objective score, weighted MaxCount multiplicity, and one unit of distinct-assignment support. Infeasible assignments are the additive zero.

Value = None | (score, multiplicity, support)
zero = None
one = (0, 1, 1)
add: keep the higher score; sum multiplicity and support on ties
multiply: add scores; multiply multiplicity and support

The enriched value is a commutative semiring over nonnegative count components. Python uses arbitrary-precision integers; Rust uses checked finite-width operations.

### Factor-table semantics

Each original factor-table cell denotes one local assignment. A feasible cell with MaxCount value (w,m) is lifted to (w,m,1); an infeasible cell is zero. After eliminating a variable, an output factor represents a set of partial assignments to the eliminated-variable block conditioned on its separator assignment.

All factors in a bucket summarize pairwise disjoint sets of already-eliminated variables. Therefore support counts multiply across those factors. States of the currently eliminated variable are disjoint alternatives; support counts add across tied maximizing states.

**Invariant:** in every active factor row, support is the number of distinct assignments to exactly the eliminated-variable block summarized by that factor, conditional on its visible scope, which attain the stored score.

The final scalar support is the number of distinct optimal full assignments. It is separate from multiplicity, which weights derivations using local factor multiplicities.

### Necessary boundary

The derivation assumes every input factor cell represents one local assignment and the input API has not pre-aggregated overlapping latent assignment sets. An API that accepts intermediate factors must carry support provenance/represented-variable blocks and prove disjointness at multiplication, or reject this output contract. It must not assume an arbitrary integer weight is a number of assignments.

Rust keeps a parallel u128 support table and checks that weighted and support factors have identical counts, scopes, cell lengths, and zero/nonzero feasibility. Support addition and multiplication are checked; overflow is an error.

### Lean proof obligations

1. Prove the triple algebra's identities, commutativity, associativity and distributivity.
2. Define the original input-factor lift.
3. Define the eliminated-variable block represented by each active factor.
4. Prove bucket blocks are pairwise disjoint and their union is the output block.
5. Prove support multiplication counts Cartesian products over disjoint blocks.
6. Prove support addition sums disjoint maximizing states of the eliminated variable.
7. Induct over elimination order to prove each message's support invariant.
8. Prove final support equals the cardinality of the distinct optimal-assignment set.
9. Separately prove weighted multiplicity equals ordinary MaxCount.
10. Prove the executable Rust implementation refines the formal definitions, including table indexing and alignment guards.

A proof of the triple algebra alone does not prove the input lift or disjoint-block semantics. Those are the algorithm-level obligations.

### Executed reference evidence

- reference/maxcount_dual_count_oracle.py: 20,000 randomized exact algebra-law checks; 512 mixed-cardinality factor graphs with three elimination orders each (1,536 runs); all MaxCount outputs matched direct enumeration and 512/512 distinct-support values matched direct optimum-assignment enumeration.
- Local v2 certificate reference src/maxcount_witness_oracle.py: 240/240 randomized cases, 240/240 direct support matches, 240/240 certificate replay passes, and 11/11 rehashed semantic mutants rejected.
- Handcrafted controls: triangle MaxCut (2,6,6); weighted unique optimum (6,4,1); unequal-weight tie (5,6,2), ordered as (score, multiplicity, distinct assignments).

These are Python exact references. They are not Rust execution, Lean proof or hardware performance evidence.