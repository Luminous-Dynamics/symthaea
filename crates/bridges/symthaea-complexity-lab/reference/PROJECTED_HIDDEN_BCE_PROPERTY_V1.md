# ETH-028 — seeded projected-BCE semantic property corpus v1

**Claim ceiling:** seeded finite exhaustive property tests on small formulas. This is not a formal proof, D4 trace replay, or large-scale solver experiment.

## Experiment

- Seed: 0xE7028.
- 2,000 generated CNFs with 3–5 variables.
- A designated clause contains a hidden blocker; each opposing clause containing the complementary blocker is constructed with a complementary literal from the deleted clause, making each resolvent tautological.
- Hidden/visible partitions vary, including additional hidden variables.
- Each case is checked syntactically for complete blocked-clause conditions, then exhaustively evaluated over every full assignment.
- Exact signed-rational visible weights are generated per case; source and target projected support and PWMC are compared.

## Results

- 2,000 / 2,000 positive blocked-hidden-variable cases preserve projected support.
- 2,000 / 2,000 preserve the exact visible weighted value.
- Full-model count changes in 1,420 / 2,000 cases. This is expected and demonstrates that the theorem profile is projected support, not full-model equivalence.
- A nonblocked negative control is rejected as blocked and demonstrates a required support change: source support {1}, after deletion {0,1}.

## Interpretation

This exercises blocker checks and exact projected semantics across multiple small formula shapes and signed rational visible weights. The generated formulas are still small, constructed to satisfy BCE's syntactic precondition, and exhaustively enumerated. This is reference testing, not a proof of the general theorem.

## Reproduction

Run python3 projected_bce_property_microbench.py. The runner writes projected_hidden_bce_property_results.v1.json.

## Next

Extend to dynamic context traces by generating a branch decision tree, conditioning formula at each node, verifying branch-local BCE receipts and checking backtrack restoration. Then differentially replay event sequences against an unchanged D4 BCEi build.


## Exact uploaded-script execution

The uploaded property runner was fetched and reconstructed byte-for-byte; its Git blob SHA-1 matched before execution.

- Script Git blob SHA-1: f13a13fcb31bfd3aef993d1204e3b50424118271
- Script SHA-256: 21d61838130b10e70f68a3638186c78bc118986dd1b0258d554522daa2bc67d8
- Runtime: Python 3.13.5, Linux.
- Result: 2,000/2,000 positive cases passed; 1,420 changed full-model count; the nonblocked counterexample correctly changed projected support from {1} to {0,1}.

The result manifest records the exact script identities and uploaded-script execution. The finite small-formula claim ceiling is unchanged.
