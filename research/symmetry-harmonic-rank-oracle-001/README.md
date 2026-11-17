# Exact SO(3) cyclic / rotational-dihedral rank oracle v0.1

This crate implements [SYMM-HARMONIC-RANK-ORACLE-001 #12169](https://github.com/Luminous-Dynamics/symthaea/issues/12169), a bounded representation-theoretic companion to [SYMM-SPECTRAL-001 #11634](https://github.com/Luminous-Dynamics/symthaea/issues/11634).

It is **not** a reproduction of the SymFP paper. It does not construct an equivariant neural representation block, measure singular values from data, compute SymFP leakage, or infer an observed physical symmetry. It generates exact rank templates for a clearly delimited group family.

## Exact formulas

For the degree-l irreducible representation V_l of SO(3), with dimension 2l+1:

- Proper cyclic rotations C_n about a declared axis fix weights m divisible by n:

  `dim(V_l^C_n) = 1 + 2 floor(l/n)`.

- The proper rotational dihedral group D_n has order 2n and consists of those n axial rotations plus n perpendicular half-turns. The character average gives:

  `dim(V_l^D_n) = (1 + 2 floor(l/n) + (-1)^l) / 2`.

  An independent weight-pair count gives one invariant combination for each positive multiple of n not exceeding l, plus the m=0 mode exactly when l is even.

All rank construction uses integer arithmetic. There are no floating-point cosines, SVDs, rank tolerances, or numerical singular-value thresholds in this oracle.

## Group-family conventions

- C_n is the order-n proper rotation subgroup about one fixed axis.
- D_n is the proper rotational dihedral group of order 2n, with the same canonical axis and a fixed perpendicular half-turn generator.
- The implementation checks the co-axial subgroup relations C_d <= C_n, C_d <= D_n, and D_d <= D_n when d divides n.
- D_n is not D_nh, a planar reflection group, or an arbitrary embedding in SO(3).
- Other orientation embeddings, polyhedral groups, products, and disconnected mixtures are outside v0.1.

## Ambiguity is a valid result

Given a truncated degree-resolved signature, `match_signature` returns the *complete set* of C_n and D_n hypotheses that match. It reports NO_MATCH, UNIQUE, or AMBIGUOUS. It never forces a label when signatures coincide.

The proposal receipt reports the first degree at which any two candidates in the declared candidate family differ, if that divergence exists within the selected cutoff. It also has a proposal-only authority type; no action, physical-equivalence, or pruning capability is exposed.

Example:

    cargo run --locked --manifest-path research/symmetry-harmonic-rank-oracle-001/Cargo.toml --example template_ambiguity

The degree-2 C3 template `[1,1,1]` is also consistent with C4, C5 and C6; degree 3 is the first distinguishing degree within that cyclic family. A separate test demonstrates the distinct degree-2 D3 template `[1,0,1]`, which is ambiguous across D3, D4, D5 and D6. The example prints the cyclic ambiguity and its cutoff resolution rather than hiding it.

## Reproduction

    cargo fmt --manifest-path research/symmetry-harmonic-rank-oracle-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/symmetry-harmonic-rank-oracle-001/Cargo.toml
    cargo clippy --locked --manifest-path research/symmetry-harmonic-rank-oracle-001/Cargo.toml --all-targets -- -D warnings

## Frozen correctness corpus

The tests cover all parameters n<=64 and degrees l<=32 for:

- cyclic closed-form dimensions versus explicit weight enumeration;
- dihedral character-average dimensions versus independent weight-pair counting;
- multiplicity bounds 0<=dim(V_l^G)<=2l+1;
- degree-wise subgroup monotonicity for all divisor pairs in the declared canonical family;
- intentionally ambiguous truncated signatures, including both cyclic and dihedral matches;
- cutoff extension and first-divergence reporting;
- exact deterministic proposal-byte replay;
- zero/invalid order, empty signature/family, and work-budget rejection.

The 64-by-32 fixture grid is a prospective exact-oracle acceptance target. Authored tests must still pass the dedicated exact-head workflow before being described as qualified.

## Claim boundary and integration

This crate supplies exact *candidate templates* to a spectral or group-space proposal lane. It does not show that a measured equivariant block has those ranks, does not prove a representation is physically realized, and does not qualify a group action.

Every candidate still needs independent finite-group and matrix-action qualification; relevant continuous components need separate Lie-algebra/connected-group evidence; Symtropy must independently qualify physical step-map, constraint, momentum and observable behavior; and the complete search task must be invariant before hard quotienting under [Symthaea #11616](https://github.com/Luminous-Dynamics/symthaea/issues/11616).

No hard pruning is authorized by a rank template or spectral score.
