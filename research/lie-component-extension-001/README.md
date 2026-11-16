# Lie / finite-component extension reference v0.1

This is the executable first slice of [SYMM-LIE-COMPONENT-001 #11799](https://github.com/Luminous-Dynamics/symthaea/issues/11799).

## The missing structural distinction

The Lie algebra describes the identity-connected component of a Lie group. It does not, by itself, describe disconnected components. A full Lie group with finitely many components fits into

\[
1 \longrightarrow G^0 \longrightarrow G \longrightarrow \pi_0(G) \longrightarrow 1.
\]

This reference covers only **split extensions** \(G \cong G^0 \rtimes H\), where \(H\) is a finite component group. It never assumes that an arbitrary extension splits.

## What is checked

Given a bounded matrix representation of a Lie-algebra generator set, exact integer structure constants, a finite component-group table, the state-space representation \(\rho_H\), and its induced action \(\mathrm{Ad}_H\) on the Lie algebra, the verifier checks:

1. The finite group table: Latin property, two-sided identity, inverses and associativity.
2. The Lie algebra: exact matrix commutator closure and the Jacobi identity.
3. The finite state representation: identity and group composition.
4. The finite adjoint representation: identity and group composition.
5. Each adjoint map preserves the Lie bracket.
6. Representation compatibility:

\[
\rho_H(h) A_i \rho_H(h)^{-1}
=
\sum_j \mathrm{Ad}_H(h)_{ji} A_j.
\]

The checker uses exact bounded integer arithmetic for v0.1; arithmetic overflow fails closed.

## Exact fixtures

- **O(2):** \(G^0 = SO(2)\), \(H=C_2\), \(J=\begin{bmatrix}0&-1\\1&0\end{bmatrix}\), reflection \(R=\mathrm{diag}(1,-1)\), and \(RJR^{-1}=-J\).
- **Direct-product control:** the component acts as \(-I\) on the state space and trivially on the one-dimensional Lie algebra. This is a different valid extension representation.
- **Affine Lie-algebra fixture:** \([A_1,A_2]=A_2\), with a discrete component acting by \(A_1\mapsto A_1,\ A_2\mapsto -A_2\).
- **Negative controls:** wrong adjoint sign, non-involutive state representation, non-homomorphic adjoint map, bracket-nonpreserving map, and a Latin but non-associative multiplication table.

## Reproduction

    cargo fmt --manifest-path research/lie-component-extension-001/Cargo.toml --all -- --check
    cargo test --locked --manifest-path research/lie-component-extension-001/Cargo.toml
    cargo clippy --locked --manifest-path research/lie-component-extension-001/Cargo.toml --all-targets -- -D warnings

## Claim ceiling

v0.1 verifies exact structural compatibility for bounded integer matrix representations of split finite-component extensions. It does not evaluate global exponentials, certify non-split extensions or extension cocycles, handle infinite component groups, prove physical step-map equivariance, or authorize task pruning. Approximate floating-point Lie bases and learned symmetry proposals need a separate numerical/uncertainty contract.
