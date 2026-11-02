# symthaea-complexity-lab

A dependency-free, exact-arithmetic research crate for the ETH-001 / ETH-002 programme.

## Scope of this first slice

The crate implements the tensor-product binomial/Newton transform on the finite grid
\(\{0,\ldots,d\}^n\), using checked i128 arithmetic:

- forward differences at the origin;
- inverse Newton reconstruction;
- checked base-(d+1) indexing;
- checked binomial coefficients;
- explicit overflow, dimension, and table-length errors;
- deterministic unit tests including degenerate dimensions and overflow cases.

The separate design module also verifies exact rational row-space witnesses for
integer measurement matrices. It checks
lambda^T C = target by checked cross multiplication, rejects malformed
dimensions/non-positive denominators, and fails closed on i128 overflow.
It verifies supplied witnesses only; it does not search for a design.

The separate matrix module verifies exact rational factorizations
`I - M0 = U * V`, giving a certificate for
`rank(I - M0) <= U.columns`.

The permutation-polynomial module independently reconstructs a rational
`M0` from a supplied polynomial in explicitly listed commuting permutation
generators. It verifies bijectivity, pairwise commutativity, exponent shape,
common-denominator construction, coefficient scaling, and exact matrix
equality. Exponents are reduced modulo point-cycle length, so huge exponents
do not cause unbounded repeated application.

These are supplied-certificate verifiers only. They do not search for a
suitable M0, prove the paper's generic matrix-existence theorem, or establish
the separate support/sparsity and degree properties.

The Python independent oracle checked 2,000 deterministic transform tables
(49,800 coefficient comparisons and 49,800 reconstructed entries), 2,006
row-space certificate cases (1,000 valid witnesses, 1,000 one-coordinate
mutations, and six malformed/overflow controls), 2,004 rational
`I - M0 = U * V` cases (1,000 valid factorizations, 1,000 mutated matrices
rejected, and four shape/overflow controls), and 2,004 polynomial-in-commuting-
permutations cases (1,000 valid generated polynomials, 1,000 mutated M0
claims rejected, and four structural/composition controls). These are finite
checks only.

It is a finite-table reference implementation. Its table size is \((d+1)^n\),
and the transform takes \(O(n(d+1)^{n+1})\) arithmetic operations. It is
**not** a subexponential counting algorithm and does not implement the
OpenAI construction.

## Run

From the repository root:

    cargo test -p symthaea-complexity-lab

This package has no third-party dependencies and is kept as a separate bridge
crate so the experiment does not pull in the cognitive core.

## Arithmetic boundary

The implementation uses i128 so every operation can report overflow rather
than silently wrap. It is only a bounded exact oracle. Large model counts need
an arbitrary-precision backend; the future algorithm must account for bit
complexity and cannot treat an arithmetic-operation bound as a Turing-time bound.

## Finite-domain scope correction

For fixed alphabet size q and fixed extensional constraint arity k, a q-ary CSP
already has a constant-overhead parsimonious encoding into bounded-width
Boolean #SAT: encode each domain value with ceil(log2(q)) bits, exclude unused
codewords, and forbid each invalid local tuple. Therefore this crate does not
claim a new q-ary subexponential theorem. It provides a direct exact transform
and a certificate primitive whose representation overhead can be measured;
stronger asymptotic claims require a parameter regime not already covered by
that Boolean reduction.

For d=1 the transform is Boolean subset Möbius inversion. For d>1, the precise
name is the tensor product of one-dimensional binomial/Newton transforms,
not the ordinary Möbius function of a finite-chain poset.

## Claim boundary

- the q-ary binomial/Newton inversion identity is not a finite-domain subexponential algorithm;
- passing finite tests is not a formal proof;
- i128 reference arithmetic is not an arbitrary-precision implementation;
- exact interpolation does not guarantee low degree or low coefficient growth
  after a replacement-form construction.

The formal theorem target remains ETH-002. Proof-carrying design search and the
separate verifier are tracked in ETH-003:
https://github.com/Luminous-Dynamics/symthaea/issues/10089

The paper's sharper quantitative measurement-design / Taylor-recovery proof is tracked in ETH-005:
https://github.com/Luminous-Dynamics/symthaea/issues/10095

The weighted/algebraic exact arithmetic qualification is tracked in ETH-006:
https://github.com/Luminous-Dynamics/symthaea/issues/10096

Dependency-aware reuse of exact lemmas and certificates is tracked in ETH-008:
https://github.com/Luminous-Dynamics/symthaea/issues/10113

The broader work plan is tracked in ETH-001.
