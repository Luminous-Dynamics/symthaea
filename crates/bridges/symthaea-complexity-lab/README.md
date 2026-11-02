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

The Python independent oracle checked 2,000 deterministic transform tables
(49,800 coefficient comparisons and 49,800 reconstructed entries) and 2,006
row-space certificate cases (1,000 valid witnesses, 1,000 one-coordinate
mutations, and six malformed/overflow controls). These are finite checks only.

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

## Claim boundary

- the q-ary binomial/Newton inversion identity is not a finite-domain subexponential algorithm;
- passing finite tests is not a formal proof;
- i128 reference arithmetic is not an arbitrary-precision implementation;
- exact interpolation does not guarantee low degree or low coefficient growth
  after a replacement-form construction.

The formal theorem target remains ETH-002. Proof-carrying design search and the
separate verifier are tracked in ETH-003:
https://github.com/Luminous-Dynamics/symthaea/issues/10089

The paper's sharper recursive tensor-contraction proof is tracked in ETH-005:
https://github.com/Luminous-Dynamics/symthaea/issues/10095

The weighted/algebraic exact arithmetic qualification is tracked in ETH-006:
https://github.com/Luminous-Dynamics/symthaea/issues/10096

The broader work plan is tracked in ETH-001.
