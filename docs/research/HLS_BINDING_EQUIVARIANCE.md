# HLS Binding Equivariance: Construction and Linear Mixing Boundary

Status: research theorem / implementation contract

## 1. Role group

Let the real unitary HDC role group be

\[
G = \{D_r = \operatorname{diag}(r_1,\ldots,r_D) : r_i \in \{-1,+1\}\}.
\]

Binding a state `h` by role `r` is

\[
B_r(h) = D_r h = r \odot h.
\]

Every `D_r` is orthogonal, self-inverse, and diagonal. Therefore it preserves L2 norm, inner products, and cosine similarity.

## 2. Target temporal symmetry

For an HLS transition `F`, the strongest same-role binding theorem is

\[
F(B_r(h), B_r(x), \Delta t) = B_r(F(h,x,\Delta t))
\]

for every role `r`, state `h`, input `x`, and valid elapsed time `dt`.

The theorem-bearing `HolographicLiquidCell` realizes this construction by making gate and tau controls depend only on role-invariant magnitudes while restricting the signed equilibrium path to odd activations.

## 3. Linear no-go theorem

**Theorem.** If a linear operator `A : R^D -> R^D` commutes with every bipolar role binding,

\[
A D_r = D_r A \quad \forall D_r \in G,
\]

then `A` is diagonal.

**Proof.** Consider any off-diagonal entry `A_ij` with `i != j`. Choose a role whose signs satisfy `r_i = +1` and `r_j = -1`. Then

\[
(A D_r)_{ij} = A_{ij}r_j = -A_{ij}
\]

while

\[
(D_r A)_{ij} = r_i A_{ij} = A_{ij}.
\]

Commutation requires `-A_ij = A_ij`, hence `A_ij = 0`. Because this holds for every `i != j`, all off-diagonal entries vanish. Conversely, any diagonal `A` commutes with every diagonal `D_r`. QED.

## 4. Consequence for HLS mixing

A naive cross-dimensional linear mixer such as

\[
M(h) = \sum_k W_k \odot \rho_k(h)
\]

cannot, in general, preserve the full same-role theorem. Under role binding,

\[
\rho_k(r \odot h) = \rho_k(r) \odot \rho_k(h),
\]

and generally `rho_k(r) != r`.

Therefore exact equivariance to the full independent-sign role group and arbitrary non-diagonal linear mixing cannot both be claimed.

This is a design boundary, not a failure of HDC. There are at least three principled options:

1. weaken/restrict the role group;
2. weaken the equivariance claim to a covariant transformation law;
3. retain full role equivariance and introduce cross-dimensional interaction only through role-invariant nonlinear context.

The current HLS research line chooses option 3.

## 5. Invariant-context construction

Define a cross-dimensional magnitude context

\[
c_i(h,x) = \frac{1}{K}\sum_{k=1}^{K} w_{k,i}
\frac{|h_{i+o_k}| + |x_{i+o_k}|}{2},
\]

with cyclic indices and fixed offsets `o_k`.

Because bipolar binding changes only signs,

\[
|D_r h| = |h|, \qquad |D_r x| = |x|,
\]

so

\[
C(D_r h, D_r x) = C(h,x).
\]

The context can therefore modulate a signed local channel, gate, or liquid time constant without itself breaking role equivariance. Computing K offset contributions for D output coordinates costs O(KD).

## 6. What this does and does not establish

Established by construction and executable tests:

- `UnitaryRole` binding is an isometry and is self-inverse;
- the theorem-bearing HLS cell is designed for same-role binding equivariance;
- the invariant context mixer is exactly sign-role invariant;
- cross-dimensional magnitude information can influence a local context coordinate.

Not established yet:

- that invariant-context mixing improves predictive or reasoning performance;
- that the architecture beats dense RNNs, CfC, SSM, Mamba-family, or Transformer baselines;
- that full permutation equivariance holds;
- that unitary role binding is the optimal HDC algebra for every task;
- that the resulting architecture is sufficient for general reasoning.

These remain experimental questions.

## 7. Next falsifiable experiment

Integrate `InvariantContextMixer` into `HolographicLiquidCell` as an optional gain/gating channel and compare:

1. diagonal HLS,
2. HLS + invariant context,
3. unconstrained signed-permutation mixer,
4. conventional dense recurrent mixing.

Measure both task accuracy and symmetry error. This creates a direct Pareto experiment between expressive mixing and algebra preservation instead of assuming that either objective dominates.
