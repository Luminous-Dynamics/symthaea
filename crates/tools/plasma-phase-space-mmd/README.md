# Plasma phase-space characteristic MMD oracle

A standalone, dependency-free Rust reference implementation of a characteristic-kernel maximum mean discrepancy (MMD) for finite weighted empirical measures on the 1D1V cylinder \(S^1_L \times \mathbb{R}\).

## Kernel

For periodic position and unbounded velocity,

\[
k((x,v),(x',v')) =
\exp\!\left[-2\kappa\sin^2\!\left(\frac{\pi(x-x')}{L}\right)\right]
\exp\!\left[-\frac{(v-v')^2}{2\ell_v^2}\right],
\]

with \(L>0\), \(\kappa>0\), and \(\ell_v>0\).

The position factor is the von Mises kernel. Its Fourier series has strictly positive coefficients for every integer spatial harmonic. The Gaussian velocity factor has a strictly positive Fourier transform over the full real frequency axis. Their product therefore has full spectral support on the dual group \(\mathbb{Z}\times\mathbb{R}\), making the product kernel characteristic on the declared cylinder for finite signed Borel measures. The equivalent small-separation spatial length scale is \(\ell_x=L/(2\pi\sqrt{\kappa})\).

## Comparison

For \(\mu=\sum_i a_i\delta_{z_i}\) and \(\nu=\sum_j b_j\delta_{z_j}\), the oracle evaluates

\[
\operatorname{MMD}_k^2(\mu,\nu) =
\sum_{i,i'} a_i a_{i'} k(z_i,z_{i'})
+\sum_{j,j'} b_j b_{j'} k(z_j,z_{j'})
-2\sum_{i,j} a_i b_j k(z_i,z_j).
\]

Before computing MMD, co-located periodic phase-space coordinates are canonicalized and their signed coefficients are aggregated. Total variation is the sum of absolute aggregated atom coefficients, not the sum of absolute raw particle weights. This preserves normalization under signed split/merge representations, including cancellation of opposite signs at the same support point. Weights are divided by the larger total variation of the two represented measures before pair accumulation. The zero-measure convention uses scale one.

The unnormalized MMD is the RKHS metric for this characteristic kernel. The reported ratio is a pair-normalized discrepancy; because its denominator depends on both inputs, we do not claim that the normalized ratio itself satisfies the triangle inequality.\n\nThe implementation uses compensated summation and checks that the squared discrepancy is nonnegative within a conservative roundoff tolerance. Materially negative values fail closed. It reports kernel pair-evaluation count, unique support atom count and raw particle counts as metadata.

## Scope and limits

- Exact kernel / direct empirical sums; \(O(N^2+M^2+NM)\) worst-case time and \(O(N+M)\) temporary memory.
- Designed as an exact-small-N assurance oracle and reference for future \(O(ND)\) random-feature or spectral approximations.
- Not Wasserstein distance, total-variation distance, a statistical p-value, field error, a plasma solver, or a physics certificate.
- Characteristicness is an injectivity statement; it does not guarantee finite-precision sensitivity at every bandwidth or on every finite sample.
- This package covers only one periodic spatial coordinate and one unbounded velocity coordinate.

## Tests

The unit tests cover permutation and split invariance, periodic representatives, a finite-Fourier alias resolved by the characteristic oracle, position/velocity correlation changes, signed measures including co-located cancellation, mass rescaling, empty and zero-weight measures, invalid inputs, overflow, and pair-evaluation metadata.

The focused GitHub Actions workflow runs Rust 1.96 format, test and strict Clippy checks at the exact pull-request revision. The source being authored does not imply that those checks have passed.
