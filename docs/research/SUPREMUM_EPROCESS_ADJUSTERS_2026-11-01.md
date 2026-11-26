# Admissible supremum adjusters for projective e-closure

**Status:** deterministic implementation of the published adjuster transform; no new statistical theorem is claimed for the transform itself. Its role is to provide an explicit auxiliary path with expected running supremum at most one for the anchored construction.

## 1. Published theorem and admissible adjusters

Tavyrikov, Goeman and de Heide (2026) show that if M_t=max_{s<=t} E_s is the running maximum of an e-process, and A:[1,infinity]->[0,infinity] is increasing/right-continuous, diverges at infinity, and satisfies

    integral_1^infinity A(x)/x^2 dx = 1,

then A(M_t) is again an e-process. The adjuster is nondecreasing in time. Since an e-process has expectation at most one at every deterministic time and the adjusted path is nondecreasing, monotone convergence gives

    E[sup_t A(M_t)] <= 1.

The running maximum itself must **not** be passed directly to e-BH; the paper gives an arbitrary-dependence counterexample. Adjust the running maximum first. Source: https://doi.org/10.1214/26-EJS2546

## 2. A normalized one-parameter family

For beta in (0,1), define

    A_beta(x) = ((1-beta)/beta) * (x^beta - 1).

Then

    integral_1^infinity A_beta(x)/x^2 dx
      = ((1-beta)/beta) * (1/(1-beta) - 1)
      = 1.

The continuous beta->0 limit is A_0(x)=log(x), also normalized. At beta=1/2,

    A_(1/2)(x) = sqrt(x) - 1.

Different beta values change how the adjuster responds to the size of the running maximum. A fixed ex-ante mixture over beta values is valid because convex mixtures of e-processes remain e-processes. The mixture is still nondecreasing, preserving the expected path-supremum bound.

This gives a transparent tuning family. Beta must be fixed before confirmatory data; selecting the largest adjusted path after seeing the same evidence is not valid. Instead, use a fixed prior over beta experts.

## 3. How this plugs into the anchored projective construction

For each intersection S, take a valid raw projective e-process E_S,t from #9695 (arbitrary-dependence affine merger) or a model-specific valid intersection process from #9712/#9758 under its explicit common-filtration and conditional-mgf assumptions.

Form its running maximum

    M_S,t = max_{u<=t} E_S,u

and the adjusted auxiliary

    F_S,t = sum_j pi_j A_beta_j(M_S,t),  sum_j pi_j=1.

Then E_HS[sup_t F_S,t] <=1. This supplies the auxiliary process required by the supremum-anchored local certificate

    E_S,t^anchored = A_S,t^baseline + (1-c_S) F_S,t.

Because the raw projective process has equal historical paths under nested future-universe extensions, and the adjuster/portfolio is a fixed deterministic function, adjusted historical paths remain projective.

Important: the adjuster repairs the **time-supremum** issue for a given local e-process. It does not create the coordinate baseline budgets q_i, prove countable-intersection validity, repair an invalid filtration, or prove that the family of raw intersection processes satisfies dynamic e-closure. Those obligations remain separate.

## 4. Implementation

- validation/supremum_eprocess_adjusters.py implements running maxima, the normalized power family, the beta=0 logarithmic limit, fixed-prior mixtures and per-intersection path transforms.
- validation/test_supremum_eprocess_adjusters.py checks normalization, the square-root special case, monotonicity, fixed-prior reproducibility, key/horizon preservation and invalid-input controls.

The code accepts numerical e-process paths starting at exactly one. It verifies the deterministic transform only; stochastic validity is inherited from the caller's raw e-process theorem and the published admissible-adjuster result.

Tests are authored but not run from a connected repository checkout here. No CI pass is claimed.

## 5. Further research

1. Compare beta values and fixed mixtures on matched stopped-FDR/SupFDR power benchmarks, with disjoint tuning/confirmatory seeds.
2. Derive a pathwise conditional calibration of beta from predictable nuisance/metadata, or use a countable fixed prior if adaptive selection is needed.
3. Optimize adjuster choice under explicit alternative families while preserving the area normalization.
4. Combine adjusted #9695 intersection processes with the #9780 coordinate baseline, then test projective extension and SupFDR on full dynamic closure.
5. Keep the additive e-BH special case as the scalable baseline; measure what power is gained by intersection-specific local evidence.

## 6. Reference

Tavyrikov, Y., Goeman, J. J., & de Heide, R. (2026). Carefree multiple testing with e-processes. Electronic Journal of Statistics, 20(2), 3178–3189. https://doi.org/10.1214/26-EJS2546
