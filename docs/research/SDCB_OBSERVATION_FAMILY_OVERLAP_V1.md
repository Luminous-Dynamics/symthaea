# SDCB observation-family overlap v1

**Status:** analytic synthetic reference with exact rational pole-swap witness. The Rust reference and harness are authored but not compiled or executed in this environment. This is not a physical sensor uncertainty result.

Machine-readable companion: docs/research/SDCB_OBSERVATION_FAMILY_OVERLAP_V1.json.

## 1. Why fixed-pair power is not enough

The fixed-pair covariance LRT in #14110 answers a conditional question:

> If the two fully specified covariance laws are exactly the nominal World A and nominal World B, what are the finite-sample size and power of the optimal simple-hypothesis test?

That is valuable, but it does not answer what happens when the candidate mechanisms have uncertain nuisance parameters.

A scientific claim often concerns two composite families:

\[
\mathcal P_A=\{P_{A,\eta}:\eta\in\Theta_A\},\qquad
\mathcal P_B=\{P_{B,\xi}:\xi\in\Theta_B\}.
\]

Before computing a robust power envelope, SDCB must ask whether these sets of complete observation distributions intersect.

## 2. An exact overlap hidden inside a one-percent uncertainty interval

The nominal #14065 World A is:

- physical decay \(\lambda_A=0.25\);
- sensor response rate \(\mu_A=4\);
- physical stationary variance \(\sigma^2_{x,A}=9\).

Its zero-lag output variance parameter is

\[
q_A=\frac{\mu_A\sigma^2_{x,A}}{\lambda_A+\mu_A}
=\frac{4\cdot9}{0.25+4}
=\frac{144}{17}.
\]

The near-swap nominal World B is:

- \(\lambda_B=4.04\);
- \(\mu_B=0.25\);
- process variance selected to preserve \(q_A\), namely \(61776/425\), approximately \(145.35529411764706\).

Now define a synthetic uncertainty interval around World B's physical decay:

\[
\lambda_B\in[3.9996,4.0804],
\qquad \mu_B=0.25,
\]

and preserve the same q by setting

\[
\sigma^2_{x,B}(\lambda_B)
=q_A\frac{\lambda_B+\mu_B}{\mu_B}.
\]

This interval contains \(\lambda_B=4\). At that exact point,

\[
\lambda_B=4=\mu_A,\qquad
\mu_B=0.25=\lambda_A,\qquad
\sigma^2_{x,B}=144.
\]

That is the exact #14002 pole-swap world.

## 3. Why the entire passive observation laws are identical

For the stationary OU process followed by a first-order low-pass instrument, the covariance kernel is

\[
C(h)=q e^{-rh}\left(1+rh\,\psi(gh)\right),
\]

where

\[
r=\min(\lambda,\mu),\quad
g=|\lambda-\mu|,\quad
\psi(x)=\frac{1-e^{-x}}{x},\quad \psi(0)=1.
\]

The kernel depends on the two rates only through their minimum and absolute difference, plus q. Both are invariant under swapping \(\lambda\) and \(\mu\).

At the uncertainty witness, both worlds have the same q and swapped rates. Therefore

\[
C_A(h)=C_B(h)\quad\text{for every }h\ge0.
\]

Both output processes are zero-mean stationary Gaussian processes. Equality of the complete covariance kernel therefore gives equality of their complete finite-dimensional distributions for every common passive sampling design.

Adding the same readout-noise covariance \(R_D\) to each design preserves the equality:

\[
\Sigma_A(D)=K_A(D)+R_D
=K_B(D)+R_D=\Sigma_B(D).
\]

This is an exact family intersection—not merely a small KL value at one sample count.

## 4. Consequence for every uniformly valid passive test

Let \(\phi(Y)\in[0,1]\) be any possibly randomized test that rejects the composite null family B. Uniform size control requires

\[
\sup_{Q\in\mathcal P_B}\mathbb E_Q[\phi(Y)]\le\alpha.
\]

If a distribution \(P^*\) belongs to both families, then

\[
\inf_{P\in\mathcal P_A}\mathbb E_P[\phi(Y)]
\le \mathbb E_{P^*}[\phi(Y)]
\le \alpha.
\]

So no uniformly size-controlled test can guarantee worst-case power above \(\alpha\) over the full composite alternative family.

This remains true with arbitrarily many passive observations. It is not an estimator weakness or a failure of the LRT implementation. It is a property of the declared observation families.

The implication is important: **robust practical power must be preceded by an observation-family overlap check.** Otherwise, a nominal point estimate can make a model look distinguishable even though the declared uncertainty set contains an exact indistinguishable case.

## 5. An intervention changes the observation protocol

The sensor-inlet unit-step response is

\[
s(t)=1-e^{-\mu t}.
\]

At \(t=0.5\), for the two response rates:

- World A, \(\mu=4\): \(s_A(0.5)=0.8646647167633873\).
- World B, \(\mu=0.25\): \(s_B(0.5)=0.1175030974154046\).

The difference is approximately 0.74716. This is a strong synthetic positive control because the intervention directly probes the instrument response rather than relying only on the ambiguous passive output.

It is not yet a physical measurement. Real stimulus amplitude uncertainty, inlet effects, calibration ancestry, readout noise and intervention safety must be declared and included in the observation contract.

## 6. What the Rust reference does

scripts/research/sdcb_family_overlap.rs is a narrow dependency-free algebraic witness checker.

It uses normalized positive rational numbers and checked i128 arithmetic for:

- exact equality of the swapped poles;
- exact equality of q;
- the q-preserving process-variance calculation;
- membership of the rational alias point inside the nuisance interval;
- distinction between the nominal near-swap and the exact alias.

The identity decision is not made with a floating-point tolerance. Floating point is used only for the separate unit-step response calculation.

The reference proves a sufficient condition for this OU/first-order-instrument symmetry. It is not a general symbolic structural-identifiability engine and it does not prove that arbitrary model families are disjoint when this particular witness is absent.

Run locally with scripts/research/run_sdcb_family_overlap_tests.sh.

The current environment has no rustc or Cargo, so tests are authored but not compiled or run here.

## 7. Required SDCB result states

- FAMILY_INTERSECTION: exact full-observation-law witness found.
- FAMILY_DISJOINT_WITHIN_DECLARED_SET: only with a valid global argument or an exhaustive finite-set contract. A sampled parameter grid is not a continuous proof.
- OVERLAP_UNRESOLVED: numerical, symbolic or nuisance-bound uncertainty prevents a defensible conclusion.

Every witness should bind both model identities, both parameter settings, units, full observation-law identity, proof method and code/fixture digest.

Finite-design KL, LRT power and interval sensitivity remain valuable secondary metrics. They must not replace family-level equivalence.

## 8. Paradigm shift and benchmark implications

SDCB should become **set-aware before it becomes test-aware**:

1. Carry bounded nuisance sets, calibration uncertainty and common-cause/reference states into the declared model family.
2. Search for exact or approximate observation-law intersections before declaring robust identifiability.
3. When exact intersection exists, identify the additional measurement operator or intervention required to split the family.
4. When families are provably disjoint, only then evaluate robust size/power over the complete nuisance envelope.
5. When proof is unavailable, abstain from a family-wide uniqueness claim.

Proposed benchmark gates:

| Track | Target |
|---|---|
| Exact family overlap | 100% detection on algebraic known-answer witnesses |
| False disjointness | Zero false disjoint-family claims on sealed overlap cases |
| Nominal-vs-family consistency | Never promote a nominal point result to a robust family result without an overlap audit |
| Robust power | Report worst-case power over the full declared nuisance set; target at least 80% for material alternatives or label UNDERPOWERED |
| Intervention value | Identify the lowest-cost permitted protocol that breaks an exact equivalence class |
| Decision sufficiency | Do not acquire a discriminating measurement when all compatible worlds imply the same terminal action |
| Evidence integrity | An overlap witness is a statistical result, not a probe authorization or evidence receipt |

The ±1% and ±0.5% ranges here are synthetic challenge intervals only. The narrower interval excludes this particular lambda=4 witness, but that does not prove all other possible observation-law intersections are absent.

## 9. Integration and next steps

Compose with the current owners; do not build a parallel planner or evidence ledger.

| Concern | Owner |
|---|---|
| Exact pole-swap identity | #14002 |
| Fixed-pair covariance KL and Pinsker | #14065 |
| Fixed-pair finite-sample LRT | #14110 |
| Uncertain parameter-family overlap | #14133 |
| Open-world structural/measurement omission | #11197 / #11262 |
| Adaptive falsification power | #12375 |
| Common-cause and reference-chain context | #13805 / #13951 |
| Decision sufficiency and exact policy value | #14038 / #13567 |
| Selector tournament | #13739 |

Next research increment after this algebraic witness: construct a finite discrete nuisance-family tournament with uniform null size calibration, explicit least-favourable cases, and held-out continuous-parameter stress tests. Only after that should we claim robust composite-hypothesis power. Unknown calibration/reference distributions must be composed from #13805/#13951 rather than encoded as unexplained diagonal noise.

## 10. Qualification and claim boundary

- The exact pole-swap identity is derived analytically.
- Independent rational arithmetic confirms \(q_A=q_B=144/17\), \(\sigma^2_{x,B}=144\) at \(\lambda_B=4\), and the ±1% interval contains the witness.
- Independent floating-point evaluation gives identical covariance values at representative lags from 0 through 100 time units.
- The Rust reference and tests are authored, not compiled or executed.
- Hosted CI, physical sensor uncertainty, metrological traceability, QUANT analysis and SOTA superiority are not claimed.

This establishes a synthetic exact-overlap result for one declared model family. It does not establish that real instruments have this uncertainty range or that the proposed intervention is physically qualified.

