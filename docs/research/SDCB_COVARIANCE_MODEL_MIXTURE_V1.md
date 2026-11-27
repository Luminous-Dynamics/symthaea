# SDCB finite Bayesian covariance-model averaging oracle v1

**Status:** bounded exact Gaussian-mixture reference. Ten Rust tests are authored but not compiled or executed in this environment. Frozen numerical values are finite Gaussian calculations, not measured sensor results.

## 1. Why this is a separate capability

#13882's finite-scenario maximin optimizer chooses one common action subset that minimizes the worst-case posterior variance over a declared finite scenario set. That is a robust optimization guarantee only over the supplied set.

It is not Bayesian model averaging. It does not assign posterior probabilities to covariance models, nor compute a mixture posterior mean or between-model variance.

This reference addresses the separate question: after an observation arrives, how should scientific belief change when competing covariance/dynamics models predict it differently?

## 2. Model contract

Each candidate Gaussian model \(s\) supplies:
- prior model probability \(w_s\);
- prior scalar scientific mean and variance \((m_{\theta,s},P_{\theta,s})\);
- predictive observation mean \(\mu_s\);
- full positive-definite innovation covariance \(S_s\);
- cross-covariance \(c_s=\operatorname{Cov}(\theta,y\mid s)\).

For observation \(y\), the innovation is \(\nu_s=y-\mu_s\). The prequential predictive density is

\[
p(y\mid s)=\mathcal N(y;\mu_s,S_s).
\]

The posterior model weight is

\[
w_s'=
\frac{w_s p(y\mid s)}
{\sum_r w_r p(y\mid r)}.
\]

The implementation evaluates the multivariate Gaussian log density using Cholesky factorization and normalizes model weights using log-sum-exp. A model with zero Bayesian prior weight remains at posterior weight zero but may still belong to the separate robust support set.

## 3. Within-model scientific update

For each model, exact Gaussian conditioning gives:

\[
m_{\theta,s}'=
m_{\theta,s}+c_s^\top S_s^{-1}\nu_s,
\]

\[
P_{\theta,s}'=
P_{\theta,s}-c_s^\top S_s^{-1}c_s.
\]

The model's joint scientific/observation moments must be positive semidefinite. Invalid cross-covariance is rejected; no silent repair is applied.

## 4. Mixture total variance

The Bayesian mixture mean is

\[
\bar m_\theta=\sum_s w_s'm_{\theta,s}'.
\]

The total variance has two separate components:

\[
P_{\mathrm{mix}}=
\underbrace{\sum_s w_s'P_{\theta,s}'}_{\text{within-model}}
+
\underbrace{\sum_s w_s'(m_{\theta,s}'-\bar m_\theta)^2}_{\text{between-model}}.
\]

Omitting the second term is a false-certainty failure. The implementation reports both terms and their sum.

## 5. Robust support set stays independent

The caller separately supplies robust support model IDs \(\mathcal S_R\). No probabilities are assigned to this set by the robust calculation, and a low or zero Bayesian posterior weight does not silently remove a model from it.

For the common Bayesian squared-error action \(\hat\theta=\bar m_\theta\), the report calculates:

\[
L_R(\hat\theta)=
\max_{s\in\mathcal S_R}
\left[
P_{\theta,s}'+(m_{\theta,s}'-\hat\theta)^2
\right].
\]

This is the worst component's posterior squared-error risk at the Bayesian mixture decision. It is not a minimax-optimal decision over the robust set. A future robust decision layer may choose a different action by minimizing worst-case loss.

Bayesian model averaging and robust support-set reporting remain two distinct outputs.

## 6. Frozen two-model fixture

Two models have priors \(0.7,0.3\), with one scalar observation \(y=0.5\).

Model A:
- predictive observation \(N(0,1)\);
- prior scientific mean 0 and variance 2;
- scientific/observation covariance 0.8.

Model B:
- predictive observation \(N(1,2)\);
- prior scientific mean 2 and variance 3;
- scientific/observation covariance 1.

Expected results:
- posterior weights: 0.7560919066448245 and 0.2439080933551755;
- posterior scientific means: 0.4 and 1.75;
- within-model mixture variance: 1.6380552264249;
- between-model variance: 0.3360998646772312;
- total mixture variance: 1.9741550911021313.

A within-model-only calculation would understate variance by 0.3360998646772312 in this constructed fixture.

## 7. Negative controls and tests

Ten tests cover:
- predictive-density model-weight updates;
- between-model variance inclusion;
- zero-weight model retained in robust support;
- full multivariate innovation covariance;
- no scientific mean movement for zero cross-covariance;
- log-space normalization under severe likelihood differences;
- invalid predictive covariance and invalid joint moments;
- invalid prior mass and support-set IDs;
- one-model reduction.

A zero Bayesian weight does not mean a model is physically impossible; it means it has zero mass under the declared Bayesian prior. Robust support-set inclusion is independent and must be explicit.

## 8. Limits and next work

This reference assumes each model's predictive Gaussian and joint theta/observation moments are already supplied by a separately validated state-space model. It does not estimate \(A,Q_c,R\), identify physical common causes, or authenticate observation receipts.

Next:
1. feed predictions from the vector state-space oracle;
2. construct scenario/model sets using chronological residuals and check standards;
3. evaluate prior/posterior model-weight calibration on held-out time blocks;
4. compare plug-in, Bayesian mixture and finite-scenario maximin decisions;
5. extend from scalar theta to vector scientific targets only with exact small-instance evaluators.
