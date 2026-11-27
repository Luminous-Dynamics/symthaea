# SDCB finite-scenario robust full-covariance design v1

**Status:** exact bounded maximin reference; five additional Rust tests authored in the full-covariance module, not compiled or executed here.  
**Implementation:** \`scripts/research/sdcb_full_covariance_gaussian.rs\` (\`optimize_maximin\`)  
**Fixture:** \`SDCB_ROBUST_COVARIANCE_V1.example.json\`  
**Expected result:** \`SDCB_ROBUST_COVARIANCE_V1.expected.json\`

## 1. Motivation

The full-covariance Gaussian oracle (#13881) assumes one known prior covariance and one known residual covariance. Real covariance estimates are uncertain, may vary by operating regime, and can be misspecified.

A selector that chooses a different measurement subset for each possible covariance model is not a deployable robust policy unless it can know which model is true before measurement. The robust reference therefore chooses **one shared action subset** and evaluates that same subset under every declared covariance scenario.

## 2. Finite-scenario maximin objective

Let \(s\in\mathcal S\) index the finite set of plausible covariance scenarios. Each scenario supplies prior covariance \(\Sigma_{0,s}\) and candidate residual covariance \(R_s\). For a common action subset \(A\), the target posterior variance in scenario \(s\) is:

\[
v_s(A)=
\left[
\Sigma_{0,s}^{-1}
+H_A^\top R_{s,A}^{-1}H_A
\right]_{jj}^{-1}
\]

where the bracket notation denotes the \(j,j\) element of the full posterior covariance matrix.

The robust selection objective is:

\[
A^*=\arg\min_{A:\sum_{a\in A}c_a\le B}
\max_{s\in\mathcal S}v_s(A).
\]

The exact reference enumerates all subsets (the same 12-action cap as the full-covariance oracle) and applies deterministic tie-breaking by lower cost, then sorted action IDs.

This is **finite-scenario maximin**. It does not guarantee robustness to covariance models outside the declared set, nor does it construct a statistically valid confidence region for the true covariance.

## 3. Frozen two-scenario fixture

Two scenarios share the same latent priors and measurement loadings. They differ only in which science/reference residual pair has a 0.003 covariance:
- scenario A: science-A residual is correlated with the reference;
- scenario B: science-B residual is correlated with the reference.

All other residual variances are 0.01. Each action costs one unit; budget is two.

A single shared subset, science-A plus reference, gives:
- scenario A target variance: 0.013759519417707996;
- scenario B target variance: 0.019512668672944374;
- worst-case target variance: 0.019512668672944374.

Science-B plus reference has the same worst-case value; lexical tie-breaking selects science-A. Two science channels without the reference have worst-case target variance approximately 0.50124688.

This is a constructed analytical example, not an empirically estimated covariance envelope or a physical sensor result.

## 4. What the guarantee means

The report preserves every scenario's posterior target variance, along with the maximum. Users can inspect which scenario is binding and whether a small change in the scenario set changes the selected policy.

The result supports the claim:
> Within the supplied finite covariance scenario set and declared linear-Gaussian model, the selected subset minimizes worst-case posterior variance among all budget-feasible subsets.

It does **not** support:
- worst-case guarantees over all possible covariance matrices;
- correctness of the scenario set or covariance estimates;
- correct attribution of residual correlation to a physical common cause;
- physical traceability or calibration validity;
- robust performance under unmodelled non-Gaussian or time-varying effects.

## 5. Integration path

1. Generate candidate covariance scenarios from independently documented uncertainty estimates, regime differences, historical residuals, or sensitivity analysis.
2. Preserve each scenario's covariance matrix and provenance in the experiment plan.
3. Select one common action subset using finite-scenario maximin.
4. Compare it with plug-in nominal design, Bayesian model averaging, distributionally robust methods, and worst-case bounds from a justified covariance uncertainty set.
5. After observing measurements, update the covariance/scenario weights only through a separately validated inference path.
6. Bind physical evidence to receipt, reference ancestry, capture time and calibration context before changing EGBT state.

For larger or continuous covariance uncertainty, replace the finite scenario set only after validating the uncertainty-set construction and optimization gap. The finite exact oracle remains the small-instance correctness reference.

## 6. Evidence status

The five additional Rust tests are authored but not compiled or executed in this environment. The two-scenario fixture is independently enumerated using a separate matrix calculation. No real covariance uncertainty set was estimated from sensor data, and no physical evaluation was conducted.
