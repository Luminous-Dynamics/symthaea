# SDCB open-world covariance-model adequacy gate v1

**Status:** bounded diagnostic reference. Four Rust tests are authored but not compiled or executed here. The gate is only as valid as its externally calibrated threshold; no physical or real-data result is claimed.

## 1. The closed-world Bayesian trap

Bayesian model averaging normalizes posterior probabilities over the supplied candidate set:

\[
\sum_{s\in\mathcal S}P(s\mid D)=1.
\]

That identity remains true when every candidate model is badly misspecified. A normalized weight vector therefore cannot certify model-set adequacy.

This increment adds an explicit open-world alarm. It preserves the Bayesian model update but independently asks whether the best candidate with positive prior support predicts the new observation within a pre-calibrated predictive envelope.

## 2. Rule and trust boundary

Each observation is scored prequentially under each positive-prior model. Let \(\ell_s(y)=\log p(y\mid D_{<t},s)\). Let \(c_\alpha\) be a threshold fitted on an independent chronological validation corpus using the same observation units, dimensionality, transformation and prediction horizon.

The diagnostic is:

\[
\operatorname{adequate\_candidate}(y)
\iff
\max_{s:w_s>0}\ell_s(y)\ge c_\alpha.
\]

If the condition fails:
- the Bayesian weights may still be reported for diagnosis;
- the model set is flagged inadequate;
- downstream scientific decisions must abstain or enter an explicitly authorized model-expansion/review path;
- a zero-prior-weight scenario in the robust support set does not rescue the Bayesian model set.

A passing result means only that at least one registered model passed this threshold. It does not prove that model is physically true, that all aspects of the covariance model are correct, or that no unmodeled process exists.

The mixture predictive log density is also reported, but the adequacy rule uses the best positive-prior candidate's score. Both are sensitive to observation units and dimension. The threshold is therefore external and must never be fit using the observation being judged.

## 3. Frozen negative controls

### All registered models wrong

Two models predict \(y\sim N(0,0.1)\) and \(y\sim N(0,0.2)\), with equal prior weights. Observation \(y=20\).

- Bayesian posterior weights still sum to one.
- The less-bad model can receive almost all posterior mass.
- Its prequential log density is far below the preregistered \(-20\) threshold.
- The model set is flagged inadequate and is not eligible for a scientific-decision update.

### One candidate predicts adequately

A second model predicts \(y\sim N(20,1)\). For the same observation \(y=20\), the best candidate passes the frozen \(-20\) threshold.

This only tests the diagnostic mechanics. The \(-20\) threshold is illustrative and is not a universal or metrologically meaningful threshold.

## 4. Open-world escalation policy

An inadequacy flag should not auto-invent a physical common cause. It should:
1. preserve the observation and rejected/flagged model-set version;
2. prevent a silent confidence increase in the scientific claim;
3. request an independently authorized model review or discrepancy-model expansion;
4. retain old models and their evidence for replay;
5. version the expanded model set and validate it on future chronological data before it can influence decisions.

This is an assurance gate, not causal discovery.

## 5. Evaluation programme

Use held-out chronological sequences, blocked bootstrap and controlled omitted-model generators. Report:
- false inadequacy alarm rate on correctly specified in-envelope data;
- detection rate and delay on omitted-cause / regime-shift data;
- model-weight calibration and rank stability;
- total scientific interval coverage;
- rate of inappropriate decision updates prevented;
- abstention rate and time-to-recovery after model expansion.

Compare single best model, Bayesian model averaging, predictive stacking, generalized Bayes/Gibbs updating, robust support-set worst-case design, and the open-world gate.

Threshold selection must be nested inside training/validation data. The final chronological test period cannot be used to tune the threshold.

## 6. Limits

This v1 uses a caller-supplied fixed threshold over Gaussian predictive log densities. It does not estimate a universal threshold, infer unknown causes, or provide a calibrated probability that the model set is wrong. Future work should assess standardized predictive residuals, PIT-based checks, multivariate tail diagnostics, change-point detection, and sequential alarm control.
