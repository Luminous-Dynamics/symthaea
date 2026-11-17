# Optical ISRU v0.2 — noise-aware EIG and decision-value reference v0.1

**Status:** research reference implementation; not yet compiled or executed in Rust  
**Parent:** Symthaea #12483 (sealed BO/EIG tournament)  
**Related:** Symthaea #12496 (v0.2 preregistration), #12497 (regime boundaries / transfer), #12511 (LHS/Sobol); Symtropy #5612 (typed metrology evidence) and #5656 (dynamic optical contamination)

## Why this tranche exists

The existing Symthaea experiment selector scores the population variance of candidate-hypothesis predictions. That is a useful hypothesis-disagreement heuristic, but it is not Bayesian expected information gain: it does not use a prior, an observation likelihood, a measurement-noise model, or the probability of each possible observation.

This tranche adds an independently inspectable finite-hypothesis reference for:

1. likelihood-based expected information gain (EIG);
2. expected value of sample information (EVSI), derived from an explicit decision-loss matrix;
3. cost-normalized versions of both scores; and
4. deterministic action ranking with stable tie-breaking.

The implementation is dependency-free and can be tested directly with rustc, without adding a Cargo dependency or wiring it into a live execution path.

## Mathematical contract

For hypotheses H, prior p(h), candidate action a, and possible outcomes y, each action supplies a complete likelihood matrix:

\[
p(y\mid h,a),\qquad \sum_y p(y\mid h,a)=1.
\]

The predictive outcome probability and posterior are:

\[
p(y\mid a)=\sum_h p(h)p(y\mid h,a),
\qquad
p(h\mid y,a)=\frac{p(h)p(y\mid h,a)}{p(y\mid a)}.
\]

Information gain is the mutual information between the hypothesis and the prospective observation:

\[
\operatorname{EIG}(a)
=H[p(h)]-\sum_y p(y\mid a)H[p(h\mid y,a)].
\]

For a declared loss table L(h,d), where d is an engineering decision, the current Bayes risk and expected post-measurement Bayes risk are:

\[
R_0=\min_d\sum_h p(h)L(h,d),
\]

\[
R_a=\sum_y p(y\mid a)\min_d\sum_h p(h\mid y,a)L(h,d).
\]

The expected value of sample information is:

\[
\operatorname{EVSI}(a)=R_0-R_a.
\]

The reference reports EIG in nats and EVSI in caller-supplied loss units. It reports EIG/cost and EVSI/cost only when the caller supplies comparable positive scalar cost units. It does not invent a universal exchange rate between hours, joules, sample mass, consumables, risk, or money.

## The critical distinction: information is not always decision value

One frozen unit-test example has three hypotheses. H0 and H1 lead to the same engineering decision; H2 is a low-prior, high-consequence alternative. An action that separates H0 from H1 produces more entropy reduction but zero EVSI. An action that detects H2 produces less entropy reduction but a much larger expected reduction in decision loss.

Independent arithmetic for that fixture gives:

| Candidate | EIG (nats) | EVSI (loss units) |
|---|---:|---:|
| Distinguish H0 from H1 | 0.360703 | 0 |
| Detect H2 | 0.077347 | 1.882 |

This is the reason the tournament must report both metrics. A policy that maximizes disagreement or entropy reduction can spend rig time resolving a scientific distinction that does not change an engineering decision, while missing a low-probability failure mode that does.

A second analytic binary-channel fixture uses prior (0.5, 0.5) and likelihoods (0.9, 0.1) / (0.1, 0.9). Its EIG is 0.368064 nats and EVSI is 0.4 loss units under unit misclassification loss.

## Why this matters for optical contamination

The likelihood model should eventually represent joint explanations, not assume that every measured change is a material effect. Example latent hypotheses could pair:

- material regime: spallation, melting/sintering, volatile release, or unrepresented;
- optical state: clean, molecular film, particulate deposition, or uncertain;
- metrology state: valid calibration, drift, common-cause reference failure, or revoked/expired calibration.

Candidate actions need not all process material. They may include a material exposure, QCM readout, optical witness measurement, clean-reference comparison, calibration cross-check, controlled cleaning/shielding experiment, requalification, or campaign pause. A maintenance or reference measurement can therefore win on EVSI when it protects a high-consequence decision or resolves a suspected common cause.

Important boundaries:

- A QCM measures deposition at its own crystal surface; it does not alone identify deposit chemistry or prove the optical state of another surface.
- Witness measurements must be representative of the optical surface in location, orientation, temperature, exposure, and spectral response.
- Agreement between channels does not establish independence if they share a source, calibration, optical path, power rail, or clock.
- A model prediction is not a physical observation; only Symtropy's qualified evidence records may update a physical campaign.
- Intervention authority remains external. This reference only ranks supplied candidate models; it does not command instruments, authorize cleaning, or bypass safety controls.

The code can accept joint-state likelihoods, but it does **not** yet implement a factored Bayesian network, state-transition model, particle filter, hidden Markov model, causal identification, or a dynamic apparatus twin. Those are later work.

## Validation contract

The Rust file contains ten authored tests covering:

- analytic EIG and EVSI for a symmetric binary channel;
- zero information under identical likelihood rows;
- a case where decision value reverses the EIG ranking;
- cost-normalized action ranking;
- zero-predictive-probability outcomes;
- rejection of unnormalized priors and likelihood rows;
- malformed matrix dimensions and non-finite costs;
- malformed decision-loss matrices; and
- deterministic tie-breaking.

No probability is silently renormalized. Negative/non-finite probabilities, invalid costs, ragged arrays, and unsupported dimensions are rejected. Zero-prior hypotheses are permitted; outcomes with zero predictive probability are skipped rather than assigned a fabricated posterior.

### Reproduce locally

From the repository root:

```sh
rustc --edition=2021 --test research/asteroid-optical-isru-v0.2/noise_aware_eig.rs -o /tmp/isru-eig-tests
/tmp/isru-eig-tests
```

**Validation honesty:** the authored tests have not been compiled or executed in the authoring environment because rustc/cargo are unavailable there. The worked EIG/EVSI values above were independently recalculated from the stated probability and loss tables; that arithmetic check is not a Rust test run. No CI result, physical validation, instrument qualification, or SOTA advantage is claimed.

## Integration plan for the sealed tournament

1. Keep the existing prediction-variance selector as a separately named baseline.
2. Add likelihood-based EIG and EVSI as separate policies; do not relabel one as the other.
3. Use the same candidate action set, prior evidence snapshot, admissibility gate, and declared cost basis for all eligible policies.
4. Add noisy, heteroscedastic, censored, failed-capture, and common-cause likelihood fixtures. Include cases where optical degradation mimics a material regime change.
5. Include measurement/maintenance actions and compare their downstream decision value with another material exposure.
6. Evaluate on sealed worlds with paired seeds and whole-batch/material-family holdouts. Keep hidden physical values inaccessible to the selector.
7. Report EIG, EVSI, decision regret, interval calibration, false regime-transition rate, abstention, accepted-product decision accuracy, and the full campaign resource vector. Use paired cluster-aware intervals at the batch or laboratory level.
8. Before integrating a dynamic model, pin a baseline and negative controls that demonstrate whether the state factors are identifiable from the available measurements.

This reference is a first executable algorithmic tranche, not a finished optimizer. Beating NASA GP-EI, BALPI, PhaseTransfer, or cost-aware BO requires matched, sealed, independently reproduced comparisons under their relevant task assumptions—not a unit-test pass or a comparison against random search alone.
