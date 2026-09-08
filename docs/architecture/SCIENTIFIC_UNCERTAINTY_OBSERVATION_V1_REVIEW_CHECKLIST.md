# SCI-008 Review Checklist — Scientific Uncertainty-Bearing Observation v1

Use this checklist to review the SCI-008 architecture contract only. It does not qualify any measurement, uncertainty method, calibration procedure, or scientific claim.

## A. Observation boundaries

- [ ] Scientific quantity/construct is distinct from measurement specification.
- [ ] Measurement specification is distinct from observation event.
- [ ] Observation event is distinct from reported value.
- [ ] Reported value is distinct from uncertainty representation.
- [ ] Uncertainty representation is distinct from uncertainty-method validity.
- [ ] Uncertainty is distinct from calibration.
- [ ] Calibration is distinct from scientific authority.

## B. Exactness

- [ ] Exactness requires an explicit exactness basis.
- [ ] Small uncertainty does not mint exactness.
- [ ] Zero standard error does not mint exactness.
- [ ] Zero-width interval does not mint exactness.
- [ ] Display precision does not mint exactness.
- [ ] Exact-by-construction semantics can remain domain-owned.

## C. Value semantics

- [ ] The shared kernel does not force one universal scalar/value type.
- [ ] Domain value schema/profile identity is retained.
- [ ] Units/domain semantics are bound.
- [ ] Exact bits/bytes alone do not imply same scientific meaning.
- [ ] Same printed decimal does not imply same exact numeric artifact.

## D. Measurement classes

- [ ] Direct, derived, proxy, latent, model-derived and simulation outputs remain distinguishable where used.
- [ ] Measurement class is not treated as a universal evidence ranking.
- [ ] Simulation output cannot silently become physical observation.
- [ ] Model prediction cannot silently become observed measurement.

## E. Uncertainty taxonomy

- [ ] Heterogeneous uncertainty classes remain explicit.
- [ ] Standard error is not automatically converted to confidence interval.
- [ ] Confidence interval is not treated as credible interval.
- [ ] Credible interval is not treated as frequentist coverage guarantee.
- [ ] Revision range is not treated as a probability interval.
- [ ] Instrument resolution is not treated as sampling uncertainty.
- [ ] Ensemble dispersion proxy is not called calibrated uncertainty.
- [ ] Unknown uncertainty remains representable.

## F. Calibration

- [ ] Uncertainty representation is distinct from calibration procedure.
- [ ] Calibration procedure is distinct from calibration execution.
- [ ] Calibration execution is distinct from calibration evidence.
- [ ] Nominal coverage is distinct from realized coverage.
- [ ] OOD coverage does not silently inherit in-domain guarantees.
- [ ] `NoExchangeabilityGuarantee`-style states remain possible.

## G. OOD / applicability

- [ ] In-domain, OOD and unknown-domain states can remain explicit.
- [ ] OOD does not automatically subtract a confidence amount.
- [ ] Boundary conditions/applicability are not hidden inside one scalar.
- [ ] Downstream policy may refuse/attenuate without rewriting the original observation.

## H. Dependency / covariance

- [ ] Observations may share uncertainty/error dependencies.
- [ ] Different observation IDs do not imply independent errors.
- [ ] SCI-006 dependency identities can be referenced.
- [ ] A covariance/correlation artifact can be bound when qualified.
- [ ] The shared kernel never assumes zero covariance by default.

## I. Replicates

- [ ] Replicate identities can be retained.
- [ ] Aggregate summary does not erase raw replicate evidence.
- [ ] Aggregation method identity is retained.
- [ ] Exclusion/missing replicate policy can be retained.
- [ ] Within/between replicate structure can remain explicit.

## J. Censoring / detection limits

- [ ] Censoring states remain distinct from ordinary observed values.
- [ ] Detection-limit values are not silently substituted as exact observations.
- [ ] Any substitution/imputation is an explicit transformation lineage.
- [ ] Interval censoring can remain explicit.
- [ ] Truncation-by-design can remain explicit.

## K. Missingness

- [ ] Missingness remains evidence-bearing.
- [ ] Different missingness reasons can remain distinct.
- [ ] Missing does not become zero.
- [ ] Missing does not become negative evidence automatically.
- [ ] Missing does not become falsifier `NotTriggered` automatically.
- [ ] SCI-004 prospective missingness policy remains separate from actual SCI-008 missingness outcome.

## L. Transformations

- [ ] Measurement-affecting transforms receive explicit lineage.
- [ ] Source uncertainty is not copied forward automatically through arbitrary transforms.
- [ ] Propagated uncertainty method has its own identity.
- [ ] No universal Gaussian/root-sum-square propagation is implied.
- [ ] Derived artifacts retain source dependency links.

## M. Falsifier integration

- [ ] SCI-007 can consume uncertainty-bearing observations.
- [ ] Threshold-straddling evidence can yield `Inconclusive`.
- [ ] Missing required uncertainty can yield `NotEvaluable`.
- [ ] Invalid measurement can yield `MeasurementInvalid`.
- [ ] Point-only evaluation requires an explicit preregistered policy when allowed.

## N. Causal boundary

- [ ] Precise estimate does not imply causal identification.
- [ ] Uncertainty interval does not prove assumptions such as exchangeability/no-confounding.
- [ ] Identification strategy/diagnostics remain separate evidence coordinates.

## O. Provenance boundary

- [ ] Calibrated uncertainty does not imply authentic source.
- [ ] Verified bytes do not imply valid uncertainty method.
- [ ] SCI-002/SCI-003/provenance layers remain independent.

## P. First implementation slice

- [ ] First Rust slice is additive/non-authorizing.
- [ ] It avoids universal numerical calculations.
- [ ] It can introduce observation/uncertainty enums/envelopes only.
- [ ] Existing domain adapters preserve stronger local semantics.
- [ ] No qualification transfers from Economics/Matter/Physical Agency.

## Q. Conjecture Engine migration

- [ ] Existing simple `(x,y)` paths remain available for explicitly simple/synthetic/exact uses.
- [ ] Real scientific data can use a richer uncertainty-bearing adapter.
- [ ] The richer path retains measurement/provenance/sample/dependency information.
- [ ] The legacy adapter does not silently assert real-world exactness.

## R. Anti-shortcuts

Reject the architecture if it introduces or implies any of these:

- [ ] universal `confidence: f64` as observation authority;
- [ ] automatic Gaussian conversion;
- [ ] `uncertainty < epsilon -> exact`;
- [ ] model dispersion called posterior uncertainty without evidence;
- [ ] nominal interval coverage called calibrated truth;
- [ ] OOD converted to one confidence penalty;
- [ ] missing/censored values silently imputed without lineage;
- [ ] independence inferred from separate observation IDs;
- [ ] point estimates treated as exact falsifier inputs by default;
- [ ] uncertainty state becoming scientific/action authority.

## S. Review question

The PR is ready for architectural review only if the answer is yes:

> Does SCI-008 stop reported points and heuristic dispersion from masquerading as exact/calibrated scientific state, while preserving heterogeneous uncertainty semantics, dependency/correlation, OOD applicability, censoring, missingness, and transformation lineage without forcing them into one probability/confidence model?
