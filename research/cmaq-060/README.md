# CMAQ-060 — Compute-sparse predictable expert selection

Status: first executable scalar e-process implementation and deterministic probe. No Monte Carlo qualification or SOTA claim is made.

## Implemented architecture

CMAQ-060 now contains a Rust monitor that:
- verifies the direct conditional paired-equality typed null plan and the canonical expert manifest;
- samples one expert before current response transformations using a predeclared ChaCha20 selector seed;
- uses EXP3-style exponential weights with an explicit uniform exploration floor and bounded normalized log-factor rewards;
- evaluates only the selected expert's response/covariate Fourier maps;
- forms the selected expert's direction from its selected-subsequence history only;
- computes one positive factor (F_t=1+\lambda\langle u_{J_t,t},r_{J_t,t}\rangle);
- updates one global log e-process with one alpha threshold;
- stages the random-number generator, selector weights, selected-expert state, replay ID and evidence trace, then commits them transactionally;
- records feature-coordinate evaluations per pair as a direct compute metric.

The EXP3 selector is an optimization policy, not the source of statistical validity. The single e-factor is conditionally mean-one when the chosen expert is selected using only the past and independent pre-pair randomness, and its bounded direction is based on that expert's pre-pair history.

## Mathematical boundary

Let (J_t) be selected from the past and independent pre-pair randomness. The current covariate may also be used if it is available before (X_t,Y_t). Under direct conditional paired equality, every registered expert has a conditionally centered paired feature difference. Conditioning on (J_t) preserves that centering.

With feature norm and direction norm at most one, the selected factor is positive for (0<\lambda<1), so the product of selected factors is one global e-process. The selector may adapt based on past log-factor rewards without a second alpha allocation.

This is not the same strategy as CMAQ-057:
- C057 evaluates all experts and maintains full-information history for each.
- C060 evaluates only the selected expert and maintains its state only on selected observations.
- C057/C058 are full-information performance comparators, not the regret comparator for C060.
- EXP3 regret is meaningful only against counterfactual rewards defined under the same selected-subsequence state semantics. Report paired power/delay gaps against C057/C058 separately from any bandit regret statement.

C060 currently uses an EXP3-style importance-weighted update with explicit exploration. It does not claim an adversarial regret theorem in the crate; the theoretical regret bound, exact constants and comparator should be independently reviewed before making such a claim.

## Compute target

For K experts and per-expert product feature dimension D:
- C057/C058 factor production evaluates O(KD) feature coordinates per observation.
- C060 evaluates one expert's maps, approximately O(D), plus O(K) selector-state bookkeeping.
- The update receipt records feature-coordinate count so this can be measured rather than inferred.

Prospective target: at least 4× fewer feature-map evaluations than C057/C058, with no more than 2 percentage points of worst-family power regression at matched alpha. This is a target only.

## Integrity and validation

- Selector randomness is seeded before the stream and is independent of current pair values.
- Selection probabilities and direction are computed before current response feature transforms.
- Selected feature state is updated only after the factor is prepared.
- Prepare/commit is non-mutating until a complete staged update is accepted.
- The probe uses CMAQ-059 common seeded streams and writes stream digests, selected expert, probability, factor, reward, evidence and trace receipts.

The deterministic probe is a wiring check, not a Type-I study. Full confirmation still requires the CMAQ-059 preregistered null/alternative grid, exact confidence intervals, worst-family results and faithful SCPT comparison. No SOTA, deployment or safety claim is made.
