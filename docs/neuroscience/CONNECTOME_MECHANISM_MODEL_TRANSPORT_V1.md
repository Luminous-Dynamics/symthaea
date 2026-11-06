# Mechanism-model predictive transport across independent animals V1

**Tracking:** #10792 — SCI-CONNECTOMICS-095
**Composes:** #10739, #10753, #10675, #10682, #9255
**Status:** synthetic integration contract and exact paired-animal known answer; no biological mechanism or population claim.

## 1. Research question

Does the predictive advantage of a model family associated with a candidate mechanism persist on independently held-out animals, under an exact query/intervention/condition binding, without turning response-model fit into proof of the underlying biological mechanism?

#10739 measures separation of finite declared mechanism hypotheses under finite declared interventions and nuisance profiles. V1's executable contract is synthetic-IID, not an animal-clustered biological guarantee. #10753 measures model calibration/response prediction on held-out animals. This bridge keeps those results separate and joins them by explicit identities.

## 2. Claim separation

Keep four conclusions distinct:
- **Synthetic query gate:** exact #10739 query identifiability on its synthetic finite model family.
- **Synthetic mechanism gate:** exact #10739 mechanism-class identifiability on that same synthetic family.
- **Held-out predictive transport:** whether one pre-registered mechanism-associated response-model family has lower animal-macro Brier on shared, determinate lockbox observations.
- **Biological mechanism truth:** remains `NOT_ESTABLISHED_BY_PREDICTIVE_TRANSPORT` in V1, regardless of which model wins.

A model that predicts effective response better does not prove a direct synapse, a polysynaptic path, an extrasynaptic mediator, or a causal effect. Functional response is not anatomical edge truth.

## 3. Exact upstream binding

The integration workflow runs the #10739 Rust audit and independently replays it with its separate tuple-product Python oracle. It separately runs the #10753 cohort verifier and replays it with its independent data-contract oracle.

The bridge receipt binds:
- raw mechanism input/report SHA-256 and the report's embedded input digest;
- raw cohort input/report SHA-256 and the transport oracle's verified split digest;
- exact mechanism query ID/specification digest and selected intervention ID;
- exact animal split, lockbox animal identities, target pair and model identities;
- pinned upstream mechanism Rust/oracle and transport-oracle Git blob identities;
- exact condition/protocol mapping records and per-animal comparison values.

The upstream mechanism `replicate_count=7` remains a synthetic IID software fixture. It is never re-labelled as seven animals or as a biological clustered guarantee.

## 4. Cross-source mapping

V1 binds the `orthogonal-joint` mechanism-audit action to the frozen `WT_BACKGROUND` primary transport condition and an `UNC31_MUTANT` OOD-abstention control. This mapping is explicit and synthetic. Similar strings or a shared neuron-pair name are not enough to infer equivalence.

The mechanism-to-predictor mapping is also explicit:
- `direct` / `DIRECT_SYNAPTIC` -> `anatomy-only`;
- `extra` / `EXTRASYNAPTIC_LATENT` -> `wired-plus-extrasynaptic`.

The bridge evaluates one pre-registered model pair and one target pair (`RID-URX`) to avoid post-hoc model or outcome selection.

## 5. Animal-level statistic

For each lockbox animal, compute Brier loss over the shared, determinate observation set for the target pair. First aggregate observations within animal. Then compare the paired animal-level losses.

Let `d_i = Brier(model_B, animal_i) - Brier(model_A, animal_i)`. A negative value favors model B. The exact one-sided paired sign test counts independent animals, not pairs, trials, frames or timepoints:

`p = 2^-n * sum(C(n,k), k = wins, ..., n)`.

V1 preregisters one comparison at alpha 0.05. Five evaluable non-tied independent animals is the smallest possible n for which all five wins can yield p <= 0.05 (p = 1/32). Fewer than five returns `NOT_ESTIMABLE_INSUFFICIENT_ANIMAL_SUPPORT`.

If a model abstains or is out of domain, the observation is not scored for that model. Pairwise comparison uses only the shared evaluation subset and separately reports asymmetric coverage/abstention. The lower-coverage model is never declared better just because its difficult cases were omitted.

`INDETERMINATE`, `NOT_MEASURED`, `FAILED` and `INACCESSIBLE` are not binary negatives.

## 6. Known-answer synthetic fixture

The integrated cohort has one train animal, one validation animal, one calibration animal, five independent WT lockbox animals and one unc-31 mutant lockbox animal. The same target pair is observed in each lockbox animal.

- The target pair `RID-URX` has `NO_DIRECT_EDGE` in its structural record.
- All five WT animals have a determinate functional-response outcome for that pair.
- `anatomy-only` predicts 0.1 functional probability; `wired-plus-extrasynaptic` predicts 0.8.
- Per-animal Brier values are 0.81 vs 0.04, so all five paired lockbox animals favor model B.
- The exact one-sided sign-test p-value is 1/32 = 0.03125.
- The mutant lockbox animal has a separate equivalently non-functional observation; the wired-plus-extrasynaptic model has no support for that condition and must abstain on both measured mutant pairs. The mutant is not included in the paired WT contrast.
- Missing and indeterminate observations remain outside binary scoring.

Expected predictive transport: `MODEL_B_FAVORED`. Expected mutant OOD abstentions by model B: 2. Biological mechanism status remains `NOT_ESTABLISHED_BY_PREDICTIVE_TRANSPORT`.

## 7. Required adversarial controls

1. Cross-query or stale source report: reject.
2. Mechanism audit status or selected action differs from the frozen manifest: reject.
3. Animal split digest, lockbox membership, or model train/calibration identities differ: reject.
4. Duplicate animal/pair observation or repeated trials counted as independent animals: reject.
5. Target pair wiring status differs from the frozen mapping: reject.
6. Action-to-condition protocol binding absent/duplicated: reject.
7. Missing, indeterminate, failed, or inaccessible observations become negative: reject.
8. Model B prediction is changed or removed for a WT lockbox observation: report asymmetric coverage or reject the known-answer fixture; never count the animal as a win.
9. OOD mutant prediction without a support receipt/abstention: reject.
10. Fewer than five paired non-tied animals: return not estimable, not a positive transport result.
11. Sign-test direction or alpha changes after outcomes are visible: reject frozen manifest digest.
12. Model B win is promoted to an extrasynaptic mechanism fact: reject.

## 8. Benchmarks and promotion targets

Software contract targets:
- 100% exact upstream replay agreement;
- 0 cross-source identity or split mismatches accepted;
- 0 pseudoreplication in paired tests;
- 100% correct abstention and insufficient-sample outcomes;
- exact sign-test agreement with brute-force enumeration for bounded n <= 20.

Biological research targets (not current results):
- compare anatomy-only, wired-plus-extrasynaptic, state/history-aware and hierarchical animal models on the same preregistered lockbox;
- animal-macro Brier and calibration on determinate labels, continuous response interval coverage, abstention and coverage reported separately;
- require independent-animal uncertainty intervals and sufficient paired lockbox animals before promoting transport support;
- replicate across a second condition/species only as a separate pre-registered transport track;
- prospective perturbation confirmation before making causal-mechanism claims.

## 9. Claim ceiling

A passing V1 integration receipt establishes byte-bound replay of two synthetic owner contracts and an exact animal-level scoring calculation. It does not establish population transportability, causal mechanism, direct synapse identity, extrasynaptic causation, or biological calibration.