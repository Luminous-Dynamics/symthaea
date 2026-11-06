# Adversarial expansion of connectome causal/measurement model sets V1

**Tracking:** #10765 — SCI-CONNECTOMICS-096
**Composes:** #10757/#10761 (shared-policy robust design), #10762/#10764 (static-model sequential policies), #10603/#10661 (append-only hypothesis challenges and replay)
**Status:** exact finite synthetic model-set expansion; no biological performance claim is made.

## 1. Research question

Can a robust scientific certificate become *more* confident after a new plausible causal/measurement model is admitted, even though the query, action space and experiment policy space did not change?

It must not. This is a direct assurance property for distributionally robust decision systems.

## 2. Exact monotonicity result

Let V be the current finite set of joint causal/measurement-model vertices, Π the frozen set of deployable non-anticipative policies, and R(π,m) the expected decision loss of policy π under model m.

The exact robust risk is:

\[ R^*(V)=\min_{\pi\in\Pi}\max_{m\in V}R(\pi,m). \]

For an append-only expansion V' = V ∪ {m_new}, each fixed policy has non-decreasing worst-case loss:

\[ \max_{m\in V'}R(\pi,m)\geq\max_{m\in V}R(\pi,m). \]

Since Π is unchanged, minimizing both sides over the same policy space gives:

\[ R^*(V')\geq R^*(V). \]

Consequences:
- robust optimum loss is non-decreasing;
- a previously valid certificate may be invalidated;
- a previously invalid certificate cannot become valid solely by adding a model vertex;
- any apparent improvement means the policy class, alphabet, query, action universe, costs or evaluator changed, or the implementation is wrong.

## 3. Critical precondition: freeze the policy space

The theorem is only valid when the same policy set is used before and after expansion. Freeze and compare exactly:

- query specification, decision rule and terminal loss;
- reconstruction-world set and each world's query decision;
- action IDs, feasibility, safety and effort costs;
- first-stage history alphabet;
- second-stage outcome alphabet;
- history-conditioned action menus and branch-cost constraints;
- finite horizon, stopping rules and allowed non-anticipative policy class;
- split manifest and evaluation semantics.

The new model vertex may change its own priors and response-kernel probabilities, but it may not add outcome labels, alter existing worlds, replace actions, or mutate an existing model vertex.

Zero-probability labels remain in the frozen common alphabet if the contract declares them. Missing response support is not an excuse to change the decision-policy domain after seeing the challenge.

## 4. Synthetic known-answer expansion

Base set: MODEL-A and MODEL-B. A calibration action emits CAL-A or CAL-B to identify the fixed model from an observable history. The policy selects ACTION-A or ACTION-B accordingly and resolves the query with zero worst-case loss.

Append MODEL-C without changing any observable category. Under MODEL-C:
- calibration emits CAL-A or CAL-B with probability 0.5 each, independent of the query world;
- every action's second-stage POSITIVE/NEGATIVE response is 0.5/0.5, independent of the world;
- priors remain 0.5 / 0.5.

MODEL-C therefore makes the query unidentifiable under the frozen intervention/policy space.

| Quantity | Base V | Expanded V' |
|---|---:|---:|
| Model vertices | 2 | 3 |
| Static-model robust optimum | 0 | 0.5 |
| Certificate tolerance | 0.01 | 0.01 |
| Robust certificate | PASS | NOT_CERTIFIABLE |
| Added vertices | — | MODEL-C only |
| Frozen query/world/action/policy space | identical | identical |

The expansion must pass monotonicity: 0.5 ≥ 0. A certificate that was valid under the base model set is correctly revoked after the new model is admitted.

## 5. Audit design

1. Read the exact base and expanded input bytes, plus a frozen challenge manifest.
2. Verify their Git object identities and emit SHA-256 over the exact bytes.
3. Load the independent sequential-policy oracle used by #10764 and validate both full input matrices.
4. Verify the expanded model list is the exact base list plus one declared vertex; no mutation, deletion, substitution or duplication.
5. Verify query, worlds, actions, feasibility/cost, history/outcome alphabets, policy space and split are identical.
6. Recompute the exact static robust policy-tree optimum on both sets using the same exhaustive algorithm and common policy class.
7. Check robust-loss monotonicity and certificate monotonicity.
8. Emit an exact audit receipt containing input hashes, model-set identities, before/after optima, certificate transition, ancestry overlaps and claim ceiling.

## 6. Required adversarial controls

- model vertex deletion or replacement;
- mutation of an existing model vertex under the same ID;
- two additions where one is declared;
- stale base or expanded input bytes;
- query/world/action/cost/feasibility/policy-space mutation;
- first-stage or second-stage alphabet expansion;
- incomplete stage-one or history-conditioned stage-two kernel matrix;
- unnormalized prior or response probabilities;
- apparent optimum decrease after expansion;
- certificate reappearing after an expansion when it was already invalid;
- shared ancestry mislabeled as independence.

## 7. Benchmarks and acceptance targets

These are acceptance targets, not biological results:

| Evaluation | Target |
|---|---:|
| Exact base/expanded optimum agreement with independent policy-tree oracle | 100% |
| False monotonicity passes on deliberate mutations | 0 |
| False certificate promotions after append-only expansion | 0 |
| Frozen-universe mutation detection | 100% |
| Raw byte and model source identity checks | 100% |
| Shared model-ancestry overlap reporting | 100% |

## 8. Claim ceiling

A passing audit establishes monotonicity only for the declared finite policy space and appended model vertices. It does not prove that the ambiguity set is biologically complete, that MODEL-C is biologically plausible, that source evidence is authentic, or that model vertices are independent.