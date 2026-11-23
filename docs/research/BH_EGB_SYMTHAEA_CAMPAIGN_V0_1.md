# Symthaea × Symtropy: 5D EGB black-hole scientific campaign v0.1

**Tracking issue:** [Symthaea #13406](https://github.com/Luminous-Dynamics/symthaea/issues/13406)  
**Symtropy dependencies:** [#7131 response operator](https://github.com/Luminous-Dynamics/symtropy/issues/7131), [#7135 local branch theorem](https://github.com/Luminous-Dynamics/symtropy/issues/7135), [#6805 Jacobian sensitivity](https://github.com/Luminous-Dynamics/symtropy/issues/6805), [#6915 Ma–Lü cross-check](https://github.com/Luminous-Dynamics/symtropy/issues/6915), [#6955 perturbative-validity envelope](https://github.com/Luminous-Dynamics/symtropy/issues/6955).  
**Status:** integration design and source audit; no Symthaea EGB campaign runtime or performance result is claimed.

## 1. Design decision

Use Symthaea substantially more—but do not make it the authority for physical calculations.

- **Symthaea:** formulate hypotheses, retrieve related structures, propose derivations, select the next research action, forecast action value, preserve campaign memory, and challenge claims.
- **Symtropy:** own the equation-derived model, numerical BVP and continuation calculations, boundary response, original-equation residuals, Noether checks, and solver receipts.
- **Independent qualification:** verify exact source blobs and fixtures, challenge derivations, check interval bounds, and decide what each result licenses.
- **Human/research authority:** chooses scientific priorities and adjudicates claims that require domain judgement.

The target is an evidence-bounded scientific campaign, not an autonomous agent with authority to promote its own discoveries.

## 2. Source audit: what Symthaea already contributes

The following public-main source blobs were inspected on 23 November 2026. Re-fetch the exact blobs at implementation time; the hashes below are audit identities, not a claim that all code has passed a clean build.

| Source | Observed role | Correct use in this campaign |
|---|---|---|
| crates/core/symthaea-core/src/physics/general_relativity.rs — blob SHA cae917e114407316bf263065d13a59dda9794e49 | Explicitly exploratory HDC encodings for GR concepts | Semantic retrieval/indexing only; not metric, geodesic, field-equation or waveform computation |
| crates/bridges/symthaea-physics-bridge/src/integrated_discovery.rs — blob SHA e840a4e316b8770dd0293154e9e116f30c049b9b | Factorial active-discovery loop over typed generation, experiment selection and CfC smoothing | Reuse its ablation discipline; do not reuse its energy-law task or assume its smoother applies to unlike solver outputs |
| src/language/research_plugin.rs — blob SHA 9aa62780576f9937ce6310c9c0176fbfa51bb9d5 | Research-domain entity/intent extraction and research prompts | Retrieval/query front-end only; it is not a full autonomous scientific-policy engine |
| src/z3_bridge.rs — blob SHA 28b802030718c299105f6e337be3a94db699f1f7 | SMT/Z3 subprocess bridge with an internal propositional fallback | Finite algebraic/logic obligations where semantics fit; not proof of PDE existence, interval bounds or functional analysis |

The GR encoder's own module documentation labels it exploratory. The HDC representation is useful as a semantic index, but a vector similarity is not a physical calculation. The Z3 bridge's fallback is not a substitute for Z3, Lean kernel checking, or rigorous continuous analysis.

The current integrated-discovery source explicitly motivates its 2×2×2 factorial and excludes an unqualified Phi-based factor. It also warns that temporal smoothing must not blend unrelated experimental rounds. That caution is especially important here: do not smooth residuals from different equation rows, parameter values, endpoint bases, meshes, units or precision levels into one pseudo-observation.

## 3. Compose existing scientific-policy owners

This campaign should implement a thin adapter over the following owners rather than creating replacements:

- [#6874 / #6875](https://github.com/Luminous-Dynamics/symthaea/issues/6874): Ramanujan hypothesis generation → EUREKA independent challenge.
- [#6879 / PR #6880](https://github.com/Luminous-Dynamics/symthaea/issues/6879): solver-independent, provenance-bound ScientificModel / PhysicsIR. Check exact merged status before importing it.
- [#10812 / PR #10867](https://github.com/Luminous-Dynamics/symthaea/issues/10812): black-hole reasoning and semantic-HDC / physical-calculation separation.
- [#8181](https://github.com/Luminous-Dynamics/symthaea/issues/8181): downstream Symtropy receipt consumption.
- [#11171](https://github.com/Luminous-Dynamics/symthaea/issues/11171): actual research-policy adapter.
- [#8721](https://github.com/Luminous-Dynamics/symthaea/issues/8721): prospective computation/tool allocation.
- [#12375](https://github.com/Luminous-Dynamics/symthaea/issues/12375): adaptive falsification power.
- [#8920](https://github.com/Luminous-Dynamics/symthaea/issues/8920): evidence-carrying derivation chains.
- [#8667](https://github.com/Luminous-Dynamics/symthaea/issues/8667): unit and coordinate metamorphic invariance.
- [#11042 / #11197](https://github.com/Luminous-Dynamics/symthaea/issues/11042): structure-first discovery and model misspecification.
- [#10723](https://github.com/Luminous-Dynamics/symthaea/issues/10723): disconfirm-first review.

The general receiver, scientific state, action schema, evidence identities and evaluation custody remain with those owners. This document defines the EGB-specific composition and tests only.

## 4. Capability-by-capability assignment

### Ramanujan / symbolic discovery

Propose exact identities, alternative derivations, asymptotic coefficient patterns, candidate response-map simplifications, and proof lemmas. Every proposal carries its expression/model digest, type judgment, assumptions, source observations, search configuration and seed, complexity, and non-claims.

A generated identity is a conjecture until an exact independent derivation or checked proof establishes it. A numerical pattern is a lead, not evidence.

### HDC

Use hypervectors to retrieve prior campaign episodes by dimensionless equation skeleton, symmetry class, endpoint-mode signature, operator pattern and proof-obligation metadata.

Keep the exact AST, units, assumptions, bases, projectors and source hashes beside the vector. HDC binding is not generally invertible; never use a vector as the canonical equation or field representation.

### CfC

Use CfC as an optional model of the research campaign's evolving state: action/result histories, correction difficulty, tool cost/latency and regime-change indicators. A possible use is forecasting whether the next correction is likely to converge or whether more source/endpoint analysis is needed.

Do not use CfC to solve the radial EGB equations, replace continuation, smooth heterogeneous equation residuals, or infer a fold from a temporal trend. Compare it against simple history baselines such as fixed thresholds and EWMA/AR-style predictors.

### FEP / active inference / value of computation

Select the next action by prospective decision value. Candidate action families include:
- exact-source replay of the action-row extractor;
- derive/replay one homogeneous sensitivity column;
- apply a predeclared sign, normalization or projector mutation;
- request a small-positive-coupling BVP point;
- increase spectral resolution or arithmetic precision;
- compare direct horizon-to-infinity transport with independent endpoint matching;
- derive a finite-coupling indicial recurrence;
- check a Ma–Lü ensemble/observable conversion;
- pursue a theorem obligation or counterexample.

Before execution, bind the evidence-prefix identity, predicted outcomes, which decision could change, estimated cost, uncertainty source and stop/pivot/replicate/abstain rule. Reuse #8721/#5476/#5478 and #11171. A retrospective rationale is not a calibrated prospective forecast.

### Causal reasoning

Use interventions to distinguish physical coupling effects from changes to gauge, horizon normalization, asymptotic projector, source sign, row extraction, precision, mesh or chart. Predeclare expected signatures and negative controls. Correlation between a small singular value and a plotted branch event is not a causal classification.

### EUREKA / independent challenge

Ramanujan proposes; EUREKA selects the challenge; Symtropy executes; independent verifiers check the source, numerical and proof obligations; only qualified evidence updates future policy state. The proposal mechanism must not choose or score its own hidden holdout.

## 5. EGB campaign data path

Use the existing canonical scientific model and receipt contracts:

ScientificModel / PhysicsIR candidate  
→ Ramanujan hypothesis envelope  
→ EUREKA challenge + prospective action forecast  
→ Symtropy versioned request  
→ Symtropy execution and original-row / Noether / endpoint checks  
→ request/source/output receipt verification  
→ independent challenge and held-out evaluation  
→ evidence-bounded campaign update.

Keep these states distinct:
- symbolic conjecture;
- exact symbolic identity;
- exact public-source replay;
- numerical candidate;
- numerically refined;
- interval validated;
- local existence / uniqueness theorem;
- continuation segment validated;
- PDE hyperbolicity qualified;
- observationally validated.

Do not infer one state from another. In particular, a stationary O(a²) BVP pass does not establish full nonlinear uniqueness, stability or strong hyperbolicity.

## 6. Initial episode manifest

1. **Action-source integrity:** replay the current public Symtropy action-row extractor against the committed O(λa²) fixture. Preserve any source/fixture discrepancy. Do not promote J1 to complete action-source replay until this passes.
2. **Einstein-limit map:** recover J0 = diag(1/2, 2) and both horizon-mode projections.
3. **First sensitivity:** challenge J1 and the affine offset derivative as separate objects; check all original rows and the differentiated Noether identity.
4. **Ma–Lü comparison:** check the equal-spin parameter, ensemble and charge map only to their declared spin and coupling orders.
5. **Basis covariance:** reproduce the #7131 exact synthetic controls and distinguish basis normalization from a physical response change.
6. **Finite-positive-coupling:** choose small positive λ points and compare both response columns against J0 + λJ1 using independently corrected BVPs.
7. **Local theorem:** help attack analytic coefficient dependence, endpoint mode counts, Fredholm index and the O(a²) local existence theorem in Symtropy #7135.
8. **Claim boundary:** explicitly challenge any inference from stationary BVP regularity to PDE hyperbolicity.

The GW250114/QNM observational campaign and Symthaea #11021 Geometry-to-Observation Identifiability Atlas are separate research programmes. They should not be merged into this stationary EGB benchmark.

## 7. Confirmatory ablations and outcomes

Use frozen episodes, exact source identities, paired starting states and matched inclusive budgets. Compare:
1. researcher-directed baseline;
2. fixed action schedule;
3. random matched-cost actions;
4. static residual/uncertainty threshold;
5. qualified existing active-experiment selector;
6. Symthaea research-policy incumbent through #11171;
7. incumbent with HDC retrieval disabled;
8. incumbent with CfC campaign modelling disabled;
9. incumbent with prospective VoC disabled/shammed;
10. exact tiny-world action oracle only as a diagnostic upper bound.

Keep the backbone, prompt, context, tool access, memory, compute and solver budget equal. Report separately:
- false-promotion and unsupported-singleton rates;
- exact/source claim disposition;
- verified derivation and theorem-obligation completion;
- cost to first independently replayed result and first interval/theorem result;
- falsification power by mutation/omission class at a declared false-alarm rate;
- forecast calibration, realized utility and decision regret;
- branch/parameter coverage versus validated branch arclength;
- missed defects, candidate rejections and negative transfer;
- solver calls, precision, memory, CPU/GPU, wall time and monetary cost.

No single score may hide false qualification or poor performance in a subgroup.

### Proposed gates (not achieved measurements)

- 100% rejection of frozen stale-source, wrong-sign, changed-projector, swapped-normalization and receipt-tampering controls.
- Zero promotion of HDC similarity, solver convergence, missing workflow status or sampled-only invertibility into theorem/physical claims.
- Correct recovery of every exact anchor in the frozen task set.
- Prospective action calibration and a better validated-result-per-cost Pareto frontier than the strongest simple policy, with paired uncertainty intervals and no false-acceptance regression.
- Require a preregistered minimum cost-to-qualified-result improvement before claiming an HDC/CfC/active-inference benefit. Remove a component from this path if its ablation shows no value.

All confirmatory tests must use hidden holdouts and evaluator custody outside the research-policy process. Publish failures and denominators.

## 8. External baselines

Keep benchmark families separate:
- AstaBench / E2E-Bench: broad research-workflow performance and cost.
- FIRE-Bench: verifiable rediscovery and evidence-backed conclusions.
- AutoformBench / a selected CAM-Bench subset: advanced Lean formalization infrastructure and proof-task difficulty.
- NewtonBench / Unsatz / FrontierPhysics: scientific-law and structure-first discovery.
- Akribeia plus matched spectral/continuation references: the actual 5D nonlinear numerical-physics problem.

A broad agent leaderboard is not a matched black-hole physics benchmark. Match theory, dimension, spin/coupling order, ensemble, conventions, resources and observables before claiming superiority.

## 9. Claim ceiling

A successful campaign adapter would show that Symthaea can allocate research effort and maintain evidence-bounded hypotheses on this declared black-hole task family. It would not prove that Symthaea discovers new physics, that HDC/CfC is superior, that all EGB branches are known, or that a stationary branch is dynamically stable or strongly hyperbolic.


## 10. Machine-readable episodes, source pins and live benchmark audit

The campaign now has a machine-readable episode manifest and schema:

- [Episode manifest](BH_EGB_CAMPAIGN_EPISODES_V0_1.json)
- [Manifest JSON Schema](BH_EGB_CAMPAIGN_EPISODES_V0_1.schema.json)
- [External benchmark landscape, 23 November 2026](BH_EGB_AGENT_BENCHMARK_LANDSCAPE_2026_11_23.md)

The manifest freezes eight episodes, ten legal research actions, ten policy arms and fourteen hostile controls. It deliberately leaves the confirmatory budgets, policy/model pins, evaluator identity, hidden-holdout digest and power/stopping plan null. Therefore its state remains design_frozen_execution_not_started; it is not eligible for confirmatory performance claims.

### Exact public source-pin replay

The manifest's thirteen source-file Git blob SHA-1 identities were independently fetched from their declared public repositories and refs. All thirteen returned blob IDs exactly matched the manifest. JSON parsing, top-level required-key presence, unique episode/action IDs, episode-to-action references, policy-arm count and negative-control count also passed the local structural check.

- Manifest blob: e2f44c570dfc28986a6ce05c35c66a787465f094.
- Schema blob: 963c5d4ff8606bbf3877fa621348752394d8612e.
- Source pins verified: 13/13.
- Episodes / actions / policy arms / hostile controls: 8 / 10 / 10 / 14.

This is a local, source-backed structural replay. It is not a formal JSON-Schema validator run, a Symthaea runtime integration test, a Rust build, hosted CI, a research-policy execution, or a physics result. The exact checks and claim ceiling are recorded in BH_EGB_CAMPAIGN_VALIDATION_RECEIPT_V0_1.json.

### Updated external benchmark implications

The 17 November 2026 SciWorkBench preprint reports a best mean score of 57.38/100 on 98 real scientific workflow tasks and finds that performance depends strongly on the model–harness pair. FIRE-Bench reports sub-50 F1 for its strongest tested agents. AstaBench's 30 April 2026 update reports only 3% perfect completion on E2E-Bench-Hard, even though the leading agent completed many individual workflow steps. These are distinct suites and should never be pooled into a single score.

For EGB, the differentiator to test is not merely better ideation. It is **source-bound, self-correcting research that selects useful actions, catches its own errors, preserves uncertainty and produces independently qualified mathematical claims**. That remains a research target, not an achieved Symthaea result.
