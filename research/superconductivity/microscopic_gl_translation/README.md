# Microscopic-to-Effective Superconductivity Evidence Bridge

**Contract:** SC-MICROSCOPIC-GL-TRANSLATION-0.1  
**Status:** source metadata and identifiability auditor; no microscopic calculation has been reproduced  
**Tracking issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12840

## Purpose

This workstream connects material-specific calculations and measurements to the effective GL models in #12793 and #12826 without treating phenomenological coefficients as free, source-less numbers.

It connects:
- conventional SCDFT / Eliashberg / DFPT-EPC calculations;
- correlated and unconventional many-body calculations;
- experiment-calibrated specimen-linked observables (#12677 and #12811);
- scalar amplitude-phase GL (#12793);
- multicomponent/interband/PDW GL (#12826);
- synthesis / phase realization (#12671), prospective discovery (#12634), and capability-aware follow-up (#12819).

The model family remains explicit. Conventional SCDFT results are not automatically applicable to cuprates, nickelates, or PDW states.

## Source benchmark motivating this bridge

Kawamura et al., *First-Principles Calculation of Coherence Length and Penetration Depth Based on Density Functional Theory for Superconductors*, Phys. Rev. Lett. 137, 196001 (published 3 November 2026), DOI https://doi.org/10.1103/lrhl-j9bt.

The paper reports a SCDFT method including finite-momentum Cooper-pair response and calculations of Tc, microscopic coherence length xi0 and London penetration depth lambdaL. Its reported application set includes elemental superconductors, V3Si and high-pressure H3S. The first fixture records only this paper-level method summary and material coverage; exact material tables, supplementary details and numerical inputs/outputs have not yet been extracted or reproduced.

The PNAS perspective *The path to room-temperature superconductivity: A programmatic approach* (2026), DOI https://doi.org/10.1073/pnas.2520324123, motivates a closed loop between prediction, synthesis thermodynamics, engineering controls and experiment. A model that predicts a high Tc but cannot identify a realizable phase or useful coherence/response is not a complete discovery result.

## Non-equivalence rules

Keep these quantities distinct:
- microscopic/BCS/Pippard coherence length xi0;
- near-Tc GL coherence length xi_GL(T);
- London penetration depth lambdaL(T), scalar only under an explicit isotropy assumption or otherwise a tensor;
- superfluid stiffness/helicity modulus under a declared normalization;
- GL gradient coefficient K;
- quartic GL coefficient b;
- microscopic pairing gap and coarse-grained |psi|.

The GL expansion is represented as

\[
F-F_n=\int d^dr\left[
a(T)|\psi|^2+\frac{b}{2}|\psi|^4+
K_{\mu\nu}(D_\mu\psi)^*(D_\nu\psi)+\cdots
\right].
\]

The normalization of psi changes the numerical coefficients. Therefore every coefficient requires a basis/field normalization, units, temperature and field regime, method, evidence status, source linkage and uncertainty/covariance.

Neither Tc, xi0 nor lambdaL alone determines the full set a(T), b, K_mu_nu, multiband coupling matrices and current response. The auditor rejects unsupported coefficient promotion.

## Evidence status vocabulary

- **derived**: extracted from an explicit, versioned microscopic free-energy, pair-susceptibility or response calculation with declared normalization and fit procedure.
- **constrained**: bounded by evidence but not uniquely identified.
- **phenomenological_fit**: fitted to experimental observations through a declared observation model/prior and state linkage.
- **not_identifiable**: current evidence does not identify the coefficient; numeric value must be null.
- **not_applicable**: coefficient is outside the chosen model family; numeric value must be null.

Paper-reported values are not independently reproduced values. The source metadata fixture preserves this distinction and intentionally carries zero numeric derived-observable rows.

## Implemented input contract and auditor

Files:
- schema: `data/schemas/superconductivity_microscopic_gl_translation_v0_1.schema.json`
- fixture: `data/research/superconductivity_microscopic_gl_translation_fixture_v0_1.json`
- binary: `sc_microscopic_gl_audit` in `tools/superconductivity-microscopic-gl-translation-harness`

The Rust auditor validates:
- explicit material state identity and state linkage for every observable and GL coefficient;
- calculation family and mechanism scope;
- finite scalar/vector/matrix observable values and non-negative finite uncertainty;
- scalar lambda_L only with an explicit isotropy assumption;
- lambda_L tensor only as a matrix with at least two declared axes;
- xi0_microscopic and xi_GL as distinct typed quantities with distinct length-scale definitions;
- derived a(T) slope only with quadratic pair susceptibility / free-energy expansion evidence;
- derived b only with quartic free-energy response;
- derived K only with finite-q pair response, a q-gradient expansion and basis normalization;
- phenomenological-fit coefficients only with experimental evidence and an experimental-effective model family;
- no promotion of conventional SCDFT outputs as correlated-unconventional GL coefficients;
- exact schema and fixture SHA-256 values in the audit receipt.

The source-method starter record contains:
- source method: SCDFT with finite-q pair response;
- paper-reported material coverage descriptions for elemental cases, V3Si and high-pressure H3S;
- no invented numerical values, meshes, cutoffs, pressure points or raw output digests;
- a(T) slope, b and K marked not_identifiable until the full quantitative response and coefficient normalization are extracted.

The K status is deliberately not_identifiable, rather than constrained, because the metadata-level source record does not include a quantitative finite-q response curve or fitted gradient coefficient.

## Next executable gates

1. Retrieve and freeze the full PRL article, methods, supplementary material and any author-provided inputs/outputs.
2. Extract exact pseudopotential/basis, XC, SOC, k/q mesh, energy cutoff, phonon treatment, Coulomb kernel and convergence.
3. Reconstruct at least one elemental case, V3Si and high-pressure H3S with exact material states and pressure.
4. Compare Tc, xi0, lambdaL and type-I/type-II classification with the paper and experiment under matched conditions.
5. Where the microscopic source actually provides the required response, fit the near-Tc GL expansion and derive only identifiable a, b and K parameters.
6. Verify GL xi_GL(T) only in the GL regime; never relabel xi0.
7. Pass identified/constrained coefficients into #12793/#12826 with covariance and provenance. Preserve non-identifiable values as null.
8. Evaluate held-out cross-modal predictions via #12811 and choose high-value next measurements/computations via #12819.

## Research targets (not results)

- Reproduce at least one published SCDFT material case with method-aligned inputs and reported uncertainty.
- 100% of exported GL coefficients have declared normalization, units, regime and evidence status.
- Zero silent xi0 ↔ xi_GL conversions.
- Zero cross-state joins across pressure, specimen, phase, doping or strain.
- Correct abstention for non-identifiable coefficients.
- Synthetic parameter recovery with nominal 90% intervals achieving 85–95% coverage for identified parameters; non-identifiable parameters must not be forced into narrow intervals.
- A held-out cross-modal prediction that beats a simpler baseline and is calibrated on grouped holdouts.

## Claim ceiling

This is a provenance and identifiability auditor. It does not reproduce the PRL calculations, derive coefficients from missing data, establish a room-temperature superconductor, prove any unconventional mechanism or authorize experiments.


## Basis-normalization non-identifiability

There is a structural ambiguity in phenomenological GL coefficients unless the order-parameter basis is fixed. Under a positive rescaling

\[
\psi' = s\psi,
\qquad
F = \int d^dr\left[
a|\psi|^2+\frac b2|\psi|^4+K|D\psi|^2
\right],
\]

the same physical functional expressed in the new field coordinate has

\[
a'=\frac a{s^2},\qquad
b'=\frac b{s^4},\qquad
K'=\frac K{s^2}.
\]

For a uniform state with \(a<0,b>0\), the amplitude obeys \(|\psi_0|^2=-a/b\), while the scalar GL length proxies satisfy

\[
\xi^2 = \frac K{|a|},
\qquad
\lambda^{-2}\propto K\frac{|a|}{b}.
\]

Both proxies are unchanged by the field rescaling even though the numerical tuple \((a,b,K)\) changes. Thus Tc, xi and lambda-like responses cannot by themselves identify absolute GL coefficient magnitudes without a field/basis normalization. The evaluator reports a synthetic factor-of-three rescaling example and checks these invariants numerically. The convention-specific proxy is not a substitute for a complete electromagnetic derivation.

The coefficient export adapter includes only numeric coefficients with status 'derived', 'constrained' or 'phenomenological_fit'. It excludes 'not_identifiable' and 'not_applicable'. Any exported coefficient must carry units, uncertainty/covariance or an explicit constraint bound; a derived record additionally requires an explicit basis normalization. The adapter preserves the full coefficient record, including state ID, method, supporting observation IDs, basis and uncertainty. It never reconstructs missing coefficients from Tc/xi/lambda.

## External benchmark comparison: what SOTA currently means

### BEE-NET (npj Computational Materials, 27 January 2026)

Gibson et al., *Developing a complete AI-accelerated workflow for superconductor discovery*, DOI https://doi.org/10.1038/s41524-026-01964-8.

The published workflow reports:
- BEE-NET MAE of 0.87 K against DFT-based Allen–Dynes Tc labels, and a 99.4% true-negative rate for its rare-property screening task;
- an initial generated pool above 1.3 million structures, a low-cost ML cascade, then about 5.6k candidates sent to DFT structural optimization;
- 866 expensive alpha-squared-F calculations in the final stage and 741 dynamically/thermodynamically stable calculated superconductors, with reported 86% final computational-screening precision;
- two synthesized follow-ups, Be2Hf2Nb and Be2HfNb2.

This is a strong conventional electron–phonon computational and discovery-pipeline baseline. Its 0.87 K metric is not experimental Tc error. The paper's final DFT estimates for the two follow-up compositions (9.3 K and 5.1 K) differ from measured resistive onsets (3.18 K and 4.24 K). The reported heat-capacity evidence also differs: Be2Hf2Nb shows a single jump consistent with bulk superconductivity, whereas Be2HfNb2 shows two jumps and evidence of multiple phases. Candidate prediction, synthesis, onset and bulk confirmation must therefore be separate scores. See the primary source and experimental discussion:
https://www.nature.com/articles/s41524-026-01964-8

### AutoSuperCon (preprint posted 11 November 2026)

Manium et al., *Machine-accelerated extraction of experimental superconducting parameters for physical insight and prediction across thousands of materials*, arXiv:2611.15627.

The preprint reports 20,354 entries for 6,793 superconductors and eight parameters: Tc, Hc, Hc1, Hc2, Jc, xi, lambda and gap Delta. It validates extracted Tc against NIMS SuperCon on 2,439 overlap records (R-squared 0.96; the subset at or below 30 K has R-squared 0.83). Its composition-based within-family predictors report log-space R-squared 0.83 for Tc and 0.69 for Hc2, but the paper explicitly reports failure to extrapolate to withheld cuprate and iron-based families.

This is an important multi-property evidence source, but it is an LLM-extracted preprint dataset, not unquestionable ground truth or a family-generalizing predictor. Record-level source, units, specimen, conditions and extraction status must remain available. Official data licensing and exact-release reproducibility must close before bulk admission. Source:
https://arxiv.org/abs/2611.15627

### First-principles response benchmark (PRL 137, 196001, 3 November 2026)

Kawamura et al. report SCDFT calculations of xi0, lambdaL and Tc using finite-momentum Cooper-pair response, applied to elemental superconductors, V3Si and high-pressure H3S. This extends the benchmark from Tc-only regression toward a jointly calculated coherence/stiffness response bundle. It remains a conventional-superconductor method and does not by itself provide the normalized GL coefficients a, b and K required by every effective model:
https://doi.org/10.1103/lrhl-j9bt

## How Symthaea should compare and aim to improve

Do not claim general SOTA superiority from a synthetic auditor, a random split, or a DFT-label regression result. Admission requires a matched task, exact data/split/model identity, and the same evidence authority.

The external benchmark ladder is:

1. **Closed-set reproduction.** Reproduce BEE-NET on its own frozen DFT/Allen–Dynes target and AutoSuperCon on its own exact released data/splits before comparing extensions. Do not mix their label authorities.
2. **Family / source / time generalization.** Add leave-family-out, leave-prototype-out, chemistry-held-out, pressure-held-out, source/lab-held-out and forward-time tests. Report selective risk versus coverage, calibration, OOD detection and abstention. The AutoSuperCon leave-family failure makes this a first-class metric, not a footnote.
3. **Multi-property, state-aware prediction.** Predict response bundles such as Tc, Hc2, Jc, xi, lambda and gap only when the sample, pressure, temperature, field orientation and measurement definition match. Keep microscopic xi0 distinct from xi_GL.
4. **Synthesis-aware discovery utility.** On a frozen prospective candidate universe and matched compute budget, compare BEE-NET's cascade, random, chemistry heuristics, diversity selection and Symthaea's capability/value-aware policy. Count independent physical validations, not just DFT survivors.
5. **Model-to-measurement closure.** Derive/constrain GL coefficients only under their stated identifiability and basis contract, then predict held-out same-specimen modalities through #12811. Do not fit one Tc and claim multi-observable prediction.
6. **Engineering translation.** Compare validated conductor/device outputs against application-matched NbTi, Nb3Sn, MgB2 and REBCO baselines only after material and specimen evidence qualifies.

Research targets for a future prospective cohort (not achieved):
- at K <= 50, aim for at least 1.5x independently validated candidate enrichment over the strongest admitted baseline, with paired 95% confidence intervals and no worse family-specific false-negative/calibration behaviour;
- at a fixed high-fidelity compute budget, aim for at least 2x fewer expensive alpha-squared-F/DFT follow-ups per independently confirmed result;
- require 85–95% empirical coverage for nominal 90% intervals on identified synthetic parameters; abstain on non-identifiable and out-of-domain cases;
- preserve all failures, non-detections and rejected candidates so recall can be audited;
- when the physical validation cohort is too small for a meaningful confidence interval, report the result as inconclusive.

The most defensible differentiator is not a larger model. It is an **evidence-executing discovery loop**: mechanism-specific physics, state-preserving data, capability-aware compute/measurement choice, calibrated abstention, synthesis outcome feedback, and independent physical verification. This is aligned with the field's stated prediction/engineering gap rather than a claim that the gap has already been solved.
