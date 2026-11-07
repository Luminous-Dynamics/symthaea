# Drosophila coordination mechanism-identifiability audit V1

**Tracking:** #11043 — DROSOPHILA-WALKING-002  
**Composes:** #9934, #10670, #9587, #9522, the phase-response surface (#10922), familywise FWER (#10959), and closed FWER (#10961).  
**Status:** exact synthetic model-equivalence audit only. No biological experiment or reproduction is claimed.

## 1. The scientific distinction

Rhythm generation and inter-leg coordination are separate estimands. Pugliese et al., Nature (28 October 2026), report a DNg100-driven rhythm-generating connectome circuit and show that the DNg100 simulations did not reproduce the tripod inter-leg pattern; the discussion names proprioceptive feedback, biomechanics and other pathways as plausible missing contributors.

A May 2026 bioRxiv preprint by Sapkal et al. reports central coupling and task-dependent proprioceptive/descending modulation of inter-leg coordination. This is a preprint and remains an independent challenger, not a settled or peer-reviewed ground truth.

Primary source links:
- Nature version of record: https://www.nature.com/articles/s41586-026-11195-y
- Pugliese analysis code: https://github.com/smpuglie/Pugliese_2026
- Pugliese simulation/Anipose data: https://doi.org/10.5281/zenodo.22260923
- Sapkal et al. bioRxiv v1/v2: https://doi.org/10.64898/2026.04.29.721658

## 2. V1 purpose

This crate is an **observational-equivalence auditor**, not an intervention selector or biological model. It compares exact integer signatures across:
1. summary-only baseline observations;
2. full baseline phase-offset / stance-distribution observations;
3. the full declared feasible intervention-response portfolio.

The four mechanism families are non-exclusive:
- local CPG autonomy;
- central neural coupling;
- proprioceptive/load feedback;
- body/environment mechanics.

A fifth omitted-mechanism challenger represents shared drive. All five synthetic worlds are deliberately calibrated to identical mean frequency, aggregate phase-locking and tripod-occupancy summaries.

## 3. Exact known answer

Five worlds yield 10 unordered model pairs.

- Summary-only baseline: all 5 worlds occupy one equivalence class; 0/10 pairs are separated.
- Full baseline trajectories: 2 classes; 4/10 pairs separated. The mechanical challenger changes pairwise phase offsets while retaining the chosen baseline summary metrics.
- Three feasible interventions: four equivalence classes; 9/10 pairs separated.
- The unresolved pair is W-central / W-shared-drive. They give identical predictions under every currently feasible intervention in the fixture.
- A candidate central-bridge block would separate them, but its feasibility receipt explicitly marks it unavailable.

The correct result is \`NOT_IDENTIFIABLE_UNDER_DECLARED_FEASIBLE_PORTFOLIO\`. If the bridge-block intervention is hypothetically marked feasible, the synthetic worlds become pairwise distinguishable—but the report still issues no biological mechanism certificate.

## 4. What this proves and does not prove

It proves exact equivalence partitioning for a frozen synthetic signature matrix, that aggregate summaries can hide full trajectory differences, and that a model-identity label cannot break an unresolved equivalence class.

It does not prove the candidate worlds are biologically complete, that an intervention is experimentally feasible, that model parameters are identified, or that a direct synaptic mechanism is true. Equal integer signatures represent synthetic observational equivalence, not a calibrated statistical equivalence test. Statistical uncertainty, distributions, animal/session clustering and test error control remain the responsibility of the upstream benchmark layers.

## 5. SOTA / industry comparison

This audit is complementary to:
- Pugliese et al. (Nature 2026): connectome-derived local rhythm generation and DNg100/DNb08 pathways;
- Sapkal et al. (bioRxiv 2026 preprint): sensory-deprivation, central coupling and task-dependent inter-leg coordination;
- FlyGym 2.1.1 and Symtropy's articulated-body/contact models: embodied mechanics and closed-loop environment;
- Symthaea #10670: decision-critical intervention selection;
- Symthaea #10922: source-bound six-leg phase-response measurement algebra.

The benchmark gap is not another per-leg frequency score. It is the **joint, intervention-conditioned discrimination** between neural coupling, peripheral feedback, mechanics and common drive, while preserving an honest non-identifiability result.

## 6. Promotion gates

Synthetic:
- exact equality partition agreement;
- 100% rejection of duplicate/incomplete intervention matrices;
- 100% exclusion of infeasible interventions from equivalence resolution;
- no mechanism certificate when any relevant pair remains equivalent;
- summary collision adversary must preserve mean frequency and aggregate phase locking while changing the full phase trajectory;
- exact-horizon and source digest mutations fail closed.

Biological, after exact source reproduction and held-out validation:
- held-out per-leg and pairwise phase-response distributional calibration;
- tripod/tetrapod occupancy calibration and full-trajectory scores;
- animal/session-level uncertainty and effective replicate accounting;
- query-critical mechanism recall and false-attribution rates at matched familywise alpha;
- cost per resolved decision-critical alternative, including reviewer, simulator, acquisition and assay time;
- all failed, unavailable, non-response, out-of-support and infeasible outcomes remain in denominators.

No biological target is promoted to a result until source-bound replication and a blinded lockbox are available.

## Additional embodied and mechanosensory comparators

- Özdil et al., ICLR 2026 introduce a 3D Drosophila leg musculoskeletal model in OpenSim/MuJoCo, with Hill-type muscles estimated from high-resolution X-ray scans and muscle-actuated behavioral replay. It is a strong muscle/limb mechanics comparator, not a connectome-derived neural controller: https://proceedings.iclr.cc/paper_files/paper/2026/hash/222843b731e2c7813291586e2b39fe89-Abstract-Conference.html
- FlyGym / NeuroMechFly 2.1.1 (released 24 October 2026) exposes joint angles, actuator forces and contact forces and includes GPU simulation state access and a split trochanter/femur model. Pin the exact release, assets, MuJoCo/Warp and solver settings: https://github.com/NeLy-EPFL/flygym/releases and https://neuromechfly.org/2.1.1/
- Kuo et al., bioRxiv preprint (16 September 2026), reports localized leg contact, subsequent-leg-contact patterns, landing probability, and connectome/optogenetic analysis of mechanosensory pathways including ascending Moonlander neurons. This is a separate landing-transition task and remains a preprint: https://doi.org/10.64898/2026.09.14.751322
- FlyBody (Nature 2025) supplies a 67-body / 102-DoF whole-body physics reference with learned controllers and proprioceptive feedback. It is a strong embodiment baseline, not a source-grounded neural mechanism: https://www.nature.com/articles/s41586-025-09029-4

Keep walking rhythm, inter-leg coordination, mechanosensory landing transition, and embodied muscle/physics evaluation as separate strata. Do not merge them into one accuracy number.


## 7. Statistical and biological promotion boundary

This exact-signature crate is a **cheap deterministic prefilter**, not the finite-sample mechanism-identifiability certificate.

The full promotion chain is owned by existing work and must remain separate:

1. **Exact signature prefilter (this PR):** find identical versus different deterministic synthetic fingerprints under the frozen intervention registry. A difference is only a candidate for statistical separation; equality is a strong counterexample within this declared fixture.
2. **Nuisance-family distributional and finite-sample test (#10737 / PR #10739):** convert the candidate model worlds into complete post-observation outcome distributions, include indicator/sensor/artifact nuisance profiles, compute robust total-variation separation, and calculate exact finite-sample testing risk at a declared independent-replicate budget. Nonzero TV alone is not enough to claim adequate finite-sample power.
3. **Held-out animal transport and calibration (#10752 / PR #10753):** evaluate calibrated response predictions on genuinely held-out animals and conditions, with animal-level aggregation, missing/indeterminate/OOD states and abstention. Frames, legs, neuron pairs and repeated trials are not independent animals.
4. **Source-bound mechanism-to-cohort bridge (#10792):** replay both owners and require explicit mapping from mechanism ID to transport model ID, selected intervention to exact measurement condition, and model observation kernel to source-bound lockbox observations. A model-family predictive win is not biological mechanism identification.
5. **Sequential risk and stopping (#10831 / #10959 / #10961):** use the existing coherent joint generative-model policy and familywise inference layers for adaptive repeated measurements and multiple mechanism claims. Do not silently switch nuisance identities or treat running suprema as e-values.

Accordingly, V1 emits \`SYNTHETIC_EXACT_SIGNATURE_PREFILTER_ONLY\` and one of:
- \`EXACT_SIGNATURE_EQUIVALENCE_REMAINS\`: at least one declared pair has identical exact signatures under the feasible portfolio;
- \`PAIRWISE_DISTINCT_EXACT_SIGNATURES\`: every declared pair has a different exact fingerprint under that portfolio.

Neither status is a biological mechanism certificate. \`mechanism_certificate_issued\` is always false.

## 8. Drosophila composition contract

The biological evaluation must keep four different task/evidence strata:
- Pugliese et al. peer-reviewed connectome-derived rhythm generation and DNg100/DNb08 pathway evidence;
- Sapkal et al. preprint walking inter-leg coordination;
- Kuo et al. preprint mechanosensory flight-to-landing transition;
- FlyGym 2.1.1, FlyBody and Özdil et al. ICLR 2026 body/muscle/contact mechanics.

The candidate mechanism family is non-exclusive in biological evaluation. Local CPG, central coupling, proprioceptive/load feedback and body/environment mechanics may coexist. Add direct, polysynaptic, extrasynaptic, shared-drive, state-mediated and observation-kernel alternatives from #10737; do not force a single winner if a mixture remains plausible.

The complete promotion statement must separately report:
- query decision status;
- deterministic-signature prefilter status;
- finite-sample mechanism discrimination risk at the declared animal-level sample size;
- held-out animal calibration/transport status;
- biological mechanism status (which remains unestablished by prediction alone).

A conclusion may legitimately be \`QUERY_IDENTIFIED_MECHANISM_UNIDENTIFIED\`, \`NOT_ESTIMABLE\`, or \`OUT_OF_DOMAIN\`. Never upgrade \`PAIRWISE_DISTINCT_EXACT_SIGNATURES\` to \`QUERY_AND_MECHANISM_IDENTIFIED\` without the later statistical, biological and source-bound gates.
