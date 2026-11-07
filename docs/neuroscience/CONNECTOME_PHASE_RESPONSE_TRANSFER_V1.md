# Connectome phase-response transfer benchmark V1

**Tracking:** #10919 — SCI-CONNECTOMICS-099  
**Composes:** #9934, #9592, #9575, #10880, #10897 and PR #10898.  
**Status:** standalone Rust fixed-point measurement algebra and synthetic known-answer tests; no biological result is claimed.

## Objective and paradigm shift

Move from scoring steady-state coordination to measuring the dynamical response operator of coordination. The primary measurand is:

    G[i,j](phi,u,c,tau)
      = wrap(theta_i_perturbed(t0 + tau)
             - theta_i_matched_control(t0 + tau))

This six-leg phase-reset transfer tensor records how a perturbation to a sender leg or declared global input changes every recipient leg relative to a matched control, at a registered perturbation phase, class, context and delay.

The diagonal response is local phase reset. Off-diagonal responses describe system-level transfer. Neither is direct synaptic causation: shared input, central coupling, proprioceptive feedback, contact/load pathways, biomechanical coupling and latent state can produce similar signatures.

A model that matches mean stepping frequency and steady-state PLV can still have the wrong response to a local disturbance. The synthetic known-answer control makes that failure explicit.

## Evidence and source boundaries

Pugliese et al., *Connectome simulations identify a central pattern generator for fly walking*, Nature version of record (28 October 2026), establishes a source-bound connectome-driven rhythm result and a separate DNb08 experimental pathway. Its results motivate a distinction between rhythm generation, all-leg phase, gait coordination and physical locomotion; these endpoints must not be collapsed into a single score.

Sapkal et al., *Central versus peripheral neural control of a coordinated walking pattern in Drosophila*, bioRxiv v2 (preprint; not peer-reviewed), provides a useful challenge-generation structure for BALL_CONTACT, SLIPPERY_SURFACE, AIR_STEPPING and FeCO-silenced air-stepping conditions. It describes central inter-leg coupling alongside sensory, descending and contact/load modulation. Its findings are not independent confirmatory truth for this implementation.

## Registered contexts and probe classes

Contexts remain distinct:
- BALL_CONTACT
- SLIPPERY_SURFACE
- AIR_STEPPING
- AIR_FECO_SILENCED

Probe classes remain distinct:
- LOCAL_PHASE_KICK
- LOAD_OR_CONTACT
- PROPRIOCEPTIVE_CHANNEL
- DESCENDING_COMMAND
- MECHANICAL_BODY_OR_SUBSTRATE

A biological row must bind the exact source revision/materialization, animal/session, leg-axis identity, task/condition, phase estimator, perturbation receipt, delay and raw kinematic evidence. The experiment-design vocabulary is not authorization for any wet-lab procedure.

## Fixed-point semantics

Each phase sample preserves an integer cycle index and a signed fractional phase in microturns:
- one cycle = 1,000,000 microturns;
- canonical fractional interval = [-500,000, 500,000);
- unwrapped delta = (perturbed cycle index - control cycle index) × 1,000,000 + (perturbed fractional phase - control fractional phase);
- wrapped reset = wrapped fractional perturbed phase minus fractional matched-control phase;
- cycle slip = (unwrapped delta - wrapped reset) / 1,000,000.

A full-cycle advance can therefore have zero wrapped phase reset but a cycle slip of +1. A small fractional advance across the canonical fractional-phase boundary is not a cycle slip. The two quantities remain separate and are never collapsed into one scalar.

The canonical six-leg axis is left fore, left middle, left hind, right fore, right middle, right hind. Any source matrix/annotation ordering must be independently mapped to stable leg IDs before populating this axis.

Descending-command probe keys must carry an exact source identity: DNg100, DNg97 or DNb08. Every probe key also records a bounded magnitude bin; the receipt digest binds its mapping to physical dose. These are separate probe keys, not interchangeable labels for generic descending stimulation. The schema and Rust validator reject a descending-command probe without this identity.

For a measured row, the crate computes one wrapped phase reset and one cycle-slip count per leg. Non-response, failed assay, missing tracking, out-of-support and unassessed rows remain typed statuses and cannot be silently turned into zero responses.

## Synthetic known-answer tranche

The standalone no-dependency Rust crate under `tools/connectome-phase-response` provides:
- exact circular wrapping across cycle boundaries;
- typed six-leg source/recipient axes;
- local and off-diagonal phase-reset extraction;
- duplicate/missing probe grid rejection;
- unavailable measurement status preservation;
- same-steady-state-frequency/PLV versus different-phase-response model signatures;
- exact circular L1 response separation on identical probe grids.

The response-distance diagnostics are measured in microturns and cycle counts. They are not probabilities, likelihoods, effect sizes, causal coefficients or biological model scores. The crate reports per-probe separation and joint-campaign separation over the concatenated registered probe portfolio, so one phase bin can be blind while multiple probe cells jointly distinguish finite model signatures. This does not prove global identifiability.

## Model challenge matrix

The eventual source-bound comparison should include:
- LOCAL_CPG_ONLY;
- CENTRAL_INTERLEG;
- PROPRIOCEPTIVE_MEDIATED;
- CONTACT_LOAD_MEDIATED;
- BODY_MECHANICAL;
- DESCENDING_MODULATED;
- HYBRID_COMBINED;
- SHARED_INPUT / LATENT_STATE negative controls.

Every model must predict the same registered probe grid for a pairwise comparison. An omitted model row cannot improve a model's apparent separation. Distinguish identification of a model class, evidence about a mechanism, and direct causal attribution.

## Industry/SOTA comparison and target

Existing locomotor connectomics and CPG studies provide important structural reconstruction, neural rhythm and steady-state phase-locking outcomes. Optimal perturbation design (including Ogino et al., eLife version of record 13 October 2026) adds parameter-identification and waveform-design objectives. The proposed phase-response surface is complementary: it turns the perturbation outcome into a context- and phase-conditioned transfer operator across all legs.

Baselines at equal observation/perturbation budget:
1. mean frequency-only;
2. steady-state PLV/tripod score only;
3. pairwise phase correlation;
4. single-leg local oscillator;
5. pinned Pugliese rate-model reproduction;
6. central-coupled phase model;
7. proprioceptive/contact/load augmented models;
8. body-mechanical coupling model;
9. hybrid sensorimotor model;
10. random and expert-prior probe design;
11. #9592 Fisher D/A-optimal and EIG design;
12. #10880 robust model-discrepancy probe selection.

Primary outcomes:
- circular phase-reset error by source, recipient, phase, context and delay;
- local vs off-diagonal response calibration;
- recovery/re-lock time;
- held-out source-leg, phase-bin, condition and animal generalization;
- minimum pairwise response separation over the declared model challenge set;
- missing/non-response/out-of-support rate and complete denominators;
- all-in cost, preparation, reviewer and acquisition time.

Biological promotion thresholds must be chosen after a source-bound pilot estimates independent-animal variability and phase-estimator uncertainty. Do not set a headline percentage before that pilot.

## Execution boundary and next tranche

Run the standalone crate tests independently of the workspace. Next, add a source adapter only after #9934 resolves exact upstream artifact custody and #9943 proves stable axis identity. Then consume blinded trajectories and phase estimates, use #9592 to select probes and #10880 / #10897 for active discrepancy selection and sequential monitoring.

No direct synaptic causal, universal CPG or biological SOTA claim is licensed by this synthetic tranche.
