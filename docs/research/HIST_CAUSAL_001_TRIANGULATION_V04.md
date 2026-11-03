# HIST-CAUSAL-001 v0.4: causal triangulation across mechanisms and apparatus

## Motivation

A randomized effect is not enough to identify a mechanism. The same primary measurement can arise from a genuine specimen response, a sensor artefact, partial actuation, latent confounding, or a common-mode calibration source shared by multiple apparent witnesses.

This v0.4 exact finite slice extends the historical-effect benchmark into general scientific experimentation. It is designed to interface with Symtropy SIM-REL-013 (joint specimen/sensor state), SIM-REL-017 (seeded challenge suite), SIM-REL-014 (paired stochastic benchmark), and SIM-REL-025 (action realization / apparatus state). It does not duplicate those planners.

## Six candidate models

1. Genuine specimen mechanism.
2. Latent common cause.
3. Primary sensor artefact.
4. Actuator non-compliance.
5. Common-mode reference artefact (a second instrument shares a biased calibration path).
6. Null effect.

The synthetic generator deliberately uses exact rational parameters so several mechanisms produce identical likelihoods under weaker measurement designs.

## Results

There are 15 unordered model pairs.

| Evidence available | Separated pairs | Unresolved |
|---|---:|---:|
| Observational primary measurement | 5/15 | 10 |
| Randomized primary measurement | 9/15 | 6 |
| Add secondary reference measurement | 12/15 | 3 |
| Add realized-actuation readback | 14/15 | 1 |
| Full witness set | 15/15 | 0 |

Critical hard cases:
- Randomization separates the genuine mechanism from a latent common cause.
- A primary sensor artefact exactly mimics the genuine model on the randomized primary measurement; the secondary reference separates it.
- Partial actuation exactly mimics the genuine model on both primary and secondary outcomes; actuator readback separates it.
- A common-mode calibration artefact mimics the genuine model on both primary and secondary outcomes; a blank/control measurement through the shared path separates it.

The exact weighted set-cover oracle finds a minimum-cost diagnostic set of 5 probes costing 23/4 = 5.75 arbitrary units: observational primary, randomized primary, actuator readback, sensor self-test, blank common-mode control. The stated costs are invented fixture weights, not experimental cost estimates.

## Qualification

- 11 unit tests pass locally.
- Every model/probe distribution sums to exactly one using rational arithmetic.
- The code uses only Python standard library for a transparent reference oracle.
- No physical data, real apparatus calibration or causal discovery is claimed.
- The exact result is conditional on the six candidate models and the finite declared probe menu.
- A complete investigation still requires model-misspecification tests and evidence that the supposedly independent sensors/controls are genuinely independent.

## Paradigm shift

Move from “replicate the observation with another instrument” to “triangulate across causal mechanisms and measurement pathways.” Two sensors are not independent merely because they have different names. Shared power, calibration, timing, software, operators, sample preparation or environmental conditions can create common-mode errors.

The scientific agent should plan a sequence of interventions that isolates:
- assigned versus realized action;
- specimen/process state versus sensor state;
- independent versus shared measurement paths;
- randomization versus observational association;
- expected effect versus negative-control behavior.

When those pathways cannot be distinguished, it must return an equivalence class and abstain.

## External benchmark alignment

EmbodiedLab (26 Oct 2026) tests agents that operate 30 quantitative experiments across five disciplines and 14 apparatus platforms; the reported failure analysis attributes most failures to apparatus interaction, not reasoning over readings already obtained. SafeLab (ICML 2026) evaluates 64 laboratory-manipulation tasks and reports unsafe-but-successful trajectories, arguing for a separate trajectory-level safety metric. AutoSciBench (4 Oct 2026) adapts tasks using solver trajectories and reports lower solver accuracy on generated tasks while receiving higher quality ratings. These motivate an adaptive challenge suite, not a claim of superiority.

Sources:
- https://arxiv.org/abs/2610.35385
- https://proceedings.mlr.press/v306/bai26e.html
- https://arxiv.org/abs/2610.05140
- https://github.com/HexaFuture/EmbodiedLab

## Implementation

The v0.4 reference and tests are the golden finite-model fixture. Rust production/verification is tracked separately in Symthaea #10081. Integrate this fixture into Symtropy SIM-REL-017 and #3021 as a separate causal-triangulation family, keeping its scores distinct from the joint physical/sensor state benchmark.

Claim ceiling: exact identifiability within six finite toy models and six probes. No empirical historical effect, quantum inference, Chain Theory derivation, physical law or SOTA win is claimed.
