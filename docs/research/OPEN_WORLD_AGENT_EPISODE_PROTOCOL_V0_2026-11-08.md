# Blind open-world research episode protocol v0

Date: 2026-11-08  
Status: executable protocol implementation in draft PR #11060; no agent-performance result claimed.

## Purpose

The three reference worlds now have a common agent-facing interaction contract. This closes the gap between evaluator-side synthetic simulators and a future agent adapter. The interface gives the agent a public task manifest, a bounded intervention API, returned observations, and a terminal structured claim. It does not give the agent the generating model or evaluator oracle.

## Protocol loop

```mermaid
flowchart TD
  E[Evaluator creates hidden episode] --> V[Public AgentView]
  V --> A[Agent selects one action]
  A -->|Probe| W[Runner executes one intervention]
  W --> O[Observation + evidence ID + updated prefix digest]
  O --> V
  A -->|Submit| C[Validate claim, citations and prefix digest]
  C --> S[Terminal submission receipt]
  S --> R[Evaluator computes model and structure scores]
```

The agent receives only `AgentView` and returns `AgentAction` values. A probe consumes one budget unit and produces one observation. A submission closes the episode. The runner rejects wrong-family interventions, out-of-budget probes, unsupported hypotheses, duplicate model claims, malformed dispositions, duplicate or unknown evidence references, and stale evidence-prefix digests.

## A0/A2 treatment integrity

The public manifest identifies the treatment:
- **A0:** one correct structural family is supplied; the parameter, initialization and sensor values remain hidden.
- **A2:** the structural family is withheld.

Evaluator constructors reject an A0 manifest whose supplied structure does not match the generating world's actual structure. A0/A2 agents must receive matched observation budgets, task seeds, interventions and environment parameters for a valid comparison. This protocol does not itself perform that randomization or seed assignment; the external benchmark harness must own it.

## Evidence binding

Each returned observation receives a monotonically assigned evidence ID (`E0001`, ...). The agent cites only IDs it actually received. Every submission must echo the SHA-256 evidence-prefix digest returned by the latest view. The evaluator verifies that the digest matches the exact task/observation prefix at submission time.

The digest is computed over the v0 Rust `serde_json` serialization of the public task and observation list. It is a versioned protocol digest, not an assertion that arbitrary JSON serializations from every programming language will hash identically. Cross-language harnesses should echo the digest rather than independently reconstruct it until a canonical cross-language serialization is standardized.

## Claims and scoring

A claim is a set of compatible typed hypotheses plus one disposition: identified, ambiguous, no-compatible-model, or abstain. It is not forced to nominate a single equation when the evidence leaves multiple models compatible.

After submission, the trusted runner can compute:
- exact parameterized-model version-space match;
- projected structural-family set match and structural identifiability, scored independently from coefficient/sensor identifiability;
- claimed/oracle/intersection counts (supporting precision and recall calculation);
- disposition correctness;
- safe abstention when the exact compatible model set is non-singleton, versus unnecessary abstention when the exact model is uniquely identified;
- evidence-reference validity;
- action count and cited-evidence count.

The score is evaluator-side and must not be returned to the agent before submission. The agent process must be isolated from the runner process. The Rust type boundary and private evaluator state are not an operating-system security sandbox.

The evaluator uses separate trace-filter implementations and does not call the agent-facing `discover()` or `predict()` functions. It shares the typed candidate-family definitions, but independently recomputes which members fit the observed trace. Differential tests compare evaluator and inference results over every candidate model in all three finite worlds; negative controls verify that omitted or incompatible claimed models do not pass exact-set scoring. This is implementation-level independence, not a fully independent benchmark implementation or external oracle.

## Process-facing JSONL adapter

The standalone `open_world_agent` executable accepts one serialized `AgentView` per input line and emits one serialized `AgentAction` per output line. It supports the finite enumerator and the explicit probe-policy baselines via a fixed CLI strategy selector. It emits no human-readable output on stdout, checks the protocol version and open status, and bounds each input frame to 1 MiB.

This closes the process-facing wire-format gap for the **reference baselines**. It does not yet connect the learned Symthaea research orchestrator, and it is not an OS sandbox. A real benchmark runner must execute the agent under a separate restricted identity/container and deny access to evaluator state, hidden ledgers, host filesystem, network, parent process memory and pre-submission scores.

## Executable coverage

The protocol module covers the boolean hidden-lag world, numeric direct/leaky-memory world, and two-input interaction world. Tests cover:
- no truth-field serialization in A2 AgentView;
- intentional disclosure of supplied structure in A0 without disclosing sensor truth;
- one-step budget accounting and terminal submission;
- wrong intervention family rejection;
- supported-model and evidence-reference validation;
- exact-prefix binding and stale-prefix rejection;
- matched A0 structure validation;
- full-factorial interaction episodes and evaluator scoring.

## Open-support evaluation mode

The default Boolean, numeric and two-input constructors are **closed-candidate-grammar** episodes. If the evaluator's independent version-space filter omits the generating truth, post-submission scoring fails with `GeneratingTruthMissingFromOracle`; a verifier or candidate-catalogue omission must not masquerade as a valid empty version space.

The evaluator now reports both `generating_model_in_version_space` (exact parameterized-model identity) and `generating_truth_behaviorally_represented` (at least one legal candidate has the same complete length-4 output signature over the finite numeric intervention language). These are not synonyms. Gain 4 + identity sensor is outside the literal candidate parameter set but is observationally equivalent to gain 2 + double sensor. The correct report is exact-membership=false, behavioral-representation=true and a singleton candidate/equivalence-class claim. Gain 7 + identity sensor has no such candidate: both flags are false and the correct exact compatible set is empty.

The evaluator-only `BlindEpisode::new_numeric_open_support` constructor is a distinct F1 parameter-support control. It permits a valid world-generator gain in 1..=16 while the agent-facing candidate grammar remains {1, 2, 3}. The agent sees the ordinary numeric task contract and receives no OOD flag. When the response is incompatible with all declared candidates, the exact empty version space and `NoCompatibleModel` disposition can be scored as correct; the evaluator separately records `generating_model_in_version_space = false`.

Do not use the open-support constructor for the closed-grammar finite census. It is a bounded parameter-support extrapolation control, not proof of structural-family discovery.

A separate evaluator-only `BlindEpisode::new_numeric_structural_shift` constructor now adds a bounded nonzero quadratic response term (q u^2) to an otherwise direct-response world while keeping the public candidate catalogue unchanged. The current test uses gain 2 and q=1, producing [-1, 0, 3] for inputs [-1, 0, 1]. The independent evaluator signature includes that extra term and compares it against the original complete candidate grammar. The correct claim is the exact empty version space with `NoCompatibleModel`; there is no behaviorally equivalent legal candidate. The constructor rejects q=0 and non-direct-response bases.

This is a deliberately small **F2 structural-support fixture**, not open-ended grammar discovery or model repair. It does not yet test arbitrary missing terms, latent variables, interactions outside the candidate language, changes in measurement process, dynamic regimes or stochastic discrepancies. F3–F5 remain future evaluator generators under #11197.

## Parent-runner / child-agent transport

The library module `process` implements a JSONL process bridge around the same blind protocol. The trusted parent owns the `BlindEpisode` and evaluator score. It starts a child with cleared inherited environment, explicit arguments and an explicit working directory. The child receives one serialized public `AgentView` frame and returns one `AgentAction` frame for each action.

The process receipt binds:
- executable-byte SHA-256, argument vector, working directory and frame limits;
- each exact serialized `AgentView` and `AgentAction` JSONL frame SHA-256, including delimiters;
- the evidence-prefix digest before and after each action;
- the terminal action, child exit code and post-submission evaluator score.

Transport bounds are 1 MiB by default, 16 MiB maximum per frame and 1,024 maximum probes per process episode. Blank/malformed/oversized frames, unsolicited trailing stdout after terminal submission, episode budget violations and non-zero exits fail closed.

**Security boundary:** this is process separation and an evidence-binding transport, not a security sandbox. It does not impose CPU/memory/wall-clock limits or restrict filesystem/network access. Untrusted agents still require an external OS/container isolation boundary. The executable hash does not commit to shared libraries or the complete runtime image, and does not authenticate the host.

The `process_episode_smoke` executable drives matched-world A0/A2 arms through the same child binary and policy arguments. It verifies exact set scoring, generating-model retention, evidence references, frame hashes and action-budget compliance. The default child is the finite reference policy; this smoke test is not a learned Symthaea researcher evaluation.

## Finite enumerator reference policy

`policy::FiniteEnumeratorBaseline` consumes only `AgentView` values and emits one probe or a terminal submission. It uses exact consistency filtering over the declared finite model family and the finite partition heuristic. This provides a transparent reproducible reference against which the learned Symthaea research policy can be compared; it is not itself learned and does not demonstrate scientific discovery beyond the finite grammar.

`strategies::ProbeStrategy` makes the experimental-design comparison explicit: maximum partition count, minimax worst-case survivors, maximum entropy, minimum expected remaining candidates under a uniform prior, fixed-grid and seeded-random selection. `policy::StrategyEnumeratorBaseline` pairs these choices with the same inference/submission logic. The exact expected-survivor statistic is the sum of squared outcome-partition sizes divided by the candidate count; entropy and expected survivors are uniform-prior controls, not calibrated Bayesian posteriors or decision-focused EVSI. The random control is deterministic under its seed.

Its current policy tests exercise paired numeric A0/A2 episodes using the same generating model and probe budget, plus a full-factorial interaction discovery episode. The evaluator separately asserts that the hidden generating model belongs to the computed version space. This is a test harness for the protocol and baseline, not a reported benchmark score.

## Current limitations / next falsifiable milestones

This is an interface and protocol-conformance layer, not an evaluated research agent. The evaluator score currently uses exact finite candidate sets and reports structural-family recovery separately from full parameterized-model recovery. It scores abstention as safe only when the exact reported version space is complete and non-singleton; abstention on a uniquely identified exact model is unnecessary. This is not yet a probabilistic posterior score, decision-focused EVSI, support-omission benchmark or continuous-world verifier. It does not add stochastic noise, drift, saturation, unknown delay, hidden spatial state, adaptive instrument selection, or external paper-derived tasks.

Next milestones:
1. Connect a Symthaea research-policy adapter that consumes only serialized AgentView messages and emits AgentAction messages.
2. Freeze the incumbent policy/config and preregister A0/A2 runs with evaluator-owned seeds.
3. Compare random, fixed-grid, partition-maximizing, entropy and decision-focused EVSI policies under the same inclusive budget.
4. Add noise and instrument transformations with proper scoring rules and calibration.
5. Replicate one external benchmark task family before making any cross-benchmark performance claim.
