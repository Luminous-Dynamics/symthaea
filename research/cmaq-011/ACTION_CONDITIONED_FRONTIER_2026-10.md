# Action-conditioned world-model evaluation frontier (2026-10-31)

This note updates the external comparison strategy for CMAQ-011. It separates four neighbouring but non-identical evaluation problems:

1. **Fixed-action rollout fidelity:** given one initial observation/history and one supplied action sequence, does the generated rollout follow that sequence and match the realized physical response?
2. **Action-effect / counterfactual fidelity:** from one identical history, does the model predict how outcomes change under alternative interventions, and rank those alternatives correctly?
3. **Closed-loop decision value:** does access to the model improve realized control outcomes over a direct policy when budgets are matched?
4. **Policy improvement from imagined branches:** can rewind/branch comparisons be used to improve the controller without teaching it to exploit simulator/model artefacts?

No one benchmark below covers all four.

## External benchmark and algorithm map

| Work | What it measures / contributes | How CMAQ should use it | Important boundary |
| --- | --- | --- | --- |
| **WorldSimProbe** | Five released action-conditioned video diagnostics: local action calibration; global trajectory coverage; behaviour-source preservation; interaction grounding; interaction dynamics. Tests whether the causal chain from supplied action to robot motion/object response is preserved. | First external visual manipulation adapter candidate. Keep all five task scores separate. Report Task 1 response-ratio calibration; Tasks 2–3 flow/action-source fidelity; Task 4 hallucinated interaction; Task 5 primitive accuracy. | Public test inputs and a separate local validation set are released; hidden official scoring references remain evaluator-side. It is a video/action-following suite, not a full same-history set of alternative physical interventions. |
| **WorldOracle** | Action-conditioned generated rollouts are evaluated against paired real executions using the same initial observation and action sequence; measures physical-state correctness, interaction alignment and multi-view coherence over longer episodes. | Candidate fixed-action real-execution fidelity track, once a public and licensed evaluation split is actually available. | Current validation/test ground-truth datasets are labelled private/pending release. Do not access, mirror, or infer their hidden labels. No adapter until official public release/access policy and data license are verified. |
| **MiraBench** | Human-judgement-oriented evaluation of action-following fidelity and optimism bias across multiple world-model configurations. It probes whether the model obeys the requested action and overstates plausible outcomes. | Use as an external action-following/optimism diagnostic reference; compare definitions and failure taxonomy, not raw scores across incompatible datasets. | Human annotation protocol and model/task interface differ from CMAQ's deterministic paired causal branches. |
| **Rollback** | A policy-improvement method using rewind/branch/counterfactual comparisons to obtain action-level supervision and improve robot policies. | Later downstream algorithmic comparator for a *frozen* counterfactual evaluator; report policy lift over its strongest eligible direct baseline. | It is an algorithm/method paper, not a benchmark metric or substitute for verifying branch truth. Its reported task gains must not be treated as a portable baseline score without reproducing its task/protocol. |
| **RoboWM-Bench** | Video-to-executable-action evaluation: generated motion must be grounded in a physically executable action sequence in the simulator/robot task. | Add as a separate manipulation/action-execution lane after the task interface, license and evaluator are pinned. | It tests executable action translation, not the same estimand as model-based action-effect ranking. |

### Pin: WorldSimProbe public code release

- Repository: https://github.com/pxxq25/WorldSimProbe
- Exact source commit: `6df13422cef04ea41492a5c662bd7a60b9e64f1a` (2026-09-30).
- License: MIT, as of the pinned commit.
- Public project page: https://evophys.com/WorldSimProbe/
- Public submission inputs: https://huggingface.co/datasets/petersonco/worldsimprobe
- Public local validation references: https://huggingface.co/datasets/petersonco/worldsimprobe_validation
- Hidden official scoring references and private simulator metadata stay on the official evaluator; the public repository explicitly excludes them.
- Pinned contract files at the commit:
  - `README.md` blob `de3e005d858febdd44acc7b7e63f38f1d8d1f341`
  - `docs/tasks.md` blob `3c7f8d7e5f3b139faac70ad9a09a4a8ae480c784`
  - `docs/metrics.md` blob `c14a13c4a3f997d7707bf1a66c1c9da263d7805d`
  - `docs/evaluator_setup.md` blob `2c7c2c5662878aa60537d9d4ad0aae4161e3fbcd`
  - `LICENSE` blob `fe9aea5ecfde5e168f19009e890549e4cd95f98f`
  - `pyproject.toml` blob `a4499fea6b4996dc23446c815a8fb745834c2022`

WorldSimProbe task details confirmed in the pinned task/metric files:

- Task 1 compares the model's small/large action-perturbation response ratio with a hidden simulator response ratio.
- Tasks 2–3 use robot-masked optical flow, with Task 3 distinguishing expert/policy/human control sources; synthetic human-like profiles are explicitly not treated as human teleoperation.
- Task 4 checks distractor/fake-contact/proximity interaction hallucinations and requires both object stability and robot-arm motion.
- Task 5 asks a VLM to classify eight interaction primitives on a shared physical-time sampling grid, with motion agreement checks.

The evaluation worker needs CUDA plus RobotSeg, DPFlow/PTLFlow and TAPNext++ for Tasks 2–4; Task 5 currently uses `Qwen/Qwen3-VL-8B-Instruct`. The pinned setup document names RobotSeg revision `dafb8c0d507276e2f96d2b07ac3661a7b3a41a5f`, TAPNet revision `c2cbab81cc06092b5f05bfe2da7bfec54e2079c9`, PTLFlow `0.4.1` and Hydra `1.3.2`. Pin model/checkpoint revisions and all transitive runtime packages in the adapter's own lock before attempting a reproduction. The repository's MIT license does not relicense third-party checkpoints or simulator assets.

## What is new in the CMAQ measurement design

CMAQ should connect these external tracks without collapsing their estimands:

- **CMAQ-011:** matched closed-loop utility and compute/cost frontier, including D0/D1/W1/W2/W3. Paper-native and wrapper-normalized comparisons remain distinct.
- **CMAQ-017:** adaptive planner-induced state/action distribution shift and multi-step exploitability; measures when more model-based search makes real outcomes worse.
- **CMAQ-018:** same-history alternative-action effects and branch ranking. The candidate is given the same observed history and action interface; it must predict `Y(do(a_i)|h)-Y(do(a_j)|h)`. This is not inferred from one fixed action sequence or observational logs.
- **WorldSimProbe / WorldOracle adapters:** external action-following and paired physical rollout fidelity, only on the public and licensed splits.
- **Rollback-style comparison:** downstream policy-improvement algorithm, evaluated only after the branch evaluator is frozen and qualified.

This gives a traceable ladder from action fidelity → interventional effect fidelity → decision advantage → policy improvement.

## External comparison protocol

For any adapter:

1. Freeze exact repository commit, public dataset revision, simulator build, wrapper stack, action cadence, observation transforms, frame rate/timestamps, task set and reward/metric implementation.
2. Keep each benchmark's own native score and split unchanged for reproduction.
3. Add CMAQ-normalized comparisons only when the environment permits identical reset/state, action definitions, horizon, intervention opportunity and common random-number pairing. Label adapted runs clearly.
4. Keep hidden validation/test data hidden. Public validation is for local evaluator testing, not permission to access private leaderboard references.
5. Report every task/primitive/failure family separately; do not create a pooled number that hides a contact, termination or action-following failure.
6. Record model/video generation cost, candidate branches, simulator steps, evaluator forward passes, latency p50/p95, memory and GPU/energy where measurable.
7. Preserve unsuccessful outputs, invalid actions, terminal failures and evaluator errors in the assigned denominator.

## Decision on immediate work

WorldSimProbe is the first practical external video/action-conditioning adapter candidate because its code, MIT license, public submission format, task definitions and a local validation set are available at a pinned revision. It should be integrated as a **separate visual track**, not silently merged with MRS.Q/WIMLE DMC scores or the synthetic CMAQ-018 branch results.

WorldOracle is a high-value follow-on, but its private validation/test ground truth makes an immediate public adapter inappropriate. Keep its release/access status as a hard stop.

## Claim ceiling

The source pins and adapter plan are not a reproduction. A public test submission is not a leaderboard result. An adapter smoke pass is not a matched comparison. No model/SOTA claim is permitted until exact-head execution, compatible baselines, held-out splits, compute accounting and independent reproduction are complete.
