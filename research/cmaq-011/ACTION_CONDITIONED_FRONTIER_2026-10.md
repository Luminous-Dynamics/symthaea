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
| **WorldOracle** | Action-conditioned generated rollouts are evaluated against paired real executions using the same initial observation and low-level action sequence; measures physical-state correctness, interaction alignment and multi-view coherence over long, successful and failed episodes. | **Highest-priority external real-execution track:** reproduce on its disclosed validation set, then submit to the official withheld leaderboard test through the portal. | The official repository announces the validation release (Oct 15, 2026), but the Hugging Face dataset card still contains stale/private wording. The disclosed validation GT is described as 32 AgiBot G2 episodes (20 human-operated + 12 policy-rollout; ~18.6 GiB). Confirm data access and CC BY-NC-SA 4.0 terms before use. Test inputs may be released, but test GT remains withheld and evaluated by the official service. |
| **MiraBench** | Human-judgement-oriented evaluation of action-following fidelity and optimism bias across multiple world-model configurations. It probes whether the model obeys the requested action and overstates plausible outcomes. | Use as an external action-following/optimism diagnostic reference; compare definitions and failure taxonomy, not raw scores across incompatible datasets. | Human annotation protocol and model/task interface differ from CMAQ's deterministic paired causal branches. |
| **Rollback** | A policy-improvement method using rewind/branch/counterfactual comparisons to obtain action-level supervision and improve robot policies. | Later downstream algorithmic comparator for a *frozen* counterfactual evaluator; report policy lift over its strongest eligible direct baseline. | It is an algorithm/method paper, not a benchmark metric or substitute for verifying branch truth. Its reported task gains must not be treated as a portable baseline score without reproducing its task/protocol. |
| **RoboWM-Bench** | Video-to-executable-action evaluation: generated motion must be grounded in a physically executable action sequence in the simulator/robot task. | Add as a separate manipulation/action-execution lane after the task interface, license and evaluator are pinned. | It tests executable action translation, not the same estimand as model-based action-effect ranking. |

### Pin: WorldOracle public validation release

- Official repository: https://github.com/WorldOracle-benchmark/WorldOracle
- Exact source commit: `8636e00ba221ed031a8fba5f7384116a8b10b5fa`.
- Code license: Apache-2.0.
- Official release announcement: Oct 15, 2026; online submission portal opened Oct 16.
- Validation dataset: https://huggingface.co/datasets/WorldOracle/worldoracle-validation
- Exact observed validation dataset revision: `0ada1ba08b905de0801b7ffd6bdfb4dc4da761ef` (2026-10-29; pin current files and SHA-256 manifest before any local evaluation).
- Data license on the Hugging Face dataset card: CC BY-NC-SA 4.0. This is non-commercial/share-alike data licensing; it is not the same license as the Apache-2.0 evaluation code. Do not assume that code-license permissions apply to the videos/annotations or to commercial model training.
- Officially documented current AgiBot G2 validation split: 20 human-operated episodes (five each across `hangtowel`, `pickfruit`, `pourwater`, `pulldrawer`) and 12 policy-rollout episodes (four each across `hangtowel`, `pickfruit`, `pourwater`). Disclosed ground truth totals approximately 18.6 GiB. Other platform entries may be present in the dataset manifest; do not infer their task counts from the AgiBot-only README table.
- Test-set inputs: https://huggingface.co/datasets/WorldOracle/worldoracle-testset. The 268 test inputs are for inference only; test GT remains private and leaderboard-evaluated.
- Official submission portal: https://worldoracle-benchmark.github.io/submission/
- Pinned source README blob: `c7d3c7e814ce99e86a304a8da5ae7d7027bf4717`.
- Pinned validation results blob: `db184e72e4eb7aa43c1272f5ba7048adb1c5e698`.

**Published public-validation baselines (WorldOracle validation results file):**

| Track / embodiment | GE-Sim 2.0 SV | GE-Sim 2.0 MV | Next reported SV baseline |
| --- | ---: | ---: | --- |
| WorldOracle-Core / AgiBot G2 | 72.25 | 78.50 | Cosmos 3: 54.83 |
| WorldOracle-Core / YAM | 69.05 | 68.50 | Cosmos 3: 56.64 |
| WorldOracle-Policy / AgiBot G2 | 65.99 | 73.59 | IRASim: 42.10 |
| WorldOracle-Long / AgiBot G2 | 61.04 | 69.98 | IRASim: 37.83 |
| WorldOracle-Long / YAM | 52.87 | 51.97 | IRASim: 35.65 |

These are reported public validation results, not hidden test-set or live-leaderboard results. The official repository warns that the validation set is substantially smaller and less diverse than the test set and that its results can have considerable variance. The paper used GPT-5.5 for physical-state judging; the live leaderboard re-evaluated baselines and uses GPT-5.6 Luna. **Do not compare a local run across evaluator versions as though the score were identical.** Freeze the exact code/dataset revision, physical-judge endpoint/model, and all metric settings.

Before adapter execution:
1. Verify the HF dataset's currently downloadable files and SHA-256 manifest against the recorded revision.
2. Review and accept the data terms for this project/use. CC BY-NC-SA 4.0 is not a permissive commercial-training license.
3. Pin the benchmark code, V-JEPA2/DINOv3 checkpoints and source commits, VLM evaluator, Python/CUDA dependencies, and task/distribution manifests.
4. Start with a one-episode smoke test and local validation scoring, then reproduce published baselines on the exact split before any model comparison.

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

WorldOracle is now the highest-value external real-execution adapter because the official code and a disclosed validation split are released. The contradictory Hugging Face card wording makes access/license verification the first gate, not a reason to misclassify the entire validation release as private. Test GT remains withheld.

## Claim ceiling

The source pins and adapter plan are not a reproduction. A public test submission is not a leaderboard result. An adapter smoke pass is not a matched comparison. No model/SOTA claim is permitted until exact-head execution, compatible baselines, held-out splits, compute accounting and independent reproduction are complete.
