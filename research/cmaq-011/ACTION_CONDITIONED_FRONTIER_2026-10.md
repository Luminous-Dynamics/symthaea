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
| **WBench** | 39 video world models, 5 dimensions and 22 metrics; 289 cases / 1,058 turns with a 158-case Navi subset. Supports text, camera/6-DoF and discrete-action interfaces. | Add as a broad multi-turn video/navigation and action-conditioning quality track. Keep quality, consistency, interaction, setting and physical dimensions separate; action-conditioned comparisons use only the discrete-action Navi interface. | Mixed outputs and judge provenance matter. Latest VWA-World Navi 81.7 is 16 submitter-evaluated plus 3 organizer-evaluated VLM metrics; do not treat this as fully independent organizer scoring. Scores across interface types or VLM judges are not directly comparable. It does not measure physical paired execution or exact alternative-action effects. |
| **WorldOracle** | Action-conditioned generated rollouts are evaluated against paired real executions using the same initial observation and low-level action sequence; measures physical-state correctness, interaction alignment and multi-view coherence over long, successful and failed episodes. | **Highest-priority external real-execution track:** reproduce on its disclosed validation set, then submit to the official withheld leaderboard test through the portal. | The official GitHub README announces a validation release and documents an AgiBot G2 subset, but the current Hugging Face validation card labels the dataset private. The observed manifest revision lists 67 episodes across AgiBot (20), YAM (15), Human Hand (20), and Policy (12), with about 60.6 GB total data. The GitHub README's 20 human-operated + 12 policy-rollout breakdown is the AgiBot G2 slice, not the whole corpus. Data are CC BY-NC-SA 4.0. **Current effective access is unresolved/restricted: do not download or use the data until explicit authorized access reconciles the GitHub release announcement with Hugging Face visibility.** Test inputs (268 episodes) remain private and contain no ground truth; test GT remains withheld and official-service evaluated. |
| **MiraBench** | Human-judgement-oriented evaluation of action-following fidelity and optimism bias across multiple world-model configurations. It probes whether the model obeys the requested action and overstates plausible outcomes. | Use as an external action-following/optimism diagnostic reference; compare definitions and failure taxonomy, not raw scores across incompatible datasets. | Human annotation protocol and model/task interface differ from CMAQ's deterministic paired causal branches. |
| **Rollback** | A policy-improvement method using rewind/branch/counterfactual comparisons to obtain action-level supervision and improve robot policies. | Later downstream algorithmic comparator for a *frozen* counterfactual evaluator; report policy lift over its strongest eligible direct baseline. | It is an algorithm/method paper, not a benchmark metric or substitute for verifying branch truth. Its reported task gains must not be treated as a portable baseline score without reproducing its task/protocol. |
| **ActiveArena** | 35 active-perception tasks across five categories, multi-round visual evidence acquisition, memory-based reasoning and ID/OOD splits. Includes 13 VLA configurations for controlled ablations. | External active information-acquisition reference for CMAQ-002 EVoI and VIS-DVPC viewpoint/compute-allocation work; report evidence cost, task success, memory use and OOD transfer separately. | Active perception / information acquisition, not an action-conditioned world-model rollout or same-history causal effect benchmark. Verify data access, code revision and license before adapter work. |
| **DEEP** | Decision Evaluation for Escalated Perception; evaluates when escalating from cheap to expensive perception improves downstream decision loss under latency/energy budgets. | Cross-domain methodology reference for VIS-DVPC and CMAQ cost accounting: all-cheap, all-full, random, fixed and learned allocators plus an evaluator-only budgeted oracle; charge allocator overhead. | Perception-compute allocation and downstream planning, not a generic world-model benchmark. Results on KITTI/nuScenes/nuPlan do not automatically transfer to Symthaea. |
| **RoboWM-Bench** | Video-to-executable-action evaluation: generated motion must be grounded in a physically executable action sequence in the simulator/robot task. | Add as a separate manipulation/action-execution lane after the task interface, license and evaluator are pinned. | It tests executable action translation, not the same estimand as model-based action-effect ranking. |

### Pin: WBench public benchmark release

- Repository: https://github.com/meituan-longcat/WBench
- Exact source commit: `edc4e80e642baac4f2b5764dec9f1a491200af8f`.
- Code/data license in repository: MIT; model weights retain their original licenses.
- Paper: https://arxiv.org/abs/2605.25874.
- Public data: https://huggingface.co/datasets/meituan-longcat/WBench.
- 289 cases / 1,058 interaction turns / four interaction types (navigation, subject action, event editing, perspective switching).
- The pinned repository README says 39 models; the live public leaderboard page observed 2026-10-31 lists 45 entries (13 text, 21 camera, 11 action). These are distinct snapshots and must not be silently merged.
- 158-case Navi subset for camera-conditioned (6-DoF) and discrete-action conditioned interfaces. Keep text-conditioned 289-case comparisons separate.
- Five dimensions / 22 metrics: quality (6), consistency (8), interaction (4), setting (2), physical (2). Exact metric names and source hashes are in `wbench-baseline.lock.json`.
- Live leaderboard page (last updated 2026-10-12, observed 2026-10-31) shows the overall Navi table led by Alaya-EVOKE-Turbo 82.0, XGEN-Jing 81.9, and Enuma 81.7; VWA-World is rank 4 at 81.7. The table includes multiple model interfaces, so these are not an action-only ranking.
- Published Oct 12 provenance examples:
  - VWA-World, action-conditioned, Navi, 158 cases: 81.7. Provenance states 16 submitter-evaluated metrics + 3 organizer-evaluated VLM metrics (causal fidelity, scene adherence, subject adherence); the 16 local metrics had aggregate schema/count checks only, with no per-case local scores supplied.
  - gd_world, camera-conditioned, Navi, 158 cases: 81.2, organizer evaluation. It is a camera interface, not an action-conditioned result.
- Judge model listed in the pinned provenance: `doubao-seed-2-0-lite-260215`. WBench warns that scores from different judges are not directly comparable.
- Before reproduction, pin judge version, exact dataset subset, generation frame/timestep contract, source model/checkpoint and original checkpoint license. Measure video generation and benchmark evaluation cost separately.
- These aggregate submissions are references, not independently reproduced results. No Symthaea score or SOTA claim exists.

### WBench capability findings (official live project page, observed 2026-10-31)

These findings are more useful than treating the aggregate ranking as a single target:

- No model dominates every dimension.
- Navigation and semantic interaction are distinct: strong navigation does not guarantee event-edit or subject-action competence.
- Camera movement does not guarantee subject/perspective consistency.
- The project reports video-quality/physical-correctness correlation of `rho = 0.82`, but near-zero correlation between control ability and physics scores. Visual quality can be a proxy for some physical metrics while being a poor proxy for controllability.
- Navigation accuracy falls by about 21 points between turn 1 and turn 4 in the aggregate analysis; this is direct evidence that errors compound across interaction turns.

The leaderboard page is last marked updated 2026-10-12 and was viewed 2026-10-31. Its 45 model entries are split into 13 text, 21 camera and 11 action interfaces. The all-types Navi ranking must not be presented as an action-only result. The official project page exposes type filters, but any action-only comparison must export/filter the exact action entries, preserve their per-metric sample counts and exclude mixed-provenance scores from independent-performance claims.

### Pin: WorldOracle public validation release

- Official repository: https://github.com/WorldOracle-benchmark/WorldOracle
- Exact source commit: `8636e00ba221ed031a8fba5f7384116a8b10b5fa`.
- Code license: Apache-2.0.
- Official release announcement: Oct 15, 2026; online submission portal opened Oct 16.
- Validation dataset: https://huggingface.co/datasets/WorldOracle/worldoracle-validation
- Exact observed validation dataset revision: `0ada1ba08b905de0801b7ffd6bdfb4dc4da761ef` (2026-10-29; pin current files and SHA-256 manifest before any local evaluation).
- Data license on the Hugging Face dataset card: CC BY-NC-SA 4.0. This is non-commercial/share-alike data licensing; it is not the same license as the Apache-2.0 evaluation code. Do not assume that code-license permissions apply to the videos/annotations or to commercial model training.
- Current Hugging Face validation manifest: 67 episodes total — AgiBot 20, YAM 15, Human Hand 20, Policy 12 — approximately 60.6 GB. The GitHub README's 20 human-operated + 12 policy-rollout breakdown describes the AgiBot G2 slice, not the full corpus.
- Test-set inputs: https://huggingface.co/datasets/WorldOracle/worldoracle-testset. The current Hugging Face card labels the 268-episode input dataset private; it contains no ground truth. Do not access or mirror it. Test ground truth remains withheld and leaderboard-evaluated.
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
4. Use only the public validation split under its CC BY-NC-SA 4.0 terms; keep private test inputs and hidden test ground truth inaccessible. Public validation does not grant permission for commercial use.
5. Report every task/primitive/failure family separately; do not create a pooled number that hides a contact, termination or action-following failure.
6. Record model/video generation cost, candidate branches, simulator steps, evaluator forward passes, latency p50/p95, memory and GPU/energy where measurable.
7. Preserve unsuccessful outputs, invalid actions, terminal failures and evaluator errors in the assigned denominator.

## Decision on immediate work

WorldSimProbe and WBench are separate visual/action-conditioning tracks. WBench is the broad multi-turn video/navigation suite; WorldSimProbe adds simulator-grounded local/global action-following and interaction diagnostics. Neither replaces WorldOracle's paired real-execution track, nor CMAQ-018's exact same-history alternative-intervention tests. Do not merge visual scores with MRS.Q/WIMLE DMC outcomes or synthetic CMAQ branch metrics.

WorldOracle is the highest-value external real-execution adapter candidate because its official evaluator code and protocol are released. The 67-episode validation manifest is documented by the official repository, but the current Hugging Face card marks the dataset private; effective access is unresolved and no dataset use is authorized by this lock. If access is explicitly granted, CC BY-NC-SA 4.0 still restricts use to its non-commercial/share-alike terms unless separate permission is obtained. The 268 test inputs remain private and test ground truth withheld.

## Claim ceiling

The source pins and adapter plan are not a reproduction. A public test submission is not a leaderboard result. An adapter smoke pass is not a matched comparison. No model/SOTA claim is permitted until exact-head execution, compatible baselines, held-out splits, compute accounting and independent reproduction are complete.
