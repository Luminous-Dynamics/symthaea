# v15 synthetic result interpretation

Run: `python3 run_benchmark.py`
Scenario corpus: 12 authored synthetic episodes.
Tests: 6/6 pass.

| Arm | Safe episodes | Safe rate | Interventions | False interventions | Compute cost units |
|---|---:|---:|---:|---:|---:|
| No action | 2/12 | 16.7% | 0 | 0 | 0 |
| Reactive threshold | 3/12 | 25.0% | 9 | 0 | 0 |
| Predictive reflex | 4/12 | 33.3% | 5 | 2 | 31 |
| Shuffled-precursor control | 5/12 | 41.7% | 5 | 2 | 31 |
| Evaluator oracle | 11/12 | 91.7% | 11 | 1 | 0 |

Predictive vs reactive: one episode was safe only under predictive reflex; none were safe only under reactive threshold. The paired net difference is +1/12 (= +8.3 percentage points) on this authored corpus.

Predictive vs shuffled-precursor control: zero episodes favored predictive and one favored the sham; the paired net difference is -1/12 (= -8.3 percentage points). Thus this toy corpus **does not establish that the true precursor signal adds value beyond the shuffled timestamp control**; the sham performed better on this authored corpus. This is an important negative result, not something to hide. It motivates stricter source-bound precursor identity, randomized sham construction within disturbance strata, and a larger held-out episode corpus before claiming causal value.

The 2/10 synthetic baseline-failing episodes intercepted by predictive reflex (20%) is not an estimate of real-world preventable-event interception. These 12 scenarios were authored to exercise branches, not sampled from a physical population.

The oracle is evaluator-only, knows hazard onset, and must not enter the deployable leaderboard. The compute/intervention units are declared arbitrary units and cannot be called joules or measured processor cost.

## Why the result matters

A synthetic benchmark can show that the reference code implements the intended decision semantics. It cannot prove that its plant abstraction predicts a real joint, gripper or robot. More importantly, a predictor that beats a reactive baseline but is beaten by a shuffled precursor control has not shown that its **specific predictive information** is load-bearing. Future evaluation must demonstrate that the correct time-aligned signal—not just an early trigger, more compute or extra interventions—causes the gain.
