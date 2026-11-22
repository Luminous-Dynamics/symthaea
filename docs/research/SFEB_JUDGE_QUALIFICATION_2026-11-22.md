# SFEB evaluator qualification: interaction-grounded judging without authority collapse

**Date:** 2026-11-22  
**Status:** research protocol and schema authored; experiments pending

## 1. New industry direction

Arena's November 12, 2026 Interaction-as-a-Judge research evaluates generated applications with computer-use agents instead of relying only on code and screenshots. The authors report improvements in broken-requirement detection and iterative repair under their own task/model setup. The released ArenaWebdevPreferences and ArenaWebdevCorrectness datasets provide preference and requirement labels with interaction traces.

The useful insight for SFEB is not to replace deterministic tests with an LLM judge. It is to add a richer observation channel: a computer-use evaluator can explore states, attempt journeys, identify broken requirements, and supply evidence-linked diagnostic hypotheses.

## 2. Authority separation

Three actors must remain distinct:

1. Execution worker: captures browser observations and signs the immutable run bundle under a pinned workflow identity.
2. Independent oracle verifier: replays raw evidence and derives deterministic contract/security/evidence outcomes.
3. Interaction judge: explores the application and produces diagnostic or preference judgements, but cannot change the task policy, evidence, signature or deterministic verdict.

A judge result may affect prioritization, repair hypotheses or subjective quality metrics. It cannot override critical access-control, identity, evidence-integrity or required-journey failures.

## 3. Controlled evaluator comparison

J0–J5 compare deterministic oracles, code-only judgement, screenshot-only judgement, code+image judgement, interaction judgement and a bounded ensemble.

Each arm sees identical frozen task inputs and receives the same time/tool budget. Arm order is randomized. Human annotators are blind to arm identity. A third reviewer adjudicates disagreements. Evaluation uses held-out task families and no optional stopping.

Adversarial items include prompt injection through the page, visible-but-inoperable controls, fake success toasts, stale accessibility/DOM observations, manipulated screenshots, incomplete exploration and copied self-authored preference explanations.

## 4. Dataset use

The ArenaWebdevPreferences dataset card identifies 6,390 pairwise web-development battles and lists CC-BY-4.0 for user prompts, while explicitly leaving model outputs under their provider terms. The ArenaWebdevCorrectness dataset card lists 189 rows and CC-BY-4.0. The paper describes 362 human-verified requirements; do not confuse row count with requirement-label count.

Neither dataset is downloaded or committed by this change. Both remain reference-only until component-level source/assets, privacy, attribution and commercial-use constraints are reviewed.

## 5. Targets

Proposed judge promotion targets:
- ≥0.85 held-out balanced accuracy for non-critical broken-requirement classification;
- ≥0.95 recall for critical defect labels;
- ECE ≤0.10 and Brier score reported;
- ≥0.80 inter-annotator agreement before relying on subjective labels;
- explicit per-category false-accept and false-reject rates;
- 100% deterministic rejection of seeded critical security/evidence controls.

The judge is promoted only for the narrow capability actually qualified. Requalify after model, prompt, tool, environment or policy changes.

## 6. Related work

- Parent programme: PR #13162
- Independent oracle replay: Issue #13247
- Signed execution and trace verification: Issue #13228
- Judge implementation and comparison: Issue #13250
- Arena Interaction-as-a-Judge: https://arena.ai/blog/interaction-as-a-judge
- Paper: https://arxiv.org/abs/2610.17895
- ArenaWebdevPreferences: https://huggingface.co/datasets/lmarena-ai/arena-webdev-preferences
- ArenaWebdevCorrectness: https://huggingface.co/datasets/lmarena-ai/arena-webdev-correctness
