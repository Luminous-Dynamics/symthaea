# Symthaea RSI: research taste, direction selection and experimental judgment

**Version:** v0 · **Date:** 2026-11-07  
**Owner:** #10877 (research taste)  
**Related:** #10497 (R4), #10717 (causal attribution), #10756 (evaluator research), #9230 (direction board), #9332 (paper-to-environment), #10630 (evidence continuity)

## Executive position

The next frontier is not merely producing more ideas or running experiments faster. It is choosing *which* problems deserve scarce research resources, turning them into concrete mechanisms and discriminating experiments, then interpreting the resulting evidence without fooling ourselves.

This is research taste. It should be decomposed into:
1. **Direction taste:** what is worth studying?
2. **Approach taste:** what specific mechanism or method could advance it?
3. **Experimental taste:** what is the most informative experiment per unit cost?
4. **Interpretive taste:** what do the results actually justify doing next?
5. **Transferable taste:** does the procedure improve its next research decisions on unfamiliar problems?

Scientific validity remains a separate empirical requirement. A persuasive proposal, high human rating, future-paper match, or efficient experiment does not by itself establish discovery.

## 1. New frontier results

### IdeaTaste (2026-11-02)

IdeaTaste evaluates six models across four areas, with more than 90,000 proposals. Average model populations reached about 26% of the human-community level on citation matching but only 18% on idea matching. The most important gap is between recognizing problems/motivations and proposing the concrete approach/research plan. Its blind review found the largest proposal-to-paper deficit in feasibility and expected success. The authors also find larger populations and richer field context do little to close the idea gap. [Paper](https://arxiv.org/abs/2611.03424).

Interpretation: the missing capability is not simply literature retrieval or direction discovery. It is specific mechanism construction and credible plans. However, matching future human publications is a retrospective community-alignment reference, not a universal truth criterion. Symthaea must preserve a separate track for independently validated original discoveries that do not match later papers.

### TasteVal (2026-10-05)

TasteVal isolates experiment design and interpretation by giving the researcher a fixed Coder. It reports Opus 5.5 at 2.30x expert-baseline compute multiplier (95% CI 1.15–4.37), from eight private tasks, 24 human experts, six seeds/model/task and 40 H100 hours/attempt. The metric is compute to reach the recruited experts' score, not universal scientific taste; tasks are private and single-H100, and transfer to large-scale parallel research is unestablished. [Paper](https://arxiv.org/abs/2610.06824).

Interpretation: this is the most direct experimental-taste baseline. We should match its separation of researcher decisions from coding/implementation, and target >2.3x on a comparable task suite only after matching its protocol. Never claim direct reproduction on our own tasks.

### Related evidence

- TASTE measures agreement with experienced researchers on paired AI-safety research proposals and finds model judgments below human judgments. High-confidence human labels and discussion improve the reference. [Anthropic](https://alignment.anthropic.com/2026/taste/).
- EurekaBench distinguishes predictive accuracy from mechanism-derived insight; outcome prediction alone does not establish explanation. [Paper](https://arxiv.org/abs/2609.30152).
- InnovationEval shows that novelty-oriented research assessment is sensitive to information access, run selection and task scope. [Epoch analysis](https://epoch.ai/blog/can-ai-automate-ai-r-and-d-yet).
- RRSI reports up to 14.1 points in-distribution and up to 4.7 OOD gains with a regularized harness search and 30% fewer policy tokens; it motivates edit-budget regularization and held-out transfer. [Paper](https://arxiv.org/abs/2609.24972).
- MASS reports 1.2–1.6x higher performance per output tokens after two cycles of multi-agent workflow search and self-supervision; this is an orchestration/training comparator, not by itself a prospective research-taste or R4 test. [Paper](https://arxiv.org/abs/2610.12176).
- REUSE gives statistical protection for repeated adaptive evaluation reuse, claiming 0% false promotions in its live experiments versus up to 20.7% for compared frameworks, with comparable final true population performance. This should inform the adaptive-promotion and evaluation-feedback contract; do not invent a parallel statistical authority. [Paper](https://arxiv.org/abs/2609.33180).
- SelfImprove-Bench introduces a G0–G4 generalization ladder across exploring, planning and reasoning game environments; it is useful as a structured transfer comparator, not as a complete scientific-research taste benchmark. [Paper](https://arxiv.org/abs/2611.02797).

## 2. Paradigm shift: research as sequential value-of-information

The strongest experiment is not necessarily the one with the highest expected score. It may be the experiment that most efficiently distinguishes between competing explanations and changes the next decision.

A candidate experiment should therefore carry:
- competing hypotheses and which outcomes discriminate them;
- predicted effect distribution, not just one point guess;
- information gain or expected decision value;
- resource cost and opportunity cost;
- failure, null, missingness and stopping policies;
- an explicit action that would follow each result.

Score the experimenter's *prospective prediction* before observing the result, then assess calibration, information gain, realized value and next-action correctness after unblinding. This makes hindsight reconstruction less useful.

## 3. Proposed benchmark architecture

Use a fixed Coder to separate research judgment from implementation. Run four stages:
1. **Problem selection:** choose from a frozen portfolio of opportunities, with importance, neglectedness, feasibility, constraints and opportunity cost.
2. **Concrete approach:** propose a mechanism, exact solution object, intermediate predictions, failure modes and falsifying observations.
3. **Experiment design:** choose paired/controlled experiments, determine sample size/stopping, and allocate the inclusive budget.
4. **Interpretation:** update confidence from noisy results and choose accept, replicate, continue, pivot, stop, abstain or escalate.

Only then test whether learned decision procedures transfer and improve a successor's research process.

Comparison arms: human experts, frontier model, frozen Symthaea incumbent, successor Symthaea, structured value-of-information baseline, and random/popularity controls. Keep the number of agents separate from the research policy and direction-allocation treatment.

## 4. Non-Goodhart scorecard

Do not use one research-taste scalar. Report:
- problem-selection realized value and calibration;
- concrete approach and mechanism feasibility;
- experiment discrimination / information gain per fully inclusive cost;
- effect-direction/range prediction calibration with proper scoring rules;
- correct next-action decisions;
- false claims and overclaim rates;
- independently confirmed useful improvements;
- future-paper match and citation match separately from independent discovery value;
- fresh-family/domain transfer;
- human disagreement and inter-rater reliability.

Future-paper matching can reward imitation of historical consensus. A novel approach with no human-paper counterpart must be judged by feasibility, falsifiability and realized evidence—not automatically marked wrong. Similarly, a proposal can be highly rated yet fail experimentally; ratings and outcomes must stay separate.

## 5. Prospective targets

These are pilot hypotheses, not current Symthaea results:

- **TasteVal-like experimental efficiency:** target a compute multiplier >2.3x the recruited expert baseline, with lower 95% interval above 1.0, on a comparable prospectively evaluated task suite. Report inference, Coder, human, evaluator and energy costs. If private TasteVal tasks are unavailable, claim only performance on our distinct suite.
- **Idea construction:** target at least 0.5 of the human level on concrete approach/research-plan matching as a first milestone, while tracking genuinely original but validated ideas in a separate lane.
- **Experiment judgment:** calibrated effect predictions and positive skill over base-rate, random and popularity-only controls.
- **Decision quality:** lower inclusive cost per independently confirmed useful improvement than human-designed and structured baselines, with no critical integrity failure.
- **Transfer:** positive gains on fresh problem/method families and at least one fresh domain without retuning.
- **R4:** only claim better improver after same-start, matched-total-cost, repeated full-loop superiority and fresh-family transfer.

## 6. Ownership and claim boundaries

Compose with #10497, #10717, #10756, #9230, #9332 and #10630. Existing receipt, evaluator, statistics, promotion and evidence-currentness authorities remain controlling. Do not create duplicate infrastructure.

A research-taste result is not a scientific discovery. Future-paper matching is not proof of truth or novelty. Human preference is not an empirical oracle. Experimental efficiency is not R4. The outer-loop claim remains governed by #10497.

## Sources

1. IdeaTaste (2026-11-02): https://arxiv.org/abs/2611.03424
2. TasteVal (2026-10-05): https://arxiv.org/abs/2610.06824
3. TASTE (Anthropic, 2026-08-28): https://alignment.anthropic.com/2026/taste/
4. EurekaBench (2026-09-30): https://arxiv.org/abs/2609.30152
5. InnovationEval (Epoch, 2026-10-07): https://epoch.ai/blog/can-ai-automate-ai-r-and-d-yet
6. RRSI (2026-09-21): https://arxiv.org/abs/2609.24972
7. MASS (2026-10-08): https://arxiv.org/abs/2610.12176
8. REUSE (2026-09-29): https://arxiv.org/abs/2609.33180
9. SelfImprove-Bench (2026-11-02): https://arxiv.org/abs/2611.02797
