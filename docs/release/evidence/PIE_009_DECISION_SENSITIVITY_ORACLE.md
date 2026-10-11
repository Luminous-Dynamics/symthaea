# PIE-009C decision-sensitivity / experiment-priority oracle

## Purpose

Provide implementation-independent semantics for identifying which uncertain parameters actually change PIE planning decisions.

The oracle is `scripts/pie-decision-sensitivity-oracle.py` and imports no Symthaea code.

It deliberately avoids assigning probabilities or collapsing multiple objectives into one hidden score. Instead it evaluates Pareto membership over the corners of declared uncertainty intervals.

## Executed evidence

The final candidate self-test was executed locally on 2026-09-12 and returned `ok` before the checked-in reference was created.

The synthetic fixtures verify:

1. alternatives may be `AlwaysPareto`, `ConditionalPareto`, or `NeverPareto` across an uncertainty envelope;
2. fixing one parameter at its nominal value reduces that uncertainty axis without changing the other evidence;
3. in the fixture, recovery uncertainty is deliberately decision-critical: resolving it removes a conditional Pareto ambiguity;
4. a deliberately minor-loss parameter does not provide the same ambiguity reduction;
5. multiple nondominated alternatives remain visible at the nominal point—there is no hidden scalar score;
6. invalid uncertainty envelopes and unknown parameter references fail closed.

## Interpretation

`ambiguity_reduction` is not Bayesian expected value of information and is not a probability of mission success.

It is a structural planning signal:

> if this parameter were resolved to its nominal value, how many currently conditional Pareto alternatives would stop changing membership across the remaining uncertainty envelope?

This can help prioritize experiments, field measurements, scale-up tests, or dataset improvements that are most likely to change architecture choices.

## Intended future use

After real PIE evidence records exist, a decision-sensitivity layer can ask questions such as:

- does uncertainty in lunar beneficiation recovery actually change the preferred industrial seed architecture?
- is MRE energy intensity decision-critical, or are transport/spares dominating instead?
- would better Mars water-access evidence change the Pareto set?
- which process test would eliminate the most architecture ambiguity before committing mass to a mission?

## Limitations

All values and objective response coefficients are synthetic. The current oracle uses interval corners and simple affine response fixtures; it is not a chemistry, economics, reliability, or nonlinear process simulator.

The oracle does not prove that the parameter with the largest ambiguity reduction is the most scientifically valuable experiment. It only identifies decision sensitivity under the declared planning model.

Tracks PIE-009 #1641, Phase-0 gates #1647, integrated audit #1700, evidence lineage #1712, and master #1604.
