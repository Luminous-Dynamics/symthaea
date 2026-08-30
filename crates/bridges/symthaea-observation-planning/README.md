# Symthaea Observation Planning

This crate defines evidence contracts for **next-best-observation** research in Planetary Perception.

The planner's job is to make the measurement trade space legible. It does not grant permission to acquire data, drill, deploy a robot, task a spacecraft, or otherwise act.

## Separate dimensions

Each candidate may retain:

- expected information gain;
- expected uncertainty reduction by hypothesis;
- delay;
- energy;
- downlink/data volume;
- human effort;
- monetary cost and currency;
- operational risk;
- intrusiveness;
- model/version/artifact provenance;
- assumptions.

There is deliberately no hidden weighted `best_score`.

## Conservative Pareto filtering

`ObservationPlan::pareto_frontier` can remove a candidate only when another candidate has at least as much expected information gain, is no more intrusive, and is no worse on every comparable cost/risk dimension with at least one strict improvement.

If a cost is unknown, it is not treated as zero. Unknown/incomparable cost prevents automatic dominance.

## Research requirement

Future learned/HDC observation planners should be benchmarked against at least:

- random selection;
- fixed sensor sequence;
- simple entropy/information-gain selection;
- conventional Bayesian experimental design where feasible.

A learned planner earns promotion only if it improves uncertainty reduction or discovery efficiency under held-out conditions without worsening calibration, safety, or hidden-resource use.

## Authority boundary

High information gain is not authorization. Invasive measurements, satellite tasking, field surveys, robotic inspection, or other consequential acquisitions must pass their own human/governance/authority boundary outside this crate.
