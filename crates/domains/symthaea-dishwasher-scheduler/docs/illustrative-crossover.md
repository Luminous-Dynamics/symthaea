# Illustrative QuickCell / FullLoad crossover analysis

Status: arithmetic example only. The inputs below are hypothetical target values used by the simulator, not measured appliance performance and not a manufacturer comparison.

## Assumptions

QuickCell:
- 300 seconds per item.
- 0.05 kWh per item.
- 0.7 L fresh water per item.

FullLoad:
- 7,200 seconds per batch.
- 14-item maximum capacity.
- Energy = 0.12 kWh base + 0.019285714285714285 kWh per item.
- Fresh water = 2.0 L base + 0.3357142857142857 L per item.

The model assumes the FullLoad process can run at any occupancy and that its base + marginal resource coefficients are linear. The partial-load measurement programme in Symthaea issue #10136 exists specifically to replace this assumption with empirical occupancy curves.

## Crossover

| Occupancy | QuickCell energy | FullLoad energy | QuickCell water | FullLoad water | QuickCell serial duration | FullLoad duration |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 0.050 kWh | 0.139 kWh | 0.70 L | 2.336 L | 5 min | 120 min |
| 2 | 0.100 kWh | 0.159 kWh | 1.40 L | 2.671 L | 10 min | 120 min |
| 3 | 0.150 kWh | 0.178 kWh | 2.10 L | 3.007 L | 15 min | 120 min |
| 4 | 0.200 kWh | 0.197 kWh | 2.80 L | 3.343 L | 20 min | 120 min |
| 5 | 0.250 kWh | 0.216 kWh | 3.50 L | 3.679 L | 25 min | 120 min |
| 6 | 0.300 kWh | 0.236 kWh | 4.20 L | 4.014 L | 30 min | 120 min |
| 10 | 0.500 kWh | 0.313 kWh | 7.00 L | 5.357 L | 50 min | 120 min |
| 13 | 0.650 kWh | 0.371 kWh | 9.10 L | 6.364 L | 65 min | 120 min |
| 14 | 0.700 kWh | 0.390 kWh | 9.80 L | 6.700 L | 70 min | 120 min |

## Interpretation under the simulator's three-objective rule

- At occupancies 1-3, QuickCell is lower on energy, water and total serial service duration. It Pareto-dominates the illustrative FullLoad partial cycle.
- At occupancies 4-5, FullLoad is slightly lower on energy, while QuickCell is lower on water and duration. Neither dominates; the scheduler retains FullLoad rather than inventing a weighted preference.
- At occupancies 6-14, FullLoad is lower on both energy and water; QuickCell is still faster in total serial service time under these assumptions. Neither dominates across all three objectives. Deadlines may still justify QuickCell.
- At full capacity, FullLoad is substantially better on resource use in this example.

This creates a useful hypothesis: QuickCell may be most valuable for *very sparse* demand, not as a replacement for batch washing. The actual crossover may move materially when partial-load sensing, variable pump speeds, heat-up, standby energy, inlet-water heating and programme selection are measured on real machines.

## Required calibration before product decisions

See [issue #10136](https://github.com/Luminous-Dynamics/symthaea/issues/10136). Measure exact reference machines at multiple occupancy levels and programmes. Replace the linear model with measured curves and uncertainty bounds. Keep cleaning, drying and rinse endpoints as hard constraints.
