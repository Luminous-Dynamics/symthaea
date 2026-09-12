# Symthaea Band-Gap Discovery Evidence

This bridge adapts the crystal-ablated composition-only random-forest predictor into a generic `symthaea-discovery::Prediction` for Tier-1 functional-performance screening.

It emits:

- metric `band_gap`;
- unit `eV`;
- fidelity `Surrogate`;
- curated training-table dataset evidence;
- composition-RF surrogate-model evidence.

The training table is content-addressed directly from its runtime entries: formula, composition, experimental gap and crystal system. The model and training source Git blob identities are also recorded.

The crystal-system feature is fixed to `Unknown` for every training and inference row, so it is a constant ablation feature rather than an inference-time guess.

The RF's inter-tree standard deviation is retained as review telemetry only. It is not inserted into generic calibrated uncertainty; epistemic uncertainty remains fully unknown and no interval is attached.

Band gap alone does not establish stability, absorptivity, transport, defect tolerance, synthesis, toxicity, abundance, cost, device efficiency or deployment fitness.

## CLI

`bandgap-discovery-evidence <candidate-id> <composition.json>`

The composition JSON is an array of `[atomic_number, fraction]` pairs that must contain unique atomic numbers and sum to 1.0.
