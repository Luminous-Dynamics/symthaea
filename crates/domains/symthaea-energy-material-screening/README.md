# symthaea-energy-material-screening

Evidence-complete, multi-objective screening contracts for energy materials.

This crate is the boundary between a single-property screening benchmark and a broader energy-material discovery process. It deliberately does **not** define a universal material score.

## Seven required evidence dimensions

Tier-1 screening requires one explicit metric contract for each of:

1. functional performance;
2. thermodynamic stability;
3. critical-material burden;
4. supply resilience;
5. human/environmental hazard;
6. circularity;
7. manufacturability.

A metric contract declares:

- exact metric name;
- exact unit;
- optimization direction (`maximize`, `minimize`, or target+tolerance);
- minimum acceptable prediction fidelity;
- accepted evidence kinds.

The seven metric/unit pairs must be distinct. One convenient score cannot silently stand in for several independent evidence dimensions.

## Metric-pluggable by design

The crate provides a few concrete metric names such as `band_gap`, `energy_above_hull`, `critical_material_mass_fraction`, `supply_concentration_hhi`, and `recyclability_fraction`.

It does **not** invent a universal toxicity, supply-risk, or manufacturability score. A future adapter may use a regulator-backed hazard metric, a particular critical-mineral dataset, a circularity dataset, or fabrication qualification evidence, but the exact metric/method must be declared by policy and carried by prediction provenance.

This is intentional because existing Symthaea capabilities already contain local critical-mineral, circularity, and fabrication concepts with different scopes. They should be adapted through an explicit contract rather than silently treated as equivalent global truth.

## Missing evidence is not zero

For each dimension, assessment can be:

- `Available`;
- `Missing`;
- `BelowMinimumFidelity`;
- `UnsupportedEvidenceKind`;
- `AmbiguousHighestFidelity`.

A candidate becomes evidence-complete only when all seven dimensions have exactly one admissible highest-fidelity prediction.

Missing hazard evidence therefore does not become hazard `0`. Missing stability evidence does not become stable. Equal highest-fidelity conflicting predictions do not get averaged or selected by insertion order.

## Evidence kind and fidelity are separate

A prediction's **fidelity** describes the model/evidence level used to produce the value.

Its evidence references separately describe **what kind of evidence artifacts** support it.

For example, a policy may require at least surrogate fidelity while accepting evidence references to an external dataset. These are independent axes and are not interchangeable.

## Completeness, feasibility, Pareto rank are different states

When all required dimensions are available, the crate constructs the existing generic `symthaea_discovery::Evaluation`.

Hard constraints are then assessed by the generic discovery contract at the unique highest-fidelity matching prediction.

Thus:

- incomplete evidence means **not yet evaluable**;
- complete + violated hard constraint means **infeasible under this policy**;
- complete + feasible means **eligible for downstream comparison**;
- Pareto rank remains unset until a Pareto adapter compares a valid cohort.

Feasible does not mean best, validated, novel, synthesizable, or deployable.

## Policy identity

The complete screening policy receives a domain-separated SHA-256. Contract ordering is canonicalized by evidence dimension for identity so semantically identical dimension declarations can be reviewed consistently.

Downstream adapters should preserve this canonical dimension order when constructing evidence artifacts and Pareto cohorts.

## Authority boundary

This crate performs no network access, model execution, experiment, synthesis, procurement, manufacturing, deployment, or physical actuation.

Its output is a screening/evidence contract only and cannot certify a material or authorize real-world action.

## Intended next adapters

Useful follow-on adapters include:

- composition RF / DFT -> functional property evidence;
- Materials Project / external electronic-structure solvers -> stability evidence;
- Symthaea critical-mineral + external resource datasets -> criticality/supply evidence;
- regulatory/toxicology datasets -> hazard evidence;
- circular-economy datasets -> circularity evidence;
- fabrication-kernel/process models -> manufacturability evidence.

Each adapter should preserve source/version/digest provenance and should not upgrade its evidence class merely because its numerical confidence is high.
