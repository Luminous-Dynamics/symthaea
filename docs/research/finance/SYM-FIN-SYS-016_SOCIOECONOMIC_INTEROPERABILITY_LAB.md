# SYM-FIN-SYS-016 — Socioeconomic Interoperability Laboratory v1

Status: research specification / deterministic synthetic laboratory

Parent issue: #7188
Related:
- #7186 — Creditism capacity, issuance, and capital-allocation stress model
- #7046 — bounded endogenous institutional evolution laboratory
- #7044 — Creditism issuance-capacity and essential-access calibration
- Mycelix #4804 — Civilization OS socioeconomic protocol profiles and interoperability kernel
- Mycelix #3343 — external economic interoperability fabric
- Mycelix #4578 — open-economy external settlement

## 1. Research thesis

A civilization should be modeled as a shared physical, epistemic, rights, governance, coordination and settlement substrate on which heterogeneous socioeconomic institutions can coexist.

The experiment must therefore compare mechanisms rather than validate a doctrine.

Initial regime profiles:

- debt/equity market capitalism;
- market + UBI;
- cooperative / pooled capital;
- mutual credit;
- Creditism;
- commons / polycentric allocation;
- public-budget planning;
- configurable hybrids.

A regime is a versioned mechanism profile, never a single label.

## 2. Shared civilization state

All profiles operate against the same world state:

- people and households;
- skills and task capacities;
- local/global resource stocks;
- labor capacity;
- energy;
- water;
- materials;
- land/space;
- housing;
- machines and productive assets;
- inventories and logistics;
- ecological floors;
- projects and maintenance liabilities;
- communication/observation network;
- institutional/authority state.

Economic instruments are not physical resources.

## 3. Non-equivalence invariants

The laboratory must preserve these distinctions:

\`instrument != entitlement != ownership != authority\`
\`economic authorization != physical capacity\`
\`valuation != settlement\`
\`settlement != legal discharge\`
\`foreign recognition != local issuance\`
\`evidence validity != political legitimacy\`
\`simulation recommendation != authorization\`
\`unknown capacity != available capacity\`

Any violation is a model/accounting failure, not an interesting outcome.

## 4. Regime profile contract

Each profile binds:

- claims and instruments;
- issuance/deletion/settlement semantics;
- allocation mechanism;
- contribution recognition;
- price formation;
- capital formation;
- property/access regime;
- risk-sharing mechanism;
- public/common allocation;
- housing/resource allocation;
- governance;
- entry/exit;
- failure and recovery;
- external trade;
- conversion/migration;
- privacy and observability costs;
- enforcement/dispute mechanism.

Profiles may share substrate primitives but may not silently share semantic authority.

## 5. Mixed-regime world

Use the baseline 100-person / 20-project / 8-resource / 4-community world.

Each community may select a different regime.

Required starting mixed-world example:

- Community A: Creditism
- Community B: debt/equity market
- Community C: mutual credit
- Community D: commons/polycentric

People may:

- trade across communities;
- migrate;
- join/leave institutions;
- contribute across boundaries where allowed;
- use permitted external settlement;
- participate in cross-community projects.

Some resources remain globally coupled so that local institutional choices cannot hide physical interdependence.

## 6. Interoperability bridge profiles

### B0 — No bridge

No economic interoperability.

Purpose: isolate institutional behavior.

### B1 — Trade-only

Goods/services cross the boundary.

Local money/credit balances remain local.

Purpose: distinguish market exchange from monetary integration.

### B2 — Explicit conversion

Conversion requires:

- source instrument identity;
- target instrument identity;
- issuer/authority;
- quantity/unit;
- rate/valuation profile;
- validity interval;
- settlement identity;
- source finality;
- correction/reversal path.

No numeric 1:1 parity is sufficient on its own.

### B3 — Synchronized settlement

Cross-ledger exchange uses atomic/conditional settlement semantics where the selected profile permits them.

The second leg cannot become final merely because the first leg was observed or transmitted.

### B4 — Mutual recognition

A receiving regime may recognize an external evidence/credential claim without importing:

- external money semantics;
- external governance authority;
- external legal ownership;
- external institutional legitimacy.

### B5 — Shared commons

Ecological/resource floors are global constraints across all economic regimes.

No bridge may bypass the common physical constraint.

### B6 — Transition corridor

Two regimes coexist during migration.

Parameters include:

- parity conversion;
- non-parity conversion;
- legacy claim treatment;
- default/haircut;
- dual-accounting period;
- capital-claim conversion;
- exit/rollback rules.

Transition is itself an institution and must be versioned.

## 7. Required experiments

### I — Baseline comparison

Run each regime against identical seeded worlds.

### II — Pairwise interoperability

For every ordered pair of regimes, test B1-B3.

### III — Multi-party bridge

Test at least four regimes sharing a common external settlement layer.

### IV — Transition

Migrate one community from market/equity into Creditism and another from mutual credit into the same shared world.

### V — Contagion

Compromise or fail one subsystem and measure propagation through:

- economic claims;
- settlement;
- governance;
- resource reservations;
- external suppliers.

### VI — Institutional evolution

Compose with #7046. Permit bounded mutations to:

- pooling;
- reservation;
- monitoring;
- verification;
- risk sharing;
- governance;
- capital commitment;
- external settlement.

Proposal, adoption and implementation remain separate.

## 8. Adversarial campaign

At minimum:

1. ticker collision;
2. same nominal unit, different redemption promise;
3. balance promoted to entitlement;
4. entitlement promoted to ownership;
5. ownership promoted to authority;
6. foreign recognition mints local issuance;
7. local deletion mistaken for external settlement;
8. external settlement mistaken for legal discharge;
9. stale FX/rate profile;
10. stale authority/verifier;
11. bridge representation treated as native;
12. partial settlement treated as complete;
13. replayed conversion;
14. double conversion across bridges;
15. closed conversion loop creates net claims;
16. captured subsystem mints externally;
17. partition creates divergent frontiers;
18. rollback resurrects deleted claims;
19. migration silently changes historical semantics;
20. bridge concentrates enough volume to become a new financial chokepoint.

## 9. Closed-loop conservation test

For a closed sequence of conversions with no declared subsidy:

\`net_claim_creation <= declared_external_subsidy\`

Under a pure exchange cycle with no subsidy:

\`net_claim_creation = 0\`

Any positive creation indicates one of:

- mint-without-consumption;
- double settlement;
- precision/rounding exploit;
- rate/semantic mismatch;
- replay;
- rollback;
- hidden subsidy.

The oracle must report the exact edge and settlement identity that created the discrepancy.

## 10. Stress families

Use identical shocks across regimes:

- supply shock;
- energy shock;
- demand surge;
- productivity boom;
- demographic transition;
- migration;
- ecological tightening;
- long-horizon capital project;
- maintenance shock;
- verification failure;
- governance capture;
- financial contagion;
- strategic import cutoff;
- network partition;
- compound multi-shock event.

## 11. Metrics

Emit a vector; never collapse the primary result to a single welfare score.

### Human/access

- essential-service coverage;
- unmet essential demand;
- housing access;
- queue/wait;
- switching/migration cost.

### Physical

- capacity utilization;
- idle capacity;
- inventory;
- bottleneck severity;
- maintenance backlog;
- ecological overshoot.

### Distribution

- purchasing-power concentration;
- productive-asset concentration;
- housing concentration/segregation;
- authority concentration;
- verifier concentration;
- debt/claim burden.

### Innovation/dynamics

- project completion;
- project failure;
- entry/exit;
- skill formation;
- automation response;
- institutional diversity.

### Interoperability

- bridge volume;
- bridge concentration;
- settlement failures;
- replay incidents;
- double-settlement incidents;
- contagion radius;
- external leakage;
- unresolved conversions.

### Governance/privacy

- monitoring burden;
- privacy/observability cost;
- enforcement burden;
- dispute burden;
- participation concentration.

## 12. Causal attribution

Every run records:

- regime profile hash;
- world seed hash;
- shock schedule hash;
- agent model hash;
- governance/adoption profile hash;
- resource-state hash;
- evidence-policy generation;
- bridge profile hash;
- settlement/finality profile;
- random seed;
- oracle version.

Every material outcome must identify the causal bridge/path by which it occurred.

## 13. Scientific controls

Mandatory:

- multiple random seeds;
- multiple starting institutional states;
- fixed-institution controls;
- no-bridge controls;
- proposal-generator ablation;
- adoption-process ablation;
- random-mutation control;
- holdout shocks;
- parameter perturbation;
- mutation-order permutation;
- replay determinism;
- counterfactual bridge removal;
- independent accounting/oracle path.

Do not train, tune or select the model using the same holdout shocks used to evaluate it.

## 14. Claim ceiling

A PASS establishes only that the declared synthetic mechanism, bridge and oracle behaved as specified under the frozen world/profile.

It does not establish:

- real-world economic superiority;
- macroeconomic equilibrium;
- political legitimacy;
- historical inevitability;
- legal interoperability;
- ecological sufficiency;
- incentive compatibility;
- real-world human welfare.

## 15. Success criterion

The laboratory succeeds if it can produce reproducible mechanism maps such as:

problem + physical regime + information regime + governance regime
-> mechanism set
-> outcome vector
-> failure boundary
-> interoperability cost
-> transition cost

The desired output is a **Civilization Mechanism Map**, not a winning ideology.


## 16. Versioned run receipt and semantic export boundary

The machine-readable run receipt is defined in [`SYM-FIN-SYS-016_run_receipt_v1.schema.json`](SYM-FIN-SYS-016_run_receipt_v1.schema.json). An illustrative shape is in [`SYM-FIN-SYS-016_run_receipt_example_v1.json`](SYM-FIN-SYS-016_run_receipt_example_v1.json); the negative-control corpus contains 26 cases (10 schema-level and 14 cross-field semantic mutations) in [`SYM-FIN-SYS-016_run_receipt_negative_controls_v1.json`](SYM-FIN-SYS-016_run_receipt_negative_controls_v1.json). The test-only [validator](../../../scripts/validate_socioeconomic_run_receipt.py) checks both layers, uses strict JSON parsing to reject duplicate keys and non-standard non-finite numeric constants, recomputes `scenario_registry_digest` against the exact bytes of the versioned scenario registry, requires `run.scenario_id` to appear in the unique `scenario_ids` catalogue, and checks each metric ID and dimension against the registry's metric catalogue, re-hashes the manifest's pinned source artifacts, and requires exactly one manifest binding for every SHA-256-shaped receipt field. Other digest-shaped receipt fields are not yet verified against their source artifacts; catalogued metric semantics must not be confused with cryptographic binding of the data behind those metrics. A small job in the existing `.github/workflows/ci.yml` is deliberately not draft-gated. A separate focused workflow listens for relevant pull-request changes and relevant-path pushes on any branch, and also supports manual re-validation after its `workflow_dispatch` trigger is available on the default branch.

A receipt binds:

- exact source revision, scenario-registry digest, environment digest, deterministic seed, model version and initial/final world-state digests;
- each community's explicit regime profile and each bridge's versioned profile;
- governance/evidence-policy generation and settlement/finality profile;
- ordered synthetic events, their payload digests, and scoped metric values with units, population scope, period and measurement status;
- oracle/evaluator identity, applied controls and counterexample-corpus digest.

The receipt's authority boundary is intentionally non-configurable in v1:

- it is not a real-world observation;
- it cannot authorize real action or reserve real resources;
- it cannot settle real assets;
- it cannot be imported into a production ledger as an instruction.

A simulated reservation/commitment/delivery/verification/settlement is only a state transition inside the frozen model. Its event name must retain the `simulated_` prefix where applicable, and every event records `authority_effect = none`, `external_effect = none`, and `settlement_effect = none`.

### Importing from Symthaea into Mycelix

The initial integration is export-only. Mycelix may ingest a receipt as research provenance and compare it to the exact scenario/profile/run digest, but it must not infer an account balance, resource reservation, credential, entitlement, payment instruction or governance authority from it.

Any later bridge into Mycelix production-oriented records needs a separately versioned adapter contract, an explicit evidence-class mapping, an independent verifier, and a capability check owned by the receiving system. Importing the receipt must be idempotent and must not have economic side effects.

Keep the following relations machine-visible:

`simulation result != observation of physical delivery`

`simulation verification != independent real-world verification`

`simulation authorization != real authorization`

`simulation settlement != external settlement`

### Evidence status

The schema, illustrative fixture, negative-control corpus and test-only validator are reference artifacts. Exact validator dependency versions are pinned. The draft-PR-capable job in the existing CI workflow checks out the exact pull-request head; the standalone workflow is retained for future maintenance/dispatch. Until one of these exact-head executions completes successfully, the schema/fixture pair is not qualified as a validated interchange format. The example metrics are synthetic placeholders, not observed results and not evidence for the efficacy of any regime.

## 17. Qualification benchmarks and targets

The staged qualification ladder, proposed numerical thresholds, reproducibility requirements, adversarial campaign sizes and provisional performance targets are specified in [SYM-FIN-SYS-016_BENCHMARK_TARGETS.md](SYM-FIN-SYS-016_BENCHMARK_TARGETS.md). These are targets, not measured results; the document separates current receipt-checker evidence from unexecuted simulator campaigns.

## 18. Artifact manifest and digest-binding policy

The versioned [artifact manifest](SYM-FIN-SYS-016_artifact_manifest_v1.json) binds the validator, tests, schema, negative corpus, dependency pins, workflow files, and scenario registry to SHA-256 digests of their exact repository bytes. The receipt binds the exact manifest bytes via `run.artifact_manifest_digest`. Every other SHA-256-shaped value in the receipt has exactly one pointer-specific manifest binding. Repository bytes are directly re-hashed; run-generated payload digests whose underlying bytes are absent are marked `unverified` with an explicit reason. The current fixture has 19 digest-shaped fields: one verified repository-artifact binding and 18 unverified placeholders. The manifest excludes itself and the receipt fixture to avoid circular hashing. This checks identity and binding completeness; it does not authenticate the issuer or prove the truth of a simulation or physical event.
