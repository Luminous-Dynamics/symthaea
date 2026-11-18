# Home Expression grammar adapter — Symthaea v0.1

Parent: intent-to-design compiler #7790.
Implementation issue: DESIGN-INTENT-GRAMMAR-001 #12765.
Upstream contract: Sol-Atlas PR #935, issue #934.
Status: adapter specification and synthetic envelopes; no candidate generator or geometry exporter implemented by this document.

## 1. Purpose and authority boundaries

Sol-Atlas owns resident-facing expression and decision receipts. Symthaea owns reasoning, candidate generation and Pareto exploration. Symtropy and qualified specialists own exact realized geometry, simulations and physical-domain findings. Mycelix owns consent, governance, provenance authority and decision rights.

The interface must preserve resident intent without smuggling design authority into style labels. A resident request for stone does not specify a structural assembly. A preference for shading does not specify solar heat-gain coefficients. A Cape reference does not grant heritage authorization. A preferred courtyard does not prove buildability.

NIST's building digitization work identifies semantic interoperability and machine-readable lifecycle metadata as persistent barriers across design, systems and operations. Carry source identity, exact revisions, semantics, units and owner-qualified evidence rather than rely on screenshots or loosely mapped fields: https://www.nist.gov/programs-projects/building-digitization-and-semantic-interoperability.

## 2. Pinned upstream snapshot

Sol-Atlas PR #935 head at authoring time: 73730cbcc2b02e2ef045bd6c068cf59e4b4f41df.

Catalogue HOME-EXPRESSION-CATALOG-0.2.0 blob SHA: 0ba1af38685636c51d19fa399dee315d648585a6.
Catalogue JSON Schema v0.2 blob SHA: fd995c652c690ecdd54f24b3b5d7b8ed695d7288.
Grammar schema v0.2 blob SHA: 661c2c5a96e13c7678b80714bdb9e59d787c5a7f.
30-case conformance corpus (5 positive / 25 negative) blob SHA: 61ff73fca82af4905bb9d5dd2a020bfaf438e94f.
Corpus schema v0.2 blob SHA: 918ab3b936b45528360a705975f3f50328aea293.

These references are a draft snapshot, not a stable main-branch dependency. Before implementation or CI use, verify the current source PR head, exact file blob SHAs, catalogue revision and schema/corpus identity. A branch name alone is insufficient.

## 3. Compiler pipeline

Resident grammar → identity/provenance validation → semantic compilation → authority separation → versioned mapping rules → candidate set → exact geometry handoff → specialist evaluation → resident comparison.

1. Validate and bind: verify catalogue IDs, grammar schema, reference sources, exact Git blob SHA-1 identities for both catalogue JSON and catalogue JSON Schema, and all 30 upstream conformance cases.
2. Preserve the source: keep the original grammar and all five axes. Do not normalize away AVOID, ACCEPTABLE, UNDECIDED, incomparable or conflicting entries.
3. Classify requirements: keep resident preferences separate from explicit resident non-negotiables, jurisdictional constraints, professional-review requirements and system invariants. Every requirement carries a source class, source reference and verification state.
4. Map only through explicit rules: each mapping rule identifies source axis/option, target parameter path, mapping version, scope, evidence/source, reviewer and validity range. No default mapping from a style label to geometry, construction property or physical outcome.
5. Generate alternatives: preserve stable candidate identity, parentage, parameter deltas, seed, generator build and unresolved requirements. A failed hard gate blocks a candidate; UNKNOWN remains conditional.
6. Evaluate downstream: geometry is accepted only from the qualified Symtropy geometry owner. Physics, accessibility, code, heritage and cost remain separate domain receipts.
7. Return a Pareto set: do not collapse dimensions into one beauty, meaning, comfort or utility number. No candidate chooses for the resident.

## 4. Typed envelopes

HOME_EXPRESSION_DESIGN_PROBLEM_V0.1.schema.json defines source bindings, grammar input, authority classes, mapping state and generation permissions. HOME_EXPRESSION_DESIGN_PROBLEM_SYNTHETIC_V0.1.json demonstrates a problem with no owner-reviewed mappings and geometry generation blocked.

HOME_EXPRESSION_CANDIDATE_SET_V0.1.schema.json defines lineage, generator identity, candidate provenance and claim ceilings. HOME_EXPRESSION_CANDIDATE_SET_SYNTHETIC_V0.1.json contains three illustrative search directions explicitly marked NOT_EXECUTED / NOT_GENERATED. They are not generated candidates and have no geometry or physical result.

## 4.1 Resident preference graph input

The resident-facing source of pairwise concept choices is Sol-Atlas [HOME-EXPRESSION-DECISION-RECEIPT-001 #917](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/917). Its v0.1 receipt distinguishes NOT_RECORDED, UNDECIDED, PREFERENCES_RECORDED, CONCEPT_DIRECTION_SELECTED and INCOMPARABLE, with pairwise decisions PREFER_LEFT, PREFER_RIGHT, EQUIVALENT, INCOMPARABLE and UNDECIDED.

Symthaea will consume this owner-native graph through [RESIDENT-PARTIAL-ORDER-INPUT-001 #12796](https://github.com/Luminous-Dynamics/symthaea/issues/12796). It must preserve exact candidate/profile/brief/design-revision identity, reject duplicate/reversed/self pairs and directed cycles, and leave every unmentioned comparison unranked. It must not create scalar utility, fill undecided comparisons or choose a candidate when the upstream state is NOT_RECORDED or INCOMPARABLE.

A decision event remains PENDING_MYCELIX_5149_VERIFICATION until the Mycelix authority/consent owner verifies it. A participant reference, event URI, digest or reviewer name does not authenticate authority. Household disagreement and private/shared intent remain under Sol-Atlas #907.

This is a **planned consumer integration**, not an implementation claim for the current v0.1 crate. The current synthetic grammar fixture contains no resident decision and stays NOT_RECORDED.

## 5. Search-method baseline update

Do not use NSGA-II as the only incumbent. Public [FloorPlanBench](https://floorplanbench.vercel.app/paper) is pinned here to full commit `7dfcca3dda6b11b4808a259499143e9bd75ed878` (MIT). It compares seven classical methods over 28 programmes (249 rooms), four tiers, ten seeds and a 50,000 evaluation-equivalent budget. Its canonical paper reports Greedy→SA at 92% solved runs and median five-objective hypervolume 0.355; the current live page reports 0.363. NSGA-II is 0.098 in the paper but 0.196 on the current live page. Reproduce the pinned core/task/run hashes and resolve the discrepancy before making comparative claims. This is an exact software floor-plan benchmark on integer-decimetre axis-aligned room rectangles, not a physical-building or regulatory certificate. [Pinned commit](https://github.com/Gariyuuu/floorplanbench/commit/7dfcca3dda6b11b4808a259499143e9bd75ed878).

Talami et al. report 93.8% fewer simulations and 85–92% true-front coverage in a Nottingham office case. Reproduce the exact coverage definition and case before cross-paper claims: https://www.sciencedirect.com/science/article/pii/S0378778826014775.

Other 2026 residential comparators are separate cases, not pooled SOTA scores:
- Amani et al.'s MO-ISDO EnergyPlus case in Tabriz reports 17 non-dominated options and 32.36% lower annual energy / 22.84% lower GHG than that case's reference building, after reducing 14 variables to 10 using standardized regression coefficients; it compares MO-ISDO with NSGA-II and MO-GWO. [Energy, 1 October 2026](https://www.sciencedirect.com/science/article/pii/S0360544226021870).
- Romeos et al.'s Mediterranean EnergyPlus/NSGA-II study reports 59 non-dominated residential options and an approximately 14% reduction for its selected energy/cost compromise; retain the entire front and measure pruning loss. [Energy and Buildings, October 2026](https://www.sciencedirect.com/science/article/pii/S0378778826009370).
- A Central African study evaluates eight representative climate zones with EnergyPlus + NSGA-II, reporting 53.13–59.85% savings against its reference buildings and PV-enabled net-negative operational carbon. Treat this as adjacent regional evidence, not a South African validation. [Buildings, 11 September 2026](https://www.mdpi.com/2075-5309/16/18/3626).
- LayoutGen's reported 92% violation reduction (3.7 to 0.3 per plan across 200 runs) is limited to its single-storey rectangular-room, selected dimensional and topology checks; setbacks, FAR, egress, stairs, parking and ventilation are not part of its generation loop. [Discover AI, 15 October 2026](https://link.springer.com/article/10.1007/s44163-026-02481-9).

Across all performance papers, replay the exact case, weather file, model, solver build, baseline, variables, uncertainty set and budget before claiming superiority. Percent savings against different reference buildings are not comparable.

Minimum matched algorithm set: constructive greedy/GRASP with backtracking; simulated annealing; greedy-seeded SA; purpose-built constraint search / exact enumeration for tiny finite cases; NSGA-II with declared constraint-domination; one suitable additional MOO strategy only where the representation supports it.

Charge candidate checks, invalid geometry proposals, solver calls, repair and partial work consistently. Invalid candidates never enter objective calculations or Pareto archives.

## 6. Proposed benchmark targets (not achieved)

- 30 upstream grammar cases pass at the exact pinned snapshot, plus at least 10 compiler-specific cases for missing mappings, unknown authorities, seed drift, candidate identity, stale revisions and change invalidation.
- Zero false-PASS and zero invalid candidates admitted to the Pareto archive.
- 100% deterministic candidate-manifest replay for identical pinned inputs, generator build and seed.
- On a held-out FloorPlanBench extension, preserve at least 92% solved-run rate and target at least 10% higher median five-objective hypervolume than reproduced Greedy→SA at identical 50,000 evaluation-equivalents; report per-task/tier distributions and paired confidence intervals.
- Separately, target at least 95% exact-front coverage and 95% hypervolume at no more than 5% of exhaustive high-fidelity calls after reproducing Talami et al. Do not conflate this with FloorPlanBench evaluation-equivalents.
- 100% preservation of resident non-negotiables, avoidance, undecided state and explicit dissent.

## 7. Initial scope boundary

Until mapping rules and downstream owner interfaces are accepted, generation is symbolic/conceptual only; no geometry, price, quantity, energy, daylight, structural, accessibility, water, carbon, code or heritage results are produced. Authority requirements are never generated by a language model or inferred from resident identity. Missing evidence cannot improve a candidate objective. No candidate winner or household consensus is manufactured.

## 8. Implementation sequence

1. Validate the pinned source schemas, catalogue and 30-case corpus.
2. Implement typed Rust envelopes and deterministic replay receipt.
3. Implement source-preserving semantic compiler and requirement classes.
4. Add explicit, reviewed mapping-rule registry with deny-by-default status.
5. Build exact enumerator for tiny candidate spaces and an invalidity-gated candidate archive.
6. Reproduce FloorPlanBench methods and artifacts before extending its input grammar.
7. Add Symtropy geometry/physics handoff only against qualified exact interfaces.
8. Run consented resident/architect trials only when the upstream interface becomes an explicitly authorized, privacy-reviewed channel.

Claim boundary: this is an adapter contract, not evidence that Symthaea currently generates buildable homes or has an architectural capability benchmark result.
