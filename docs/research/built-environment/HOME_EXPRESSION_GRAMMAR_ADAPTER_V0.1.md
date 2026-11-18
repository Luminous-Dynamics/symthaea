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

Sol-Atlas PR #935 head at authoring time: 42caacb8b3e00f87fcf46c0dd8ceae87763aa1d6.

Catalogue HOME-EXPRESSION-CATALOG-0.2.0 blob SHA: d784b285a706068aeba7bf7a88bdb857ed2053cc.
Grammar schema v0.2 blob SHA: 661c2c5a96e13c7678b80714bdb9e59d787c5a7f.
30-case conformance corpus (5 positive / 25 negative) blob SHA: 61ff73fca82af4905bb9d5dd2a020bfaf438e94f.
Corpus schema v0.2 blob SHA: 918ab3b936b45528360a705975f3f50328aea293.

These references are a draft snapshot, not a stable main-branch dependency. Before implementation or CI use, verify the current source PR head, exact file blob SHAs, catalogue revision and schema/corpus identity. A branch name alone is insufficient.

## 3. Compiler pipeline

Resident grammar → identity/provenance validation → semantic compilation → authority separation → versioned mapping rules → candidate set → exact geometry handoff → specialist evaluation → resident comparison.

1. Validate and bind: verify catalogue IDs, grammar schema, reference sources, exact blob identities and all 30 upstream conformance cases.
2. Preserve the source: keep the original grammar and all five axes. Do not normalize away AVOID, ACCEPTABLE, UNDECIDED, incomparable or conflicting entries.
3. Classify requirements: keep resident preferences separate from explicit resident non-negotiables, jurisdictional constraints, professional-review requirements and system invariants. Every requirement carries a source class, source reference and verification state.
4. Map only through explicit rules: each mapping rule identifies source axis/option, target parameter path, mapping version, scope, evidence/source, reviewer and validity range. No default mapping from a style label to geometry, construction property or physical outcome.
5. Generate alternatives: preserve stable candidate identity, parentage, parameter deltas, seed, generator build and unresolved requirements. A failed hard gate blocks a candidate; UNKNOWN remains conditional.
6. Evaluate downstream: geometry is accepted only from the qualified Symtropy geometry owner. Physics, accessibility, code, heritage and cost remain separate domain receipts.
7. Return a Pareto set: do not collapse dimensions into one beauty, meaning, comfort or utility number. No candidate chooses for the resident.

## 4. Typed envelopes

HOME_EXPRESSION_DESIGN_PROBLEM_V0.1.schema.json defines source bindings, grammar input, authority classes, mapping state and generation permissions. HOME_EXPRESSION_DESIGN_PROBLEM_SYNTHETIC_V0.1.json demonstrates a problem with no owner-reviewed mappings and geometry generation blocked.

HOME_EXPRESSION_CANDIDATE_SET_V0.1.schema.json defines lineage, generator identity, candidate provenance and claim ceilings. HOME_EXPRESSION_CANDIDATE_SET_SYNTHETIC_V0.1.json contains three illustrative search directions explicitly marked NOT_EXECUTED / NOT_GENERATED. They are not generated candidates and have no geometry or physical result.

## 5. Search-method baseline update

Do not use NSGA-II as the only incumbent. FloorPlanBench is a public benchmark with an exact validator for integer-decimetre rectilinear room layouts; it reports 28 programmes, four tiers, ten seeds and a 50,000 evaluation-equivalent budget across random, greedy, SA, greedy-seeded SA, NSGA-II, constraint search and analytic placement. Its paper reports Greedy→SA at 92% solved runs and median five-objective hypervolume 0.355; NSGA-II is weaker on that specific representation. The current live overview and paper differ in headline values, so pin the canonical source commit and per-run artifacts. This is a reproduction target, not a universal SOTA certificate. Paper: https://floorplanbench.vercel.app/paper. Source: https://github.com/Gariyuuu/floorplanbench.

Talami et al. report 93.8% fewer simulations and 85–92% true-front coverage in a Nottingham office case. Reproduce the exact coverage definition and case before cross-paper claims: https://www.sciencedirect.com/science/article/pii/S0378778826014775.

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
