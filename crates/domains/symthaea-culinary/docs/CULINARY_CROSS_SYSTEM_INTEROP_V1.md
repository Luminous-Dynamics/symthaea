# Culinary cross-system interop contract v1

**Status:** design contract on a draft branch stacked on Symthaea PR #7951. Not yet adopted or independently implemented by Mycelix, Symtropy, or Sol Atlas.
**Machine-readable envelope:** [`CulinaryDecisionBundleV1.schema.json`](../schemas/CulinaryDecisionBundleV1.schema.json)

## Decision

Yes, integrate the culinary work more deeply with Mycelix, Symtropy and Sol Atlas—but integrate the *evidence flow*, not the codebases into one tightly coupled runtime. Each project already owns a distinct job. Culinary decisions should compose those outputs through an immutable, versioned receipt and read-only projections.

The interface should remain JSON-compatible and implementable in Rust without importing the other repositories' crates. A schema version is the compatibility boundary; producer identity, exact commit SHA, input-manifest digest, scenario IDs/digests, units, provenance references, observation/currentness state, and privacy scope travel with the result.

## Ownership boundaries

| System | Owns / supplies | Must not be treated as |
|---|---|---|
| **Symthaea Culinary** | Candidate generation/evaluation; typed objective definitions; hard-gate dispositions; baseline/scenario frontiers; explanations; rank-instability and robust-dominance results. | A source of live stock, legal authority, verified food safety, clinical advice, or physical outcomes. |
| **Mycelix Commons / supplychain** | Canonical cooperative/product/lot/order/fulfilment events; producer claims and source references; consent/access policy; correction, supersession and recall lineage; rights-governed disclosure. | Proof that a physical claim is true merely because it is signed, replicated, attested or has a reputation score. An order or forecast is not fulfilled inventory. |
| **Sol Atlas Leptos** | Read-only geospatial and temporal context: producer/pickup catchments, market areas, access corridors, seasonal/availability context where actually sourced, and privacy-preserving aggregate demand/coverage. | Canonical product identity, live inventory, nutrient truth, food-safety certification, or permission to expose exact household/farm locations. |
| **Symtropy** | Deterministic, bounded what-if rehearsal of declared preparation/workcell, storage/transport or harvest mechanics where an adequate model exists; resource/time/energy estimates and synthetic trajectories with receipts. | Physical observation, a calibrated food-process model by default, food-safety validation, a forecast guaranteed to occur, or authorization to actuate hardware. |

Existing work to compose rather than duplicate:
- Mycelix [MYC-FOOD-001 #5001](https://github.com/Luminous-Dynamics/mycelix/issues/5001).
- Sol Atlas Leptos [ATLAS-FOOD-001 #105](https://github.com/Luminous-Dynamics/sol-atlas-leptos/issues/105).
- Symtropy [NUTRI-PORTFOLIO-001 #1649](https://github.com/Luminous-Dynamics/symtropy/issues/1649).
- Symtropy [AGRI-VOI-001 #1661](https://github.com/Luminous-Dynamics/symtropy/issues/1661).
- This branch's dependency: [Symthaea robust Pareto PR #7951](https://github.com/Luminous-Dynamics/symthaea/pull/7951), built on [point evaluator PR #7929](https://github.com/Luminous-Dynamics/symthaea/pull/7929).

## Bundle meaning

`CulinaryDecisionBundleV1` is a portable **decision receipt**, not a general-purpose canonical database. It states what a producer computed from which frozen inputs, at which commit and declared time/scope. It does not inline licensed or private source datasets; it links to authorized evidence references and source/input manifests.

The envelope includes:
- schema version, bundle ID, generated-at time, producer system/component/repository/version/40-character commit SHA;
- declared purpose, geographic granularity, validity window and known-at time;
- input manifest digest, digest algorithm, evidence refs and parent bundle refs;
- named scenario IDs, exact scenario digest, baseline marker and assumption references;
- objective keys, minimize/maximize direction, units and objective-definition reference;
- per-candidate gate disposition and gate evidence;
- per-scenario objective values with explicit evidence status and references;
- rank-instability counts, robust-rankability and robust dominance relationships.

Schema validation is necessary but not sufficient. Consumers must also enforce semantic invariants listed below. Use new schema versions for incompatible changes; do not silently reinterpret old fields.

## Non-negotiable semantic invariants

1. **Known-at vs valid-at:** retain when the fact applied and when it became known. Historical replay cannot use data that was not known at that time.
2. **Source evidence vs computed value:** every objective value is either directly observed or explicitly derived/estimated/simulated with a transformation or evidence reference. Provenance pointers and digests bind records/bytes; they do not establish source truth.
3. **Missing/stale/conflicted/suppressed are not numeric zeros.** They cannot be made to look observed by a serializer or UI.
4. **Scenario identity binds joint assumptions.** Metric blocks must match their registered scenario ID/digest. Comparing candidates under different implicit assumptions is invalid.
5. **Units and state matter.** Per-100 g, per serving, package size, raw/cooked/drained state, edible fraction, currency, market, and time window must be explicit and compatible before numeric comparison.
6. **Hard gates are non-compensatory.** Unknown or failed applicable safety/allergen/rights constraints do not become eligible because another objective is excellent. A source record or model output is not a safety certification.
7. **Simulation stays simulated.** Symtropy outputs carry a `simulated` evidence status and simulator/model/run identities. They must not overwrite physical observations or be relabelled as observed price, inventory, safe temperature, harvest or household outcome.
8. **Privacy is part of the schema boundary.** Bundles carry area granularity and a privacy-policy reference, not household names, addresses, or raw purchase history by default. Sol Atlas receives only the precision and aggregates permitted by the source policy.
9. **Correction and revocation remain discoverable.** Every source reference must permit consumer-side freshness / supersession checks. A cached bundle is not automatically current after a source correction, lot recall or consent change.
10. **Authority is read-only by default.** The bundle is a recommendation/evidence product, not an order, payment, dispatch, refrigeration command, crop-control command, or human-consent grant.

## Example end-to-end flow

```text
Mycelix canonical product / lot / offer / fulfilment and consent references
                         |
                         v  (authorized, read-only projection)
Sol Atlas spatial context + source dates + permitted geography/coverage
                         |
                         v  (versioned input references, not duplicated records)
Symthaea candidate + gate evaluation + scenario-aligned robust Pareto report
                         |
              optional, declared what-if request
                         v
Symtropy synthetic run + model/profile identity + run-health/trajectory receipt
                         |
                         v
Symthaea compares simulated results as simulated evidence,
updates the decision receipt, and keeps physical outcomes separate
                         |
                         v
Sol Atlas visualizes trade-offs/coverage; Mycelix governs source access,
producer claims, corrections, fulfilment and community consent
```

Not every decision requires every system. A pantry-only recipe plan can run without Sol Atlas or Symtropy; no inventory assumption should be inferred just because those services are absent. Mycelix is not a runtime prerequisite for synthetic unit tests.

## South Africa / Johannesburg pilot considerations

The first local pilot should report source coverage before making local-affordability claims. PMBEJD publishes monthly household food basket and affordability material, including a Johannesburg basket; it is a useful external benchmark series, but it is not equivalent to current SKU-level retail prices or a nutrient database ([Household Affordability Index](https://pmbejd.org.za/index.php/household-affordability-index/)). FAO FPMA publishes domestic commodity retail/wholesale and international series where available; verify the market and commodity series' scope before projecting it onto a household ([FPMA](https://www.fao.org/giews/food-prices/price-tool/en/)). USDA FoodData Central is useful for eligible food records and nutrient fields, but must not be assumed to match every South African product or food preparation state ([API guide](https://fdc.nal.usda.gov/api-guide/)). Open Food Facts can support product identity and contributed packaged-food records, but its own documentation warns that contributed data may be inaccurate, incomplete or unreliable; check its ODbL/Database Contents License obligations before redistribution ([API overview](https://openfoodfacts.github.io/openfoodfacts-server/api/), [licensing guide](https://openfoodfacts.github.io/openfoodfacts-server/api/tutorials/license-be-on-the-legal-side/)).

The benchmark should separately measure coverage and error for SKU matching, package/unit conversion, freshness, local availability and meal nutrient calculations. The monthly affordability index can contextualize the result, but cannot be substituted for per-item prices. A U.S. nutrient record cannot silently resolve a South African branded SKU or prepared dish.

## Contract conformance corpus

The following should be frozen as synthetic fixtures before optimization tuning. They are acceptance requirements, not passing results:

- **Baseline vs price shock:** scenario-aligned price shift reverses candidate dominance; both alternatives remain visible if neither robustly dominates.
- **Stale local price:** stale price is marked stale and excluded from the current observed basket; no silent latest-known carry-forward.
- **Correction/recall:** updated Mycelix source lineage makes an old bundle detectable as superseded; affected lots/candidates do not look current merely because the old content digest remains unchanged.
- **Private geography:** exact producer/household coordinates are absent from a consumer-facing bundle unless the source policy specifically authorizes that disclosure; suppressed low-count cells are not re-identifiable via differencing.
- **Simulation separation:** a Symtropy temperature/resource trajectory is labelled simulated and cannot satisfy an observation-only safety gate.
- **Missing scenario:** absent scenario/objective evidence yields incomplete/not comparable, not a copied baseline.
- **Unit/state mismatch:** kg vs each, ZAR/kg vs ZAR/pack, raw vs cooked, or different serving denominators fail or require an explicit reviewed conversion receipt.
- **Consent withdrawn / licence incompatible:** consumer refuses access or blocks redistribution; no best-effort fallback bypasses the restriction.
- **Offline replay:** duplicate/out-of-order events do not duplicate stock or fulfilment totals; event time and receive time stay distinct.
- **Objective order / input order:** canonical output remains identical after allowed permutations.
- **Negative-control authority:** no candidate can cause payment, order placement, dispatch, actuator change or safety certification through bundle interpretation.

## Proposed integration acceptance targets

- 100% of contract fixtures pass an independent shape + semantic checker.
- Zero observed/simulated/evaluated state conflations in negative controls.
- 100% of bundle objective values can be traced to the frozen input manifest and their claim-specific evidence/transformation references.
- 100% invalid scenario digest, unit, time scope or objective schema mismatches fail closed.
- 100% unauthorized/private fields omitted or rejected in privacy fixtures.
- 100% stale/corrected/recalled-source fixtures surface a currentness or supersession disposition.
- 100% independent recomputation of synthetic Pareto scenario reports matches expected frontier and robust-dominance relations.
- Measure consumer latency, memory, bundle size and source-resolution failure rates only after correctness tests have an executable baseline.

## Implementation order

1. Qualify Symthaea #7929 and #7951 on exact heads.
2. Review and version the bundle schema here; freeze one valid and several intentionally invalid synthetic fixtures.
3. Agree on field semantics with owners of Mycelix #5001, Sol Atlas #105, Symtropy #1649 and #1661 before adding adapters.
4. Implement a single producer/exporter and one independent consumer/conformance checker; do not build four implementations in parallel.
5. Add source adapters and currentness checks after license/coverage review.
6. Add the first measured South African basket experiment; publish missingness and uncertainty before claiming affordability or outcome improvements.

## Claim boundary

This contract is a design proposal on a draft branch. It does not prove the other repositories implement it, that any live data source covers the pilot, that scenario probabilities are calibrated, or that a recommendation is safe, nutritious, culturally accepted, affordable in practice, or environmentally superior. Those claims require their own source evidence, legal/domain review where applicable, and measured evaluation.
