# ENV-MOBILITY-006: Evidence-bounded microfactory make-versus-buy economics

**Status:** Cost-model and benchmark contract. All embedded numeric cases are explicitly synthetic test data, not supplier quotes or local price estimates. No purchase recommendation or manufacturing saving is established.  
**Parent:** [ENV-MOBILITY draft PR #7870](https://github.com/Luminous-Dynamics/symthaea/pull/7870).  
**Canonical owners to compose:** [MFG-VALUE-001 #7608](https://github.com/Luminous-Dynamics/symthaea/issues/7608), [MFG-PROC-000 #5686](https://github.com/Luminous-Dynamics/symthaea/issues/5686), [factory scheduler #7677 / PR #7678](https://github.com/Luminous-Dynamics/symthaea/pull/7678), [repair closure #5775](https://github.com/Luminous-Dynamics/symthaea/issues/5775), and [digital thread #6005](https://github.com/Luminous-Dynamics/symthaea/issues/6005).

## 1. Paradigm: don't buy a machine because it looks like a factory

Make-or-buy should be an auditable decision over **conforming outputs, lifecycle costs, service constraints and uncertainty**—not a vote for local manufacturing as ideology and not a shopping list ranked by sticker price.

For the first survey sensor/logger article (SPM-0), compare supplier-fabricated adapters and harnesses against a small local cell. Evaluate a printer, drill press, crimper/test fixtures, CNC equipment or automation only when a real bottleneck and repeated workload justify the capability. The right first factory may be a workbench plus local print/CNC services and commodity modules.

The machine itself is a dependency: it consumes capital, floor space, energy, maintenance, calibration, operator learning time and spare parts. Buying a machine may reduce unit costs while increasing fixed costs and operational fragility.

## 2. Separate feasibility from price

Apply hard gates before comparing economics:

1. **Function and quality profile:** make and buy options must target the same part definition, tolerances, lifetime/use envelope and acceptance checks.
2. **Process/part qualification:** a cheaper unqualified local process is not eligible to replace a qualified supplier for a safety-relevant application. It can still be a candidate for non-critical fixtures while its process is qualified.
3. **Source and measurement evidence:** supplier quote, currency, date, quantity breaks, freight/duties, lead time, lot traceability and inspection/return terms must be identified. Local process costs need observed cycle/labor data, not only machine nameplate data.
4. **Serviceability and supply resilience:** time-to-repair, local substitute availability, downtime risk and minimum order quantity are separate decision axes, not hidden bonuses inside an opaque score.
5. **Economics:** compare only after the first four gates establish comparable alternatives. If a required input is missing, output an abstention/range—not a fabricated zero.

## 3. Transparent cost model

For each alternative \(a\), compute annual fixed cost as

\[
F_a = C_a\,\mathrm{CRF}(r_a,n_a) + M_a + S_a + K_a
\]

where \(C_a\) is equipment/tooling capital assigned to the option, \(r_a\) is annual capital cost/discount rate, \(n_a\) is economic life in years, \(M_a\) is annual maintenance, and \(S_a+K_a\) covers separately declared site/support and calibration/qualification overhead. The capital recovery factor is

\[
\mathrm{CRF}(r,n)=
\begin{cases}
1/n & r=0,\\
\frac{r(1+r)^n}{(1+r)^n-1} & r>0.
\end{cases}
\]

Use consistent, explicitly stated real or nominal currency assumptions. Do not combine real cash flows with nominal discount rates.

Sum variable cost per attempted unit \(V_a\) from its stated components: materials or supplier quote, direct labor, energy/consumables, logistics, variable inspection, and expected rework/rejection/scrap handling. Keep component-level terms in the report rather than only exposing the total.

Let \(y_a\in(0,1]\) be the observed or explicitly assumed accepted yield. Under the initial one-good-unit-per-attempt model, the expected variable cost per conforming unit is \(V_a/y_a\). At annual demand \(D>0\) for conforming units, the first-order total annual cost per conforming unit is

\[
C_a(D)=\frac{F_a}{D}+\frac{V_a}{y_a}.
\]

This formula is a screening model, not a production cost guarantee. It assumes stable average yield and doesn't capture batch-size discontinuities, nonlinear overtime, learning curves, demand seasonality, multi-output processes or correlated failures. Model those separately when evidence warrants it.

If the make and buy options are both qualified and use the same output profile, a positive break-even demand exists when their cost curves intersect at a positive \(D\). For \(V_{make}/y_{make} < V_{buy}/y_{buy}\) and \(F_{make}>F_{buy}\):

\[
D^*=\frac{F_{make}-F_{buy}}{V_{buy}/y_{buy}-V_{make}/y_{make}}.
\]

Report “no positive break-even” if the curves do not cross at positive demand, and report “not comparable / abstain” if qualification or cost evidence is incomplete. Do not extrapolate below minimum order quantities or beyond declared capacity.

## 4. Required alternative inputs

### In-house option
- initial equipment and tooling capital, economic life and discount rate;
- annual maintenance, occupied-space allocation, training/support, calibration and quality-system cost;
- bill of materials and approved substitutions;
- operation-specific observed setup/touch/queue/rework times and labor rates;
- measured energy/consumables, inspection cost, scrap/rework distribution and accepted yield;
- capacity and utilization assumptions, maintenance downtime, and capability limits.

### Supplier option
- exact part/revision and approved supplier source;
- quote currency, timestamp/validity, unit price and quantity break;
- minimum order, freight, duties/tax treatment and delivery lead-time distribution;
- incoming inspection, carrying cost, nonconformance/returns, rework and supplier-yield treatment;
- qualification, lot genealogy, repair/replace commitments and alternate-source availability.

For both options, bind the acceptance profile and evidence maturity. Where local in-house process capability cannot meet the same declared profile, the options are not equivalent and must not be compared as if they were.

## 5. Starter synthetic corpus and how to read it

The companion JSON fixture env-mobility-make-buy-v1.json contains eight numerical test cases built only to exercise the equations and gating rules. The last case combines an unqualified in-house process with a missing supplier quote to ensure qualification gates do not become an unpriced purchase recommendation. The fixture labels every price-like number as synthetic; it includes low-demand outsource preference, a higher-demand make-preference hypothesis, yield sensitivity, an unqualified local process, missing supplier quote data, and a no-positive-break-even case.

These cases check arithmetic and decision semantics only. They are **not** estimates of the price of a 3D printer, local labor, South African suppliers, shipping, component cost, or actual factory utilization. Replace synthetic data with timestamped quotations, measured cycle times and a declared currency only after an exact SPM-0 BOM and local sourcing review exist.

## 6. Benchmark and decision gates

### A. Calculation correctness
- Evaluate the synthetic fixture independently from any production purchase/design selector.
- Match each expected decision exactly when evidence and qualification status are complete.
- Compare numeric outputs against hand calculations to a declared tolerance (initial synthetic arithmetic tolerance: \(10^{-6}\) currency units for values and \(10^{-4}\) for the break-even demand).
- Test zero discount rate, one-year life, invalid/negative costs, non-finite inputs, out-of-range yield, missing quote, missing calibration/qualification evidence, zero demand and unit/currency mismatch.
- Missing inputs never become numerical zero by default.

### B. Decision integrity
- Zero recommendations to use an ineligible/unqualified production process in place of a qualified alternative.
- Zero make-or-buy recommendations when required cost evidence is missing or alternatives don't meet the same product profile.
- Report every component's annual fixed/variable cost, yield assumption, evidence source, uncertainty/range and decision reason.
- Break-even demand and annual-cost comparisons are diagnostics, not proof that enough real demand exists.

### C. Physical/economic value
- Freeze a baseline supplier quote set and at least five observed prototype build/repair tasks before tuning the model.
- Compare low/base/high demand, accepted yield, labor and lead-time scenarios; use sensitivity intervals rather than a single best guess.
- Track conforming-unit cost, assembly/inspection labor, scrap/rework, lead time, downtime, tool maintenance, quality escape and time to restore service.
- At 25 cumulative units, a target of at least 30% lower assembly touch labor is only evaluated against the frozen five-unit measured baseline with identical acceptance criteria.
- A lifecycle-cost target (for example, at least 30% lower cost per valid observation) is considered only after measured platform, maintenance, field-support and observation-quality data exist. It is not inferable from an adapter's unit cost.

### D. Equipment ownership gate
Before buying a machine, require a named repeated operation that it enables, an observed or defensible annual workload, a process capability/inspection path, trained operator coverage, safe workspace, maintenance and replacement-part plan, and a sensitivity result that stays favorable across a plausible demand band. If these are absent, outsource, borrow, rent or use a qualified maker facility while collecting evidence.

## 7. Research anchors

- NIST's [Digital Thread for Manufacturing](https://www.nist.gov/programs-projects/digital-thread-manufacturing) describes product-definition conformance, lifecycle traceability and linking design with manufacturing/quality data. This work should preserve those identities while comparing supplier and local routes.
- NIST's [QIF technology survey](https://doi.org/10.6028/NIST.IR.8127) describes quality-information exchange across design, inspection planning, measurement, analysis and reporting. Use the existing QIF/field owners for metrology evidence rather than creating new quality data semantics.
- NIST's [requirements for integrating design, manufacturing and inspection information](https://doi.org/10.1080/00207543.2021.1920057) reports that full linking across the lifecycle has real implementation hurdles. The microfactory digital thread should start small, record its gaps and expand through conformance-tested interfaces, not assume digital continuity merely because a record has an ID.

## 8. Exit criteria

A real equipment-buy decision is not ready until the BOM and tolerance/acceptance profile are frozen, qualified alternatives and supplier quotes are traceable, local process times and yields are observed, labor/space/maintenance assumptions are explicit, and the make/buy comparison is repeatable by an independent reviewer.

The intended paradigm is a **right-sized factory**: spend capital only where evidence shows it improves conforming output, repair latency or supply resilience; use commodity modules and external manufacturing when they are better; and optimize lifecycle cost per valid observation rather than factory ownership, machine utilization or unit price in isolation.
