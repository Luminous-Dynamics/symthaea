# TEXTILE-DESIGN-TO-RECOVERY-BENCHMARK-001
## Design features → sorting → disassembly → fibre opening → yarn → receiver acceptance

Status: benchmark protocol proposal. No Luminous physical trial, industrial matched-lot comparison, or real-data model reproduction is claimed by this document.

Related:
- [TEXTILE-SORT-FUSION-001 #11743](https://github.com/Luminous-Dynamics/symthaea/issues/11743)
- [TEXTILE-DATA-INTAKE-001 #11873](https://github.com/Luminous-Dynamics/symthaea/issues/11873)
- [Symthaea sensing/design/recovery contracts PR #11744](https://github.com/Luminous-Dynamics/symthaea/pull/11744)
- [Symtropy multi-stage factory verifier PR #4927](https://github.com/Luminous-Dynamics/symtropy/pull/4927)
- [Symtropy attachment-removal contract PR #4945](https://github.com/Luminous-Dynamics/symtropy/pull/4945)

## 1. Core paradigm shift

The target is not maximum material diverted, nominal fibre-label accuracy, or maximum shredder throughput in isolation. It is **maximum accepted downstream value per unit of incoming textile**, with uncertainty, damage, labour, utilities, reuse and waste visible.

The causal chain to measure is:

`garment construction
→ observable material identity
→ safe route decision
→ reuse / repair / component recovery / pre-shredder disassembly
→ opening process
→ fibre-length distribution and usable fibre yield
→ spinning preparation and yarn performance
→ fabric performance
→ independent receiving-process acceptance`

A result at one stage cannot be promoted into a claim about a later stage. A camera classification does not prove fibre composition. A fibre-composition estimate does not prove recyclability. A high opening yield does not prove yarn quality. A yarn trial does not establish commercial acceptance.

A separate **lot-lineage ledger** is now part of the chain. It binds every incoming lot, physical transformation, split/merge, residual, terminal output and receiver decision. This prevents an intermediate sorted lot from being counted again as final fibre output and prevents one lot from being consumed by two later routes without an explicit split.

## 2. What current research says — and does not say

### 2.1 Construction affects the quality of recycled fibre

A controlled cotton study published 7 November 2026 tested 13 bespoke 100% cotton fabric constructions and isolated fibre type, yarn spinning configuration, yarn count and fabric structure. It reported that interlacing density was the primary predictor of opening efficiency in woven structures (reported correlation r = -0.94); under its laboratory conditions, up to 70% of recovered fibre mass from low-interlacement woven samples retained lengths above 12.7 mm. The authors used the manual Fibre Array method (ASTM D1440) to capture short-fibre content that Fibrogram-based approaches can miss.

This is a reason to make construction and fibre-length distribution first-class benchmark variables—not a universal 12.7 mm pass/fail threshold. Receiving mills and yarn specifications must define acceptance criteria for each intended yarn route.

Reference: Rotheram et al., Waste Management (2026), DOI 10.1016/j.wasman.2026.115980; https://pubmed.ncbi.nlm.nih.gov/42998391/

### 2.2 Downstream preparation changes achievable recycled content

A 5 May 2026 industrial spinning study reported that combing and compact ring spinning improved the handling of mechanically recycled cotton in ring-spun yarns, with trials up to 50% recycled cotton and reductions in neps/thick places for the specified combed 50% recycled-cotton condition.

The benchmark therefore must not use one generic yarn process as the sole receiving test. Carded, double-carded, combed and compact spinning are separate downstream routes with their own costs, yield losses and quality distributions.

Reference: Aslan et al., Waste Management (2026), DOI 10.1016/j.wasman.2026.115502; https://pubmed.ncbi.nlm.nih.gov/41936747/

### 2.3 Elastane-containing and stretch waste is a distinct hard case

An open-access industrial-reintegration study dated 22 October 2026 examined one post-industrial feedstock reported as 68 wt% polyester, 29 wt% viscose and 3 wt% elastane. A selected shredding–carding sequence produced approximately 68% usable fibrous material; recovered fibre was incorporated at 47.5 wt% in ring-spun yarn, whose tenacity was about 25% below its virgin-yarn reference. The paper explicitly notes the absence of independently replicated trials, an elastane-free control and an equivalent virgin fabric.

Use this as a **replication target and adversarial case**, not as a general performance baseline or a proven causal comparison.

Reference: Silva et al., Journal of Cleaner Production (2026), DOI 10.1016/j.jclepro.2026.149662; https://www.sciencedirect.com/science/article/pii/S0959652626022043

## 3. Design-time fabric architecture contract

The design dossier now has an explicit `OpeningCharacterized` intent. For every textile-bearing shell, lining, trim and padding component, it requires a type-appropriate construction descriptor with its source reference and basis weight.

- **Woven:** weave pattern, yarn system/count, warp density, weft density and interlacement density.
- **Knitted:** knit structure, yarn system/count, wales per centimetre and courses per centimetre. Woven interlacement is not fabricated for knitted goods.
- **Nonwoven:** declared bonding process.
- **Laminated:** lamination method and a reference to its substrate architecture.
- **Unknown/other:** remains incomplete until the construction is resolved; the system does not infer a process route from a generic label.

These are as-designed/specification fields used to stratify trials, not physical measurements or guarantees of opening behaviour. A mill/pattern source reference is required; the physical pilot must independently verify the sample's actual construction before opening. The 7 November cotton study's interlacement result is woven-specific and must not be transferred to knitted or nonwoven constructions.

### Construction descriptors and opening strata

The static design contract requires type-specific inputs only when `OpeningCharacterized` is declared:
- woven: weave pattern, yarn system/count, warp/weft density and interlacement density;
- knitted: knit structure, yarn system/count and wale/course density;
- nonwoven: bonding method;
- laminated: lamination method and substrate-structure reference.

The 20-case synthetic design corpus now exercises missing and unknown construction descriptors. The corpus checks metadata completeness only. An independent specimen inspection must verify that the physical article matches the declared construction before it enters a fibre-opening trial.

## 3. Benchmark arms

All arms must use the same frozen incoming specimens/lots, acceptance methods and declared end points.

| Arm | Process | Comparison question |
|---|---|---|
| A. Incumbent sorting + clipping separation | Fibersort NIR/RGB sort; cut into 20–100 mm clippings; Trimclean optical/metal detection and pneumatic separation | What accepted material and trim stream does the incumbent route actually produce? |
| B. Manual whole-garment route | Trained operator condition/sorting, component harvest and declared panel disassembly | What is achievable before shredding under a documented manual baseline? |
| C. Scripted guarded fixture | Fixed procedure with force/stop instrumentation on an admitted attachment family | What does a controlled, non-learning automation baseline achieve? |
| D. Sensing-assisted disassembly | Visual + calibrated NIR + force/tactile observations with explicit abstention | Does sensing reduce intervention, damage or residual contamination without increasing false-safe actions? |
| E. Learned robotic disassembly | Guarded robot with training and held-out construction families | Does learning improve over manual/scripted baselines under matched safety and specimen conditions? |
| F. Conventional fibre opening | Frozen incumbent opening process and settings | What is the matched reference fibre-length and usable-yield distribution? |
| G. Structure-aware opening | Process settings selected using the declared weave/knit/yarn/interlacing descriptors | Can design/structure information improve fibre retention and downstream yarn quality? |
| H. Downstream spinning matrix | Carded, double-carded, combed and compact routes as available | Which recovered fibre distributions are actually accepted by the target yarn process? |

Do not compare the pre-shredder Luminous route to Trimclean as though both perform the same operation. Trimclean is a clipping-level trim separation process after cutting. The pre-shredder route tests whether larger panels, fibres or reusable components can be preserved. Both should be measured on matched feedstock, but their output claims and process boundaries differ.

Industry source references:
- Valvan Fibersort/Trimclean overview: https://www.valvan.com/en/solutions/textile-sorting-recycling
- Trimclean leaflet (20–100 mm clipping process, optical/metal detection and pneumatic separation): https://www.valvan.com/uploads/general/VALVAN_trimclean_leaflet-A4_08_lowres.pdf?v=1685433923

## 4. Experimental matrix

### 4.1 Design variables

At minimum, stratify by:

- fibre family and blend: cotton, polyester, viscose/regenerated cellulose, polyamide, elastane-containing blends, unknown/mixed;
- yarn route: ring-spun, open-end/rotor, compact, other declared systems;
- yarn count and ply where available;
- fabric construction: woven, knit, nonwoven, multilayer/laminated;
- interlacing or loop structure, density and mass per area;
- finishes/coatings, dyes, print, membranes, seam tape and bonded layers;
- garment barriers: lining, interlining, padding, pockets, labels, zippers, fasteners, decorative trims and electronics;
- condition: new/pre-consumer, post-consumer wear/repair, contamination, moisture and damage;
- joining method: stitch class, bartack, rivet, zipper, button, snap, hook-and-loop, adhesive bond, thermal fusion, unknown.

Avoid a full Cartesian explosion. Use a preregistered fractional factorial or blocked design that preserves material/process interactions and reports which combinations were not tested.

### 4.2 Specimen and lot controls

- Assign immutable IDs to source lot, garment, exact design revision, panel, component, physical specimen, fibre output lot, yarn lot and finished fabric lot.
- Preserve sample custody and sampling weights. Do not count repeated images, replicate spectra or multiple coupons from one physical specimen as independent garments.
- Keep source garment and material family grouped across train, calibration, validation and final test.
- Split by complete garment/specimen group; separately hold out material-family and instrument groups where sample support permits.
- Freeze process settings and receiver specification before each confirmatory run.
- Record conditioning, environmental state, sampling location, measurement uncertainty and instrument calibration.
- Use blinded duplicate measurements and inter-operator/inter-laboratory checks where practical.

## 5. Measurement protocol

### 5.1 Incoming garment and design declaration

Capture:
- component/joint topology with exact product/design revision;
- declared material identity, mass and source reference for each component;
- joining method, reversibility, access zone, removal operation, required tools and dependency graph;
- visual garment family/presentation, NIR observation and any independent lab reference as separate evidence planes;
- moisture/contamination/condition observations and uncertainty.

A static design audit means only that the declared topology is internally coherent. It is not physical evidence that a garment can be repaired or disassembled.

### 5.2 Sorting and route selection

Report:
- per-class and worst-class macro-F1, balanced accuracy and confusion;
- instrument-held-out and facility-held-out performance;
- calibrated uncertainty, selective risk/coverage, open-set false acceptance and abstention rate;
- true/false acceptance and rejection at each downstream route;
- operator intervention, remeasurement and exception handling;
- latency and throughput under the actual measurement boundary.

A system that abstains safely on hard cases must report its lower coverage and increased labour cost. Do not hide those outcomes.

### 5.3 Disassembly and component recovery

Per attempt, record:
- exact garment/specimen/design revision/component/joint/tool IDs;
- action sequence, elapsed time, force trace, tool changes and human interventions;
- outcome: removed, partial, not removed, aborted or abstained;
- independently inspected residual attachment, damage area, holes, tears and severed yarns;
- recovered component identity, mass and reuse/receiving decision;
- textile output, attachment output, reject/residual mass and unexplained mass.

A partial removal is not a successful complete removal. A closed mass balance is not proof of purity. An evidence reference does not authenticate its issuer or reviewer.

### 5.4 Fibre opening and fibre quality
The fibre-length distribution must be bound to the exact physical process trial and accepted output lot, and must record:
- measurement method/standard and version;
- a versioned sampling plan (sampling locations, subsampling, conditioning and specimen selection);
- total sampled mass in milligrams;
- number of independent subsamples/replicates represented;
- expanded measurement uncertainty and the receiving-policy ceiling;
- exact accepted output lot ID and source trial evidence ID.

A valid histogram summing to 1,000,000 ppm is not sufficient if its sampling plan is absent, the replicate count is zero, the measurement uncertainty exceeds the declared receiver limit, or the sample mass exceeds the accepted output-lot mass. Record repeatability and between-subsample variation separately when available; do not treat repeated reads of one specimen as independent physical samples.


For each accepted fibre output lot, record:
- source process-trial evidence ID and exact accepted output-lot ID;
- process configuration/settings, sequence, machine, batch and energy;
- usable fibre yield, rejects, dust/fines and residual components;
- fibre-length distribution with method and uncertainty, explicitly capturing short-fibre content;
- fibre diameter/fineness, neps, trash/contamination and other properties required by the intended yarn route.

ASTM D1440 is used in the cited 2026 cotton study; confirm the current applicable edition and laboratory method before adopting it as the common protocol. Compare Fibrogram and manual/alternative methods if measurement limits could truncate the short-fibre tail.

### 5.5 Yarn, fabric and receiver acceptance

For each output yarn route, measure:
- recycled mass fraction and recipe;
- yarn tenacity/elongation, neps, thick places, thin places, hairiness and mass irregularity;
- process stability, waste extraction, ends-down/breakage and production speed;
- fabric tensile/tear, abrasion, pilling, dimensional stability, colour and functional requirements relevant to the intended product.

The receiver must record the exact specification version, pass/fail/conditional outcome and a distinct receiving-evidence reference. “Potentially spinnable” or “manufactured a sample” is not equivalent to commercial-grade acceptance.

## 6. Primary metrics

Material-flow reporting must count terminal outputs only. Publish source input mass, accepted receiver mass, rejected mass, waste, awaiting-receiver mass and unaccounted process residual separately; retain parent/child lot identities across every sorting, split, merge and recovery event. Intermediate lot masses are not final recovery yield.

Do not collapse these into one score. Report per stratum and with uncertainty intervals.

1. **Accepted downstream yield**: kg accepted by the receiving process / kg incoming material.
2. **Fibre-preservation index**: measured fibre-length distribution and mass fraction above the receiver-defined threshold, relative to the matched feedstock/reference.
3. **Useful output yield**: accepted yarn or fabric mass / incoming mass.
4. **Trim and component recovery**: recovered reusable component mass, residual trim mass and verified reuse/acceptance.
5. **Quality non-inferiority**: confidence intervals for yarn tenacity, neps, irregularity and fabric tests against the receiving specification and matched baseline.
6. **Damage**: damaged panel mass/area, severed yarn count, strength retention and rejects caused by disassembly.
7. **Resource intensity**: energy, water, steam, chemicals, labour minutes and machine time per accepted kg.
8. **Operational performance**: throughput, uptime, bottleneck utilisation, setup labour, rework, recirculation and plan churn under disruptions.
9. **Evidence completeness**: percentage of output claims bound to exact specimen, trial, measurement, output lot and independent receiving acceptance.
10. **System economics**: cost per accepted kg with explicit labour, capital, energy, maintenance, transport, depreciation and reject boundaries.

## 7. Targets — stretch objectives, not achieved results

Set numerical targets only after the matched incumbent pilot and measurement repeatability study. Initial research targets:

- zero critical false-safe disassembly actions on preregistered unsafe/unknown construction controls;
- ≥95% completion within one named, held-out attachment family, with force/damage stop rules fixed before testing;
- zero accepted-output lots with unexplained mass beyond the declared uncertainty;
- 100% fibre-length records linked to the exact process trial and accepted output lot;
- improve accepted downstream yield or fibre-length retention without violating the receiver’s yarn/fabric specification;
- at least 10% relative improvement in accepted output per incoming tonne against a matched incumbent route, only if a powered trial can establish the difference;
- no increase in worst-stratum rejection, damage or safety-critical event rate;
- quantify, rather than presume, whether pre-shredder separation preserves reusable components or longer textile fractions compared with clipping-level trim separation.

A 10% target is a programme hypothesis, not a claim of likely or currently achieved improvement. The primary win is receiver-accepted quality-adjusted output, not the arbitrary threshold itself.

## 8. Statistical and reproducibility rules

- Pre-register primary outcomes, non-inferiority margins, exclusions, stopping criteria and multiplicity treatment.
- Use independent physical garments/specimens as the unit of resampling; use garment/lot-level clustered confidence intervals or hierarchical models.
- Separate exploratory factor screening from confirmatory matched-lot comparisons.
- For multiple process arms, publish all arms and negative controls, not only the best run.
- Share schemas, scripts and non-sensitive aggregate receipts; do not upload raw restricted imagery, human identifiers or unreviewed supplier data.
- Maintain source dataset version, license, file digests, derived-data transform versions and exact code revisions.
- Report missing, unknown, abstained, failed and aborted trials as first-class outcomes.
- Require independent receiving-process evidence for any route-eligibility claim.

## 9. Digital product passport and legal boundary

The European Commission currently indicates planned Q4 2027 adoption of the ESPR delegated act for textiles, after which further guidance and technical specifications will be developed. The final textile data fields and compliance dates are not yet fixed. Use a versioned, jurisdiction-aware data model now, but do not hardcode draft requirements as current law.

Source: https://single-market-economy.ec.europa.eu/single-market/digital-product-passport/textile-apparel_en

Keep source-notice ID, legal status, effective date, jurisdiction, version and retrieval time separate from engineering provenance. A product passport must not be allowed to convert a design declaration or model prediction into a verified physical-recovery claim.

## 10. Claim ceiling

This protocol is an experimental design and benchmark specification. It does not mean the datasets have been acquired, the industrial incumbent has been measured, the human baseline has been run, a robot has executed removal, or any recovered fibre/yarn/fabric has been accepted by a commercial receiver.

A credible claim of superiority requires matched feedstock, frozen acceptance tests, independent physical measurements, reproducible analysis, uncertainty intervals, and receiving-process evidence.
