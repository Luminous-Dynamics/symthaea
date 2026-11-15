# Atelier Multi-Family Product and Manufacturing System v0.1

**Status:** product architecture and research roadmap; no style beyond the two concept briefs is physically qualified  
**Owner:** Atelier collection architecture #12182  
**Parent:** Apparel Engineering Digital Thread #11731  
**Date:** 2026-11-15

## 1. Mission

Atelier must become a complete clothing design, engineering and manufacturing system—not a single aesthetic capsule, a prompt-to-image tool, or a collection of disconnected garment prototypes.

The system should support many styles and functions while preserving the fact that different garments require different geometry, materials, processes, machines, fit criteria and useful-life models.

The governing principle is:

> One coherent engineering thread; many legitimate garment forms and manufacturing routes.

The design language remains architectural minimalism, natural tactility and subtle technical engineering. That is a family resemblance, not a universal silhouette or a restriction on cultural, expressive, ceremonial, adaptive or experimental design.

The first physically grounded anchors remain:
- **AO-01 Quiet Architecture:** woven everyday overshirt; cut-and-sew route.
- **CR-01 Aperture Wrap:** adaptable travel/rest wrap; layered woven route with optional removable warmth panel.

Neither has a dimensioned production pattern, physical toile, qualified fabric lot or completed wearer test yet.

## 2. Paradigm shift: design the product and its route together

A common failure in product development is to design a garment first and ask a factory to manufacture it later. That can create a beautiful but infeasible product, an unnecessary number of seams, excess material waste, poor repair access, expensive machinery requirements or an unsuitable care burden.

Atelier should co-design:

1. wearer intent and functional use;
2. silhouette, style and construction language;
3. body/fit profile and movement requirements;
4. textile/fibre/yarn/fabric architecture;
5. pattern and 2D/3D geometry;
6. seams, joins, closures and trims;
7. process route, equipment, tooling and skills;
8. sampling, inspection and qualification;
9. repair, care, reuse and recovery.

The route may be woven cut-and-sew, knit cut-and-sew, whole-garment knitting, layered/quilted construction, modular assembly, or a justified combination. No route is automatically superior.

Use existing authority owners:
- Symthaea: wearer intent, product requirements, design reasoning, alternatives, uncertainty and engineering-thread composition.
- Sol-Atlas: read-only spatial projection, frames, body/garment geometry and uncertainty.
- Symtropy: cloth physics, process scenarios, scheduling, robotics and machine-related benchmarks.
- Mycelix: canonical product/material/lot/process/inspection/repair/lifecycle records and authorized provenance.
- Human designer and wearer: final aesthetic intent, consent, fit and acceptance.

Do not create a second CAD kernel, textile solver, factory scheduler, machine controller, material registry or lifecycle ledger.

## 3. A reusable product architecture

### 3.1 Separate three axes

Every garment is described by three independent, composable axes:

**A. Style family** — the kind of product and its visual/construction vocabulary.

**B. Construction family** — how the material becomes an article.

**C. Functional profile** — the lived use, task, environment and wearer constraints.

Example: one product can be a relaxed woven overshirt (style family), woven cut-and-sew (construction family), with everyday indoor/outdoor comfort (functional profile). A second can use the same functional profile with a knitted construction and a very different silhouette.

Do not use a style label as a proxy for its manufacturing route, and do not use a gender category as a proxy for body dimensions or fit.

### 3.2 Initial style-family portfolio

| ID | Family | Example products | Primary construction routes | Main engineering focus |
|---|---|---|---|---|
| AT-E01 | Woven everyday tops | Shirt, overshirt, tunic | Woven cut-and-sew | Collar, sleeve pitch, placket, pocket, seated/reach comfort |
| AT-E02 | Everyday bottoms | Trousers, shorts, skirts | Woven or stretch-woven cut-and-sew | Seat, rise, crotch, waist, pocket, knee and sitting mobility |
| AT-E03 | Dresses and one-piece garments | Shirt dress, relaxed dress, jumpsuit | Woven/knit cut-and-sew; modular lining | Drape, balance, closure, coverage, don/doff and mobility |
| AT-K01 | Knitted foundations | Tee, base layer, tank | Circular knit or flat-knit panels plus linking/sewing | Gauge, stretch/recovery, skin feel, seam bulk and wash stability |
| AT-K02 | Advanced knitwear | Cardigan, sweater, seamless top | Flat knitting, fully fashioned or WHOLEGARMENT/knit-and-wear | Stitch architecture, shaping, yarn feed, machine capability and finishing |
| AT-C01 | Wraps and travel layers | CR-01, travel wrap, robe coat | Woven panels, modular lining/insulation, cut-and-sew | Coverage envelope, closure retention, seated access, packed volume |
| AT-O01 | Outer layers | Light coat, overshirt jacket, shell | Woven cut-and-sew, lined/quilted or laminated constructions | Layering, wind/rain claims only when tested, closures, seams and care |
| AT-M01 | Movement apparel | Stretch top, active trouser, mobility layer | Stretch knit/woven, flatlock/coverstitch or bonded construction | Dynamic stretch, recovery, seam extension, moisture and movement |
| AT-A01 | Adaptive apparel | One-hand closure top, seated-fit trouser, sensory-calm garment | Woven/knit or hybrid; user-specific closures | Independent don/doff, reach, access, pressure, sensory preference and adjustability |
| AT-W01 | Task/work garments | Studio jacket, maker apron, utility trouser | Woven cut-and-sew with replaceable wear components | Pockets under load, abrasion, repair and task mobility; no PPE claims |
| AT-R01 | Rest and sleepwear | Lounge set, robe, sleep top | Soft woven/knit, low-bulk seams | Tactile comfort, thermal/moisture experience, care and low-pressure construction |
| AT-X01 | Occasion, ceremonial and expressive | Formal garment, performance costume, heritage-inspired piece | Bespoke woven/knit, hand/industrial hybrid, decorative assembly | Aesthetic authorship, cultural/rightful reference, embellishment and conservation |
| AT-U01 | Accessories and soft goods | Scarf, belt, pouch, travel textile | Woven/knit, cut-and-sew, modular assembly | Contact feel, attachment, load, repair and packability |

These are planning families, not a commitment to manufacture all products immediately. The catalog should grow by evidence and capability, not by the number of names in a roadmap.

### 3.3 Functional profile library

A product may carry multiple profiles, each with distinct requirements and test plans:

- everyday indoor/outdoor;
- hot/dry, hot/humid, cool/dry or indoor/outdoor-transition use;
- travel and long-duration seated wear;
- rest/lounge/sleep;
- movement and sport-like activity (without medical or protective claims);
- task/work utility (non-PPE);
- accessible/adaptive dressing;
- sensory/tactile preferences;
- occasion, ceremonial, cultural and expressive use;
- conservation/archive/display;
- rental/shared use, repair-first or made-to-order service.

Use measured climate envelopes and declared test conditions rather than national stereotypes. Treat low-frequency, ceremonial and expressive garments as valid products: low wear frequency is not a value penalty. Preserve separate service units such as wear-hours, care cycles, events, performance runs, exhibition days or conservation years.

## 4. Manufacturing route selection

The route should be selected by product geometry, material, construction, available equipment/skills, quantity, repair route and qualified functional requirements.

| Route | Where it can fit | Advantages to evaluate | Risks/trade-offs to measure |
|---|---|---|---|
| Woven cut-and-sew | Shirts, trousers, dresses, wraps, task garments | Broad cloth availability, flexible silhouette, repairable seams | Cutting waste, seam count, fraying, fit iterations, labour/setup |
| Knit cut-and-sew | Tees, knit tops, stretch garments | Familiar garment assembly with knitted fabric | Seam bulk/extension, edge curl, differential stretch and recovery |
| Fully fashioned flat knit | Sweaters, cardigans, shaped panels | Shaping during knitting, reduced cutting waste | Gauge/yarn/needle limits, machine availability, programming and linking |
| Whole-garment / knit-and-wear | Suitable seamless or near-seamless knit products | Can eliminate many sewing operations for the knitted article | Restricted materials/geometry, machine cost, setup and finishing |
| Layered / quilted | Warmth panels, coats, wraps, insulation layers | Modular thermal construction and component replacement | Bulk, attachment pressure, wash mismatch and component complexity |
| Laminated / bonded | Selected technical outerwear and functional panels | Reduced seam bulk or specialized material behaviour | Bond durability, repair difficulty, material separation and care limits |
| Modular assembly | Removable linings, insulation, cuffs, hoods, pockets | Adaptation and component renewal | Extra interfaces, compatibility, parts inventory and attachment failure |
| Hand/industrial hybrid | Bespoke, ceremonial, decorative and low-volume work | Fine detail and cultural/expressive fidelity | Skill availability, time, repeatability and specialist care |

Never select a route because a simulator says it is fast or because a vendor markets it as sustainable. Use the exact named machine, software/backend, material, recipe, operator/skill and output inspection.

### 4.1 First route decisions

**AO-01:** woven cut-and-sew with a simple, repairable lockstitch-first assembly route. Machine sequence may use lockstitch, overlock/edge finishing, buttonhole/attachment and pressing as needed. Do not assume every machine is in the local partner's shop; capability evidence must name the actual machine.

**CR-01:** begin with woven cut-and-sew shell and modular panel construction. Compare natural-fibre warmth and technical insulation only as physical coupons. Keep the garment useful without the panel. Do not force it into whole-garment knitting.

**First knit product:** select a simple base layer or cardigan as a second route-validation anchor. Compare a real local cut-and-sew knit route with a named flat-knitting/fully-fashioned/whole-garment route only where machine access and yarn/program support are confirmed. SHIMA SEIKI and STOLL systems are industry baselines, not assumed local assets.

**First bottom garment:** a simple trouser should follow once measurement/ease and fit-toile semantics are stable. It tests rise, crotch, seat, waistband, closure, pockets, sitting, walking and machine operations that AO-01 does not cover.

## 5. Full design-to-manufacture workflow

This is a gated, branching workflow rather than a rigid waterfall. Material coupons, pattern development and construction analysis can proceed in parallel, but each accepted sample must bind the exact compatible revisions.

### Stage 0 — wearer intent and product definition

Record:
- use contexts, postures and activities;
- aesthetic direction and protected design features;
- body/fit profile and measurement source;
- hard constraints versus negotiable preferences;
- care, price, repair, accessibility and material exclusions;
- intended service unit and claims explicitly out of scope.

No photo/body scan is required. Measurements are user-controlled and private by default.

### Stage 1 — concept and style language

Create at least three distinct design directions where the problem is exploratory. Use sketch, drape study, pattern block, existing garment or text description as a starting point.

Record reference asset provenance, rights and cultural/context notes. A generated render is an ideation artifact, not a production design.

### Stage 2 — garment grammar and configuration

Specify:
- silhouette, length, volume and regional ease;
- neckline/collar, shoulder, sleeve, cuff and hem;
- closure, pocket, lining, vent, reinforcement and trim;
- modular pieces and attachment interfaces;
- color/material options chosen from physical references;
- intended configuration variants.

Preserve user-selected features and make alternatives reversible.

### Stage 3 — textile/material architecture

Material candidates carry fibre/yarn/construction evidence, supplier, SKU, lot, width, mass/area, thickness, finish, care instructions, price, stock, provenance and measurement uncertainty.

Develop matched coupons and construction trials for relevant properties. Supplier declarations, model predictions and measured laboratory results are different evidence classes.

### Stage 4 — pattern and geometry

Use a parametric block where justified, but do not force one universal block across woven shirts, trousers, knitwear, wraps and stretch apparel.

Each pattern piece needs stable identity, revision, grainline/orientation, cut quantity, seam-pair IDs, notches, darts/pleats, grade rules, seam allowances, interfacing/reinforcement maps, material role and construction notes.

Keep body measures, finished garment measures and ease in separate tables. Grade only after the declared fit block is accepted.

### Stage 5 — virtual screening and physical toile

Use specialist CAD and simulation tools for speed. Preserve source-native files and semantic sidecars. Calibrate avatars/material parameters against physical toiles and report residuals by landmark, posture and garment region.

A good render, low mesh distance or simulated tension map does not approve a pattern or establish wearer comfort. The physical toile remains a gate.

### Stage 6 — construction and manufacturing engineering

Compile the pattern and product semantics into an operation graph:
- part and edge identities;
- seam/edge correspondence and orientation;
- stitch and finish class;
- operation preconditions/postconditions;
- irreversible operations and accessibility;
- machine, tooling, thread/needle and operator-skill requirements;
- inspection, pressing, repair and bounded rework steps.

Compose the existing Sewing Plan Compiler #12134. Unknown edge pairing, operation, machine capability or safety condition must block handoff rather than be guessed.

### Stage 7 — route and machine qualification

Match the operation graph to named machine capabilities and a declared factory/resource envelope. Distinguish declared, manufacturer-specified, observed, qualified and production-qualified capabilities.

Use:
- Symtropy textile robotics #4865 for cloth handling/seam execution research;
- Symtropy knit preflight #4878 for Knitout/machine capability parsing;
- Symtropy machine observability #5311 for passive telemetry/commissioning;
- Symtropy multi-stage factory #4875 for route, machine, workforce and utility scheduling;
- MFG-PROC #5686 for general process/capability/recipe semantics.

A valid plan is not a machine command. A parser pass is not physical machine qualification.

### Stage 8 — sample, inspect and correct

Record exact garment, pattern, BOM, process, machine, material lot and sample identities. Measure as-built dimensions, seam quality, material consumption, mass, construction deviations and repair access.

Each correction creates a new revision with measured deltas and dependent-evidence invalidation.

### Stage 9 — production and quality

Start with one-off samples, then pilot batches, then small production. Select made-to-order, preorder, one-off artisanal, small-batch, rental/shared or forecast production according to the actual use and demand model.

Record first-pass yield, defects, rework, setup/changeover, accepted output, machine availability, resource consumption and workforce constraints. Do not optimize throughput by silently relaxing hard quality or worker constraints.

### Stage 10 — care, repair and lifecycle

For each style, publish care based on tested fabric/garment behaviour. Record repair instructions, replaceable components, common failure zones, reinspection and requalification. End-of-life claims require a verified receiver and material route, not merely a fibre label.

Compose Mycelix's product/lot/inspection and lifecycle owners; do not create a competing product passport.

## 6. Benchmark programme

The benchmark should be multi-layered, with separate metrics and no universal “best garment” score.

### A. Design and authorship

- hard-constraint retention;
- wearer/designer preference attainment;
- protected-feature preservation;
- alternatives/Pareto diversity;
- accepted, rejected, edited and reversed suggestions;
- time to first acceptable concept and production-ready package;
- blinded expert design evaluation under a declared style/use rubric;
- asset-rights and provenance violations.

Do not rank cultural/ceremonial expression using a generic commercial-style similarity score.

### B. Pattern and digital geometry

- seam-length matching and topology validity;
- piece/notch/grain/allowance preservation;
- grade-rule and measurement error;
- 2D/3D correspondence and unit/frame consistency;
- virtual-to-physical landmark residuals by body region and posture;
- physical correction count and expert rework.

### C. Manufacturing readiness and process

- typed plan feasibility and precedence correctness;
- machine-capability false accept/reject;
- unsupported operation detection;
- first-pass assembly acceptance;
- seam and dimensional quality;
- defects, rework, material yield and setup/changeover;
- cost per accepted functional unit, with transparent assumptions.

### D. Wearer and use performance

- separate tactile, thermal, moisture, fit/pressure, mobility and aesthetic outcomes;
- comfort trajectories over declared session times;
- movement/coverage and task completion;
- care and wash response;
- lower-tail and subgroup outcomes, not only means.

Use EverydayWearBench #12157 and TravelRobeBench under #12163 rather than one pooled clothing score.

### E. Service life and circularity

- dimensional stability and appearance retention;
- repair completion, time, cost, parts and reinspection;
- wear/care cycles or the correct style-specific service unit;
- components separability and verified end-of-life route;
- lifecycle cost with explicit lifetime assumptions.

A low-frequency ceremonial garment is not to be judged as though it were a daily T-shirt.

### F. Factory and machine

- accepted garments/coupons per hour;
- machine setup and changeover;
- availability, stoppage and recovery;
- energy/water/waste per accepted functional unit;
- equipment observation completeness and calibration burden;
- worker skill compatibility and ergonomic constraints;
- plan churn and resilience under outages or material delays.

Use Symtropy's existing scheduling and equipment owners.

## 7. Industry/SOTA comparison strategy

Atelier must compare against named, current alternatives on matched tasks.

1. **CLO and Browzwear:** compare digital design/simulation workflows, but preserve native source files and measure virtual-to-physical error. Do not try to recreate a weaker CAD suite.
2. **SHIMA SEIKI and STOLL:** compare cut-and-sew versus fully fashioned/whole-garment knit routes for products that can genuinely be made either way. Whole-garment knitting can remove seams, but it is not applicable to every fabric, silhouette or style.
3. **Robotic sewing research:** reproduce the 2026 single-arm paper's declared geometry, fabric, lighting, speed, initialization and seam metric before claiming better seam accuracy.
4. **OEM sewing solutions:** compare passive retrofit and mixed-fleet interoperability against JUKI's smart machine and production-monitoring features. The target is broader evidence-linked integration, not an invented absence of OEM digitization.
5. **Repair/lifecycle leaders:** use Patagonia, Nudie Jeans and ASKET as repair-programme comparators. Repair itself is not Atelier's novelty.
6. **Human–AI co-design research:** reproduce the exact task and rating protocol of relevant 2026 human–AI design work before comparing time or design quality. Image-only metrics are insufficient.

Competitive advantage hypothesis: a single evidence-linked workflow can connect wearer intent, pattern and geometry, material, process, named machine, physical sample, inspected quality and repair history while preserving owner boundaries and user control. This is not yet demonstrated.

## 8. Portfolio sequencing

Do not attempt a dozen full prototypes immediately.

### Wave 1 — core construction anchors
1. AO-01 woven overshirt.
2. First simple knit foundation (tee/base layer or cardigan).
3. One trouser/bottom.

These cover woven upper-body, knit material/process, and lower-body movement/fit.

### Wave 2 — differentiated use profiles
4. CR-01 travel/rest wrap.
5. Adaptive-access garment with one-hand closure or seated-fit variant.
6. Non-PPE task/work garment with repairable stress zones.

### Wave 3 — specialist routes and expression
7. Whole-garment knitted product where a named machine route is available.
8. Light outer layer or shell, with weather claims only after complete relevant tests.
9. Occasion/ceremonial or performance garment with rights-aware decoration and style-specific service units.
10. Sleep/loungewear and accessories.

Every wave may change order depending on supplier access, machine availability, sampling cost and validated wearer demand.

## 9. Proposed programme gates (not achieved)

- 100% of accepted design constraints trace from brief to as-built sample.
- Zero unsupported fit, comfort, care, weather, protective, sustainability or material-composition claims.
- Zero production handoff with unknown required machine capability or unresolved sewing-operation semantics.
- 100% of accepted physical results bound to exact pattern/BOM/process/machine/material-lot identities.
- At least one garment in each of the woven cut-and-sew, knit and modular/layered construction families physically sampled before claiming multi-route platform capability.
- Demonstrate a matched improvement in at least one wearer-valued product outcome and one manufacturing outcome without worsening declared critical requirements.
- Show a measurable reduction in avoidable sample rework/iterations only after a conventional matched baseline is frozen.
- Preserve wearer/designer agency and plural style acceptance; do not trade these away for a single style or productivity score.

## 10. South African development strategy

Start with the Johannesburg fabric and prototype enquiry network already identified under #12182. Confirm the actual capabilities and terms of each supplier before route selection.

Keep textile swatch sourcing, pattern engineering, CMT sampling, equipment access, test-lab services and manufacturing capacity as separate workstreams. Ask each partner for:
- one-off sample/toile ability;
- machine inventory and make/model;
- operation capabilities and material envelopes;
- operator skill and fitting/inspection process;
- pattern/native-file ownership and export terms;
- MOQ, setup/changeover costs and lead time;
- repair and rework capability;
- safety documentation and maintenance;
- data access and confidentiality.

South Africa's CTCP is explicitly intended to support upgrading equipment, processes, products and people in the clothing/textile value chain. Treat it as a potential later partner/funding route, not guaranteed funding or an assumption of programme eligibility: https://www.thedtic.gov.za/financial-and-non-financial-support/incentives/clothing-and-textiles-competitiveness-programme/

## 11. Immediate execution checklist

1. Keep AO-01 as the first physical garment and obtain its actual fit profile/toile.
2. Freeze the first style-family catalog and product/route vocabulary.
3. Identify one accessible lockstitch machine for the passive machine-observability pilot (#5311).
4. Identify whether a local knitting partner/machine route is available before specifying a knit foundation.
5. Build a three-garment synthetic product-thread corpus: AO-01, a knit foundation and a CR-01 modular wrap.
6. Exercise pattern identity, BOM, sewing-plan, machine-profile and lifecycle lineage with adversarial stale-revision fixtures.
7. Compare CLO/Browzwear and native OEM machine capability as baselines rather than duplicating those products.
8. Only expand physical prototypes when material, pattern, machine, operator, sample and inspection resources are available.

## 12. Claim ceiling

This roadmap establishes product-family architecture and an executable research plan. It does not establish that any family, style, pattern, route, machine, material, comfort profile or production workflow is physically qualified. Product-family breadth is not production capacity. A digital thread is not proof of fit. A machine preflight is not a physical sew. A simulated garment is not a wearer trial. Each claim needs the evidence appropriate to the exact style, construction, material lot, process and use profile.
