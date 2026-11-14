# Luminous Atelier — competitive research refresh and paradigm shifts

**Research cut-off:** 14 November 2026  
**Status:** current landscape synthesis and proposed evaluation programme. No comparative superiority is claimed.

## 1. Strategic position

The industry is no longer missing individual pieces of the digital apparel pipeline. Professional platforms increasingly integrate 2D/3D garment construction, points of measure, grading, simulation, tech packs, materials and AI-assisted ideation. Research systems are moving from image generation toward parametric fit, sewing-pattern synthesis, human–AI co-design and sewing-sequence planning.

Luminous Atelier should therefore not be positioned as another fashion image generator or as a replacement for professional CAD at the outset.

The differentiator is an **evidence-carrying, human-controlled garment lifecycle compiler**:

`human intent → editable design brief → pattern/3D semantics → material/process alternatives → verifiable assembly plan → qualified manufacturing route → physical inspection → repair/lifecycle record`

A parallel wardrobe path should begin with the user's existing garments:

`declared needs → wardrobe coverage → outfit/alteration/repair/borrow/rent/reuse options → optional new design`

Both paths preserve user authorship. "Make nothing" is valid. Aesthetic, cultural, ceremonial, sentimental and expressive value cannot be reduced to one universal utility score.

## 2. Professional industry baselines

### Browzwear VStitcher 2026.3.0 — released 30 October 2026

The newest official release combines formulas and points-of-measure in one table with 2D values, 3D values, targets and tolerances; it can export POM data to CSV and, for eligible enterprise/team accounts, apply target POM measurements to reshape a piece on the base size. It also expands static-point controls and per-piece 2D colour/allowance display.

**Implication:** Atelier cannot claim novelty for connecting measurements, 2D/3D POM and editable pattern shapes. Instead, benchmark semantic round-trip preservation, local ease-field intent, user-authored constraints, cross-tool portability, version-bound evidence and physical validation on top of those capabilities.

Official release notes: https://help.browzwear.com/en/articles/17747919-vstitcher-2026-3-0-release-notes  
Official download/release index: https://help.browzwear.com/en/articles/13065045-download-links

### CLO 2026.1 — released 13 August 2026

CLO reports 29 new features, including pattern properties with revision history, 2D grading review across sizes, additional knit stitch types, seam ripping/tearing and GPU simulation on supported platforms.

**Implication:** compare against a mature professional workflow, not a basic 3D viewer. Do not rebuild all of CLO's authoring and simulation functions. Use interoperability adapters and compete on lifecycle orchestration, user sovereignty, traceable constraints, qualified manufacturing and repair.

Official release: https://www.clo3d.com/en/resources/notices/1442

### Other established tool classes

- Browzwear and CLO are professional digital garment creation and simulation platforms; both already cover large parts of the 2D/3D design workflow.
- Parametric pattern projects and OpenSloper-like systems expose editable construction rules and pattern geometry; they are useful interoperability candidates rather than targets to clone wholesale.
- Garment generation research is increasingly pattern-aware. A generated image or plausible mesh is not equivalent to a topologically valid, graded, sewable pattern.

**Competitive stance:** use incumbent CAD for high-quality authoring and simulation where available. Make the full Luminous thread better at preserving a person's requirements across tools, proving which claims are supported, detecting unsupported operations, and connecting production to physical evidence and lifecycle.

## 3. Research frontier and what it means

### 3.1 Spatially explicit fit — EASE (2026)

EASE embeds garment geometry on a parametric human-body model and represents ease as spatially varying, anisotropic per-triangle scales. The paper reports direct editing of local material allowance, transfer of ease intent to new body shapes, pose adaptation that redistributes ease to reduce strain, and optimization of 2D sewing patterns under geometric and seam constraints.

Paper: https://arxiv.org/abs/2606.29419

**What Atelier should do better:** preserve a wearer's local ease intent as an explicit, editable field and version it independently from a body model or simulation. Benchmark regional ease error, pose-induced strain, pattern seam consistency and transfer to held-out body shapes. Do not claim that size labels or a single global fit score capture fit.

### 3.2 Human–AI sustainable fashion co-design (23 September 2026)

Wang, Sun and Anderson describe a framework with 4,200 garment design records, 350 expert-rated samples, 864 consumers and 426 production patterns. The study explicitly separates displayed source attribution from actual design provenance and evaluates grouped validation across design, expert, consumer and production-waste data.

Paper: https://doi.org/10.1177/24723444261488123

**What Atelier should do better:** measure aesthetic/brief adherence, human control, actual pattern correctness, material efficiency and consumer acceptance separately. Preserve designer edits, references and rights metadata. Keep affordability as a real constraint; do not presume customers will pay a sustainability premium or that a sustainability label compensates for price or fit.

### 3.3 Sewing sequence planning — multi-agent system (11 September 2026)

Park and Kim's study introduces semantic metadata for panels, edges and correspondences, plus generation, verification, question and revision agents in a human-in-the-loop cycle. It evaluates 100 patterns and reports improvement over a single-agent setup while still requiring expert intervention for some patterns.

Paper: https://doi.org/10.1177/0887302X261487373

**What Atelier should do better:** build the assembly sequence as a typed operation graph, not just generated prose. Make part/edge identity, interfaces, predecessor constraints, machine capability, ambiguity and proof obligations machine-checkable. Route the validated graph to Symtropy for resource scheduling and physical process work. Never treat a plan as machine authorization.

This has become a dedicated public workstream: [Symthaea #12134 — SEWING-PLAN-COMPILER-001](https://github.com/Luminous-Dynamics/symthaea/issues/12134).

### 3.4 Personalized concept/try-on/pattern pipelines — EasyFashion (2026)

EasyFashion is a relevant comparator because it connects personal references and text, avatar/size refinement, virtual try-on and pattern generation. It demonstrates that an end-to-end personalized pipeline is already a current research direction.

Paper: https://arxiv.org/abs/2609.18483

**What Atelier should do better:** local-first/manual-measurement paths, rights-aware references, editable pattern semantics, explicit uncertainty, hard-constraint preservation, accessible design and production qualification. These are targets for matched comparison, not asserted wins.

### 3.5 Textile sensing and sorting: distinct tasks must stay separate

**Visual garment category/presentation:** TexSortML provides 751 post-consumer garments across 11 categories and 2,253 annotated frames at three handling-disorder levels. It uses garment-level five-fold cross-validation. A directly fine-tuned ViT lost 9.3 percentage points between the easiest and hardest presentation stages, compared with 16.6–16.9 points for CNN baselines; external pretraining on cleaner second-hand imagery degraded robustness under crumpled/rotated presentations.

Paper: https://www.mdpi.com/1424-8220/26/19/6236

**NIR fibre/composition:** NIST NIR-SORT 2.0 provides near-infrared and microscopy data for known-provenance and post-consumer textiles, including custom blends, dye/finish variation and new handheld instrumentation. The data descriptor describes 113 fabric samples and 61 fibre specimens; NIST also reports new custom blend and dyed/pre-consumer specimens in 2.0.

NIST release: https://www.nist.gov/news-events/news/2026/03/nist-releases-nir-sort-20-enhancing-model-validation-textile-feedstock-0  
Dataset paper: https://doi.org/10.1038/s41597-026-07434-6

**Benchmark rule:** visual garment type, textile construction, NIR composition, sortability, stream purity and recovery acceptance are separate labels and outcomes. A visual class cannot prove fibre composition; NIR composition cannot prove a recycler will accept the item.

## 4. Regulatory and regional context — avoid overclaiming

### European Union textile Digital Product Passport

The European Commission currently lists Q4 2027 as the planned adoption of the textile-specific ESPR delegated act. The timeline is indicative and may change; detailed requirements, technical specifications and implementation guidance will follow adoption.

Official textile/apparel page: https://single-market-economy.ec.europa.eu/single-market/digital-product-passport/textile-apparel_en

**Engineering response:** make the Mycelix projection versioned and schema-driven. Do not hard-code an assumed final textile field list or advertise regulatory compliance before the final legal and technical requirements are published.

### South Africa circular textiles

GreenCape's May 2026 report describes a **voluntary, collaborative, industry-led circular textile initiative** informed by two years of stakeholder engagement. Its proposed pathway moves from coalition-building (2025–26) to formal initiative (2026–27), a 2027–2032 roadmap and an EPR concept note. This is not proof that a new mandatory textile EPR fee is currently in force.

GreenCape: https://greencape.co.za/library/developing-a-voluntary-circular-textile-initiative-for-south-africa-potential-pathways-and-value-proposition/

The South African Government's 5 November 2026 Gazette 55679 lists GoN 8145 as proposed general EPR regulations open for comment through 5 December 2026. Separate notices in that Gazette identify proposed scheme amendments for named sectors such as paper/packaging, electrical/electronic equipment, lighting and pesticides. Do not interpret GoN 8145 alone as a finalized textile-specific product obligation.

Official public-comment register: https://www.gov.za/nr/documents/public-comment?page=1

**Engineering response:** support voluntary provenance, repair and material-flow evidence without making legal-compliance claims. Keep jurisdiction, source date, notice scope, status and applicable product category explicit in policy records.

## 5. Paradigm shifts worth pursuing

1. **From prompt-to-image to a garment compiler.** A canonical, versioned intermediate representation connects brief, parts, edges, seams, local fit intent, materials, construction and process. Images are references/previews, not canonical engineering truth.
2. **From size labels to fit fields.** The user specifies ease, coverage, mobility and posture preferences by region; simulation estimates physical behavior; a real wearer validates comfort.
3. **From one optimal design to a feasible choice set.** Hard requirements are non-compensatory; unknown gates remain unknown; objective trade-offs are exposed as non-dominated alternatives.
4. **From AI as author to AI as negotiator and verifier.** Suggestions are reversible, attributable to exact revisions and selectively accepted or rejected. Human authorship and agency are measured outcomes.
5. **From isolated garment to wardrobe system.** Optimize declared needs using owned items, repair, alteration, borrowing, rental and second-hand routes before new manufacture.
6. **From static fabric library to active inverse material discovery.** Search candidate fibre/yarn/weave/knit/finish constructions; select high-information physical coupons; separate measured from predicted properties.
7. **From operation text to typed assembly graph.** Validate the graph and machine capability declarations before scheduling; handle ambiguities through expert clarification.
8. **From a recycling claim to observed recovery evidence.** Sortability and disassembly are design attributes; actual fibre identity, stream purity, recovery yield and lifecycle benefit require separate evidence.

## 6. AtelierBench — matched SOTA programme

| Lane | Primary metrics | Must-have controls |
|---|---|---|
| Design/co-design | Brief adherence, blinded expert quality, agency, time to accepted design | Same brief, references, time budget and designer experience |
| Fit/ease | Regional ease error, pose strain, pattern seam consistency, wearer acceptance | Held-out body profiles, poses, fabric families and garment identities |
| Pattern interoperability | Piece/edge/seam/grain/notch/unit preservation; explicit loss reports | Round trips across exact tool versions and supported format subsets |
| Sewing assembly | Operation precedence, edge references, machine capability, expert corrections | Held-out construction families; typed graph oracle; human clarification |
| Cloth simulation/robotics | Deformation, drape, manipulation task success, damage, Sim2Real delta | Named fabric specimens, measured properties, exact robot/sensor calibration |
| Textile sensing/sorting | Composition accuracy, garment category, presentation success, correct-stream yield | Distinct visual/NIR labels; specimen- and garment-disjoint splits |
| Wardrobe/circularity | Need coverage, repair outcomes, acquisition count, user satisfaction | No-purchase and repair-first arms; user-adjudicated compatibility |
| Lifecycle evidence | Provenance completeness, stale/revoked evidence detection, mass-balance closure | Physical event identity and independent evidence; no inferred recycled-content claims |

### Baselines

Compare against:
- Expert patternmaker/sewing technologist.
- CLO and Browzwear workflows for professional authoring and virtual fit.
- Parametric garment/pattern baseline.
- Image-only and multimodal generative fashion systems.
- The published multi-agent sewing-sequence approach.
- Manual wardrobe planning and a basic set-cover/greedy optimizer.
- NIST NIR-SORT and TexSortML published baseline protocols for their distinct sensing tasks.

Match the exact problem before comparing scores. Paper-reported results are not equivalent to independently reproduced baselines.

### Proposed stretch goals (not achieved)

- 100% detection of injected missing-edge, stale-digest, unknown dependency and unsupported-machine adversarial cases.
- Zero invalid operation graphs admitted to scheduling handoff.
- 100% preservation of supported pattern semantics on declared interchange paths.
- 30% fewer expert corrections than a reproduced sewing-sequence baseline without worse physical first-pass acceptance.
- 20% fewer assembly rework events in the named, physically tested overshirt scope.
- 30% lower median time to independently accepted production package than matched conventional CAD workflow without degrading quality, fit, agency or manufacturability.
- 20% less cutting waste and 25% fewer physical sample iterations only after a frozen, matched manufacturing baseline.
- Zero unnecessary retail purchase recommendations when the user explicitly forbids purchases; borrowing/rental/second-hand remain separately configurable.
- Zero unsupported physical, safety, environmental or regulatory claims.

These targets must be frozen after baseline reproduction, task agreement, measurement repeatability and appropriate power/reviewer agreement analysis.

## 7. Claim ceiling

A schema pass is not a physical test. A topologically valid operation graph is not proof of sewability. A 3D simulation is not wearer acceptance. A material supplier declaration is not measured composition. A product passport is not proof of recycling or environmental superiority. A benchmark win on one task does not establish universal fashion quality or inclusivity.

## 8. Primary references

- Browzwear VStitcher 2026.3.0 release notes (30 October 2026): https://help.browzwear.com/en/articles/17747919-vstitcher-2026-3-0-release-notes
- CLO 2026.1 official release (13 August 2026): https://www.clo3d.com/en/resources/notices/1442
- EASE: Parametric garment design with explicit and local ease control: https://arxiv.org/abs/2606.29419
- EasyFashion: https://arxiv.org/abs/2609.18483
- Human–AI co-created sustainable fashion: https://doi.org/10.1177/24723444261488123
- Multi-agent sewing sequence planning: https://doi.org/10.1177/0887302X261487373
- TexSortML: https://www.mdpi.com/1424-8220/26/19/6236
- NIST NIR-SORT 2.0: https://www.nist.gov/news-events/news/2026/03/nist-releases-nir-sort-20-enhancing-model-validation-textile-feedstock-0
- NIR-SORT dataset descriptor: https://doi.org/10.1038/s41597-026-07434-6
- European Commission textile DPP: https://single-market-economy.ec.europa.eu/single-market/digital-product-passport/textile-apparel_en
- GreenCape South African voluntary circular textile initiative: https://greencape.co.za/library/developing-a-voluntary-circular-textile-initiative-for-south-africa-potential-pathways-and-value-proposition/


## 9. Everyday wear: comfort, durability and lived garment service (14 November 2026)

A new application workstream is now tracked as [Symthaea #12157 — EVERYDAY-WEAR-001](https://github.com/Luminous-Dynamics/symthaea/issues/12157). It composes the existing textile qualification, material inverse-design, personal garment studio, pattern/3D interoperability, sewing-plan, spatial, cloth-physics and Mycelix lifecycle owners rather than creating parallel engines.

### Research update

**ISO 11092:2026, third edition (May 2026)** specifies the sweating guarded-hotplate ("skin model") method for thermal resistance (Rct) and water-vapour resistance (Ret), including multilayer assemblies. Its own introduction emphasizes that clothing comfort depends on interacting, time-dependent heat and mass transfer. This is an important material-characterization anchor, not a universal comfort score or a direct wearer-comfort test.

Official standard: https://www.iso.org/standard/85998.html

ASTM's current textile methods catalogue includes distinct methods for air permeability (D737), abrasion (D4966), pilling (D4970), tensile strength (D5034), tear strength (D1424/D5587), seam-related properties and dimensional stability. Select methods by garment/material scope and pin the exact edition; do not treat one test as a proxy for overall garment durability.

ASTM catalogue: https://store.astm.org/products-services/standards-and-publications/standards/textile-standards.html

A 2024 study proposing physical garment durability protocols demonstrates why testing must be product-specific: its jeans and T-shirt protocols use different combinations of abrasion, bursting/tensile strength, pilling, seam strength, spirality, dimensional stability and visual assessment. Repeated domestic laundering and visual inspection are part of the evidence, not optional decoration.

Study: https://doi.org/10.1016/j.clrc.2024.100245

### Competitive correction

Do not claim novelty for "breathable", "moisture-wicking", "durable" or "comfortable" fabric selection. These are established textile engineering concerns and individual test methods already exist. The proposed differentiation is the complete evidence path from wearer-authored requirements through garment geometry, material/construction alternatives, measured textile and garment tests, controlled wearer trials, service-life observations, repair and lifecycle records.

### New benchmark lane: EverydayWearBench

Keep material-coupon tests, whole-garment physical tests, instrumented task trials, wearer-reported comfort, care burden and observed service-life outcomes separate. Use a randomized crossover wearer evaluation where feasible; report subgroup and lower-tail outcomes, missingness and adverse events. Compare garments under matched wearer, size, activity, ambient conditions, wash history and price/garment class.

The first bounded demonstrator is the existing non-PPE woven overshirt. Separate material comparisons (fixed pattern/construction) from fit/construction comparisons (controlled materials) to avoid confounding. Climate profiles should be measured envelopes—hot/dry, hot/humid, cool/dry, cold/wet, high solar exposure and indoor/outdoor transitions—not country stereotypes. Johannesburg may be one pilot context, not the universal baseline.

### Explicit metric boundaries

- Rct/Ret, air permeability and moisture behavior are material/assembly properties under defined test conditions, not subjective comfort.
- Abrasion/pilling/tensile/tear/seam/dimensional-change results describe specific failure modes, not total service life.
- A successful standing fit does not establish seated, bending, reaching, dressing or walking comfort.
- A supplier fibre label or visual classifier does not establish independently measured composition.
- Accelerated laundering or abrasion does not establish years of real-world wear without validated correlation.
- "Water repellent" does not establish a complete garment is rainproof.
- Simulated performance is not physical or wearer evidence.

### Proposed hypotheses

The new issue records proposed targets only: zero accepted hard-constraint violations; zero unsupported comfort/durability/weather/composition claims; 20% reduction in selected rework/failure events; 20% improvement in a preregistered durability-retention endpoint without degrading wearer acceptance; and at least 15% improvement in a declared comfort or mobility profile without material lower-tail harm. These are not measured results, standards or industry benchmarks.

Issue and full protocol: https://github.com/Luminous-Dynamics/symthaea/issues/12157


## 10. Clothing-native exosuits and global terrestrial wear (14 November 2026)

New workstream: [EXOSUIT-GARMENT-001 #12162](https://github.com/Luminous-Dynamics/symthaea/issues/12162). This is an apparel/interface product owner, not a duplicate exosuit controller, dynamics engine, human-device safety authority or cloth solver. It composes MOBILITY-ASSIST-001 #7766, MOBILITY-ASSIST-006 #7850, MOBILITY-ASSIST-008 #7880, MOBILITY-ASSIST-009 #7890, benchmark harness #7784, Everyday Wear #12157, and the existing garment/compiler and Symtropy/Sol-Atlas/Mycelix interfaces.

### Research frontier

**Texoskeletons (first published 6 May 2026).** The *Advanced Functional Materials* paper explores textile-integrated one-dimensional pneumatic actuators and triboelectric movement sensors. In its reported setup, three actuators grouped in a textile lifted more than 300 g, and a wrist sleeve with 14 actuators and four sensors demonstrated wearable integration; wrist-motion classification was 85.71%. This is a research prototype and a useful textile-robotics comparator—not evidence of full-body load-bearing, all-weather, clinically effective or production-ready clothing.

Paper: https://doi.org/10.1002/adfm.75714

**System-level lightweight design review (7 July 2026).** A *Frontiers in Mechanical Engineering* review compares rigid, textile and hybrid lower-limb systems. It highlights that system mass alone is insufficient: distal mass, stiffness, load paths, human-device coupled dynamics and assistance mechanism placement matter. Its comparison table lists a 3.8–4.1 kg portable textile exosuit and much heavier rigid systems, but these are secondary-source representative values across different products and must not be treated as a controlled leaderboard.

Paper: https://doi.org/10.3389/fmech.2026.1835857

**Soft exosuit feasibility (2026).** A *Nature Communications* study of ten older adults reports a 2.86 kg textile-based hip exosuit, a mean 1.8 additional one-minute sit-to-stand repetitions and 13.6% lower walking metabolic cost in the powered-on versus powered-off comparison. The sample is small and task/population-specific; it is not directly comparable to consumer hiking products or proof of general benefit.

Paper: https://doi.org/10.1038/s41467-026-75528-1

**Consumer outdoor comparator (Hypershell X Ultra S).** The manufacturer lists 2,538 g total wearing weight including battery, IP54 and a -20 °C to 60 °C operating-temperature range. These are manufacturer-published specifications; the product page is not an independent environmental or biomechanical test. IP54 applies to the stated product configuration, not automatically every garment seam, connector, cable exit or wearer interface. Under IEC 60529, IP54 denotes dust-protected (not dust-tight) and splash-protected (not water-jet or immersion) enclosure performance for the exact tested configuration; it is not garment-wide or all-climate qualification.

Manufacturer page: https://hypershell.tech/en-us/blogs/news/inside-the-new-hypershell-x-series-upgrades-real-world-impact-and-faq

**Skin shear and interface.** An October 2026 article proposes a shear-isolating actuation path using layered webbing and a movable pulley; its motivation is that tendon-driven systems can cause chafing even when additional padding adds bulk. Treat this as a design hypothesis requiring independent reproduction, not as an established safety threshold.

Paper: https://doi.org/10.1002/adrr.70177

### Competitive interpretation and design hypothesis

Do not claim novelty for textile exosuits, garment-integrated sensors/actuators, lightweight outdoor exoskeletons or IP-rated consumer exoskeletons. The differentiator to test is an evidence-qualified **clothing-native platform** with separately replaceable:
1. skin-interface / comfort liner;
2. load-path garment chassis and mechanical anchors;
3. removable actuation, transmission, sensing, controller and battery modules;
4. climate / weather / abrasion shell modules.

Keep load-bearing seams distinct from decorative seams. Do not assume the outer weather shell is structurally qualified. Removable electronics and washable textile parts are preferred. Textile sensors and actuators stay optional until wash, sweat, UV, fatigue, drift, thermal and connector behaviour is qualified.

Keep passive support, powered soft assist, all-weather garment kit and specialist protective/EVA systems as different product profiles. A terrestrial outdoor garment is not an EVA pressure suit, survival garment or PPE product.

### Lightweight and environmental benchmarks

The new issue records a conditional research hypothesis of ≤2.0 kg total worn mass including battery for one future powered terrestrial hip-assist profile, but only if frozen matched testing demonstrates non-inferior assistance, task effectiveness, runtime, fit, reliability and safety against a named 2.5–3 kg comparison class. This is not an achieved result, not a release requirement, and not a reason to sacrifice function. Report proximal/distal mass, moment/inertia impact, volume, pressure/shear, thermal burden and don/doff burden—not mass alone.

Build explicit environmental profiles: hot/dry plus dust and solar load; hot/humid plus sweat and heavy rain; cool/wet plus wind; cold start and cold-weather operation; salt/coastal exposure; mud/grit/cleaning; and indoor/outdoor thermal transitions. For each, freeze exposure duration, temperature, humidity, rain/dust/UV condition, movement and recovery profile. "Global" means a tested family of terrestrial use envelopes, not a universal claim across every hazard.

Separate device-enclosure ingress tests from textile rain/water tests, garment seam/closure performance, battery safety/temperature, connector contamination and complete worn-system function. Exclude fire, chemical/biological hazard, immersion, severe storm, avalanche, toxic atmosphere, vacuum and radiation unless a separate specialist product programme exists.

### Standards and risk context

- ASTM F3474-25 is an active practice for selecting functional ergonomic parameters and metrics for exoskeleton evaluation; it is not a complete safety case: https://store.astm.org/f3474-25.html
- ASTM F3392-20 covers user information for wearing, care and maintenance: https://store.astm.org/standards/f3392
- ASTM F3527-24 is a task-specific environmental risk-assessment guide for exoskeleton deployment: https://shop.standards.ie/en-ie/standards/astm-f-3527-2024-1276363_saig_astm_astm_3509866/
- ISO 13482 Edition 2 is listed by ISO as under publication (stage 60.60, 11 November 2026) and will replace ISO 13482:2014. It covers service robots used in personal and professional/commercial applications, while excluding industrial and medical applications. At the research cut-off the ISO page says final production steps remain; verify formal publication and exact product-category applicability before making a conformity claim: https://www.iso.org/standard/83498.html

Consult licensed standards text and qualified domain reviewers before freezing a conformity or test claim.

## 11. Post-futurist travel robe

New workstream: [TRAVEL-ROBE-001 #12163](https://github.com/Luminous-Dynamics/symthaea/issues/12163). It composes the Personal Garment Studio #12093, Everyday Wear #12157, Wardrobe-First Optimization #12094, apparel digital thread #11731, pattern interoperability #11750, sewing-plan compiler #12134 and textile qualification #5984.

The concept is an elegant, comfortable, non-PPE travel robe or wrap-coat family: home/accommodation, terminal waiting, seated aircraft/train/bus travel, walking and variable indoor/outdoor conditions. The design language should be restrained and architectural rather than costume-like or gadget-heavy. Cultural and ceremonial references require user intent, respect and provenance; no single robe silhouette is treated as a universal aesthetic.

Commercial analogues already exist, so Atelier should not claim to invent the travel wrap or changing robe. MERSEA's Modern Travel Wrap advertises a securing loop and matching travel bag; Dryrobe's Tech-Lite changing robe advertises 1 kg with bag, weatherproof construction and packability. They occupy different product niches and are candidate comparators, not direct evidence of performance parity. Compare named garments under matched coverage, warmth, price, mass, pack volume, seated comfort, access, laundering and durability protocols.

- MERSEA Modern Travel Wrap: https://mersea.com/products/modern-travel-wrap
- Dryrobe Tech-Lite: https://dryrobe.com/products/dryrobe-lite

Key engineering questions:
- Can the robe remain comfortable sitting/reclining as well as standing and walking?
- Are closures stable but easy to operate, including one-handed variants when requested?
- Are passport/phone/ticket pockets reachable while seated, without distorting drape or creating pressure?
- Do hem, sleeve and belt elements avoid snagging on seats, luggage, wheels and steps?
- Can the garment layer across temperature changes without one heavy "all-season" material?
- Does it pack compactly, recover from creasing, dry under realistic conditions, survive laundering and remain repairable?
- Does the user control silhouette, fit, length, coverage, pocket map, material hand and aesthetic direction?

**TravelRobeBench v0.1** separates seated comfort, mobility, closure/pocket function, tactile/thermal comfort, pack volume, care, durability, style acceptance and rights/provenance. Compare with named garments of matched coverage, warmth and price class. A hidden pocket is not an anti-theft guarantee; a render is not a sewable pattern; a quick-dry label is not measured drying.

The first execution tranche is a rights-aware design brief, three silhouette hypotheses, a seated-use task map, editable pattern and typed sewing-plan fixture, synthetic negative controls, then physical sample and wearer evaluation.

Issue: https://github.com/Luminous-Dynamics/symthaea/issues/12163

## 12. Overall strategic conclusion

Atelier should not become a styling interface attached to a collection of disconnected devices. It can become the human-authored clothing and wearable-product compiler that connects garment intent, spatial fit, textile structure, mechanical interfaces, environmental modules, manufacturing plans, physical qualification, care, repair and product history.

The meaningful competitive question is not "Can we render a futuristic suit or robe?" It is whether a wearer can use the resulting garment in real tasks, in declared environments, with acceptable comfort and burden, while the design and evidence remain inspectable and repairable.
