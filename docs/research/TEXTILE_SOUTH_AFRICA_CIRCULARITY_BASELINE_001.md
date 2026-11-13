# TEXTILE-SOUTH-AFRICA-CIRCULARITY-BASELINE-001

**Status:** research and pilot protocol. No national textile waste tonnage, local sorting performance, LCA result or processor capacity is claimed by this document.

**Parent work:** [TEXTILE-SORT-FUSION-001 #11743](https://github.com/Luminous-Dynamics/symthaea/issues/11743) · [TEXTILE-ROUTE-QUALIFICATION-001 #11898](https://github.com/Luminous-Dynamics/symthaea/issues/11898)

## 1. Why a South African baseline is strategically useful

A route-selection benchmark built only on imported datasets risks measuring the wrong feedstock, markets, infrastructure and environmental conditions. South Africa needs a source-grounded baseline that can distinguish reuse, repair, component recovery, mechanical recycling, chemical recycling, downcycling and reject/disposal outcomes.

A June 2026 South African legal/industry analysis reports that much collected post-consumer clothing is downcycled into industrial rags, stuffing or insulation, while fibre-to-fibre recycling remains constrained by blended fabrics and limited local sorting/processing infrastructure. Treat this as a reported industry diagnosis, not a statistically complete national waste-flow measurement:
https://www.ensafrica.com/news/detail/11908/mending-south-africas-textile-recycling-gap

The opportunity is therefore not “classify more garments.” It is to discover which local streams can retain product value, which can deliver accepted secondary material, which are currently downcycled, and which cannot be routed safely or economically with the available infrastructure.

## 2. Current South African programme and policy context

### 2.1 UNEP InTex Phase II

UNEP's InTex Phase II in Africa runs through 2028 and includes South Africa. South African partners include the Centre for African Resource Efficiency and Sustainability (CARES) and the Council for Scientific and Industrial Research (CSIR). The published programme describes:
- sector dialogue, coordination and knowledge exchange;
- life-cycle assessment and circular-economy capacity building;
- support for textile manufacturers to reduce environmental impacts and adopt circular practices;
- a CSIR-led national textile Life Cycle Inventory (LCI) database;
- investigation of opportunities, challenges and practical approaches for textile EPR in South Africa.

Source: https://www.unep.org/en/intex

This is a potential collaboration and data-alignment context, not an endorsement, agreement, dataset licence or proof that a partnership is available.

### 2.2 Draft National Waste Management Strategy 2026

The DFFE published Draft National Waste Management Strategy 2026 (Draft 4.0) for comment in Government Gazette 53894 on 19 December 2025. Clothing and textiles are included among proposed priority waste streams. DFFE's February 2026 notice extended the comment deadline to 13 April 2026. A May 2026 budget address still described the strategy as a draft and said publication for implementation would follow finalisation.

Sources:
- https://www.gov.za/documents/notices/national-environmental-management-waste-act-national-waste-management-strategy-2
- https://www.dffe.gov.za/mediareleases_aucamp_closingdateforsubmission_dnwms2026
- https://www.gov.za/news/speeches/minister-willie-aucamp-forestry-fisheries-and-environment-dept-budget-vote-26-may

The proposed strategy's priority-stream listing is useful for pilot design; it must not be presented as a final adopted textile regulation.

### 2.3 Proposed EPR Regulations 2026 — current consultation window

Government Notice 8145, published in Government Gazette 55679 on 5 November 2026, proposes replacing the 2020 EPR Regulations with a consolidated 2026 version. The official notice invites comments until **5 December 2026**. The draft says it applies to products identified under section 18(1)(a) and separately published by the Minister. The 27-page draft notice contains no textual match for “textile” or “clothing”; this alone does not establish that textiles are outside all present or future section 18 notices.

The draft includes relevant design directions for any covered stream:
- lifecycle accounting and records for products placed on market, waste generation, collection, sorting, recycling and recovery;
- design for reuse/recycling/recovery without compromising function;
- integration of informal waste collectors, reclaimers and waste pickers;
- secondary-market development;
- measurement and allocation of material flows;
- LCA and environmental-toxicity considerations;
- reporting on reuse, recycling, recovery and beneficial use.

Source: https://www.gov.za/sites/default/files/gcis_document/202611/55679gon8145.pdf

The EPR schema should therefore carry `jurisdiction`, `source_notice_id`, `legal_status`, `effective_from`, `version` and `retrieved_at`. Never hardcode this draft as a final textile-specific obligation.

### 2.4 Waste-economy enterprise infrastructure

The Waste Economy Enterprises Repository of South Africa (WEERSA) describes itself as a national platform to register and connect waste-economy MSMEs and stakeholders. It is a discovery and ecosystem resource, not a verified directory of textile processors:
https://weersa.environment.gov.za/

The National Cleaner Production Centre South Africa also describes a clothing and textiles circular-economy project with UNIDO and GEF focused on sustainable chemicals and waste management:
https://www.industrialefficiency.co.za/clothingandtextile/

## 3. Baseline data collection design

Do not start by estimating a national total from fragmented public statements. Build a sample-framed flow baseline with provenance and coverage labels.

### 3.1 Sampling strata

At minimum, record:
- source channel: household collection, retailer take-back, commercial/institutional, factory cutting waste, unsold inventory, imported second-hand goods, other;
- geography and urban/rural context at the lowest privacy-safe resolution;
- collection actor and contract/custody model;
- product form: garment, footwear/accessory only if in scope, household textile, technical textile, factory offcut;
- condition: wearable, repairable, component-harvest candidate, non-wearable, unknown;
- composition: labelled declaration, supplier statement, sensor measurement, lab reference, unknown;
- material structure: woven/knit/nonwoven, interlacing/loop construction where measurable;
- attachments: fasteners, zips, labels, pads, elastane, coatings, laminates and unknown bonds;
- contamination: moisture, soil, oil, biological, coatings/finishes and unknown;
- route actually used and receiving-process outcome.

Separate pre-consumer and post-consumer samples. Never split repeat images, acquisitions or physical specimen replicates across training and evaluation.

### 3.2 Chain-of-custody and inclusion

A lot record should bind collector/aggregator, date, measured incoming mass, item or specimen IDs, transfers, route decisions, outputs, rejects and receiving party. Collect only personal data essential to a legitimate purpose, and do not expose waste-picker identities or location traces in public benchmark artefacts.

The baseline must measure collection fees, labour minutes, safety equipment, training, intervention and payout to informal collectors/reclaimers. The proposed 2026 EPR draft explicitly mentions integration and payment of registered waste collectors/reclaimers/pickers for collection services; if the proposal changes, keep that policy linkage versioned.

### 3.3 Measurement set

Per lot and route, record:
- incoming kg and item count with calibrated scale and uncertainty;
- kg and item count to reuse, repair, component recovery, textile fibre, chemical polymer, downcycle, reject, disposal and unresolved;
- composition reference, unresolved fraction and measurement/calibration identity;
- fibre-length distribution after opening; fines/short-fibre fraction;
- residual trim and contamination;
- damage and usable panel/component retention;
- downstream accepted kg and receiving specification;
- energy, water, steam, chemical inputs, wastewater and transport distance;
- labour/collector payments, safety incidents, downtime and rework.

No route is successful merely because material left the collection site. Count accepted output at the receiving process; record rejected and stockpiled material as such.

## 4. Pilot comparisons

For a matched lot or randomized matched sublots, compare:
1. current local status quo and actual receiving routes;
2. expert human reuse/repair grading;
3. pre-shredder component recovery and repair triage;
4. NIR/RGB sorting followed by a clipping-level Fibersort/Trimclean-like route where available;
5. mechanical fibre preparation and receiving-spinner trials;
6. chemical recycling only where a local/partner processor and safety controls exist.

Do not imply that the Fibersort + Trimclean line is currently installed in South Africa without site evidence. Use it as an industrial comparator/protocol reference until a local installation and trial are documented.

## 5. Local SOTA and outcome metrics

The first South African benchmark should not publish a single national “recycling rate” from incomplete data. Report:
- mass-flow coverage: share of sampled incoming mass with known source, route and downstream disposition;
- reuse/repair: share independently accepted for continued use;
- component recovery: reusable components accepted per input kg;
- mechanical recycling: accepted spinnable fibre mass per input kg, fibre-length distribution, yarn performance and rejection;
- chemical recycling: accepted polymer mass and functional quality per input kg;
- downcycling: accepted product mass and declared grade;
- reject/disposal: separate mass and reason codes;
- value retained per input kg with local selling price, transport, processing and cost boundaries;
- utility and chemical intensities per accepted kg;
- labour minutes, pay, safety and intervention per accepted kg;
- uncertainty, sample coverage and worst-case strata.

Suggested stretch objectives after the first measurement campaign—not current results:
- 100% of sampled lots have custody and mass-balance states recorded;
- 100% of route-qualified outputs have an independent receiving decision;
- no unknown, conditional or rejected route is counted as accepted recycling;
- improve accepted value per input kg without shifting hidden cost to collectors, workers, municipalities or downstream reject streams;
- publish results by feedstock/region/channel and confidence interval, not just a pooled average.

## 6. Data, governance and deployment sequence

1. Align terminology and metadata with CSIR's emerging textile LCI work; request access through proper channels rather than scraping or copying.
2. Identify existing local collection, reuse, repair, sorting, recycler and waste-picker organisations with WEERSA and sector partners.
3. Agree on a minimum privacy-preserving lot manifest and measurement protocol.
4. Run a small baseline sample before building a classifier. Establish the actual local mix of reusable, repairable, blended, contaminated and unknown items.
5. Use the Symthaea evidence gate to qualify observations and candidate routes; use Symtropy to plan only those route-qualified operations.
6. Compare route outcomes against matched status quo and actual incumbent equipment.
7. Add LCA only after local inventory data and a consistent functional unit exist. Keep climate, water scarcity, toxicity, transport and avoided-production assumptions explicit.

## 7. Claim ceiling

This document is a research protocol based on public programme and policy sources. It does not claim a final NWMS 2026, a textile-specific EPR obligation, a local Fibersort/Trimclean installation, national textile waste tonnage, an existing CSIR partnership with Luminous Dynamics, a local fibre-to-fibre capacity figure, or measured environmental benefit. Each would require direct source confirmation, written data rights, local field measurement and/or formal agreements.
