# Luminous Aqua — Evidence Corpus v0.1

**Snapshot:** 2026-11-18  
**Entries:** 15  
**Claims:** 75

> Source-classified evidence register. Manufacturer declarations, standards/programme criteria, local installation references, independent market samples and peer-reviewed results are kept distinct. Values marked unknown have not been inferred. This corpus is not a product certification or evidence of Aqua performance.

## Nebia Corre Four-Function Fixed Showerhead

- **Entry ID:** `brondell-nebia-corre-n400ro`
- **Subject type:** product
- **Manufacturer / publisher:** Brondell
- **SKU:** N400RO model family (N400ROCH, N400ROSRN, N400ROBL, N400RORB, N400ROBG)
- **Market:** United States manufacturer specification
- **Revision:** Not stated

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `brondell.nebia.super-saver.flow` | flow_rate | 4.5 L/min | manufacturer_declaration | declared | mode=Super Saver; unknown: flow_test_pressure, temperature_profile, test_method, sample_count | context_only: Published mode-flow claim; pressure-specific matched performance is not supplied in the sheet. | [Brondell](https://www.brondell.com/content/Nebia_Corre_Fixed_Showerhead_Spec_Sheet_web_010623.pdf); manufacturer_datasheet; date not stated |
| `brondell.nebia.other-modes.flow` | flow_rate | 5.7 L/min | manufacturer_declaration | declared | mode=Angel Hair / Hard Spray / Soft Spray; unknown: flow_test_pressure, temperature_profile, test_method, sample_count | context_only: Manufacturer-declared flow by mode; no matched task or spray-map measurements supplied. | [Brondell](https://www.brondell.com/content/Nebia_Corre_Fixed_Showerhead_Spec_Sheet_web_010623.pdf); manufacturer_datasheet; date not stated |
| `brondell.nebia.inlet-pressure.range` | supply_pressure_range | 137.9–551.6 kPa | manufacturer_declaration | derived | unknown: pressure datum, dynamic versus static definition, performance at each pressure point | context_only: Declared supply-water pressure range, converted from 20–80 psi. | [Brondell](https://www.brondell.com/content/Nebia_Corre_Fixed_Showerhead_Spec_Sheet_web_010623.pdf); manufacturer_datasheet; date not stated |

**Limitations — `brondell.nebia.super-saver.flow`:**
- Do not treat 4.5 L/min as a novel Aqua result.
- Do not infer coverage, force, comfort or low-pressure task performance from mode flow alone.

**Limitations — `brondell.nebia.other-modes.flow`:**
- Do not merge this value with Super Saver mode.

**Limitations — `brondell.nebia.inlet-pressure.range`:**
- This is not a measured pressure-flow or spray-performance curve.
- Aqua test points below the declared lower bound are exploratory, not supported-range comparisons.

## Raindance Select E Hand Shower 120 3jet PowderRain EcoSmart+

- **Entry ID:** `hansgrohe-raindance-select-e-28729000`
- **Subject type:** product
- **Manufacturer / publisher:** Hansgrohe
- **SKU:** 28729000
- **Market:** South Africa
- **Revision:** Not stated

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `hansgrohe.28729000.powderrain.flow` | flow_rate | 5.2 L/min | manufacturer_declaration | declared | mode=PowderRain; pressure=300 kPa; datum=manufacturer nominal 3 bar; unknown: independent flow confirmation, temperature profile, sample count | eligible_as_matched_test_reference: Exact South African SKU and declared mode/pressure can anchor a physical matched test; the declaration itself is not an independent result. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28729000); manufacturer_datasheet; date not stated |
| `hansgrohe.28729000.rain.flow` | flow_rate | 5.4 L/min | manufacturer_declaration | declared | mode=Rain; pressure=300 kPa; datum=manufacturer nominal 3 bar; unknown: independent flow confirmation, temperature profile, sample count | eligible_as_matched_test_reference: Exact South African SKU and declared mode/pressure can anchor a physical matched test; the declaration itself is not an independent result. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28729000); manufacturer_datasheet; date not stated |
| `hansgrohe.28729000.massage.flow` | flow_rate | 5.2 L/min | manufacturer_declaration | declared | mode=Massage; pressure=300 kPa; datum=manufacturer nominal 3 bar; unknown: independent flow confirmation, temperature profile, sample count | eligible_as_matched_test_reference: Exact South African SKU and declared mode/pressure can anchor a physical matched test; the declaration itself is not an independent result. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28729000); manufacturer_datasheet; date not stated |
| `hansgrohe.28729000.maximum-flow` | maximum_flow_rate | 6 L/min | manufacturer_declaration | declared | pressure=300 kPa; datum=manufacturer nominal 3 bar; unknown: independent flow confirmation, test tolerance | eligible_as_matched_test_reference: Manufacturer maximum-flow declaration for the named SKU. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28729000); manufacturer_datasheet; date not stated |
| `hansgrohe.28729000.minimum-operating-pressure` | minimum_operating_pressure | 150 kPa | manufacturer_declaration | derived | unknown: pressure measurement datum, performance at pressures below 1.5 bar | context_only: The datasheet states a 1.5 bar minimum flow pressure; converted to 150 kPa. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28729000); manufacturer_datasheet; date not stated |
| `hansgrohe.28729000.minimum-function-flow` | minimum_function_flow | 4.6 L/min | manufacturer_declaration | declared | unknown: exact test procedure, flow condition at minimum function | context_only: Manufacturer statement about function onset; not a task-quality endpoint. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28729000); manufacturer_datasheet; date not stated |

**Limitations — `hansgrohe.28729000.powderrain.flow`:**
- Reproduce on the exact market variant and mode.

**Limitations — `hansgrohe.28729000.rain.flow`:**
- Do not average Rain with PowderRain or massage.

**Limitations — `hansgrohe.28729000.massage.flow`:**
- Mode-specific comparator only.

**Limitations — `hansgrohe.28729000.maximum-flow`:**
- Do not infer exact flow at other pressure points.

**Limitations — `hansgrohe.28729000.minimum-operating-pressure`:**
- This is not an independent task-performance threshold or complete operating range.

**Limitations — `hansgrohe.28729000.minimum-function-flow`:**
- Do not confuse minimum function flow with minimum supply pressure.

## Activera Select S Shower Set 95 2jet EcoSmart with Varia 72 cm bar

- **Entry ID:** `hansgrohe-activera-select-s-28048000`
- **Subject type:** product
- **Manufacturer / publisher:** Hansgrohe
- **SKU:** 28048000
- **Market:** South Africa
- **Revision:** Not stated

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `hansgrohe.28048000.rain.flow` | flow_rate | 7.2 L/min | manufacturer_declaration | declared | mode=Rain; pressure=300 kPa; datum=manufacturer nominal 3 bar; unknown: independent flow confirmation, temperature profile, sample count | eligible_as_matched_test_reference: Exact South African SKU and declared mode/pressure can anchor a physical matched test. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-shower-set-95-2jet-ecosmart-with-shower-bar-varia-72-cm-28048000); manufacturer_datasheet; date not stated |
| `hansgrohe.28048000.impactrain.flow` | flow_rate | 6.5 L/min | manufacturer_declaration | declared | mode=ImpactRain; pressure=300 kPa; datum=manufacturer nominal 3 bar; unknown: independent flow confirmation, temperature profile, sample count | eligible_as_matched_test_reference: Exact South African SKU and declared mode/pressure can anchor a physical matched test. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-shower-set-95-2jet-ecosmart-with-shower-bar-varia-72-cm-28048000); manufacturer_datasheet; date not stated |
| `hansgrohe.28048000.maximum-flow` | maximum_flow_rate | 7.2 L/min | manufacturer_declaration | declared | pressure=300 kPa; datum=manufacturer nominal 3 bar; unknown: independent flow confirmation | eligible_as_matched_test_reference: Manufacturer maximum-flow declaration for the exact named shower set. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-shower-set-95-2jet-ecosmart-with-shower-bar-varia-72-cm-28048000); manufacturer_datasheet; date not stated |
| `hansgrohe.28048000.function-from` | minimum_function_flow | 3 L/min | manufacturer_declaration | declared | unknown: task completion at function onset, flow test method | context_only: Manufacturer function-onset statement; not a task outcome. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-shower-set-95-2jet-ecosmart-with-shower-bar-varia-72-cm-28048000); manufacturer_datasheet; date not stated |
| `hansgrohe.28048000.operating-pressure.range` | operating_pressure_range | 50–600 kPa | manufacturer_declaration | derived | unknown: pressure datum, performance at every point in the range | context_only: Published operating range converted from 0.5–6 bar. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-shower-set-95-2jet-ecosmart-with-shower-bar-varia-72-cm-28048000); manufacturer_datasheet; date not stated |
| `hansgrohe.28048000.pressure-responsive-nozzles` | pressure_compensation_feature | Pressure-responsive nozzle feature | manufacturer_declaration | declared | unknown: measured flow-pressure curve, independent durability and clogging tests | context_only: The manufacturer describes FlexPower as pressure-responsive. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-shower-set-95-2jet-ecosmart-with-shower-bar-varia-72-cm-28048000); manufacturer_datasheet; date not stated |

**Limitations — `hansgrohe.28048000.rain.flow`:**
- This is a manufacturer declaration, not an independent result.

**Limitations — `hansgrohe.28048000.impactrain.flow`:**
- Do not average the two spray modes.

**Limitations — `hansgrohe.28048000.maximum-flow`:**
- Do not infer maximum flow under other pressures.

**Limitations — `hansgrohe.28048000.function-from`:**
- Do not confuse function onset with a verified useful bathing minimum.

**Limitations — `hansgrohe.28048000.operating-pressure.range`:**
- This is a product operating-pressure declaration, not evidence of non-inferior bathing task at 50 kPa or every intermediate pressure.

**Limitations — `hansgrohe.28048000.pressure-responsive-nozzles`:**
- Feature description is not a measured pressure-compensation result.

## EPA WaterSense Specification for Showerheads Version 1.1

- **Entry ID:** `epa-watersense-showerhead-spec-v1-1`
- **Subject type:** programme
- **Manufacturer / publisher:** United States Environmental Protection Agency
- **SKU:** Not applicable / not stated
- **Market:** United States programme
- **Revision:** Version 1.1, 2018-07-26

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `epa.watersense.v11.maximum-rated-flow` | maximum_rated_flow | 7.6 L/min | standard_or_programme_criterion | declared | pressure points=140, 310, 550 kPa; pressure tolerance=±7 kPa; method=EPA WaterSense Specification for Showerheads v1.1 sections 3.1.1–3.1.2; unknown: individual three-point results, applicable current edition of referenced ASME/CSA method, individual maximum-flow result at each tested pressure | eligible_as_matched_test_reference: Programme maximum rated-flow criterion; U.S. label programme, not a South African legal limit. | [United States EPA WaterSense](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100VLRA.txt); official_standard_or_programme; 2018-07-26 |
| `epa.watersense.v11.minimum-flow-low-pressure` | minimum_flow_fraction | 60 % of rated maximum | standard_or_programme_criterion | declared | pressure=140 kPa; pressure tolerance=±7 kPa; method=EPA WaterSense v1.1 section 3.1.3.1; unknown: sample statistic and tolerance details in the specification | eligible_as_matched_test_reference: Low-pressure minimum-flow criterion in the programme specification. | [United States EPA WaterSense](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100VLRA.txt); official_standard_or_programme; 2018-07-26 |
| `epa.watersense.v11.minimum-flow-high-pressure` | minimum_flow_fraction | 75 % of rated maximum | standard_or_programme_criterion | declared | pressure points=310, 550 kPa; pressure tolerance=±7 kPa; method=EPA WaterSense v1.1 section 3.1.3.2; unknown: separate results at 310 and 550 kPa | eligible_as_matched_test_reference: High-pressure minimum-flow criterion in the programme specification. | [United States EPA WaterSense](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100VLRA.txt); official_standard_or_programme; 2018-07-26 |
| `epa.watersense.v11.spray-force-handheld` | minimum_spray_force | 0.56 N | standard_or_programme_criterion | declared | pressure=140 kPa; pressure tolerance=±7 kPa; method=ASME A112.18.1/CSA B125.1 spray-force procedure referenced by WaterSense v1.1 section 4.1.1 | eligible_as_matched_test_reference: Standardized spray-force criterion for high-efficiency showerheads and handheld showers. | [United States EPA WaterSense](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100VLRA.txt); official_standard_or_programme; 2018-07-26 |
| `epa.watersense.v11.spray-force-rain` | minimum_spray_force | 0.4 N | standard_or_programme_criterion | declared | pressure=140 kPa; pressure tolerance=±7 kPa; method=ASME A112.18.1/CSA B125.1 rain-shower spray-force procedure referenced by WaterSense v1.1 section 4.1.2 | eligible_as_matched_test_reference: Standardized spray-force criterion for high-efficiency rain showers. | [United States EPA WaterSense](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100VLRA.txt); official_standard_or_programme; 2018-07-26 |
| `epa.watersense.v11.annular-coverage` | spray_coverage_criteria | Annular-ring collection criteria | standard_or_programme_criterion | declared | method=ASME A112.18.1/CSA B125.1 spray-coverage procedure referenced by WaterSense v1.1 section 5.1 | eligible_as_matched_test_reference: Standardized coverage distribution criteria. | [United States EPA WaterSense](https://nepis.epa.gov/Exe/ZyPURL.cgi?Dockey=P100VLRA.txt); official_standard_or_programme; 2018-07-26 |

**Limitations — `epa.watersense.v11.maximum-rated-flow`:**
- Formal conformity requires the full specification, sample rules and referenced ASME/CSA method.

**Limitations — `epa.watersense.v11.minimum-flow-low-pressure`:**
- This is a fraction of the rated maximum, not a universal absolute flow.

**Limitations — `epa.watersense.v11.minimum-flow-high-pressure`:**
- This is a fraction of the rated maximum, not a universal absolute flow.

**Limitations — `epa.watersense.v11.spray-force-handheld`:**
- Not a direct measure of all aspects of human comfort.

**Limitations — `epa.watersense.v11.spray-force-rain`:**
- Not a direct measure of all aspects of human comfort.

**Limitations — `epa.watersense.v11.annular-coverage`:**
- Aqua's bather-plane spatial map is supplementary and does not replace this method.

## SANS 10252-1:2018 Edition 3.2 — Water supply installations for buildings

- **Entry ID:** `sans-10252-1-2018-edition-3-2`
- **Subject type:** installation_reference
- **Manufacturer / publisher:** South African Bureau of Standards
- **SKU:** Not applicable / not stated
- **Market:** South Africa
- **Revision:** Edition 3.2

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `sans10252.shower.table3.water-saving-design` | design_flow_and_pressure | 10 L/min design flow at 50–100 kPa; 12 L/min likely-maximum user-demand planning value; 6 L/min likely minimum. | installation_design_reference | declared | pressure range=50–100 kPa; method=SANS 10252-1:2018 Edition 3.2 Table 3; Table 3 footnote (a) defines likely maximum as the flow a user would normally allow the fitting to discharge, not necessarily the maximum attainable. Footnote (b) says the design-flow pressure values are used where a manufacturer's flow-pressure curve is unavailable. | context_only: Building installation design reference, not a product-efficiency label or product pass criterion. | [SABS (publicly accessible copy hosted by Mossel Bay Municipality)](https://www.mosselbay.gov.za/storage/documents/cb942e034d7e086691fa18d41a15b553.pdf); official_standard_or_programme; date not stated |
| `sans10252.shower.clause5332.maximum` | maximum_shower_discharge | 10 L/min | standard_or_programme_criterion | declared | method=SANS 10252-1:2018 Edition 3.2 clause 5.3.3.2 | context_only: Conservative clause-level design reference pending controlled SABS interpretation. | [Institute of Plumbing South Africa (IOPSA) notice quoting SANS 10252-1:2018 Edition 3.2](https://www.iopsa.org/news/7216337); industry_association_notice; date not stated |
| `sans10252.basin.aerated-mixing.design` | design_flow_and_pressure | 9 L/min at 50 kPa | installation_design_reference | declared | pressure=50 kPa; method=SANS 10252-1:2018 Edition 3.2 Table 3 | context_only: Fixture-specific building design reference, not a product performance certification. | [SABS (publicly accessible copy hosted by Mossel Bay Municipality)](https://www.mosselbay.gov.za/storage/documents/cb942e034d7e086691fa18d41a15b553.pdf); official_standard_or_programme; date not stated |
| `sans10252.bath.aerated-mixing.design` | design_flow_and_pressure | 20 L/min at 50 kPa | installation_design_reference | declared | pressure=50 kPa; method=SANS 10252-1:2018 Edition 3.2 Table 3 | context_only: Fixture-specific building design reference, not a product performance certification. | [SABS (publicly accessible copy hosted by Mossel Bay Municipality)](https://www.mosselbay.gov.za/storage/documents/cb942e034d7e086691fa18d41a15b553.pdf); official_standard_or_programme; date not stated |
| `sans10252.bidet.mixer.design` | design_flow_and_pressure | 9 L/min at 50 kPa | installation_design_reference | declared | pressure=50 kPa; method=SANS 10252-1:2018 Edition 3.2 Table 3 | context_only: Fixture-specific building design reference, not a product performance certification. | [SABS (publicly accessible copy hosted by Mossel Bay Municipality)](https://www.mosselbay.gov.za/storage/documents/cb942e034d7e086691fa18d41a15b553.pdf); official_standard_or_programme; date not stated |

**Limitations — `sans10252.shower.table3.water-saving-design`:**
- Table 3 is an installation demand / design-flow reference, not a product efficiency label or performance test.
- The 12 L/min likely-maximum figure is a user-demand planning value, not permission to exceed the separate clause 5.3.3.2 maximum discharge.
- The 50–100 kPa design-flow pressure is used where the manufacturer's flow-pressure curve is unavailable; it is not a universal product minimum-pressure rating.
- Confirm the controlled SABS text and project applicability.

**Limitations — `sans10252.shower.clause5332.maximum`:**
- The cited IOPSA notice is a summary, not the controlled SABS publication.
- Table 3's 12 L/min likely-maximum figure is a user-demand planning quantity, not necessarily the maximum attainable; it does not conflict with, or override, the separate clause-level 10 L/min discharge limit.
- Confirm controlled SABS text, amendments and project applicability.

**Limitations — `sans10252.basin.aerated-mixing.design`:**
- Confirm controlled SABS edition and project applicability.

**Limitations — `sans10252.bath.aerated-mixing.design`:**
- Confirm controlled SABS edition and project applicability.

**Limitations — `sans10252.bidet.mixer.design`:**
- Confirm controlled SABS edition and project applicability.

## CLASP South African Shower Heads Testing Report — 19-product sample

- **Entry ID:** `south-africa-clasp-19-showerhead-sample-2022`
- **Subject type:** market_sample
- **Manufacturer / publisher:** CLASP / report author Herman Strauss
- **SKU:** Not applicable / not stated
- **Market:** South Africa
- **Revision:** Published 2022; summarized by WRC in 2025

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `clasp.sa.sample.size` | sample_size | 19 products | independent_market_sample | measured | method=CLASP South African Shower Heads Testing Report; sample n=19 | context_only: Dated local product sample; product identities were removed from public report imagery. | [CLASP](https://www.clasp.ngo/research/all/south-african-shower-heads-testing-report/); independent_market_study; date not stated |
| `clasp.sa.pressure-compensating.mean-flow` | mean_flow_rate | 6.6 L/min | independent_market_sample | measured | pressure=300 kPa; method=CLASP report; WRC 2025 summary of 19-product sample; sample n=19 | context_only: Aggregate mean for the pressure-compensating subset. | [Water Research Commission summarizing CLASP South African testing](https://www.wrc.org.za/wp-content/uploads/mdocs/WW%20July%20-%20Sept%202025_web.pdf); independent_market_study; date not stated |
| `clasp.sa.fixed-orifice.mean-flow` | observed_flow_rate_range | 24–27 L/min | independent_market_sample | measured | pressure=300 kPa; method=CLASP report; WRC 2025 summary of 19-product sample; sample n=19 | context_only: Dated aggregate range for fixed-orifice products in the sample; not an exact-SKU mean or named product result. | [Water Research Commission summarizing CLASP South African testing](https://www.wrc.org.za/wp-content/uploads/mdocs/WW%20July%20-%20Sept%202025_web.pdf); independent_market_study; date not stated |

**Limitations — `clasp.sa.sample.size`:**
- Not a 2026 exhaustive market census.
- Does not support exact-SKU comparisons.

**Limitations — `clasp.sa.pressure-compensating.mean-flow`:**
- Not a named product result.
- Do not treat 6.6 L/min as the current whole-market mean.

**Limitations — `clasp.sa.fixed-orifice.mean-flow`:**
- Not a named product result.
- Do not interpret the lower and upper values as an uncertainty interval or confidence interval.
- Sample composition and market availability may have changed.

## Consequences of the Integration of a Hyperbolic Funnel into a Showerhead for Droplets, Jet Break-Up Lengths, and Physical-Chemical Parameters

- **Entry ID:** `hyperbolic-vortex-showerhead-study-2019`
- **Subject type:** research_study
- **Manufacturer / publisher:** Not applicable / not stated
- **SKU:** Not applicable / not stated
- **Market:** Laboratory research
- **Revision:** Water 2019, 11(12), 2446

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `vortex.showerhead.same-jet-velocity.water-reduction` | water_reduction_at_similar_mean_jet_velocity | 14.4 % | peer_reviewed_study_result | measured | method=Comparative laboratory measurements at 6, 7, 8 and 9 L/min; weighted jet-velocity analysis; unknown: human task quality in a new trial, independent replication in other showerhead geometry, full-service cleaning/purge burden | context_only: Peer-reviewed prior art for hyperbolic vortex geometry; reported reduction is based on mean jet-velocity equivalence. | [Water journal / van de Griend et al.](https://www.mdpi.com/2073-4441/11/12/2446); peer_reviewed_article; date not stated |
| `vortex.showerhead.chemistry-observations` | water_chemistry_observation | Qualitative water-chemistry changes observed | peer_reviewed_study_result | measured | method=Separate recirculating chemical-characterization rig; unknown: different water chemistries, independent hygiene meaning, microbial outcomes | context_only: Observation from the study's chemistry rig. | [Water journal / van de Griend et al.](https://www.mdpi.com/2073-4441/11/12/2446); peer_reviewed_article; date not stated |

**Limitations — `vortex.showerhead.same-jet-velocity.water-reduction`:**
- Authors inferred shower experience from earlier jet-velocity literature; no new human comfort trial was performed.
- Authors noted changed centre/peripheral spray distribution may or may not be desirable.
- Not an Aqua result or guaranteed transferable improvement.

**Limitations — `vortex.showerhead.chemistry-observations`:**
- Not evidence of improved or degraded hygiene by itself.
- Any relevance for Aqua must be decided by Mycelix / competent water-quality authority.

## Wise Choice of Showerhead Patterns: How to Save Energy during Showering While Maintaining Thermal Comfort

- **Entry ID:** `shower-pattern-thermal-transfer-study-2024`
- **Subject type:** research_study
- **Manufacturer / publisher:** Not applicable / not stated
- **SKU:** Not applicable / not stated
- **Market:** Laboratory research
- **Revision:** Water 2024, 16(7), 928

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `thermal-pattern.energy-reduction` | potential_energy_reduction | 20 % | peer_reviewed_study_result | derived | method=Five showerheads, 18 spray patterns, six showering conditions (two water temperatures × three flow rates); sample n=5; unknown: replication with Aqua geometry, whole-service water and energy accounting, independent human-factors endpoint | context_only: Study-specific comparison derived from tested spray-pattern thermal-transfer performance. | [Water journal / Zhang, Mui and Wong](https://www.mdpi.com/2073-4441/16/7/928); peer_reviewed_article; 2024-03-22 |
| `thermal-pattern.parameters` | influential_design_parameters | Nozzle area ratio, pressure and resistance factor identified as influential parameters | peer_reviewed_study_result | measured | method=Regression and statistical analyses of 18 patterns; sample n=5; unknown: Aqua material/manufacturing effects, pressure range beyond the study's conditions | context_only: Useful DOE prior art for spray/thermal co-design. | [Water journal / Zhang, Mui and Wong](https://www.mdpi.com/2073-4441/16/7/928); peer_reviewed_article; 2024-03-22 |

**Limitations — `thermal-pattern.energy-reduction`:**
- Not a universal 20% efficiency guarantee.
- Do not apply directly to Aqua without matched test, calibrated thermal boundary and independent task/comfort assessment.

**Limitations — `thermal-pattern.parameters`:**
- The paper's factor values and response relationships need exact replication before use as an Aqua design rule.

## Activera Select S Shower System 95 2jet EcoSmart with ShowerTablet Select 360 and 65 cm shower bar

- **Entry ID:** `hansgrohe-activera-system-28087000`
- **Subject type:** product
- **Manufacturer / publisher:** Hansgrohe
- **SKU:** 28087000
- **Market:** South Africa
- **Revision:** Not stated

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `hansgrohe.28087000.rain.flow` | flow_rate | 7.2 L/min | manufacturer_declaration | declared | mode=Rain; pressure=300 kPa; datum=manufacturer nominal 3 bar; unknown: independent laboratory replication, temperature profile, sample count | eligible_as_matched_test_reference: Exact SA system SKU and mode can anchor a physical matched test; declaration is not a measured Aqua comparator result. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28087000); manufacturer_datasheet; date not stated |
| `hansgrohe.28087000.impactrain.flow` | flow_rate | 6.5 L/min | manufacturer_declaration | declared | mode=ImpactRain; pressure=300 kPa; datum=manufacturer nominal 3 bar; unknown: independent laboratory replication, temperature profile, sample count | eligible_as_matched_test_reference: Exact SA system SKU and mode can anchor a physical matched test; declaration is not a measured Aqua comparator result. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28087000); manufacturer_datasheet; date not stated |
| `hansgrohe.28087000.system.maximum-flow` | maximum_flow_rate | 8 L/min | manufacturer_declaration | declared | pressure=300 kPa; datum=manufacturer nominal 3 bar; unknown: independent laboratory replication, outlet / mixed operating state | eligible_as_matched_test_reference: Manufacturer-declared maximum system flow for the exact shower system. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28087000); manufacturer_datasheet; date not stated |
| `hansgrohe.28087000.minimum-pressure` | minimum_operating_pressure | 50 kPa | manufacturer_declaration | derived | unknown: flow and spray performance at exactly 0.5 bar, pressure datum, complete operating range | context_only: Manufacturer-declared minimum flow pressure converted to kPa. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28087000); manufacturer_datasheet; date not stated |
| `hansgrohe.28087000.maximum-operating-pressure` | maximum_operating_pressure | maximum ≤ 1000 kPa | manufacturer_declaration | derived | unknown: complete product pressure range statement in this claim, performance across full operating range | context_only: Manufacturer's published maximum operating pressure converted to kPa; minimum is declared in a separate claim. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28087000); manufacturer_datasheet; date not stated |
| `hansgrohe.28087000.thermostat` | thermostat_configuration | Integrated ShowerTablet Select 360 thermostat | manufacturer_declaration | declared | unknown: independent laboratory replication, complete task performance, flow curve across operating range | context_only: Exact system component configuration. | [Hansgrohe](https://pro.hansgrohe.com/articledetail-activera-select-s-shower-system-for-exposed-installation-95-2jet-ecosmart-with-showertablet-select-360-and-shower-bar-65-cm-28087000); manufacturer_datasheet; date not stated |
| `hansgrohe.28087000.safety-stop` | temperature_safety_feature | SafetyStop at 40°C | manufacturer_declaration | declared | unknown: independent laboratory replication, complete task performance, flow curve across operating range | context_only: Manufacturer feature declaration for the named system. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/download-datasheet?lang=en_ZA&productCode=28087000); manufacturer_datasheet; date not stated |
| `hansgrohe.28087000.coolcontact` | temperature_safety_feature | CoolContact thermostat housing intended to prevent the housing from heating up | manufacturer_declaration | declared | unknown: independent laboratory replication, complete task performance, flow curve across operating range | context_only: Manufacturer feature description. | [Hansgrohe](https://pro.hansgrohe.com/articledetail-activera-select-s-shower-system-for-exposed-installation-95-2jet-ecosmart-with-showertablet-select-360-and-shower-bar-65-cm-28087000); manufacturer_datasheet; date not stated |

**Limitations — `hansgrohe.28087000.rain.flow`:**
- Keep the Rain mode distinct from ImpactRain and total-system maximum.

**Limitations — `hansgrohe.28087000.impactrain.flow`:**
- Keep the ImpactRain mode distinct from Rain.

**Limitations — `hansgrohe.28087000.system.maximum-flow`:**
- The hand-shower mode maximum is separately specified as 7.2 L/min; do not conflate the two.

**Limitations — `hansgrohe.28087000.minimum-pressure`:**
- This does not establish useful bathing task performance at 50 kPa or a complete pressure/flow curve.

**Limitations — `hansgrohe.28087000.maximum-operating-pressure`:**
- This is a system operating-pressure limit, not evidence of spray/task performance at every lower pressure.

**Limitations — `hansgrohe.28087000.thermostat`:**
- Compare full installed system burden separately from a bare handheld shower set.

**Limitations — `hansgrohe.28087000.safety-stop`:**
- This is not an independent scald-safety certification or measured outlet-temperature result.

**Limitations — `hansgrohe.28087000.coolcontact`:**
- Not an independent surface-temperature test.

## Activera Select S Hand Shower 95 2jet EcoSmart+ with WaterBoost

- **Entry ID:** `hansgrohe-activera-waterboost-28036000`
- **Subject type:** product
- **Manufacturer / publisher:** Hansgrohe
- **SKU:** 28036000
- **Market:** South Africa
- **Revision:** Not stated

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `hansgrohe.28036000.rain.base-flow` | flow_rate | 2.9 L/min | manufacturer_declaration | declared | mode=Rain; pressure=300 kPa; method=Manufacturer product-page declaration; datum=manufacturer nominal 3 bar; unknown: independent laboratory replication, temperature profile, sample count, boost duration and total permitted activation time | eligible_as_matched_test_reference: Exact SKU and base spray mode are stated; this is not an independent matched result. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |
| `hansgrohe.28036000.impactrain.base-flow` | flow_rate | 2.9 L/min | manufacturer_declaration | declared | mode=ImpactRain; pressure=300 kPa; method=Manufacturer product-page declaration; datum=manufacturer nominal 3 bar; unknown: independent laboratory replication, temperature profile, sample count, boost duration and total permitted activation time | eligible_as_matched_test_reference: Exact SKU and base spray mode are stated; this is not an independent matched result. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |
| `hansgrohe.28036000.impactrain.boost-flow` | flow_rate | 6.6 L/min | manufacturer_declaration | declared | mode=ImpactRain + Booster; pressure=300 kPa; method=Manufacturer product-page declaration; datum=manufacturer nominal 3 bar; unknown: independent laboratory replication, temperature profile, sample count, boost duration and total permitted activation time | not_eligible_until_reproduced: Manufacturer declares a mode-specific boosted flow. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |
| `hansgrohe.28036000.rain.boost-flow` | flow_rate | 9.4 L/min | manufacturer_declaration | declared | mode=Rain + Booster; pressure=300 kPa; method=Manufacturer product-page declaration; datum=manufacturer nominal 3 bar; unknown: independent laboratory replication, temperature profile, sample count, boost duration and total permitted activation time | not_eligible_until_reproduced: Manufacturer declares a mode-specific boosted flow. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |
| `hansgrohe.28036000.flow-regulator` | flow_regulator_setting | 4 L/min | manufacturer_declaration | declared | method=Manufacturer product-page declaration; unknown: pressure condition for regulator value, independent measurement of tolerance | context_only: Manufacturer declaration of regulator setting. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |
| `hansgrohe.28036000.maximum-flow` | maximum_flow_rate | 4 L/min | manufacturer_declaration | declared | pressure=300 kPa; method=Manufacturer product-page declaration; datum=manufacturer nominal 3 bar; unknown: independent laboratory replication, temperature profile, sample count, boost duration and total permitted activation time | not_eligible_until_reproduced: Manufacturer's product page lists a 4 L/min maximum while the same page lists boosted mode flows of 6.6 and 9.4 L/min. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |
| `hansgrohe.28036000.minimum-pressure` | minimum_operating_pressure | 80 kPa | manufacturer_declaration | derived | method=Manufacturer product-page declaration; unknown: flow and spray performance at exactly 0.8 bar, pressure measurement datum, complete upper operating range | context_only: Manufacturer-declared minimum flow pressure converted to kPa. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |
| `hansgrohe.28036000.function-from` | minimum_function_flow | 3 L/min | manufacturer_declaration | declared | method=Manufacturer product-page declaration; unknown: pressure condition for function onset | context_only: Manufacturer feature-onset statement, not task success. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |
| `hansgrohe.28036000.waterboost-feature` | task_selective_boost_feature | User-held side button temporarily increases flow; release ends the boost request | manufacturer_declaration | declared | mode=User-held boost; method=Manufacturer product-page declaration; unknown: quantitative boost duration, boost flow at pressures other than 3 bar | context_only: Commercial precedent for an explicitly user-held temporary flow increase. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |
| `hansgrohe.28036000.specification-scope-ambiguity` | manufacturer_specification_scope_ambiguity | Unresolved scope ambiguity: the relationship between the stated 4 L/min regulator / maximum and the separately listed boosted mode flows is not explained on the page. | manufacturer_declaration | declared | pressure=300 kPa; method=Manufacturer product-page declaration; datum=manufacturer nominal 3 bar; unknown: independent laboratory replication, temperature profile, sample count, boost duration and total permitted activation time | not_eligible_until_reproduced: This is a documentation-scope ambiguity that must be clarified or independently reproduced; it is not resolved by assuming the boost bypasses the limiter. | [Hansgrohe South Africa](https://pro.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |

**Limitations — `hansgrohe.28036000.rain.base-flow`:**
- Keep base Rain separate from boosted Rain.

**Limitations — `hansgrohe.28036000.impactrain.base-flow`:**
- Keep base ImpactRain separate from boosted ImpactRain.

**Limitations — `hansgrohe.28036000.impactrain.boost-flow`:**
- Boost activation is described as press-and-hold; published maximum/regulator scope is ambiguous.
- Boost duration and all-in task burden are not supplied.

**Limitations — `hansgrohe.28036000.rain.boost-flow`:**
- Boost activation is described as press-and-hold; published maximum/regulator scope is ambiguous.
- Boost duration and all-in task burden are not supplied.

**Limitations — `hansgrohe.28036000.flow-regulator`:**
- Do not assume this figure governs or excludes WaterBoost without clarification; possible tolerance is stated separately.

**Limitations — `hansgrohe.28036000.maximum-flow`:**
- Scope of the stated maximum relative to WaterBoost is not explained; preserve as an unresolved manufacturer specification ambiguity.

**Limitations — `hansgrohe.28036000.minimum-pressure`:**
- Not independent task-performance evidence; pressure datum is not detailed on the page.

**Limitations — `hansgrohe.28036000.function-from`:**
- Do not treat the function-onset number as proof of non-inferior task quality.

**Limitations — `hansgrohe.28036000.waterboost-feature`:**
- The product page does not state a quantitative boost-duration limit or complete-task water saving result.

**Limitations — `hansgrohe.28036000.specification-scope-ambiguity`:**
- Do not select an assumed interpretation for benchmark scoring.
- Physical comparator use requires exact SKU verification and recorded manual/boost state.

## UK Energy Technology List — Showers

- **Entry ID:** `uk-energy-technology-list-showers`
- **Subject type:** programme
- **Manufacturer / publisher:** Not applicable / not stated
- **SKU:** Not applicable / not stated
- **Market:** United Kingdom; non-domestic applications
- **Revision:** Criteria page accessed 2026-11-18

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `uk.etl.showers.maximum-flow.no-ubf` | maximum_rated_flow | 9 L/min | standard_or_programme_criterion | declared | mode=Shower head / shower valve / shower system; no UBF; method=UK ETL Showers section 1.3.2; unknown: exact UWL test output for a selected commercial product, product-specific test report; Non-domestic ETL eligibility criterion. | eligible_as_matched_test_reference: External non-domestic programme criterion; apply only to the relevant product category and ETL method. | [UK Department for Energy Security and Net Zero — Energy Technology List](https://etl.energysecurity.gov.uk/products/showers/showers); official_standard_or_programme; date not stated |
| `uk.etl.showers.maximum-flow.ubf` | maximum_rated_flow | 10.6 L/min | standard_or_programme_criterion | declared | mode=Shower head / shower valve / shower system; with UBF; method=UK ETL Showers section 1.3.2; unknown: exact UWL test output for a selected commercial product, product-specific test report; Non-domestic ETL eligibility criterion. UBF means user-behaviour feature. | context_only: Higher maximum flow allowance is conditional on defined user-behaviour features; it is not a blanket efficiency advantage. | [UK Department for Energy Security and Net Zero — Energy Technology List](https://etl.energysecurity.gov.uk/products/showers/showers); official_standard_or_programme; date not stated |
| `uk.etl.showers.pressure-independence.minimum-flow` | minimum_flow_fraction | 60 % of rated maximum | standard_or_programme_criterion | declared | mode=Same exact shower product category and declared configuration; pressure points=150, 300 kPa; method=UK ETL Showers section 1.4.3.2; Unified Water Label scheme January 2023; datum=3 bar reference and 1.5 bar lower pressure; unknown: product-specific specimen result, test uncertainty and sample size for a chosen model, exact UWL test report; Only applies where 3 bar reference flow is <=8.0 L/min and the non-domestic ETL category applies. | eligible_as_matched_test_reference: Useful 1.5 bar / 3 bar pressure-independence comparator only when the ETL product, reference-flow and Unified Water Label method conditions apply. | [UK Department for Energy Security and Net Zero — Energy Technology List](https://etl.energysecurity.gov.uk/products/showers/showers); official_standard_or_programme; date not stated |
| `uk.etl.showers.spray-coverage.method` | spray_coverage_criteria | Spray coverage tested using the Unified Water Label procedure published January 2023. | standard_or_programme_criterion | declared | mode=Shower heads and shower systems; method=UK ETL Showers section 1.4.3.3; Unified Water Label scheme January 2023; unknown: individual product spray-coverage outcome, complete test report for selected SKU; Use the scheme's controlled method; the ETL summary page does not reproduce all UWL procedures. | context_only: A standards/method reference, not evidence that any particular product passed the spray-coverage procedure. | [UK Department for Energy Security and Net Zero — Energy Technology List](https://etl.energysecurity.gov.uk/products/showers/showers); official_standard_or_programme; date not stated |
| `uk.etl.electric.maximum-flow.standard` | maximum_rated_flow | 4.5 L/min | standard_or_programme_criterion | declared | mode=Standard maximum available flow for electric showers; method=UK ETL Showers criteria section 1.3.2; updated 15 September 2026; unknown: product-specific test report, whole-system energy result for any Aqua implementation; Non-domestic UK ETL pathway. The 9 L/min allowance requires power modulation, stable output temperature and ETL-compliant wastewater heat recovery; 10.6 L/min additionally requires UBF. | context_only: Current non-domestic electric-shower programme pathway; not South African law, domestic certification or evidence of Aqua performance. | [UK Department for Energy Security and Net Zero — Energy Technology List](https://etl.energysecurity.gov.uk/products/showers/showers); official_standard_or_programme; date not stated |
| `uk.etl.electric.maximum-flow.with-dwhr` | maximum_rated_flow | 9 L/min | standard_or_programme_criterion | declared | mode=Maximum available flow where electric input is modulated to minimize electricity and maintain stable output temperature with ETL-compliant wastewater heat recovery; method=UK ETL Showers criteria section 1.3.2; updated 15 September 2026; unknown: product-specific test report, whole-system energy result for any Aqua implementation; Non-domestic UK ETL pathway. The 9 L/min allowance requires power modulation, stable output temperature and ETL-compliant wastewater heat recovery; 10.6 L/min additionally requires UBF. | eligible_as_matched_test_reference: Current non-domestic electric-shower programme pathway; not South African law, domestic certification or evidence of Aqua performance. | [UK Department for Energy Security and Net Zero — Energy Technology List](https://etl.energysecurity.gov.uk/products/showers/showers); official_standard_or_programme; date not stated |
| `uk.etl.electric.maximum-flow.with-dwhr-and-ubf` | maximum_rated_flow | 10.6 L/min | standard_or_programme_criterion | declared | mode=Maximum available flow where the qualifying DWHR/power-modulation pathway also includes user-behaviour features; method=UK ETL Showers criteria section 1.3.2; updated 15 September 2026; unknown: product-specific test report, whole-system energy result for any Aqua implementation; Non-domestic UK ETL pathway. The 9 L/min allowance requires power modulation, stable output temperature and ETL-compliant wastewater heat recovery; 10.6 L/min additionally requires UBF. | eligible_as_matched_test_reference: Current non-domestic electric-shower programme pathway; not South African law, domestic certification or evidence of Aqua performance. | [UK Department for Energy Security and Net Zero — Energy Technology List](https://etl.energysecurity.gov.uk/products/showers/showers); official_standard_or_programme; date not stated |

**Limitations — `uk.etl.showers.maximum-flow.no-ubf`:**
- Not a South African legal limit or a domestic product certification.
- Maximum available flow is not proof of task quality, spray coverage or thermal comfort.

**Limitations — `uk.etl.showers.maximum-flow.ubf`:**
- Do not compare with the 9 L/min no-UBF ceiling without preserving the UBF condition.
- This is a non-domestic programme criterion, not an Aqua target.

**Limitations — `uk.etl.showers.pressure-independence.minimum-flow`:**
- The 60% value is a programme criterion, not a universal minimum for every shower.
- Flow retention alone does not establish spray force, coverage, completed task quality, comfort or safety.
- Do not claim ETL conformity from an Aqua two-point calculation alone.

**Limitations — `uk.etl.showers.spray-coverage.method`:**
- Aqua's spatial deposition map remains supplementary and is not a substitute for the UWL method.

**Limitations — `uk.etl.electric.maximum-flow.standard`:**
- Programme ceiling / eligibility condition, not a measured product result or guaranteed useful-task flow.
- Do not apply the 9 L/min pathway without compliant wastewater heat recovery, power modulation and stable output temperature.

**Limitations — `uk.etl.electric.maximum-flow.with-dwhr`:**
- Programme ceiling / eligibility condition, not a measured product result or guaranteed useful-task flow.
- Do not apply the 9 L/min pathway without compliant wastewater heat recovery, power modulation and stable output temperature.

**Limitations — `uk.etl.electric.maximum-flow.with-dwhr-and-ubf`:**
- Programme ceiling / eligibility condition, not a measured product result or guaranteed useful-task flow.
- Do not apply the 9 L/min pathway without compliant wastewater heat recovery, power modulation and stable output temperature.
- The 10.6 L/min pathway additionally requires a qualifying user-behaviour feature.

## In situ experimental and theoretical studies of the performance and the fouling impact of shower greywater heat recovery systems

- **Entry ID:** `shower-dwhr-fouling-study-2025`
- **Subject type:** research_study
- **Manufacturer / publisher:** Not applicable / not stated
- **SKU:** Not applicable / not stated
- **Market:** 14-month in-situ commercial shower DWHR study
- **Revision:** Energy and Buildings 331 (2025), 115341

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `dhwr.2025.field-duration` | field_study_duration | 14 months | peer_reviewed_study_result | measured | mode=Study-specific shower DWHR configuration; task=Domestic hot-water / shower service; method=Study-specific in-situ measurement and/or validated model; unknown: full rig and site conditions must be taken from the paper, not transferable to another climate, topology or exchanger; The article highlights a 14-month in-situ study. | context_only: Study-specific peer-reviewed result; not an Aqua result, certification value or universal DWHR guarantee. | [Bouvenot and Beaudet, Energy and Buildings](https://doi.org/10.1016/j.enbuild.2025.115341); peer_reviewed_article; 2025-03-15 |
| `dhwr.2025.overall-exchanger-efficiency` | overall_heat_exchanger_efficiency | 66 % | peer_reviewed_study_result | measured | mode=Study-specific shower DWHR configuration; task=Domestic hot-water / shower service; method=Study-specific in-situ measurement and/or validated model; unknown: full rig and site conditions must be taken from the paper, not transferable to another climate, topology or exchanger; Study-specific overall exchanger efficiency including fouling and unsteady effects. | context_only: Study-specific peer-reviewed result; not an Aqua result, certification value or universal DWHR guarantee. | [Bouvenot and Beaudet, Energy and Buildings](https://doi.org/10.1016/j.enbuild.2025.115341); peer_reviewed_article; 2025-03-15 |
| `dhwr.2025.fouling-reduction` | fouling_nominal_efficiency_reduction | 8 % | peer_reviewed_study_result | measured | mode=Study-specific shower DWHR configuration; task=Domestic hot-water / shower service; method=Study-specific in-situ measurement and/or validated model; unknown: full rig and site conditions must be taken from the paper, not transferable to another climate, topology or exchanger; The paper reports a fouling-related reduction in nominal exchanger efficiency. | context_only: Study-specific peer-reviewed result; not an Aqua result, certification value or universal DWHR guarantee. | [Bouvenot and Beaudet, Energy and Buildings](https://doi.org/10.1016/j.enbuild.2025.115341); peer_reviewed_article; 2025-03-15 |
| `dhwr.2025.overall-system-efficiency` | overall_dhwr_system_efficiency | 52 % | peer_reviewed_study_result | measured | mode=Study-specific shower DWHR configuration; task=Domestic hot-water / shower service; method=Study-specific in-situ measurement and/or validated model; unknown: full rig and site conditions must be taken from the paper, not transferable to another climate, topology or exchanger; Study-specific system efficiency includes heat losses from showerhead to drain. | context_only: Study-specific peer-reviewed result; not an Aqua result, certification value or universal DWHR guarantee. | [Bouvenot and Beaudet, Energy and Buildings](https://doi.org/10.1016/j.enbuild.2025.115341); peer_reviewed_article; 2025-03-15 |

**Limitations — `dhwr.2025.field-duration`:**
- Do not transfer the numerical result to an Aqua design without matched whole-system testing.
- Keep fouling, dynamic use, useful receiver-side heat, standby losses and parasitics inside the accounting boundary.

**Limitations — `dhwr.2025.overall-exchanger-efficiency`:**
- Do not transfer the numerical result to an Aqua design without matched whole-system testing.
- Keep fouling, dynamic use, useful receiver-side heat, standby losses and parasitics inside the accounting boundary.

**Limitations — `dhwr.2025.fouling-reduction`:**
- Do not transfer the numerical result to an Aqua design without matched whole-system testing.
- Keep fouling, dynamic use, useful receiver-side heat, standby losses and parasitics inside the accounting boundary.

**Limitations — `dhwr.2025.overall-system-efficiency`:**
- Do not transfer the numerical result to an Aqua design without matched whole-system testing.
- Keep fouling, dynamic use, useful receiver-side heat, standby losses and parasitics inside the accounting boundary.

## Optimal coupling of shower drain water heat recovery systems with domestic electrical and heat pump storage water heaters

- **Entry ID:** `shower-dhwr-system-coupling-study-2026`
- **Subject type:** research_study
- **Manufacturer / publisher:** Not applicable / not stated
- **SKU:** Not applicable / not stated
- **Market:** Study-specific domestic hot-water systems
- **Revision:** Energy and Buildings 368 (2026), 117761; published 1 October 2026

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `dhwr.2026.global-cop-experimental` | global_dhw_cop | 6.3 COP | peer_reviewed_study_result | measured | mode=Study-specific shower DWHR configuration; task=Domestic hot-water / shower service; method=Study-specific in-situ measurement and/or validated model; unknown: full rig and site conditions must be taken from the paper, not transferable to another climate, topology or exchanger; Study-reported experimental best-case global DHW COP; configuration and boundary are paper-specific. | context_only: Study-specific peer-reviewed result; not an Aqua result, certification value or universal DWHR guarantee. | [Bouvenot and Beaudet, Energy and Buildings](https://doi.org/10.1016/j.enbuild.2026.117761); peer_reviewed_article; 2026-10-01 |
| `dhwr.2026.global-cop-simulation` | global_dhw_cop | 8 COP | peer_reviewed_study_result | modelled | mode=Study-specific shower DWHR configuration; task=Domestic hot-water / shower service; method=Study-specific in-situ measurement and/or validated model; unknown: full rig and site conditions must be taken from the paper, not transferable to another climate, topology or exchanger; Study-reported simulated result, not a measured field result. | context_only: Study-specific peer-reviewed result; not an Aqua result, certification value or universal DWHR guarantee. | [Bouvenot and Beaudet, Energy and Buildings](https://doi.org/10.1016/j.enbuild.2026.117761); peer_reviewed_article; 2026-10-01 |
| `dhwr.2026.heat-pump-cop-penalty` | heat_pump_cop_reduction | 15 % | peer_reviewed_study_result | measured | mode=Study-specific shower DWHR configuration; task=Domestic hot-water / shower service; method=Study-specific in-situ measurement and/or validated model; unknown: full rig and site conditions must be taken from the paper, not transferable to another climate, topology or exchanger; 15% is the lower endpoint of a reported 15–20% effect in some configurations, not a universal penalty. | context_only: Study-specific peer-reviewed result; not an Aqua result, certification value or universal DWHR guarantee. | [Bouvenot and Beaudet, Energy and Buildings](https://doi.org/10.1016/j.enbuild.2026.117761); peer_reviewed_article; 2026-10-01 |
| `dhwr.2026.best-coupling-topology` | preferred_dhwr_coupling_topology | Unbalanced DWHR connected to the shower mixer performed best in the tested overall system configurations. | peer_reviewed_study_result | measured | mode=Study-specific heat-pump/storage and DWHR topologies; task=Shower DHW service; method=Study-specific validated numerical/experimental model; unknown: not transferable without matched topology/system testing; The best overall system topology was not necessarily the topology with the highest component recovery efficiency. | context_only: Study-specific whole-system topology finding; not a universal ranking or Aqua result. | [Bouvenot and Beaudet, Energy and Buildings](https://doi.org/10.1016/j.enbuild.2026.117761); peer_reviewed_article; 2026-10-01 |
| `dhwr.2026.model-validation` | model_validation_cv_rmse | 7 % | peer_reviewed_study_result | measured | mode=Study-specific shower DWHR configuration; task=Domestic hot-water / shower service; method=Study-specific in-situ measurement and/or validated model; unknown: full rig and site conditions must be taken from the paper, not transferable to another climate, topology or exchanger; Study-reported model validation result; not Aqua model accuracy. | context_only: Study-specific peer-reviewed result; not an Aqua result, certification value or universal DWHR guarantee. | [Bouvenot and Beaudet, Energy and Buildings](https://doi.org/10.1016/j.enbuild.2026.117761); peer_reviewed_article; 2026-10-01 |

**Limitations — `dhwr.2026.global-cop-experimental`:**
- Do not transfer the numerical result to an Aqua design without matched whole-system testing.
- Keep fouling, dynamic use, useful receiver-side heat, standby losses and parasitics inside the accounting boundary.

**Limitations — `dhwr.2026.global-cop-simulation`:**
- Do not transfer the numerical result to an Aqua design without matched whole-system testing.
- Keep fouling, dynamic use, useful receiver-side heat, standby losses and parasitics inside the accounting boundary.

**Limitations — `dhwr.2026.heat-pump-cop-penalty`:**
- Do not transfer the numerical result to an Aqua design without matched whole-system testing.
- Keep fouling, dynamic use, useful receiver-side heat, standby losses and parasitics inside the accounting boundary.

**Limitations — `dhwr.2026.best-coupling-topology`:**
- Do not assume mixer-side DWHR always wins; receiver, storage, standby and dynamic losses matter.

**Limitations — `dhwr.2026.model-validation`:**
- Do not transfer the numerical result to an Aqua design without matched whole-system testing.
- Keep fouling, dynamic use, useful receiver-side heat, standby losses and parasitics inside the accounting boundary.

## EN 18021:2025 — Sanitary tapware, measurement of functional performance of taps and showers

- **Entry ID:** `en-18021-2025-functional-performance-testing`
- **Subject type:** standard
- **Manufacturer / publisher:** Not applicable / not stated
- **SKU:** Not applicable / not stated
- **Market:** European standard; national adoption varies
- **Revision:** Current; UNI entry into force 2 July 2025

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `en18021.2025.scope.type-tests` | functional_test_scope | Laboratory type-test standard for PN10 taps and shower systems and PN5 shower outlets and shower sets; not a manufacturing QC test. Excludes shower taps alone, taps for filling baths, combined-tap bath-filling functions, and body/side jets. | standard_or_programme_criterion | declared | mode=Taps, shower outlets, shower sets and shower systems; method=EN 18021:2025 scope; UNI catalogue summary; unknown: controlled standard text and full clauses required for formal testing; Current in UNI catalogue; entry into force 2 July 2025. | context_only: Current European laboratory type-test reference; not South African law or proof that any Aqua product passed. | [UNI — Ente Italiano di Normazione](https://store.uni.com/en/en-18021-2025); official_standard_or_programme; 2025-07-02 |
| `en18021.2025.shower-outlet-pressure` | shower_outlet_operating_pressure | For shower outlets and shower sets: dynamic pressure lower operating limit 0.05 MPa (0.5 bar); static pressure maximum 0.5 MPa (5 bar); recommended dynamic pressure 0.1–0.3 MPa (1–3 bar). | standard_or_programme_criterion | declared | mode=PN5 shower outlets and PN5 shower sets; method=EN 18021:2025 Table 2; UNI catalogue summary; unknown: full Table 2 measurement procedure and tolerances in controlled standard; Do not infer task quality or performance at each pressure from the operating-range boundary. | context_only: Category-specific operating range and recommended pressure from a current type-test standard; not a product result. | [UNI — Ente Italiano di Normazione](https://store.uni.com/en/en-18021-2025); official_standard_or_programme; 2025-07-02 |
| `en18021.2025.tap-system-pressure` | tap_shower_system_operating_pressure | For taps and shower systems: dynamic pressure lower operating limit 0.05 MPa (0.5 bar); static pressure maximum 1.0 MPa (10 bar); recommended dynamic pressure 0.1–0.5 MPa (1–5 bar), with recommended dynamic value measured at the point of discharge. | standard_or_programme_criterion | declared | mode=PN10 taps and PN10 shower systems; method=EN 18021:2025 Table 1; UNI catalogue summary; datum=point of discharge for recommended dynamic pressure per Table 1 footnote; unknown: full Table 1 measurement procedure and tolerances in controlled standard; Do not treat the point-of-discharge datum as the terminal fitting inlet. | context_only: Category-specific type-test operating range; point-of-discharge datum must be preserved. | [UNI — Ente Italiano di Normazione](https://store.uni.com/en/en-18021-2025); official_standard_or_programme; 2025-07-02 |
| `en18021.2025.temperature-limits` | temperature_operating_limits | Maximum 70°C only for short periods not exceeding one hour; recommended temperature up to 42°C for shower outlets/sets and up to 65°C for taps/shower systems. | standard_or_programme_criterion | declared | mode=Table 1 taps/systems and Table 2 shower outlets/sets; method=EN 18021:2025 Tables 1–2; UNI catalogue summary; unknown: full temperature test procedure and tolerance in controlled standard; This is a functional-test operating summary, not an anti-scald setting or user-temperature recommendation for Aqua. | context_only: Category-specific standard temperature limits and recommended test temperatures. | [UNI — Ente Italiano di Normazione](https://store.uni.com/en/en-18021-2025); official_standard_or_programme; 2025-07-02 |

**Limitations — `en18021.2025.scope.type-tests`:**
- Do not treat type testing as production quality control.
- Do not transfer pressure/temperature limits between product categories.
- Verify controlled national adoption and the exact standard before a formal claim.

**Limitations — `en18021.2025.shower-outlet-pressure`:**
- The 0.5 bar lower limit is an operating-range boundary, not a guarantee of useful task performance at 0.5 bar.
- Do not transfer this Table 2 envelope to taps or shower systems.

**Limitations — `en18021.2025.tap-system-pressure`:**
- Do not transfer this Table 1 envelope to PN5 shower outlets / sets.
- The pressure limits alone do not demonstrate task quality or user comfort.

**Limitations — `en18021.2025.temperature-limits`:**
- Never use the 70°C short-duration limit as a point-of-use user setpoint.
- Independent anti-scald and hot-water hygiene requirements remain separate.

## Unified Water Label — Taps & Showers Technical Criteria

- **Entry ID:** `uwl-taps-showers-technical-2023`
- **Subject type:** programme
- **Manufacturer / publisher:** Not applicable / not stated
- **SKU:** Not applicable / not stated
- **Market:** Unified Water Label programme
- **Revision:** January 2023

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `uwl.2023.type1.p1` | uwl_dynamic_pressure_point | Type 1 P1 = 150 kPa +10/-0 kPa, applied dynamically to each inlet. | standard_or_programme_criterion | declared | mode=Type 1 supply system; P1 lower pressure point; method=UWL Taps & Showers Technical Criteria January 2023, section 4.2 Table 1; datum=each inlet separately; unknown: full product-category-specific sequence in UWL procedure; Not a single upstream residual-pressure reading. | context_only: Method-specific dynamic inlet pressure; not interchangeable with point-of-discharge or faceplate-centre pressure. | [Unified Water Label Association](https://uwla.eu/wp-content/uploads/2023/05/TAPS-Showers-Technical-2023.pdf); official_standard_or_programme; 2023-01-01 |
| `uwl.2023.type1.p2` | uwl_dynamic_pressure_point | Type 1 P2 = 300 kPa +20/-0 kPa, applied dynamically to each inlet. | standard_or_programme_criterion | declared | mode=Type 1 supply system; P2 reference pressure point; method=UWL Taps & Showers Technical Criteria January 2023, section 4.2 Table 1; datum=each inlet separately; unknown: full product-category-specific sequence in UWL procedure; ETL flow-rate procedure additionally requires P2 at the centre of the shower faceplate. | context_only: Method-specific dynamic inlet pressure; ETL also declares a faceplate-centre P2 requirement for maximum-flow testing. | [Unified Water Label Association](https://uwla.eu/wp-content/uploads/2023/05/TAPS-Showers-Technical-2023.pdf); official_standard_or_programme; 2023-01-01 |
| `uwl.2023.type2.p3` | uwl_dynamic_pressure_point | Type 2 P3 = 10 kPa +2/-0 kPa, applied dynamically to each inlet. | standard_or_programme_criterion | declared | mode=Type 2 installation system; P3 low-pressure point; method=UWL Taps & Showers Technical Criteria January 2023, section 4.2 Table 1; datum=each inlet separately; unknown: full product-category-specific sequence in UWL procedure; Aqua's 50/75/100 kPa exploratory points do not cover this P3 test point. | context_only: Type 2 low-pressure method point, not a generic Aqua building supply condition. | [Unified Water Label Association](https://uwla.eu/wp-content/uploads/2023/05/TAPS-Showers-Technical-2023.pdf); official_standard_or_programme; 2023-01-01 |
| `uwl.2023.calibration` | test_instrument_calibration_requirement | UWL requires measurement equipment to be calibrated to international or national standards. | standard_or_programme_criterion | declared | mode=UWL tap and shower test procedures; method=UWL Taps & Showers Technical Criteria January 2023, section 4.1; unknown: scheme-specific conformity evidence for an individual laboratory; Mycelix remains Aqua's evidence owner for instrument identity, calibration and uncertainty. | context_only: Test-method quality requirement, not product-performance evidence. | [Unified Water Label Association](https://uwla.eu/wp-content/uploads/2023/05/TAPS-Showers-Technical-2023.pdf); official_standard_or_programme; 2023-01-01 |

**Limitations — `uwl.2023.type1.p1`:**
- Each required inlet must be independently supplied and measured.
- The asymmetric tolerance must not be converted to a symmetric ±10 kPa interval.

**Limitations — `uwl.2023.type1.p2`:**
- Each required inlet must be independently supplied and measured.
- Do not substitute a terminal fitting inlet or upstream residual gauge for this method datum.

**Limitations — `uwl.2023.type2.p3`:**
- Each required inlet must be independently supplied and measured.
- Asymmetric tolerance is +2/-0 kPa, not symmetric ±2 kPa.

**Limitations — `uwl.2023.calibration`:**
- Calibration status alone does not establish method applicability, measurement uncertainty or product compliance.

---

Generated from `docs/research/aqua-evidence-corpus-v0.1.json` by `.github/scripts/check-aqua-evidence-corpus.py`. Do not edit this rendering independently; change the JSON source and regenerate.
