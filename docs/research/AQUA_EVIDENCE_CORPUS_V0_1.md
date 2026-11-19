# Luminous Aqua — Evidence Corpus v0.1

**Snapshot:** 2026-11-19  
**Entries:** 21  
**Claims:** 134

> Source-classified evidence register. Manufacturer declarations, independent measurements, independent research recommendations, standards/programme criteria, local installation references and peer-reviewed results are kept distinct. Values marked unknown have not been inferred. This corpus is not a product certification or evidence of Aqua performance.

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
| `clasp.sa.test.pressure-range` | sampled_test_pressure_range | 250–550 kPa | independent_market_sample | measured | method=CLASP 2022 report; flow measured at intervals while supply pressure was swept; sample n=19; datum=Pressure transmitter connected via a take-off tee intended to avoid velocity-component influence; unknown: individual product pressure series and raw observations require full source tables, the report notes sampling is limited and not a current market census; The report photographed spray patterns at 300 kPa; tests included all stated settings. | context_only: Dated South African market sample, useful for local pressure-flow test design; not a current 2026 market census. | [CLASP / Omega Test House — South African Shower Heads Testing Report](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.flow-meter.uncertainty` | instrument_measurement_uncertainty | 0.3 % | independent_market_sample | measured | method=Endress+Hauser DMA20 mass-flow water meter as reported by CLASP; sample n=19; unknown: calibration certificate and traceability record not reproduced in the report extract; Report states the instrument uncertainty at 95% confidence. | context_only: Measurement-system detail for interpreting the CLASP data; not an Aqua instrument qualification. | [CLASP / Omega Test House — South African Shower Heads Testing Report](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.pressure-transmitter.uncertainty` | instrument_measurement_uncertainty | 2.9 % | independent_market_sample | measured | method=WIKA A-10 pressure transmitter using a pressure take-off tee; sample n=19; unknown: calibration certificate and exact pressure-point uncertainty propagation; Report states the instrument uncertainty at 95% confidence. | context_only: Measurement-system detail; pressure uncertainty is materially larger than the stated flow-meter uncertainty. | [CLASP / Omega Test House — South African Shower Heads Testing Report](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.fixed-orifice.fraction` | flow_profile_fraction | 58 % | independent_market_sample | measured | pressure=300 kPa; method=CLASP flow-profile classification of 19 sampled showerheads; sample n=19; unknown: individual classification sensitivity and confidence interval; 42% of patterns were classified as pressure-compensating. | context_only: Small, dated sample showing the prevalence of fixed-orifice flow behaviour among the selected South African market products. | [CLASP / Omega Test House — South African Shower Heads Testing Report](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.pressure-compensating.fraction` | flow_profile_fraction | 42 % | independent_market_sample | measured | pressure=300 kPa; method=CLASP flow-profile classification of 19 sampled showerheads; sample n=19; unknown: individual classification sensitivity and confidence interval; 58% of patterns were classified as fixed-orifice. | context_only: Small, dated sample showing the prevalence of pressure-compensating flow behaviour among selected products. | [CLASP / Omega Test House — South African Shower Heads Testing Report](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.pressure-compensating.mean-flow.clasp-2022` | mean_flow_rate | 7 L/min | independent_market_sample | measured | pressure=300 kPa; method=CLASP Table 3 / analysis summary; sample n=19; unknown: subset denominator and calculation precision should be confirmed from the complete raw dataset; WRC's 2025 summary of this study reports 6.6 L/min; keep these as separate source statements pending reconciliation. | context_only: Direct report figure for the selected sample; not a named product or current market value. | [CLASP / Omega Test House — South African Shower Heads Testing Report](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.fixed-orifice.mean-flow.clasp-2022` | mean_flow_rate | 25 L/min | independent_market_sample | measured | pressure=300 kPa; method=CLASP Table 3 / analysis summary; sample n=19; unknown: subset denominator and calculation precision should be confirmed from the complete raw dataset; WRC's 2025 summary of this study reports 24 L/min; retain the source distinction. | context_only: Direct report figure for the selected sample; not a named product or current market value. | [CLASP / Omega Test House — South African Shower Heads Testing Report](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.report-wrc-summary.discrepancy` | source_internal_numerical_discrepancy | Source summary difference: CLASP report = 7.0 L/min pressure-compensating / 25 L/min fixed-orifice; WRC 2025 summary = 6.6 / 24 L/min. Reconciliation of subset/rounding method is unresolved. | independent_market_sample | measured | pressure=300 kPa; method=Cross-check of CLASP 2022 report and WRC 2025 summary; sample n=19; unknown: whether the difference is rounding, subset definition or secondary-summary calculation; Keep both figures source-labelled; do not select or average one without reconciliation. | context_only: Preserves an observed difference between the primary report and later secondary summary. | [CLASP / Omega Test House — South African Shower Heads Testing Report](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |

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

**Limitations — `clasp.sa.test.pressure-range`:**
- Do not infer low-pressure behaviour below about 250 kPa from this sample.
- The report does not identify the tested brands/models in its published figures.

**Limitations — `clasp.sa.flow-meter.uncertainty`:**
- Do not transfer this uncertainty to Aqua instruments or other test arrangements.

**Limitations — `clasp.sa.pressure-transmitter.uncertainty`:**
- Do not infer combined uncertainty for a derived flow-retention statistic without its measurement model.

**Limitations — `clasp.sa.fixed-orifice.fraction`:**
- Not a statistically representative estimate of the full national showerhead market.

**Limitations — `clasp.sa.pressure-compensating.fraction`:**
- Not a statistically representative estimate of the full national showerhead market.

**Limitations — `clasp.sa.pressure-compensating.mean-flow.clasp-2022`:**
- Do not average with WRC's 2025 secondary summary without reconciling subset and rounding methods.

**Limitations — `clasp.sa.fixed-orifice.mean-flow.clasp-2022`:**
- Do not treat 24–25 L/min as a confidence interval; it is a difference between published source summaries.

**Limitations — `clasp.sa.report-wrc-summary.discrepancy`:**
- No universal benchmark should use a single selected value until the calculation basis is reconciled.

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
| `hansgrohe.28036000.water-saving-claim` | water_savings_fraction | 60 % | manufacturer_declaration | declared | mode=EcoSmart+ base spray modes; method=Manufacturer product-page marketing declaration; unknown: exact baseline SKU and test conditions, whether advertised saving includes boosted mode use; Boosted operation is separately listed. | context_only: Manufacturer claim; the exact baseline and boost-use share need matched testing. | [Hansgrohe South Africa — Activera Select S 28036000](https://www.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |
| `hansgrohe.28036000.easy-split-repair` | repairability_feature | Manufacturer-declared EasySplit disassembly to facilitate replacement of parts and separation into components. | manufacturer_declaration | declared | method=Manufacturer product-page feature declaration; unknown: repair time, tools, fastener count and exact replaceable part manifest, independent return-to-service evidence; This is a repair-oriented design feature, not a measured repairability score. | context_only: Commercial design-for-repair precedent; Aqua should compete on measured access, time, part availability and qualified return-to-service. | [Hansgrohe South Africa — Activera Select S 28036000](https://www.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |
| `hansgrohe.28036000.flexpower-nozzle` | pressure_compensation_feature | Elastic spray nozzles dynamically open and close with pressure to adapt spray delivery. | manufacturer_declaration | declared | method=Manufacturer product-page technology description; unknown: measured pressure-flow and spray-force curves, low-pressure task quality; Do not infer a pressure retention ratio from the feature description alone. | context_only: Commercial pressure-adaptive nozzle architecture; exact performance requires reproduced measurements. | [Hansgrohe South Africa — Activera Select S 28036000](https://www.hansgrohe.co.za/articledetail-activera-select-s-hand-shower-95-2jet-ecosmart-with-booster-28036000); manufacturer_product_page; date not stated |

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

**Limitations — `hansgrohe.28036000.water-saving-claim`:**
- Do not infer whole-task savings from the up-to percentage.
- Record boost frequency and duration in any comparative task trial.

**Limitations — `hansgrohe.28036000.easy-split-repair`:**
- Does not establish that every repair is non-destructive or that repair is completed successfully.

**Limitations — `hansgrohe.28036000.flexpower-nozzle`:**
- Aqua must benchmark actual flow, force, coverage and task completion under matched pressure points.

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
| `dhwr.2026.highlights-abstract-cop-discrepancy` | source_internal_numerical_discrepancy | Source-internal figure discrepancy: Highlights = 6.3 experimental / 8.0 simulation; Abstract = 6.2 experimental / 8.0 simulation. | peer_reviewed_study_result | measured | mode=Coupled shower DWHR + DHW storage / heat pump configurations studied in the article; task=Domestic hot-water system performance; method=Publisher landing-page Highlights and Abstract comparison; unknown: authoritative corrected numerical figure in article body / erratum, full configuration-specific COP conditions; Do not silently select 6.2 or 6.3 as a universal value; keep this difference source-scoped until the full article or authors resolve it. | context_only: The source page itself reports different experimental COP values in Highlights and Abstract. Preserve the discrepancy and do not use one figure as an Aqua design target. | [Bouvenot and Beaudet, Energy and Buildings](https://doi.org/10.1016/j.enbuild.2026.117761); peer_reviewed_article; 2026-10-01 |

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

**Limitations — `dhwr.2026.highlights-abstract-cop-discrepancy`:**
- Both figures are study-specific and not transferable to an Aqua design.
- Use the exact article body, configuration and boundary before any comparative model calibration.
- Simulation COP of 8.0 must remain distinct from experimental results.

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
| `en18021.2025.shower-boost-function-clauses` | shower_boost_function_scope | Separate shower-outlet (8.3.2) and shower-system (10.3.8) boost test clauses; deliberate temporary boost with automatic cancellation. Shower-specific numeric requirements require the controlled full standard. | standard_or_programme_criterion | declared | mode=PN5 shower outlets under clause 8.3.2; PN10 shower systems under clause 10.3.8; method=EN 18021:2025 terms 3.5, clause 8.3.2 and clause 10.3.8; accessible standard preview contents and definition; unknown: Full controlled text and numerical acceptance criteria for shower outlet clause 8.3.2, Full controlled text and numerical acceptance criteria for shower system clause 10.3.8, Exact specimen-specific manufacturer/laboratory results; Tap-only clause 7.3.3.4 numeric criteria must not be transferred to shower outlets or shower systems. | context_only: Confirms separate standard test categories and boost-function definition; no shower-specific acceptance threshold or Aqua/product result is asserted. | [UNI — Ente Italiano di Normazione](https://store.uni.com/en/en-18021-2025); official_standard_or_programme; 2025-07-02 |

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

**Limitations — `en18021.2025.shower-boost-function-clauses`:**
- Do not apply tap boost thresholds from EN 18021 clause 7.3.3.4 to shower outlets or systems without reviewing their separate clauses.
- UWL 2023 clause 12.1 is a separate test method with its own requirements.
- A manufacturer boost button or advertised flow does not prove conformity or user-task benefit.
- The controlled standard and exact applicable product category are required before formal testing.

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
| `uwl.2023.boost.definition` | boost_function_definition | Temporary flow increase for a specific user action; automatically cancelled when the flow control is closed. | standard_or_programme_criterion | declared | mode=UWL Taps & Showers boost-function feature; task=Specific user action; method=UWL Taps & Showers Technical Criteria January 2023, definition 3.12; unknown: product-specific boost-control mechanism, product-specific maximum boost duration; This is a feature definition, not proof that an exact model passes clause 12.1. | context_only: Common method vocabulary for classifying temporary boost; a manufacturer's boost button is not automatically a verified UWL result. | [Unified Water Label Association](https://uwla.eu/wp-content/uploads/2023/05/TAPS-Showers-Technical-2023.pdf); official_standard_or_programme; 2023-01-01 |
| `uwl.2023.boost.criteria` | boost_function_criteria | UWL 12.1 requires automatic return to regular flow after flow-control shutoff; Qmax = Qreg; Qboost < 2 × Qreg; regular flow Qreg must be labelled. | standard_or_programme_criterion | declared | mode=UWL Type 1 boost-function test; pressure=300 kPa; task=Regular / boosted / post-close flow sequence; method=UWL Taps & Showers Technical Criteria January 2023, sections 4.2 Table 1 and 12.1; datum=P2 dynamic pressure applied to each inlet separately; 300 kPa +20/-0 kPa; unknown: absolute L/min equality tolerance for Qmax = Qreg is not specified in clause 12.1, exact laboratory uncertainty decision rule; Aqua preserves measurement uncertainty and requires Qmax/Qreg agreement within combined declared measurement uncertainty; this is a software screening rule, not a substitute for scheme interpretation. | eligible_as_matched_test_reference: Method-specific boost test at the UWL P2 datum; use the complete controlled method and exact product configuration. | [Unified Water Label Association](https://uwla.eu/wp-content/uploads/2023/05/TAPS-Showers-Technical-2023.pdf); official_standard_or_programme; 2023-01-01 |

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

**Limitations — `uwl.2023.boost.definition`:**
- This does not establish task quality, energy savings or ETL user-behaviour-feature eligibility.

**Limitations — `uwl.2023.boost.criteria`:**
- Do not compare nominal manufacturer mode declarations as a UWL result.
- The numeric method check does not assess task-quality, comfort, aerosol, safety or energy outcome.

## ISO 20887:2020 — Design for disassembly and adaptability

- **Entry ID:** `iso-20887-2020-design-for-disassembly`
- **Subject type:** standard
- **Manufacturer / publisher:** Not applicable / not stated
- **SKU:** Not applicable / not stated
- **Market:** International; buildings, civil engineering works and constituent parts
- **Revision:** Edition 1; reviewed and confirmed 2025

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `iso.20887.2020.design-for-disassembly-scope` | design_for_disassembly_scope | Building/civil-engineering design-for-disassembly and adaptability principles with performance-measurement guidance; no universal numeric levels. | standard_or_programme_criterion | declared | method=ISO 20887:2020, Edition 1; ISO catalogue review confirmation 2025; unknown: No universal numeric repairability / disassembly threshold, AQS-specific component requirements and acceptance levels; Current version confirmed by ISO in 2025; applies to buildings, civil engineering works and constituent parts. | context_only: Design-for-disassembly principles guide the AQS service-spine and component access design; the standard supplies no product-specific numeric pass/fail threshold. | [International Organization for Standardization (ISO)](https://www.iso.org/standard/69370.html); official_standard_or_programme; 2020-01-01 |

**Limitations — `iso.20887.2020.design-for-disassembly-scope`:**
- Not a showerhead, plumbing fixture or bathroom product certification.
- Does not set a universal numeric disassembly or repairability threshold.
- Aqua-specific job manifests and measured trials are still required.

## EN 45554:2020 — Assessment of ability to repair, reuse and upgrade energy-related products

- **Entry ID:** `en-45554-2020-repair-reuse-upgrade`
- **Subject type:** standard
- **Manufacturer / publisher:** Not applicable / not stated
- **SKU:** Not applicable / not stated
- **Market:** European standard; energy-related products
- **Revision:** 2020 edition

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `en.45554.2020.repair-assessment-scope` | repairability_assessment_scope | Generic repair/reuse/upgrade assessment methods for energy-related products, with criteria such as disassembly depth, fasteners/connectors, tools and support. | standard_or_programme_criterion | declared | method=EN 45554:2020; unknown: Direct product-group applicability to passive plumbing fixtures, Aqua-specific repairability score or numeric thresholds; Use relevant methods as design inspiration only unless product-group applicability is confirmed. | context_only: Methodological reference for repair-job factors; its stated scope is energy-related products, so direct application to passive plumbing fixtures is not assumed. | [Royal Netherlands Standardization Institute (NEN)](https://www.nen.nl/en/nen-en-45554-2020-en-270440); official_standard_or_programme; 2020-01-01 |

**Limitations — `en.45554.2020.repair-assessment-scope`:**
- The stated scope is energy-related products; it is not automatically applicable to passive plumbing fixtures.
- It is not an Aqua product repairability result or a universal repairability score.
- Aqua must establish a product-specific fault/job manifest and measured acceptance method.

## KOHLER Anthem EvoCycle smart shower system

- **Entry ID:** `kohler-anthem-evocycle-k33481`
- **Subject type:** product
- **Manufacturer / publisher:** KOHLER
- **SKU:** K-33481
- **Market:** United States / Canada specification
- **Revision:** Technical specification dated 2026-04-15

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `kohler.k33481.cycle.fresh-water-feed` | fresh_water_flow_rate | 1.8927 L/min | manufacturer_declaration | declared | mode=Cycle Mode; task=One active shower session; method=Manufacturer product-page description; unknown: independent flow measurement, supply pressure and water-temperature conditions, fresh-water feed trace over a full task; Fresh-water input is blended with filtered recirculated water; this is not the total delivered shower flow. | context_only: Manufacturer-declared fresh-water feed; not an independent matched test or whole-task water result. | [KOHLER manufacturer product page](https://www.kohler.com/en/products/showers/anthem-evocycle); manufacturer_product_page; date not stated |
| `kohler.k33481.cycle.water-savings` | water_savings_fraction | 80 % | manufacturer_declaration | declared | mode=Cycle Mode with 2.5 US gal/min outlet; duration=10 min; task=Manufacturer's 10-minute example; method=Manufacturer comparison: 25 US gal conventional flow versus 5 US gal fresh water in Cycle Mode; unknown: independent replication, task-quality equivalence method, startup and cleaning boundary outside the illustrated example; The manufacturer states 2 US gal/min is recirculated and 0.5 US gal/min is fresh input. | context_only: Manufacturer claim based on a stated 10-minute same-flow example; not an independently verified saving. | [KOHLER manufacturer product page](https://www.kohler.com/en/products/showers/anthem-evocycle); manufacturer_product_page; date not stated |
| `kohler.k33481.cycle.rapid-clean-water` | cleaning_water_per_cycle | 2.8391 L | manufacturer_declaration | derived | mode=Rapid Clean; duration=5 min; task=After-session cleaning cycle; method=Manufacturer product-page description; US gallon conversion; unknown: measured water at drain boundary, measured electrical energy, actual use across installations; Manufacturer recommends the five-minute Rapid Clean after each shower. | context_only: Manufacturer-declared water per cleaning cycle converted from 0.75 US gallons; not a measured Aqua or independent result. | [KOHLER manufacturer product page](https://www.kohler.com/en/products/showers/anthem-evocycle); manufacturer_product_page; date not stated |
| `kohler.k33481.cycle.rapid-clean-duration` | cleaning_cycle_duration | 5 min | manufacturer_declaration | declared | mode=Rapid Clean; duration=5 min; task=After-session cleaning; method=Manufacturer product-page description; unknown: measured cleaning efficacy, actual power draw; Recommended after each shower. | context_only: Manufacturer-declared cycle duration, not an independently qualified cleaning outcome. | [KOHLER manufacturer product page](https://www.kohler.com/en/products/showers/anthem-evocycle); manufacturer_product_page; date not stated |
| `kohler.k33481.cycle.deep-clean-duration` | deep_clean_cycle_duration | 15 min | manufacturer_declaration | declared | mode=Deep Clean; duration=15 min; task=Periodic system cleaning; method=Manufacturer product-page description; unknown: water and energy consumed, consumables or service labour, independent endpoint method; Manufacturer recommendation is monthly. | context_only: Manufacturer-declared maintenance schedule; full resource and service burden must be measured. | [KOHLER manufacturer product page](https://www.kohler.com/en/products/showers/anthem-evocycle); manufacturer_product_page; date not stated |
| `kohler.k33481.cycle.same-session-boundary` | recirculation_scope | Same-session recirculation only while Cycle Mode is active; no storage for later users or connection to other household plumbing. | manufacturer_declaration | declared | mode=Cycle Mode; task=One active shower session; method=Manufacturer product-page description; unknown: independent validation of water quality, biofilm and aerosol exposure outcomes, cleaning coverage of all wetted surfaces; Manufacturer describes a washable reusable fine-mesh filter and continuous 0.5 US gal/min fresh-water refresh. | context_only: Manufacturer-described process boundary; not evidence of microbiological safety or hygiene efficacy. | [KOHLER manufacturer product page](https://www.kohler.com/en/products/showers/anthem-evocycle); manufacturer_product_page; date not stated |
| `kohler.k33481.minimum-outlet-flow` | minimum_connected_showerhead_flow | 3.4 L/min | manufacturer_declaration | derived | mode=Compatible connected showerhead / handspray; method=K-33481 technical specification; unknown: performance below the listed minimum, task performance at each outlet flow; Manufacturer compatibility statement; converted from 0.9 US gal/min. | context_only: Manufacturer compatibility floor, not an independently tested service-quality threshold. | [KOHLER K-33481 technical specification](https://techcomm.kohler.com/techcomm/pdf/K-33481_spec_US-CA_Kohler_en.pdf); manufacturer_datasheet; 2026-04-15 |
| `kohler.k33481.cycle.water-savings-after-rapid-clean` | water_savings_fraction | 77 % | manufacturer_declaration | derived | mode=Cycle Mode with 2.5 US gal/min outlet plus one Rapid Clean after the shower; duration=10 min; task=Manufacturer's 10-minute example with one recommended after-session Rapid Clean; method=Arithmetic derived from KOHLER's published 25 US gal conventional example, 5 US gal Cycle Mode example and 0.75 US gal Rapid Clean water; unknown: monthly Deep Clean water and energy use, actual startup, purge, reject water and measured usage, independent task-quality equivalence, site-specific actual consumption; This is not an independently measured result. It includes the stated Rapid Clean water only; monthly Deep Clean and other unreported flows are excluded. | context_only: Transparent arithmetic on manufacturer-published figures to illustrate the effect of the recommended per-shower cleaning cycle; not a matched or independent result. | [KOHLER manufacturer product page](https://www.kohler.com/en/products/showers/anthem-evocycle); manufacturer_product_page; date not stated |

**Limitations — `kohler.k33481.cycle.fresh-water-feed`:**
- Do not equate fresh-water feed with the total shower stream.
- Measure startup, pauses, post-use cleaning and reject water in a matched task.

**Limitations — `kohler.k33481.cycle.water-savings`:**
- Do not compare directly with direct-flow showerhead L/min without the same completed-task functional unit.
- The claim applies to Cycle Mode and the stated 2.5 US gal/min outlet configuration.

**Limitations — `kohler.k33481.cycle.rapid-clean-water`:**
- Count this water in every applicable completed-task and lifecycle boundary.
- Do not infer microbiological efficacy from the cycle's existence or name.

**Limitations — `kohler.k33481.cycle.rapid-clean-duration`:**
- Duration does not establish cleaning or disinfection efficacy.

**Limitations — `kohler.k33481.cycle.deep-clean-duration`:**
- Include periodic cleaning and downtime in lifecycle comparisons.

**Limitations — `kohler.k33481.cycle.same-session-boundary`:**
- Water-quality, surface/biofilm and aerosol outcomes require independent method-specific assessment.
- Do not describe the system as greywater reuse or cross-user stored-water reuse.

**Limitations — `kohler.k33481.minimum-outlet-flow`:**
- Do not interpret as the fresh-water input rate in Cycle Mode.

**Limitations — `kohler.k33481.cycle.water-savings-after-rapid-clean`:**
- Do not present 77% as a measured or fully all-in lifecycle saving.
- Monthly Deep Clean water/energy, startup, purge, standby, consumables and maintenance remain unquantified.
- The 80% headline claim is for Cycle Mode use; this adjusted illustration adds only the stated Rapid Clean water.

## Orbital Shower recirculating shower system

- **Entry ID:** `orbital-shower-current-family`
- **Subject type:** product
- **Manufacturer / publisher:** Orbital Systems
- **SKU:** Current product family; individual configuration / SKU not stated on cited public product page
- **Market:** Manufacturer global product and installation pages; regional market applicability not specified
- **Revision:** Installation requirements page accessed 2026-11-19

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `orbital.shower.water-savings` | water_savings_fraction | 90 % | manufacturer_declaration | declared | mode=Recirculating shower; package/configuration dependent; task=Long shower / user-selected shower duration; method=Manufacturer product-page claim; unknown: baseline product and flow, task duration and user profile, startup, purge and cleaning boundary, independent matched testing; A product-page claim, not a matched Aqua result. | context_only: Manufacturer marketing claim; baseline, task and all-in boundary require exact configuration clarification. | [Orbital Systems manufacturer product page](https://www.orbital-systems.com/products/orbital-shower); manufacturer_product_page; date not stated |
| `orbital.shower.energy-savings` | energy_savings_fraction | 80 % | manufacturer_declaration | declared | mode=Recirculating shower; package/configuration dependent; task=Long shower / user-selected shower duration; method=Manufacturer product-page claim; unknown: baseline heater and system boundary, tariff, inlet temperatures and use profile, pump, control, standby and cleaning energy, independent matched testing; A product-page claim, not a measured Aqua result. | context_only: Manufacturer marketing claim; system boundary and baseline require independent matched testing. | [Orbital Systems manufacturer product page](https://www.orbital-systems.com/products/orbital-shower); manufacturer_product_page; date not stated |
| `orbital.shower.water-volume` | water_volume_per_shower | 20 L | manufacturer_declaration | declared | mode=Recirculating shower; task=Manufacturer-described long shower; method=Manufacturer product-page statement; unknown: duration of long shower, user count and spray configuration, startup, cleaning, purge and reject water, pressure and inlet-temperature conditions; No duration is supplied in the cited headline statement. | context_only: Context-only manufacturer declaration; cannot be normalized to litres per minute or per completed task without the missing duration and accounting boundary. | [Orbital Systems manufacturer product page](https://www.orbital-systems.com/products/orbital-shower); manufacturer_product_page; date not stated |
| `orbital.shower.pressure-range` | operating_pressure_range | 200–1000 kPa | manufacturer_declaration | derived | mode=Incoming hot and cold water supplies; method=Orbital installation requirements page; datum=Incoming supply; exact dynamic/static datum not clarified; unknown: pressure measurement datum, performance at all points in the declared range; The same page includes a pressure-flow table with a 1.5 bar row; scope relative to the stated 2 bar lower range is unresolved. | context_only: Manufacturer installation requirement converted from bar to kPa; not a product performance curve. | [Orbital Systems installation requirements](https://support.orbital-systems.com/en/19639-installation-requirements.html); manufacturer_product_page; date not stated |
| `orbital.shower.pressure-flow-table` | pressure_flow_declaration | Published expected system flow: 1.5 bar = 8.7 L/min; 2 bar = 10.2 L/min; 3 bar = 11.7 L/min; 4 bar and above = 12.0 L/min. | manufacturer_declaration | declared | mode=Pressure-dependent system flow; method=Orbital installation requirements sidebar; datum=Incoming tap-water pressure; precise test datum not stated; unknown: test fixture and outlet configuration, measurement uncertainty and sample count, why 1.5 bar appears below the declared 2 bar operating-range minimum; Retain independently from the 2–10 bar stated incoming pressure range. | context_only: Manufacturer-declared pressure/flow table; values are not independently measured and do not resolve the stated 2 bar minimum. | [Orbital Systems installation requirements](https://support.orbital-systems.com/en/19639-installation-requirements.html); manufacturer_product_page; date not stated |
| `orbital.shower.inlet-capacity-table` | inlet_supply_capacity | System flow 12 L/min → 9 L/min per hot and cold inlet; 10 → 8; 8 → 6; 6 → 5 L/min per inlet. | manufacturer_declaration | declared | mode=Separate hot and cold inlet connections; method=Orbital installation requirements page; unknown: flow test pressure and temperature, independent capacity verification; Each inlet is considered separately; do not sum the two inlet capacities as a single mixed-flow measurement. | context_only: Manufacturer installation-capacity declaration; configuration-specific and not a local plumbing approval. | [Orbital Systems installation requirements](https://support.orbital-systems.com/en/19639-installation-requirements.html); manufacturer_product_page; date not stated |
| `orbital.shower.site-envelope` | installation_requirements | Fixed installation; continuously powered while idle; ambient 15–35°C during operation and ≥4°C when idle; accessible core ≤3 m from drain; separate hot/cold supplies; cold 4–25°C; hot 50–65°C; incoming pressure 2–10 bar; PN10 maximum working pressure; hardness 53–250 mg CaCO3/L; iron <0.3 mg/L; single-phase 230 VAC 50–60 Hz 16 A. Manufacturer also recommends a warm controlled core location. | manufacturer_declaration | declared | mode=Fixed residential / commercial installation; method=Orbital installation requirements page; unknown: pressure measurement datum and dynamic pressure profile, local electrical/plumbing approvals and site-specific installation scope, actual standby energy and outage behavior, availability of compliant inlet water across all sites; Current product-specific installation page accessed 2026-11-19. Older product-page variants may differ; use exact market/configuration documentation. | context_only: Manufacturer installation envelope; market-specific building and electrical approval remains separate. | [Orbital Systems installation requirements](https://support.orbital-systems.com/en/19639-installation-requirements.html); manufacturer_product_page; date not stated |
| `orbital.shower.operating-ambient-envelope` | operating_environment_requirements | Ambient 15–35°C while operating; ambient must remain ≥4°C when idle; core must be continuously powered even while not in use. | manufacturer_declaration | declared | mode=Fixed installation / exact Orbital Shower Core configuration; task=Site installation eligibility; method=Orbital Systems current installation requirements page (19639); unknown: independent confirmation of exact sensor/ambient measurement point, behavior during prolonged loss of mains power; Manufacturer operating/idle ambient and always-powered declaration. | context_only: Manufacturer-declared site operating envelope; not proof of outage safety or local suitability. | [Orbital Systems installation requirements](https://support.orbital-systems.com/en/19639-installation-requirements.html); manufacturer_product_page; date not stated |
| `orbital.shower.drain-floor-geometry` | floor_drain_geometry | Core ≤3 m from drain centre; manufacturer recommends floor slope 20 mm/m toward drain; core positioned below head shower and ≥100 mm above floor. | manufacturer_declaration | declared | mode=Fixed installation / exact Orbital Shower Core configuration; task=Site installation eligibility; method=Orbital Systems current installation requirements page (19639); unknown: site-specific drain levels and fall survey, as-built pipe route and local plumbing code; The 20 mm/m slope is a manufacturer recommendation, not an independent universal building code. | context_only: Manufacturer installation geometry declaration/recommendation; requires a competent site survey. | [Orbital Systems installation requirements](https://support.orbital-systems.com/en/19639-installation-requirements.html); manufacturer_product_page; date not stated |
| `orbital.shower.tile-thickness` | tile_thickness_limit | 10 mm | manufacturer_declaration | declared | mode=Fixed installation / exact Orbital Shower Core configuration; task=Site installation eligibility; method=Orbital Systems current installation requirements page (19639); unknown: market / base configuration compatibility, adhesive and substrate details; Manufacturer installation-page limit. | context_only: Manufacturer installation constraint for the shower zone; not a general tile standard. | [Orbital Systems installation requirements](https://support.orbital-systems.com/en/19639-installation-requirements.html); manufacturer_product_page; date not stated |
| `orbital.shower.fuse-dependent-flow` | fuse_dependent_flow_capacity | Declared fuse-dependent system flow: 10 A → 6 L/min; 13 A → 8 L/min; 16 A → 12 L/min. | manufacturer_declaration | declared | mode=Fixed installation / exact Orbital Shower Core configuration; task=Site installation eligibility; method=Orbital Systems current installation requirements page (19639); unknown: independent measured flow and uncertainty, exact voltage and load profile, competent electrical approval; Manufacturer states lower fuse ratings reduce system flow; one dedicated residual circuit breaker per core. | context_only: Manufacturer declaration linking electrical capacity to system flow; not an electrical design approval or independent measurement. | [Orbital Systems installation requirements](https://support.orbital-systems.com/en/19639-installation-requirements.html); manufacturer_product_page; date not stated |
| `orbital.shower.network-requirement-18295` | installation_requirements | Connectivity: password-protected Wi-Fi (2.4 GHz / 5 GHz) or Ethernet. | manufacturer_declaration | declared | mode=Orbital Shower installation / article 18295; method=Manufacturer installation requirements page 18295; unknown: whether connectivity is required for continuous operation, setup only, remote functions or warranty, exact current product revision / regional applicability; Article 19639 does not list this connectivity clause; confirm applicability to the exact product configuration. | context_only: Product-specific installation prerequisite declared on one manufacturer support article; exact variant applicability requires confirmation. | [Orbital Systems support portal — Installation requirements article 18295](https://support.orbital-systems.com/en/18295-installation-requirements.html); manufacturer_product_page; date not stated |
| `orbital.shower.pipe-bend-limit-18295` | installation_requirements | Maximum three bends on pipe routing. | manufacturer_declaration | declared | mode=Orbital Shower installation / article 18295; method=Manufacturer installation requirements page 18295; unknown: whether the bend limit applies to all markets and current product revisions, bend radius and pipe diameter conditions; Article 19639 does not mention a bend-count limit; confirm exact applicability rather than deleting or universally applying it. | context_only: A current manufacturer support article states a three-bend maximum; another support article omits it. | [Orbital Systems support portal — Installation requirements article 18295](https://support.orbital-systems.com/en/18295-installation-requirements.html); manufacturer_product_page; date not stated |
| `orbital.shower.install-document-scope-difference` | manufacturer_specification_scope_ambiguity | Two live manufacturer installation articles contain different scope fields: article 18295 lists Wi-Fi/Ethernet and max three pipe bends; article 19639 lists ambient/freeze protection, floor slope and fuse-dependent flow. Exact market, product revision and applicability are not declared across the two pages. | manufacturer_declaration | declared | method=Cross-check of Orbital support articles 18295 and 19639; unknown: authoritative document hierarchy, product revision and country applicability, whether omitted clauses remain required or are superseded; Preserve both source statements pending manufacturer clarification. | context_only: Cross-document scope difference requiring product/revision clarification; not resolved by assuming one page supersedes the other. | [Orbital Systems support portal — Installation requirements article 18295](https://support.orbital-systems.com/en/18295-installation-requirements.html); manufacturer_product_page; date not stated |

**Limitations — `orbital.shower.water-savings`:**
- Do not use as a verified comparator saving without reproducing the exact configuration and full service boundary.

**Limitations — `orbital.shower.energy-savings`:**
- Do not infer purchased kWh savings from recirculated water volume or component efficiency alone.

**Limitations — `orbital.shower.water-volume`:**
- Do not infer a standardized 10-minute or 15-minute result.

**Limitations — `orbital.shower.pressure-range`:**
- Preserve the 1.5 bar table row as an unresolved scope discrepancy, not as a revised minimum.

**Limitations — `orbital.shower.pressure-flow-table`:**
- Do not call the 1.5 bar point a supported operating point until the manufacturer clarifies it.
- Do not treat this table as a flow-retention or task-performance test.

**Limitations — `orbital.shower.inlet-capacity-table`:**
- Record hot and cold inlet supply capacity separately in any matched installation.

**Limitations — `orbital.shower.site-envelope`:**
- These are manufacturer installation declarations/recommendations, not local building/electrical approvals.
- Do not carry a pipe-bend limit from another installation document revision unless the exact product configuration confirms it.
- Treat continuous power while idle, ambient envelope, service access and drain location as explicit site-fit inputs.
- Do not assume this envelope matches every South African household or retrofit.

**Limitations — `orbital.shower.operating-ambient-envelope`:**
- Treat continuous power and idle temperature as separate site-fit prerequisites.
- Do not interpret the ambient range as an independent thermal-safety limit.

**Limitations — `orbital.shower.drain-floor-geometry`:**
- Do not treat the recommended slope as a universal statutory requirement.
- Measure actual drain location, levels, access and pipe route for the exact installation.

**Limitations — `orbital.shower.tile-thickness`:**
- Check the exact product and installation configuration before design freeze.

**Limitations — `orbital.shower.fuse-dependent-flow`:**
- Record fuse / protective-device configuration with actual flow and temperature tests.
- Never infer the required circuit or protection design from this declaration alone.

**Limitations — `orbital.shower.network-requirement-18295`:**
- Do not impose network connectivity on all recirculating architectures or Aqua passive direct-flow products.
- Treat connectivity as a product-specific Site Fit domain where the exact manual requires it.

**Limitations — `orbital.shower.pipe-bend-limit-18295`:**
- Do not assume it applies to every product revision or every market without manufacturer confirmation.
- Use this as a product-specific installation-envelope field, not a generic Aqua plumbing rule.

**Limitations — `orbital.shower.install-document-scope-difference`:**
- Do not treat absent information on one page as evidence that a prerequisite is waived.
- For exact Orbital comparison, request a revision-controlled installation manual for the regional product variant.

## Consumer Reports 2026 showerhead laboratory rating panel

- **Entry ID:** `consumer-reports-2026-showerhead-panel`
- **Subject type:** market_sample
- **Manufacturer / publisher:** Consumer Reports
- **SKU:** 22-model U.S. showerhead test panel; models listed individually by Consumer Reports
- **Market:** United States; not a South African market census
- **Revision:** Ratings page accessed 2026-11-19

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `consumer-reports.2026.panel.sample-size` | sample_size | 22 products | independent_market_sample | measured | method=Consumer Reports 2026 showerhead laboratory testing; sample n=22; unknown: sampling frame and market representativeness for South Africa; The ratings page reports 22 showerheads across single-setting and multisetting categories. | context_only: Independent consumer-laboratory panel; not a universal or local-market census. | [Consumer Reports 2026 showerhead laboratory ratings](https://www.consumerreports.org/home-garden/showerheads/c33121/); independent_market_study; 2026-01-27 |
| `consumer-reports.2026.panel.test-dimensions` | spray_coverage_criteria | Coverage uniformity; force relative to an ideal comfort value; measured water consumption; ease of switching settings; temperature drop measured at a container 18 inches from the showerhead. | independent_market_sample | measured | method=Consumer Reports laboratory rating methodology; sample n=22; unknown: full raw data and complete apparatus tolerances, pressure and inlet-temperature conditions for each model; Consumer Reports publishes claimed and actual water consumption on model rating pages. | context_only: Independent consumer-lab method summary; Aqua should reproduce or exceed relevant outcomes with a preregistered standard-compatible protocol. | [Consumer Reports 2026 showerhead laboratory ratings](https://www.consumerreports.org/home-garden/showerheads/c33121/); independent_market_study; 2026-01-27 |
| `consumer-reports.2026.high-sierra.measured-flow` | flow_rate | 6.0567 L/min | independent_market_sample | measured | mode=High Sierra Classic Plus Low Flow; single-setting; method=Consumer Reports published showerhead ratings; measured actual flow; unknown: pressure and temperature conditions, test repeatability and uncertainty; Actual flow converted from the published 1.6 US gal/min; manufacturer claim shown separately on the rating page. | context_only: Independent published measurement; US test sample and conditions must be matched before direct Aqua comparison. | [Consumer Reports 2026 showerhead laboratory ratings](https://www.consumerreports.org/home-garden/showerheads/c33121/); independent_market_study; 2026-01-27 |
| `consumer-reports.2026.jolie.measured-flow` | flow_rate | 5.6781 L/min | independent_market_sample | measured | mode=Jolie Filtered; single-setting; method=Consumer Reports published showerhead ratings; measured actual flow; unknown: pressure and temperature conditions, test repeatability and uncertainty; Actual flow converted from the published 1.5 US gal/min. | context_only: Independent published measurement; exact test context is not fully exposed on the public ratings summary. | [Consumer Reports 2026 showerhead laboratory ratings](https://www.consumerreports.org/home-garden/showerheads/c33121/); independent_market_study; 2026-01-27 |
| `consumer-reports.2026.sparkpod.measured-flow` | flow_rate | 6.0567 L/min | independent_market_sample | measured | mode=SparkPod High Pressure Rain Showerhead; single-setting; method=Consumer Reports published showerhead ratings; measured actual flow; unknown: pressure and temperature conditions, test repeatability and uncertainty; Actual flow converted from the published 1.6 US gal/min. | context_only: Independent published measurement; exact test context is not fully exposed on the public ratings summary. | [Consumer Reports 2026 showerhead laboratory ratings](https://www.consumerreports.org/home-garden/showerheads/c33121/); independent_market_study; 2026-01-27 |
| `consumer-reports.2026.delta-in2ition.measured-flow` | flow_rate | 5.6781 L/min | independent_market_sample | measured | mode=Delta Faucet 5-Spray In2ition; multisetting model; method=Consumer Reports published showerhead ratings; measured actual flow; unknown: specific spray setting corresponding to the summary flow figure, pressure and temperature conditions, test repeatability and uncertainty; The public rating summary gives claimed and measured flow; verify exact model and mode before replication. | context_only: Independent published measurement; use as a method/claim-calibration signal, not as a matched product superiority conclusion. | [Consumer Reports 2026 showerhead laboratory ratings](https://www.consumerreports.org/home-garden/showerheads/c33121/); independent_market_study; 2026-01-27 |

**Limitations — `consumer-reports.2026.panel.sample-size`:**
- Use as an independent-method benchmark, not as evidence about every showerhead or South African SKU.

**Limitations — `consumer-reports.2026.panel.test-dimensions`:**
- Consumer Reports' comfort-force reference is its own rating method, not a substitute for EN, UWL or WaterSense criteria.

**Limitations — `consumer-reports.2026.high-sierra.measured-flow`:**
- Do not substitute manufacturer claimed flow for measured flow.

**Limitations — `consumer-reports.2026.jolie.measured-flow`:**
- Keep claimed and measured values distinct.

**Limitations — `consumer-reports.2026.sparkpod.measured-flow`:**
- Keep claimed and measured values distinct.

**Limitations — `consumer-reports.2026.delta-in2ition.measured-flow`:**
- Freeze exact model and spray setting before any physical comparison.

## CLASP South African Showerheads Testing Report

- **Entry ID:** `clasp-south-african-showerhead-testing-2022`
- **Subject type:** market_sample
- **Manufacturer / publisher:** CLASP
- **SKU:** 19 anonymized showerhead samples (A–S)
- **Market:** South Africa; selected distributor and house-brand retail sample
- **Revision:** October 2022 report; laboratory tests at Omega Test House, Centurion

| Claim ID | Parameter | Canonical value | Evidence class | Result class | Conditions / unknowns | Comparison status | Source |
|---|---|---|---|---|---|---|---|
| `clasp.sa.2022.sample-size` | sample_size | 19 products | independent_market_sample | measured | method=CLASP South African ShowerHeads Testing Report; samples A–S; sample n=19; unknown: market representativeness beyond selected sales/house-brand sample, brand/model identities were anonymized; Sample included distributor and house-brand retail selections. | context_only: Independent laboratory market sample; not a census of the South African showerhead market. | [CLASP / Herman Strauss (Jackstra Solutions); laboratory testing by Omega Test House (Pty) Ltd](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.2022.flowmeter-uncertainty` | instrument_measurement_uncertainty | 0.3 % of reading | independent_market_sample | measured | method=Endress+Hauser DMA20 mass-flow meter, report apparatus section; sample n=19; unknown: calibration certificate and traceability record for the individual instrument, uncertainty budget outside the stated instrument specification | context_only: Instrument uncertainty stated in the report; not a complete uncertainty budget for every reported result. | [CLASP / Herman Strauss (Jackstra Solutions); laboratory testing by Omega Test House (Pty) Ltd](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.2022.pressure-transmitter-uncertainty` | instrument_measurement_uncertainty | 2.9 % of reading | independent_market_sample | measured | method=WIKA A-10 pressure transmitter, report apparatus section; sample n=19; unknown: calibration certificate and traceability record for the individual instrument, uncertainty budget outside the stated instrument specification | context_only: Instrument uncertainty stated in the report; not a complete uncertainty budget for every reported result. | [CLASP / Herman Strauss (Jackstra Solutions); laboratory testing by Omega Test House (Pty) Ltd](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.2022.test-pressure-range` | sampled_test_pressure_range | 0 | independent_market_sample | measured | pressure range=250–550 kPa; pressure points=250, 300, 500 kPa; method=Stabilized flow readings during a pressure ramp; summary values at 250, 300 and 500 kPa; sample n=19; datum=Dynamic pressure at showerhead test supply; pressure transmitter arrangement described by the report; unknown: raw per-sample data files, uncertainty propagation for interpolated summary values | context_only: Study-specific test envelope; it is not a universal South African household pressure distribution. | [CLASP / Herman Strauss (Jackstra Solutions); laboratory testing by Omega Test House (Pty) Ltd](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.2022.pressure-compensating-flow-at-300kpa` | mean_flow_rate | 7 L/min | independent_market_sample | measured | mode=Pressure-compensating profile group; pressure=300 kPa; method=CLASP report Table 3; interpolated summary value, rounded to nearest 1 L/min; sample n=19; unknown: raw group-level observations and group count, task comfort, shower force and standardized coverage equivalence | context_only: Study-specific average of profiles classified as pressure compensating; not a universal product specification. | [CLASP / Herman Strauss (Jackstra Solutions); laboratory testing by Omega Test House (Pty) Ltd](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.2022.fixed-orifice-flow-at-300kpa` | mean_flow_rate | 25 L/min | independent_market_sample | measured | mode=Fixed-orifice profile group; pressure=300 kPa; method=CLASP report Table 3; interpolated summary value, rounded to nearest 1 L/min; sample n=19; unknown: raw group-level observations and group count, task comfort, shower force and standardized coverage equivalence | context_only: Study-specific average of profiles classified as fixed orifice; not a universal product specification. | [CLASP / Herman Strauss (Jackstra Solutions); laboratory testing by Omega Test House (Pty) Ltd](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.2022.market-water-saving-label-flow-at-300kpa` | mean_flow_rate | 10.9 L/min | independent_market_sample | measured | mode=Products marked as water saving by packaging/marketing; pressure=300 kPa; method=CLASP report Table 3/category average, rounded to nearest 1 L/min; sample n=19; unknown: raw group count and raw values, whether marketed claim matched a defined task-quality method | context_only: Study-specific average for a small anonymized group marketed as water saving. | [CLASP / Herman Strauss (Jackstra Solutions); laboratory testing by Omega Test House (Pty) Ltd](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.2022.high-price-category-flow-at-300kpa` | mean_flow_rate | 8 L/min | independent_market_sample | measured | mode=High-price selected sample category; pressure=300 kPa; method=CLASP report category average; sample n=19; unknown: only two high-price products in the selected 19-head sample, population representativeness | context_only: Selected sample category average; not a statistically representative price effect. | [CLASP / Herman Strauss (Jackstra Solutions); laboratory testing by Omega Test House (Pty) Ltd](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.2022.proposed-max-flow` | proposed_maximum_shower_flow | 10 L/min | independent_recommendation | proposed | mode=Proposed national showerhead standard framework; pressure=300 kPa; method=CLASP report recommendations section; not an enacted standard; unknown: whether government/SABS later adopted this recommendation, current legal status and exact current standard text; The report recommends measurements at least at 300 kPa and 150 kPa. | context_only: Independent research recommendation for a proposed national standard; not a current legal limit or certified criterion. | [CLASP / Herman Strauss (Jackstra Solutions); laboratory testing by Omega Test House (Pty) Ltd](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.2022.proposed-preferred-wels-band` | proposed_preferred_flow_band | 4.5–6 L/min | independent_recommendation | proposed | mode=Proposed future South African water-efficiency labelling scheme; method=CLASP report recommendations section; flow band conditional on performance requirements; unknown: whether a national WELS scheme or rating band has since been adopted, final official test methods and thresholds | context_only: Independent research recommendation; not an enacted WELS rating or certification result. | [CLASP / Herman Strauss (Jackstra Solutions); laboratory testing by Omega Test House (Pty) Ltd](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.2022.proposed-test-pressures` | proposed_test_pressure_points | At least 150 kPa and 300 kPa. | independent_recommendation | proposed | pressure points=150, 300 kPa; method=CLASP report recommendations; reference to AS 3662 / proposed SANS test framework; unknown: whether a current official SANS method has since incorporated these points | context_only: Independent research proposal for future local test standard; not proof of the current normative procedure. | [CLASP / Herman Strauss (Jackstra Solutions); laboratory testing by Omega Test House (Pty) Ltd](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `clasp.sa.2022.flow-regulator-design-principle` | pressure-compensating-design-principle | Prefer a pressure-compensating flow characteristic that maintains acceptable flow across the installation pressure envelope. | independent_recommendation | proposed | pressure points=150, 250, 300, 350, 500 kPa; method=CLASP report analysis and recommendations; comparison between fixed-orifice and pressure-compensating flow profiles; unknown: formal local acceptance criteria, task-quality equivalence under the full pressure envelope | context_only: Independent design recommendation from a limited flow-profile test; not a tested Aqua design rule. | [CLASP / Herman Strauss (Jackstra Solutions); laboratory testing by Omega Test House (Pty) Ltd](https://www.clasp.ngo/wp-content/uploads/2022/10/SA-Showerheads-Testing-Report.pdf); independent_market_study; 2022-10-01 |
| `wrc.2025.clasp.pressure-compensating-flow-at-300kpa` | mean_flow_rate | 6.6 L/min | independent_market_sample | measured | pressure=300 kPa; method=Water Wheel September/October 2025 secondary summary of the October 2022 CLASP South African ShowerHeads Testing Report; sample n=19; datum=Dynamic showerhead supply pressure; unknown: secondary summary does not publish the row-level raw values, calculation/rounding basis for difference from the primary CLASP summary table; Secondary summary of the CLASP 19-head sample. The primary CLASP PDF table rounds its pressure-compensating group average to 7 L/min. | context_only: Secondary source representation of CLASP data; use alongside, not in place of, the primary CLASP PDF. | [Water Research Commission, The Water Wheel (September/October 2025)](https://www.wrc.org.za/wp-content/uploads/mdocs/WW%20July%20-%20Sept%202025_web.pdf); independent_market_study; date not stated |
| `wrc.2025.clasp.fixed-orifice-flow-at-300kpa` | mean_flow_rate | 24 L/min | independent_market_sample | measured | pressure=300 kPa; method=Water Wheel September/October 2025 secondary summary of the October 2022 CLASP South African ShowerHeads Testing Report; sample n=19; datum=Dynamic showerhead supply pressure; unknown: secondary summary does not publish the row-level raw values, calculation/rounding basis for difference from the primary CLASP summary table; Secondary summary of the CLASP 19-head sample. The primary CLASP PDF table rounds its fixed-orifice group average to 25 L/min. | context_only: Secondary source representation of CLASP data; use alongside, not in place of, the primary CLASP PDF. | [Water Research Commission, The Water Wheel (September/October 2025)](https://www.wrc.org.za/wp-content/uploads/mdocs/WW%20July%20-%20Sept%202025_web.pdf); independent_market_study; date not stated |

**Limitations — `clasp.sa.2022.sample-size`:**
- Do not infer brand-level or SKU-level superiority; sample identities were anonymized.
- The report itself states category sample sizes were limited and recommends larger samples for statistically significant price/category trends.

**Limitations — `clasp.sa.2022.flowmeter-uncertainty`:**
- Do not treat 0.3% as combined measurement-system uncertainty.

**Limitations — `clasp.sa.2022.pressure-transmitter-uncertainty`:**
- Do not treat 2.9% as combined measurement-system uncertainty.

**Limitations — `clasp.sa.2022.test-pressure-range`:**
- Use 150 and 300 kPa as proposed local design test points only where the complete rig and comparator are rated.

**Limitations — `clasp.sa.2022.pressure-compensating-flow-at-300kpa`:**
- This is a flow-profile group average, not the lowest measured individual flow.
- The report tested flow rates and spray patterns but did not establish completed-bathing-task superiority.

**Limitations — `clasp.sa.2022.fixed-orifice-flow-at-300kpa`:**
- This is a flow-profile group average, not an exact SKU result.
- The report states that fixed-orifice behavior can cause poor low-pressure spray and excessive high-pressure flow.

**Limitations — `clasp.sa.2022.market-water-saving-label-flow-at-300kpa`:**
- This finding shows that marketing language is not evidence of low flow or task performance.
- Do not generalize to all current South African products.

**Limitations — `clasp.sa.2022.high-price-category-flow-at-300kpa`:**
- The report explicitly notes limited category sample sizes and does not claim statistically significant price/flow correlation.

**Limitations — `clasp.sa.2022.proposed-max-flow`:**
- Do not describe 10 L/min as an enacted SANS limit or current statutory requirement.

**Limitations — `clasp.sa.2022.proposed-preferred-wels-band`:**
- The band is explicitly conditional on acceptable spray force, spray coverage and positive user experience.
- Do not present the proposal as an existing national rating.

**Limitations — `clasp.sa.2022.proposed-test-pressures`:**
- Keep separate from WaterSense 140 / 310 / 550 kPa test points.
- Do not label this a current regulatory requirement.

**Limitations — `clasp.sa.2022.flow-regulator-design-principle`:**
- Do not assume pressure compensation alone guarantees spray quality, comfort or hygiene.

**Limitations — `wrc.2025.clasp.pressure-compensating-flow-at-300kpa`:**
- The primary CLASP PDF reports the same category as 7 L/min after rounding to the nearest 1 L/min.
- Do not average the two summaries or infer a new exact mean without the raw group observations.

**Limitations — `wrc.2025.clasp.fixed-orifice-flow-at-300kpa`:**
- The primary CLASP PDF reports the same category as 25 L/min after rounding to the nearest 1 L/min.
- Do not average the two summaries or infer a new exact mean without the raw group observations.

---

Generated from `docs/research/aqua-evidence-corpus-v0.1.json` by `.github/scripts/check-aqua-evidence-corpus.py`. Do not edit this rendering independently; change the JSON source and regenerate.
