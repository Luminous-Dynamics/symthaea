# Luminous Aqua — Bath, Bidet and Basin Benchmark Protocol v0.1

**Status:** research / preregistration draft; no fixture has been tested under this protocol  
**Parent:** [AQUA-BATHING-SERVICE-001 #12660](https://github.com/Luminous-Dynamics/symthaea/issues/12660)  
**Workstream:** [AQUA-WASHSTATION-001 #12693](https://github.com/Luminous-Dynamics/symthaea/issues/12693)  
**Evidence owners:** Symtropy #2152/#4978/#5364/#5425; Mycelix #6025/#6029/#6034/#6048/#6054/#6057/#6100; Sol-Atlas #694/#700/#816.

## 1. Research question

Can Aqua baths, bidets/personal-wash fixtures and basins deliver a completed, independently assessed task with equal or better usability, hygiene, accessibility and reliability, while reducing whole-task water, useful heat, purchased electricity/fuel, maintenance burden and lifecycle cost?

The functional unit is the completed task, not the fixture's advertised flow, nominal bath capacity, feature count or cycle timer. Compare each fixture class separately; a bath, basin wash and bidet task are not interchangeable litres.

## 2. Market and reference baselines

### Basin / lavatory faucet

The U.S. EPA WaterSense labelled private lavatory-faucet specification currently published on its programme page uses:
- maximum 1.5 US gpm (5.7 L/min) at 60 psi (414 kPa);
- minimum 0.8 US gpm (3.0 L/min) at 20 psi (138 kPa);
- independent certification for water efficiency and product-performance criteria, including a 500,000-cycle life test and lead-free requirements under the referenced plumbing-fitting standard.

The minimum flow matters: EPA says 0.5 gpm products intended for public/commercial restrooms are generally not suitable for private bathroom tasks such as shaving and face washing. This is a U.S. programme reference, not South African law or a local product approval.
Sources:
- https://www.epa.gov/watersense/bathroom-faucets
- https://www.epa.gov/system/files/documents/2022-05/ws-products-perfomance-faucets.pdf
- https://www.epa.gov/system/files/documents/2023-08/ws-homes-TRM-3-BathroomSinkFaucetsTechSheet.pdf

**Important programme status:** EPA's page says work on the proposed WaterSense Version 2.0 private-lavatory-faucet specification was paused in February 2025 for review. The draft proposed reducing the maximum from 1.5 to 1.2 gpm (4.5 L/min), but that draft must not be described as an approved criterion or binding requirement. Treat 1.5 gpm / 0.8 gpm as the existing published label reference until EPA formally changes it; confirm status before future claims.
Source: https://www.epa.gov/watersense/bathroom-faucets

### Bidet / personal-hygiene fixture

ASME A112.4.2-2021 / CSA B45.16:21 is a relevant product-test reference for water-closet personal-hygiene devices. Its contents include:
- backflow prevention and water-source requirements;
- electrical components;
- temperature-safety devices;
- bidet-spray flowing and static-pressure tests;
- temperature tests, including shutoff temperature;
- bidet sprayer lifecycle and self-cleaning tests.

This is an international product-standard comparator, not automatically a South African legal requirement. Applicability, current edition, national adoption and installation rules must be confirmed for the target market.
Source: https://www.asme.org/getmedia/d59be223-9338-401b-ab46-97895f111887/a112-4-2_csa-b45-16_2021-table-of-contents.pdf

A current premium feature comparator is KOHLER PureWash E930. Its product information lists adjustable water temperature/pressure, heated seat, warm-air dryer, child mode, energy-save function and a self-cleaning wand. The U.S. model is specified for a 120 V / 15 A GFCI connection. These are manufacturer-listed features, not independent hygiene, electrical, energy or service-life test outcomes; confirm the exact model/voltage and supply variant before South African comparison.
Source: https://www.kohler.com/en/products/toilet-seats-bidet-seats/shop-toilet-seats-bidet-seats/purewash-e930-elongated-bidet-toilet-seat-w-remote-control-28821

### Bathing vessels

Use the locally applicable plumbing/building standard and exact manufacturer's data for overflow, waste, supply fittings and installation. A locally available copy of SANS 10252-1:2016 Edition 3.1 reports that it supersedes the 2012 edition; the controlled current edition and exact clauses must be checked in the SABS catalogue before engineering or compliance claims.
Reference copy:
- https://www.ndlambe.gov.za/wp-content/uploads/2023/05/SANS-10252-Water-installation.pdf

Nominal tub/overflow capacity is not the correct functional unit. Actual fill volume depends on the marked waterline, body displacement, bath geometry and the declared immersion task. Aqua measures actual litres at the intended bathing level and records temperature loss during the service.

## 3. Functional units and complete-service boundaries

### A. AquaBath — one completed bath task

Declare:
- bath geometry and exact product/configuration;
- intended waterline / immersion profile;
- user/task profile, including seated/assisted variant where applicable;
- inlet cold/hot temperatures, pressure and ambient conditions;
- target fill temperature and service duration.

Count from user request to drained, cleanable, ready state:
- hot and cold water admitted, actual fill volume and top-up;
- hot-water source and distribution energy;
- time-to-ready and temperature stability;
- heat loss / useful heat delivered over the named bathing interval;
- drain, overflow, clean and post-use water/energy;
- entry/exit and transfer effort, assistant clearance and slip/egress outcomes;
- cleaning time, chemical, scale/soap residue, accessible service and repair burden.

Do not shrink effective immersion or worsen thermal comfort to manufacture a lower litres-per-bath result. Report volume and task experience together.

### B. AquaWash — one completed personal-wash task

Declare:
- exact device / nozzle / seat model and market variant;
- water source, backflow architecture and intended use;
- pressure, temperature and available flow envelope;
- task profile, spray position/range, duration and user controls;
- cleaning protocol, standby state and power supply where fitted.

Count:
- startup / purge, active delivery, self-clean cycle and post-use water;
- water volume and flow/pressure trace per task;
- inlet, delivered and nozzle temperature;
- heater / seat / dryer / UV or other accessory electricity separately;
- nozzle extension/retraction, cleaning coverage and post-cycle condition;
- splash/contamination pathway and access for inspection/replacement;
- standby, warm-up, cleaning, consumables, maintenance and service downtime.

A self-clean cycle is a device/process feature, not proof of microbiological safety. Independent methods and competent assessors select any biological endpoints. The device must not use greywater, pool water, shower recirculation or toilet-cistern water as its supply. Backflow protection and the applicable local installation hazard classification require qualified plumbing review.

### C. AquaBasin — one completed basin task

Separate task classes:
- handwashing using a frozen method/time;
- face washing;
- shaving / grooming;
- assisted or seated wash;
- simultaneous multi-user basin use if designed for it.

Count:
- flow and volume at source/fixture boundary, including start/stop and time-to-hot-water;
- pressure-compensation and flow response at declared low/nominal/high supply conditions;
- water temperature and time to desired hot water;
- splash distribution, water outside the basin and cleanup burden;
- task completion, reach, control access and user preference;
- trap/aerator cleanability, scale and soap build-up;
- hot-water energy, standby and replacement/service burden.

Cold-start controls, demand-controlled hot-water delivery and compact/short branch runs are candidate design hypotheses. Any water/energy saving must be demonstrated on an equal task and must count the hot-water distribution boundary.

## 4. Matched comparator arms

Freeze exact market SKU, model, installation, supply, hardware, firmware, accessories and maintenance state before testing.

| Class | Comparator | Required evidence |
|---|---|---|
| Basin | conventional mixer tap | exact flow curve and installation; task baseline |
| Basin | WaterSense-labelled 1.5 gpm lavatory faucet | exact certified model and label; compare 414 kPa and minimum-flow 138 kPa reference points |
| Basin | pressure-compensating / laminar-flow faucet | named SKU, actual flow at low/nominal/high pressure, splash and task performance |
| Basin | cold-start / demand-controlled fixture | exact mode and hot-water distribution boundary; independent meter |
| Bidet | mechanical bidet spray / separate bidet | exact plumbing/backflow design, flow/pressure, spray and temperature method |
| Bidet | integrated bidet seat (e.g. KOHLER PureWash E930) | exact market/voltage model, cleaning instructions, energy/standby and service components |
| Bath | matched conventional tub | exact product, fill line, volume, material and insulation |
| Bath | compact/deep-soak tub | actual immersion volume and entry/exit/task outcome, not marketing capacity |
| Bath | accessible / step-through bathing vessel | declared user profile, transfer/egress and assistant space with volume/energy |

Vendor data may nominate comparators but cannot replace measured results. If local market version or operating boundary is unknown, preserve UNKNOWN and do not rank it.

## 5. Frozen conditions and metrology

Preregister:
- device/fixture serial, exact model and configuration;
- inlet water temperature, hardness/chemistry and source identity;
- pressure at static and flowing conditions, at exact fixture reference points;
- fresh-water and hot-water flow meters, resolution and uncertainty;
- electrical meter at device and relevant hot-water/room boundaries;
- temperature at source, fixture outlet, tub body-water field and any hot-water source;
- ambient temperature and humidity for bath heat loss / room comparison;
- installation geometry and outlet height/orientation;
- task duration, task procedure, operator, test day and randomized order;
- calibration / check standards, maintenance, water batch and instrument timebase.

For basin faucet comparisons, include 138 kPa (20 psi) and 414 kPa (60 psi) to match the WaterSense published reference points; include South African site-relevant conditions after the local envelope is measured. Do not treat a static pressure gauge reading as the pressure while flowing.

For bath fill and thermal assessment, use actual measured temperature and volume. A transparent analytical estimate of sensible heat is:

`Q_thermal (kWh) = V_L × 4.186 × ΔT_C / 3600`

This is an idealized water sensible-heat calculation only. It does not account for heater efficiency, hot/cold mixing, storage/distribution losses, tub heat capacity, room heat exchange or human-body displacement. Physical meters and the relevant Symtropy thermal authority remain the result source.

## 6. Primary endpoint vector

Do not produce a universal “water-saving fixture score.” Report separately by fixture and task:
- task completion and quality;
- water litres per completed task;
- useful heat delivered / purchased fuel / purchased electricity;
- time-to-ready and temperature stability;
- pressure/flow envelope and spray/cleaning outcomes;
- accessibility, reach, transfer and assistance;
- splash and cleanup burden;
- cleaning water/chemical/labour and service time;
- reliability, downtime, spares and replacement;
- lifecycle cost per independently qualified task;
- measurement uncertainty, exclusions, retries and failed attempts.

Safety, potable-source protection, backflow, temperature limits, hygiene, egress and accessibility are non-compensatory gates. A lower resource number cannot compensate for failure in those categories.

## 7. Proposed targets — hypotheses, not achieved results

### Basin
- Primary reference: ≤5.7 L/min at 414 kPa with no less than 3.0 L/min at 138 kPa, using the existing WaterSense specification only as a comparison reference.
- Stretch design: explore ≤4.5 L/min at 414 kPa only when handwashing/face-washing/grooming task quality and low-pressure usability remain non-inferior. This is a research target, not a South African standard.
- Zero hidden hot-water distribution waste in the metered task boundary; report time-to-hot-water and water/energy discarded while waiting.
- Any cold-start saving must be demonstrated using the same user task and exact hot-water boundary.

### Bidet / personal wash
- 100% of supply paths have a documented, independently reviewed backflow protection solution appropriate to the local installation hazard.
- 100% of powered models separately report standby, warm-up, active spray, dryer and cleaning energy.
- Nozzle cleaning / lifecycle tests use a selected product method and report failures, replacement intervals and resource burden.
- No human or shared-water exposure until the specific water-quality / cleaning / aerosol assessment is independently admitted.

### Bath
- 100% of bath comparisons report actual measured fill volume at the same declared immersion level and task.
- ≥20% whole-task water or energy reduction against a matched incumbent is a stretch only if the same immersion, temperature, duration, access and comfort requirements are maintained.
- Report bath fill energy and post-fill heat loss separately; do not count a water-volume change as a thermal saving without measurement.

### Lifecycle
- ≥20% lower lifecycle cost per qualified task is a stretch objective only after matched installed prices, utility costs, cleaning labour, consumables, repairs, replacements and downtime have been independently measured.
- No cleaning or maintenance burden shifted silently to the resident.

## 8. Adversarial / fault corpus

1. low inlet pressure with task flow below the usable floor;
2. high pressure and pressure-compensator failure;
3. temperature overshoot / failed independent temperature limiter;
4. hot-water distribution wait and cold-start vs hot-start;
5. bath filled to nominal overflow but fails intended immersion level;
6. tub with high heat loss / insufficient hot-water recovery;
7. bidet nozzle fails to retract or clean;
8. backflow-protection device absent, mismatched or overdue;
9. cleaning chemical/media exhausted;
10. power loss during bidet seat/nozzle operation;
11. sensor drift / missing flow or temperature observation;
12. accessible transfer path obstructed or reach outside declared envelope;
13. basin splash/overflow causes uncounted cleanup;
14. service part unavailable / destructive replacement required;
15. repeated-use scale and aerator loading.

Synthetic cases validate data/logic only; they do not demonstrate real fixture performance, hygiene or usability.

## 9. Statistical and human-factors design

Use pilot runs to estimate variance and cluster effects, then preregister the locked confirmation sample size and stopping rule. Preserve independent physical runs separately from multiple readings in the same run; account for device, installation, operator, day and water-batch clustering. Blind task assessors where practical.

Human-factors research requires independent safety review, informed consent, an appropriate protocol and any applicable ethics approval. No human trial can promote a fixture to regulatory or clinical certification.

## 10. Claim ceiling

This protocol does not establish potable-water safety, bidet hygiene, disease prevention, clinical benefit, accessibility certification, South African standards compliance, water/energy savings, affordability, reliability, product superiority or permission for occupied deployment. Targets are hypotheses until the exact product and matched task have been independently tested.
