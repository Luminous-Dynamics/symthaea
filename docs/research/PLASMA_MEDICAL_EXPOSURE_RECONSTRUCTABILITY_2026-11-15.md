# PLASMA-MEDICAL-002 — biological exposure record and reconstructability benchmark

**Issue:** https://github.com/Luminous-Dynamics/symthaea/issues/12265  
**Predecessor:** PLASMA-MEDICAL-001 / https://github.com/Luminous-Dynamics/symthaea/issues/12236  
**Date:** 15 November 2026  
**Status:** schema, synthetic fixtures and first audit implementation authored; Rust execution / hosted exact-head CI not yet verified  
**Claim ceiling:** data-contract validation and synthetic replay only; no physical or clinical validation

## 1. Why this increment exists

PLASMA-MEDICAL-001 protects the interpretation of public research sources. Its source register and 29-case claim-scope audit distinguish a trial protocol from a result, Phase I safety/feasibility from efficacy, a non-significant comparison from equivalence, and direct CAP from plasma-activated water.

That is necessary but not sufficient. A study can be described accurately and still be impossible to reproduce if the actual treatment configuration, measured interface exposure, spatial distribution, assay timing, calibration, uncertainty or liquid handling history are absent.

The next bottleneck is **biological exposure reconstructability**.

A 2026 review/meta-analysis covering more than 400 studies proposes complementary physical and biological dosimetry indices. It discusses volumetric reactive oxygen/nitrogen species (RONS) flux and redox-homeostatic status, and reflects the lack of standardized CAP dosing. This is a research framework, not a universal validated clinical dose standard.

Source: https://pubmed.ncbi.nlm.nih.gov/42143970/

## 2. Core paradigm: do not collapse distinct layers

The schema links five layers by immutable identifiers while keeping their meanings separate:

1. **Source configuration:** device and source revision, discharge architecture, gas composition, nominal settings, consumables, software and service lineage.
2. **Physical exposure:** observations at the target interface, with units, method, calibration, uncertainty, time, raw-data digest and (when spatial claims are made) registered geometry and coverage.
3. **PAL/PAW formulation lineage:** starting matrix, generation event/source, vessel/container, storage conditions, time to application, chemistry at measured times and delivery conditions. This is a separate branch from direct CAP.
4. **Biological context/response:** model class, tissue or matrix, assay, link to the exact exposure event and elapsed time after exposure. In-vitro, ex-vivo, animal, human and phantom data remain distinguishable.
5. **Clinical context:** protocol, comparator, standard-care/co-interventions, endpoint, follow-up and analysis-plan references. Patient-identifying data are outside this record.

The model does not define one scalar dose. It records a multidimensional exposure fingerprint and the evidence that supports each observation.

## 3. Measurement semantics

Every measurement has a status:
- **MEASURED:** a numeric value exists with unit, method, calibration declaration, uncertainty declaration, observation time and raw-data digest.
- **BELOW_DETECTION_LIMIT:** no quantified value is claimed; the detection limit is recorded separately. It is not zero.
- **NOT_MEASURED:** measurement was not performed.
- **NOT_APPLICABLE:** the quantity does not apply in the declared context.
- **UNKNOWN:** status/value cannot be established from available evidence.
- **WITHHELD:** data exist but are not included in this record.

The contract rejects a numeric value paired with UNKNOWN or BELOW_DETECTION_LIMIT. This guards against a common but consequential data error: encoding absent or censored measurement as a measured zero.

Calibration is explicit. A measured quantity with expired or unknown calibration is blocked from supporting a traceability claim. Measurement uncertainty must be quantified or explicitly marked not quantified; not quantified is not silently interpreted as zero uncertainty.

Raw data are not embedded in the fixture. The record binds SHA-256 digests so data custody and later replay can be handled by a controlled evidence store. A digest establishes identity/integrity of bytes, not the scientific validity of the measurement.

## 4. Spatial and hotspot-aware measurement

A point reading is not a spatial map. A claim of spatial uniformity requires:
- an identified coordinate frame;
- a declared target distance and treated area;
- a sampling plan and sample count;
- a spatial-map digest and coverage declaration;
- a review of local maxima / hotspots.

The mean must not erase local peaks. Any physical benchmark must specify the measured quantity, sensor and calibration, spatial registration, sampling density, uncertainty budget and intended application before defining a numerical acceptance threshold.

No single inter-device coefficient-of-variation target is appropriate for every analyte, geometry and instrument. Those targets must be selected per measurement purpose and capability.

## 5. PAL/PAW is its own intervention identity

For plasma-activated liquids, the exposure record requires:
- starting matrix identifier and matrix type;
- generation event and generation source;
- vessel/container material and lot;
- storage condition, temperature, stored-at and applied-at timestamps;
- elapsed minutes at application;
- measured chemistry observations;
- delivered volume and contact duration.

The chemistry of a liquid can change with matrix composition, generation conditions, container, storage, elapsed time and biological medium. Direct plasma exposure cannot be used as a proxy for PAW, and a PAW result does not transfer between formulations without an evidence bridge.

A source-description nuance to retain in future evidence updates: the 2026 BMJ Open PAW wound-rinse protocol title refers to two Australian hospitals, while its methods section describes a single-centre trial. The record should preserve the detailed design stated in the methods and flag the title/method discrepancy for source reconciliation rather than calling it multicentre without qualification.

Protocol source: https://pubmed.ncbi.nlm.nih.gov/42431657/

## 6. S1 synthetic audit corpus

Files:
- `data/research/plasma_medical_exposure_record_schema_v0_1.json`
- `data/research/plasma_medical_exposure_fixture_v0_1.json`
- `xtask/src/plasma_medical_exposure_audit.rs`

Command:
- `cargo run -p xtask -- plasma-medical-exposure-audit`

The 14 frozen cases are designed to produce:
- **2 PASS:** complete synthetic direct-CAP and PAW records;
- **8 BLOCKED:** absent physical observations, expired calibration, missing spatial map, unreviewed hotspot, missing PAW time-to-use, exposure-link mismatch, incomplete clinical context and unquantified measurement uncertainty;
- **4 REJECTED:** UNKNOWN encoded as zero, below-limit encoded as a value, malformed provenance digest and modality/type mismatch.

These are expected synthetic dispositions, not experimental results. A fixture passing means only that the data contract accepts the deliberately complete synthetic record.

The first audit implementation checks core semantic invariants with the workspace's existing Serde/JSON dependencies. The JSON Schema is a companion interchange contract; the Rust audit does not claim to be a complete JSON Schema 2020-12 validator.

## 7. What this enables later

### 7.1 Blind exposure-reconstruction challenge

A future physical research consortium could distribute frozen, anonymized source configuration and instrument data to independent teams. Teams would reconstruct exposure distributions and uncertainty without access to biological outcomes. A separate blinded evaluation could compare:
- bias and uncertainty calibration;
- repeatability across runs and devices;
- spatial coverage and hotspot detection;
- detection-limit handling;
- out-of-domain abstention;
- calibration drift detection.

This challenge must be tied to declared analytes, metrology and instrumentation. A strong software replay is not a substitute for traceable physical measurement.

### 7.2 Tissue-response surface benchmark

After approved partner-laboratory work exists, a preregistered benchmark could link physical exposure fingerprints to time-indexed biological responses for a specific model/tissue and assay. It must include both beneficial and harmful response regions, assay validity controls, biological replicates and external replication. Models must detect when they are outside their trained domain and abstain rather than recommend a patient dose.

### 7.3 Clinical translation benchmark

Clinical outcome records remain distinct. For a wound-care study, for example, microbial burden, wound area, complete healing, time to healing, recurrence, pain, adverse events, rescue intervention and patient-reported outcomes are separate endpoints. A surrogate must not silently stand in for patient benefit. Comparisons must bind wound etiology, standard care, co-interventions, follow-up and analysis plan.

## 8. Comparison with industry and current research

An established device such as kINPen MED shows that Luminous is not entering an empty market. The differentiating work should be measured against incumbent systems and the actual standard of care, not against a claim of being the first medical plasma jet.

The proposed competitive axes are:
- exposure reconstructability and measurement completeness;
- measurement uncertainty and calibration traceability;
- spatial mapping/hotspot reporting;
- treatment-to-treatment and device-to-device reproducibility;
- liquid chemistry/stability lineage for PAL/PAW;
- biological response reproducibility for a narrow model and assay;
- integration with clinically relevant endpoints;
- serviceability, consumables, gas/electricity burden and lifecycle cost;
- independent replication and honest abstention on out-of-domain evidence.

Economic comparison should eventually report cost per patient achieving a prespecified clinical endpoint—not merely price per device or treatment time. No cost-effectiveness claim is possible without validated clinical outcomes and lifecycle costing.

## 9. Regulatory and safety boundaries

The work remains preclinical research-data infrastructure. The schema is not a medical device, treatment protocol, clinical decision-support system or authorization.

South Africa's SAHPRA medical-device clinical-investigation guideline (SAHPGL-MD-17) describes a device-specific application package that includes the protocol/clinical investigation plan, investigator brochure, informed-consent documents, instructions for use, investigator GCP evidence, site/investigator agreements where applicable, trial registration evidence, ethics approval, insurance/indemnity and other supporting documents. Any future study requires specialist review of current applicable requirements.

Source: https://www.sahpra.org.za/wp-content/uploads/2025/09/SAHPGL-MD-17_v1-Guideline-on-Clinical-Investigation-of-Medical-Devices.pdf

No patient-facing treatment settings, hardware actuation, or human/animal experimentation are part of this work. Any physical or clinical validation must be planned with qualified biomedical and clinical partners, appropriate institutions, ethics committees and regulators.

## 10. Execution status and honest limits

Authored:
- exposure record JSON Schema v0.1;
- 14 synthetic known-answer cases;
- Rust audit module and five unit tests;
- this benchmark specification.

Not yet established:
- local Rust compilation, formatting or test execution;
- hosted CI exact-head PASS;
- full JSON Schema implementation conformance;
- physical metrology performance or inter-laboratory reproducibility;
- biological response, dose, safety or clinical efficacy;
- device readiness, regulatory status or Luminous medical-device capability.

The next engineering step is to run the Rust audit and independently validate the schema/fixtures; the next scientific step is to co-design an analyte-specific metrology protocol with a plasma physicist and biomedical laboratory. These remain separate gates.
