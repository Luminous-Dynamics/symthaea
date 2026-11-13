# OLI interoperability with NASA's Life Detection Knowledge Base and Life Detection Forum

**Status:** design specification; no NASA data imported and no NASA integration tested  
**Machine-readable mapping:** `benchmarks/oli/nasa-ldkb-interop-profile-v0.json`

## Strategic position

Luminous Dynamics should **complement NASA's life-detection knowledge ecosystem, not recreate it**.

NASA's Life Detection Knowledge Base (LDKB) is a community-maintained resource that organizes potential biosignatures across Chemistry, Structure and Activity. Its entries use four assessment dimensions: biological prevalence, abiotic prevalence, biological signal strength and abiotic signal strength. The LDKB preserves literature arguments supporting and opposing a measurement's usefulness, identifies knowledge gaps and links to a Measurement Technology Module and Science Traceability Tool.

The Life Detection Forum (LDF) connects science objectives, biosignature knowledge, measurement technologies and instrument requirements. NASA's Ladder of Life Detection is a discussion and mission-design framework, explicitly not a definitive final product or a universal instrument/biosignature endorsement.

Primary references:
- [NASA Life Detection Knowledge Base](https://science.nasa.gov/astrobiology/researchers/life-detection-resources/ldkb/)
- [NASA Life Detection Forum](https://lifedetectionforum.com/)
- [NASA Ladder of Life Detection](https://science.nasa.gov/astrobiology/researchers/life-detection-resources/ladder-of-life-detection/)
- Pohorille et al., *Life Detection Knowledge Base: A Community Tool for Knowledge Management and Representation*, DOI [10.1089/ast.2024.0106](https://doi.org/10.1089/ast.2024.0106)

## Where OLI should add value

The LDKB is a knowledge and argument-management resource; it is not a universal classifier. OLI should extend the scientific workflow with executable, independently checkable operations:

1. Typed, versioned evidence records with source and artifact digests.
2. Explicit support/contradiction relationships, common-cause provenance and dispute handling.
3. Quantitative measurement coverage, sampling cadence, units, detection limits and calibrated detection probabilities.
4. Model-adequacy diagnostics and calibrated abstention for unfamiliar mechanisms.
5. Deterministic abiotic controls, adversarial confidence-surface testing and open-world evaluation.
6. Prospective experiment selection with rival predictions, sham/passive controls, measurement cost, and safety/authority constraints.
7. Replayable evidence receipts connecting the claim to its raw input, model, analysis, uncertainty and independent verification.

The differentiation is **executable evidence and experiment assurance**, not a larger static catalogue.

## Preserve NASA's conceptual distinctions

The four LDKB assessment dimensions must remain separate. Biological prevalence, abiotic prevalence, biological signal strength and abiotic signal strength are not four inputs to be collapsed into a single universal life probability.

Likewise:
- A biosignature claim is not a measurement.
- Signal strength is not posterior probability of life.
- An instrument's advertised detection limit is not the probability of detecting a target in a heterogeneous sample.
- A non-detection is not evidence of absence without a coverage and false-negative model.
- A rung on the Ladder is not a context-free ranking of all possible life forms.

OLI should retain the NASA entry URI, category and original terminology when linking or importing a permitted record. Its substrate-agnostic ontology is an additional representation, not a replacement for the source taxonomy.

## Conceptual mapping

| NASA resource concept | OLI representation | Required safeguard |
|---|---|---|
| Chemistry/Structure/Activity entry | PropertyQuestion, MechanismHypothesis, ObservablePrediction | Preserve the external identifier and term |
| Literature argument for/against | EvidenceRecord, EvidenceConflict | Keep support and contradiction separate |
| Biological/abiotic prevalence | Environment-conditioned prior evidence | Unknown stays unknown; retain scope and provenance |
| Biological/abiotic signal strength | Observable response model and calibrated likelihood | Do not equate signal strength with probability of life |
| Measurement Technology Module | ObservationCoverage and instrument capability evidence | Preserve units, calibration, footprint, cadence and detection limit |
| Science Traceability Tool | Objective → property question → observable → measurement/instrument requirements | Proposals remain advisory and authority-gated |
| Ladder of Life Detection | External context reference | Do not turn rung position into a universal life score |

The machine-readable mapping defines required fields, source identity, measurement and experiment metadata, and four interoperability benchmark tasks.

## Proposed interoperability benchmarks

### 1. LDKB traceability mapping

Map a selected set of LDKB entries through objective, observable, measurement and instrument requirements without losing source IDs or terminology.

Measure mapping completeness, URI integrity, term preservation and expert adjudication agreement. Missing fields must remain explicitly unknown.

### 2. Four-dimension evidence completeness

Check whether each of the four assessment dimensions has appropriately scoped, cited evidence. Measure gap-detection recall and provenance completeness. The system must not impute a missing abiotic prevalence argument or convert an unknown biological signal-strength estimate to zero.

### 3. Traceability to executable experiment

Turn a documented evidence gap into a bounded, advisory experiment proposal with:
- at least two rival hypotheses;
- quantitative predicted outcomes or an explicit non-derivable status;
- an intervention or measurement and a matched sham/passive control where applicable;
- instrument and sample observability;
- resource costs, stopping rule and uncertainty;
- safety, planetary-protection, rights and human-authority constraints.

Compare against an exact subset oracle on small finite experiment sets. An optimizer receives no credit if it violates a non-tradable constraint.

### 4. Replay and independent review

A separate evaluator reconstructs a claim from the external entry through raw measurement, preprocessing, inference and experiment proposal. Measure exact replay, artifact-digest matching, reviewer disagreement and unresolved-claim rates.

## Data and governance policy

This specification does **not** authorize scraping, bulk importing or rehosting NASA community content. Use links/citations or a documented export/API with permission from the resource owner. Any future import must retain original identifiers, attribution, source version, access date and file-level rights.

There is no NASA endorsement, partnership or integration implied. This design does not claim that LDKB is inferior or incomplete for its intended community knowledge-management role.

## Execution status and next steps

- Interoperability mapping and machine-readable profile authored.
- No NASA LDKB records imported or rehosted.
- No API integration or interoperability test executed.
- Next implementation should begin with a small, explicitly user-authorized set of linked entries and a read-only mapping adapter. Do not build a scraper or duplicate database.

## Claim ceiling

This is a design for interoperating with NASA's community astrobiology resources and extending them with executable assurance. It is not a NASA-endorsed interface, an operational life-detection system, or a claim that OLI outperforms NASA's resources.
