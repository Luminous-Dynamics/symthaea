# HARDWARE-TOPOLOGY-001 — Synthetic assembly and ECAD fixture corpus v0.1

**Status:** DESIGN_ONLY / synthetic fixture specification  
**Tracking:** [HARDWARE-TOPOLOGY-001 #10530](https://github.com/Luminous-Dynamics/symthaea/issues/10530)  
**External comparator:** [BenchCAD 2.0](https://benchcad.com/benchcad-2)  
**Execution:** no CAD/ECAD kernel, EDA tool, renderer, benchmark runner or external task has been executed for this document.

## 1. Evaluation thesis

The corpus separates two questions:

1. Does the recovered assembly contain the right identified parts in the right reference frames, with valid mating/clearance relationships?
2. Does the recovered board connect the correct identified pins into the correct electrical nets, including layer/via semantics?

Neither question can be answered by visual plausibility alone. A correct individual shape may be misplaced. A mostly correct component inventory may still contain a critical short, open, or pin-map error.

This corpus is a proposed source/evaluator contract. It does not create a new geometry identity, component catalogue, BOM, circuit ontology, manufacturing profile or physical authority. Existing canonical owners remain authoritative.

## 2. Corpus and trust boundaries

Each eventual fixture must have three versioned artifacts:

- **Candidate-visible input:** supplied parts/drawings/views, board renders/layer evidence, task constraints and the permitted component/library identity set.
- **Evaluator-only oracle:** expected part transforms, assembly constraints, pin-to-net partitions, critical short/open controls, and accepted equivalence classes.
- **Provenance record:** exact source geometry/board revision, generator revision, unit/frame policy, evaluator revision and artifact digests.

Do not expose hidden transforms, expected net partitions, evaluator mutation controls or reference answers to the scored candidate. The markdown examples below define the public synthetic contract, not a sealed external benchmark.

## 3. Mechanical assembly fixtures

### ASM-P01 — six-part instrument support frame

**Global frame:** right-handed XYZ; origin at the centre of the bottom face of the base plate; +Z upward; dimensions in mm.

**Parts and nominal geometry:**
- **BASE-01:** 200 × 120 × 8 rectangular base plate; bottom face at z=0; centre at (0,0).
- **POST-L:** 20 × 20 × 100 square post; centre at (-80,0,58); axis parallel +Z.
- **POST-R:** same part variant; centre at (80,0,58); axis parallel +Z.
- **BEAM-T:** 180 × 20 × 20 rectangular beam; centre at (0,0,112); long axis parallel +X.
- **TIE-R:** 180 × 20 × 10 tie bar; centre at (0,45,100); long axis parallel +X.
- **GUSSET-L, GUSSET-R:** asymmetric 6 mm plate gussets with a 90 mm leg and a 30 mm leg, each with two Ø6 mounting holes. The long leg faces inward; the datum hole is 15 mm from the lower end. Left/right variants are not freely interchangeable.

**Required relationships:** posts mate to the two base datum pads; the beam seats on both post tops; the tie bar mates to the rear post faces; gussets connect the declared post/beam interfaces. No member may be reversed merely because its outer bounding box still fits.

**Negative controls:** reverse one gusset; rotate one asymmetric tie/brace 180 degrees; move one post 2 mm off its datum; substitute a visually similar but wrong left/right gusset; duplicate a bolt pattern while omitting its counterpart.

**Acceptance:** correct exact part/variant/count; per-instance pose within the frozen tolerance; all declared mating constraints satisfied; no interference beyond the declared tolerance; assembly envelope and datum graph consistent. A shape-only match is insufficient.

### ASM-P02 — bearing/shaft alignment

**Parts:** base plate with two bearing pedestals, two exact bearing variants, a Ø20 mm shaft, one keyed hub and one coupling. The shaft axis and pedestal bore axes are the primary reference.

**Intervention:** move the second bearing pedestal 0.5 mm in +Y, preserving the part geometries. The oracle must detect axis misalignment rather than treating each part as individually valid.

**Acceptance:** concentricity/coaxiality, shaft engagement length, axial spacing, key orientation and coupling interface each reported separately. Do not infer bearing load capacity or operational safety from geometric alignment.

### ASM-P03 — datum and pose ambiguity

**Input:** two identical cylindrical spacers and a symmetric plate. The source does not distinguish which spacer is the left/right instance; both assignments produce the same constrained assembly.

**Required disposition:** accept the symmetry-equivalent assignments or report the equivalent set. Do not invent a unique historical instance identity when the input cannot identify it.

### ASM-N01 — collision and clearance

**Input:** otherwise valid assembly with one spacer shifted 0.2 mm into a declared keep-out envelope.

**Required disposition:** fail the applicable non-interference gate; report the exact conflicting pair and measured penetration/clearance. A small global geometric score must not compensate for the collision.

### ASM-N02 — missing, duplicate or incompatible article

**Input variants:** omit one critical fastener; duplicate one support post; replace a specified bearing with a visually similar but incompatible bore/width variant; use a superseded component revision.

**Required disposition:** explicit missing/duplicate/incompatible identity outcome, not a geometric similarity pass. Reuse exact component identities and substitution policy from ROB-CATALOG.

## 4. ECAD connectivity fixtures

### ECAD-P01 — low-voltage sensor/controller board

**Declared board profile:** synthetic two-layer low-voltage board. Components and pin maps are explicit; dimensions and render placement are not electrical identity.

**Component identities:**
- **J1:** 2-pin input connector (1=VIN, 2=GND).
- **U1:** 3-pin regulator (IN, GND, OUT).
- **U2:** MCU (VDD, GND, PA0, PA1).
- **J2:** 2-pin signal connector (1=GPIO_A, 2=GPIO_B).
- **R1:** pull-up resistor (1=+3V3, 2=GPIO_A).
- **LED1:** indicator (A=GPIO_B, K=GND).

**Expected net partition:**
- **VIN:** J1.1, U1.IN.
- **+3V3:** U1.OUT, U2.VDD, R1.1.
- **GND:** J1.2, U1.GND, U2.GND, LED1.K.
- **GPIO_A:** U2.PA0, J2.1, R1.2.
- **GPIO_B:** U2.PA1, J2.2, LED1.A.

**Acceptance:** exact component/pin identity, exact expected net partition (after permitted alias normalization), and no undeclared shorts/opens. The candidate must preserve no-connect states separately from unknown/missing pin evidence.

### ECAD-N01 — critical rail short

**Mutation:** merge VIN, +3V3 and GND into one candidate net.

**Required disposition:** hard failure on critical short. Component recognition and footprint accuracy cannot compensate.

### ECAD-N02 — critical open

**Mutation:** split U2.VDD from the +3V3 net while retaining the visual appearance of a continuous power trace in the top render.

**Required disposition:** hard failure on missing power connection. The scorer must compare the exact pin partition, not only trace/component image overlap.

### ECAD-N03 — crossing without connection

**Mutation:** two traces cross in the same rendered view but have no junction and no shared electrical node.

**Required disposition:** keep nets separate. A visual crossing alone is not a connection.

### ECAD-N04 — cross-layer ambiguity and via control

**Positive:** a declared via connects the correct top/bottom copper nodes.

**Negative:** the same geometry is drawn on two layers without a via.

**Required disposition:** the positive case joins the net; the negative case remains disconnected. If the supplied views do not reveal the relevant via/layer evidence, return UNKNOWN/NEEDS_EVIDENCE rather than assume continuity.

### ECAD-N05 — swapped pin mapping

**Mutation:** U2 symbol pin identities are exchanged while package appearance is unchanged.

**Required disposition:** reject the incorrect pin-to-net mapping using the exact library/device revision. Do not map pins by visual proximity alone.

### ECAD-N06 — no-connect versus missing evidence

**Mutation variants:** (a) a pin is explicitly marked NC; (b) a pin label is unreadable; (c) a pin is missing from the extracted graph.

**Required disposition:** these are three different states. Only explicit NC is a positive no-connect claim; unreadable or absent evidence remains unknown/unresolved.

## 5. Evaluation metrics

Keep assembly and ECAD metrics separate.

### Assembly
- Exact component/variant/count coverage.
- Per-instance translation and rotation error.
- Datum, axis and mating-constraint satisfaction.
- Clearance/interference severity and pair identity.
- Assembly envelope and interface error.
- Symmetry-aware equivalence and abstention quality.
- Full accepted/rejected/unknown/unsupported/ungraded accounting.

For symmetric components, compare against the declared equivalence class of acceptable transforms rather than a single hidden transform. Do not use raw face/edge numbers as stable instance or feature identities.

### ECAD
- Component and footprint identity accuracy.
- Pin-number/orientation accuracy.
- Pin-to-net precision and recall.
- Exact net-partition equivalence (with only declared aliases normalized).
- Critical short/open detection, including false-negative rate.
- Via/layer correctness.
- NC versus unknown state accuracy.
- Full accepted/rejected/unknown/unsupported/ungraded accounting.

An overall edge-level F1 score may be diagnostic, but it cannot override a critical rail short/open false pass.

## 6. Metamorphic and adversarial controls

1. Reordering component records must not change the canonical assembly result.
2. Renaming arbitrary face/edge indices must not change semantic part identity or mating outcomes.
3. A global rigid transform of all parts and reference datums must preserve relative mating and clearance results.
4. A symmetric component may admit more than one equivalent pose; an asymmetric keyed component may not.
5. A geometrically identical component with a different protected part number/revision must remain a different article identity until an approved substitution says otherwise.
6. Swapping candidate order must not change the selected acceptable set.
7. A net-label alias change may preserve a graph only when the alias is declared in the exact profile.
8. A same-layer trace crossing without a node remains disconnected.
9. A via inserted/removed changes layer connectivity exactly.
10. A pin-map revision changes the pin-to-net interpretation even when the rendered footprint is unchanged.
11. Missing inner-layer evidence changes disposition to UNKNOWN, not PASS.
12. Deliberately corrupted evaluator task counts, digests or missing negative cases must reject the benchmark receipt.

## 7. Competitive baseline and scope

BenchCAD 2.0 is a distinct external benchmark, not a substitute for Parametric CAD Bench V3:
- BenchCAD 2.0 spans drawing-to-part, drawing-to-assembly, reverse-engineered part/assembly, parts-plus-assembly, and PCB render-to-wiring tasks.
- Its published v2.0 Core reference reports GPT-6 Astra max 0.59 at about $11 per case / 62 min median, and Claude Opus 5.5 max 0.57 at about $28 per case / 3.7 h.
- Its published v2.0 protocol uses 30 rounds and does not carry reasoning between rounds. The current v2.1 protocol carries reasoning, uses 60 rounds and 500k output tokens per case; do not mix the scores.
- The Public 100-case set is evaluation-only, non-reportable and disjoint from Core; Core 100 requires an evaluation-only agreement. No BenchCAD cases, drawings, renders, reference files or canaries are included in this corpus.

The Luminous synthetic fixtures should be designed to test our own canonical semantics first. External comparisons require identical task sets, tool interfaces, budgets, evaluator and harness version. The benchmark score does not establish mechanical safety, PCB manufacturability, EMC/SI/PI compliance or physical performance.

## 8. Runner-independent next work

1. Independent review of expected transforms, mating relations and pin-to-net partitions.
2. A neutral machine-readable manifest with exact board/component identifiers, units, frames and terminal dispositions.
3. Synthetic source artifact generation in a pinned CAD/ECAD environment after contract review.
4. A separate oracle/verifier implementation that never imports candidate answers.
5. Mutation testing against both production adapter and evaluator.
6. Composition map to #6040, #4877, #4859, #4883, #8223, #5633/#5661 and #6179.
7. Only after isolation/custody qualification: a pinned adapter, a matched baseline and (if agreement/access is granted) BenchCAD Core evaluation.

## 9. Claim ceiling

This document specifies synthetic fixture intent only. It contains no validated CAD/STEP files, native assembly, KiCad project, extracted netlist, scorer execution or benchmark result. It establishes no mechanical fit, electrical safety, manufacturability, EMC/SI/PI performance, compliance, reliability or physical qualification.
