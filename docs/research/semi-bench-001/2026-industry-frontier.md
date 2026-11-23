# SEMI-BENCH-001 — 2026 industry-frontier update

**Review date:** 2026-11-21  
**Related work:** [SEMI-EDA-001 #13069](https://github.com/Luminous-Dynamics/symthaea/issues/13069), [SEMI-BENCH-001 #13058](https://github.com/Luminous-Dynamics/symthaea/issues/13058)

This update sharpens two parts of the programme: end-to-end EDA validation and standards-compatible manufacturing intelligence.

## 1. EDA: final routed quality is the benchmark, not surrogate improvement

### ChiPBench — NeurIPS 2025 Datasets & Benchmarks Track

Wang et al., *Benchmarking End-To-End Performance of AI-Based Chip Placement Algorithms*:

- Proceedings: https://proceedings.neurips.cc/paper_files/paper/2025/hash/1cba8502063fab9df252a63968691768-Abstract-Datasets_and_Benchmarks_Track.html
- DOI: https://doi.org/10.52202/085713-0673

The benchmark covers 20 circuits across CPU, GPU and NPU domains and evaluates six AI placement algorithms through the OpenROAD flow. The authors report that the evaluated AI methods degraded final PPA relative to the hierarchical macro-placement baseline.

**Interpretation:** a model can optimize its chosen placement proxy and still deliver worse physical-design outcomes. Luminous must evaluate the complete flow, preserve failures and compare against the hierarchical/analytical baseline.

### PPAPlace — ICCAD 2026

Chen and Han, *PPAPlace: Differentiable Cross-Stage Objectives for Chip Placement Optimization*:

- https://arxiv.org/abs/2608.13790
- DOI: https://doi.org/10.1145/3831252.3834172

The authors report:
- Pearson correlation near -0.08 between macro HPWL and WNS in their ChiPBench analysis;
- better final-timing fidelity from post-global-routing labels than from HPWL or pre-route timing, with a practical cost tradeoff;
- on five held-out ChiPBench test circuits, reported average WNS improvement of 22% and TNS improvement of 51% over the hierarchical baseline, while preserving power and routability and without test-circuit retraining.

These are paper-reported results, not reproduced Luminous outcomes or universal SOTA.

**Luminous implication:** introduce a proxy-fidelity gate. For every learned surrogate, publish the relationship between its predicted/ranked metric and final routed WNS/TNS/PPA on held-out circuits. A proxy that fails the fidelity gate cannot become the sole optimization objective.

### FastTuner — ACM TODAES 2026

*A Hybrid Reinforcement Learning Framework for Efficient Physical Design Parameter Tuning*:

- https://doi.org/10.1145/3779423

The paper reports up to 79.38% TNS improvement, 12.22% total-power reduction and over 50x runtime reduction across seven industrial TSMC 28nm designs. These are flow/task-specific maxima and must not be treated as directly comparable with ChiPBench or PPAPlace without matched constraints, design set, toolchain, baseline and measurement procedure.

### Proposed EDA comparison matrix

| Method | Strength | Required counter-check |
|---|---|---|
| Hier-RTLMP / hierarchical baseline | Uses design hierarchy and dataflow | Preserve exact baseline and flow |
| Analytical placement | Reproducible optimization baseline | Full post-route result |
| HPWL-only / pre-route surrogate | Cheap intermediate signal | Proxy-to-final rank fidelity |
| RL / GNN / Transformer placement | Learns structure/policies | Held-out circuits, hard constraints, full flow |
| PPAPlace-style cross-stage surrogate | Uses post-global-routing evidence | Surrogate calibration and final routed PPA |
| FastTuner-style multi-stage DSE | Tunes flow parameters | Same compute budget, total runtime and final design |

The primary score is not one PPA number. It is a constrained Pareto comparison of final WNS/TNS, area, power, routing feasibility, DRC, runtime, memory and transfer to unseen circuits.

## 2. Manufacturing intelligence: NIST TIDAL is a useful external architecture reference

NIST's **Super Intelligence for Manufacturing** programme is explicitly developing measurement science, standards, evaluation methods, datasets and benchmarks for trustworthy, interoperable manufacturing intelligence:

https://www.nist.gov/programs-projects/super-intelligence-si-manufacturing

Its TIDAL concept means **Teamed, Integrated, DT-assured and Layered Intelligence**. It organizes intelligence across:
- asset level: monitoring, inspection and root-cause analysis;
- site/area level: scheduling, line optimization and predictive alerts;
- enterprise level: generative design, cross-site benchmarking and orchestration.

NIST's public description also emphasizes human roles, contextualized manufacturing data, knowledge graphs, digital twins and a shared operation test bench spanning real equipment and simulators.

**Luminous implication:** the current Symthaea/Symtropy/Mycelix division should be mapped to the TIDAL layers, not positioned as a closed-loop autonomy system that replaces human and equipment authority. Our comparative advantage can be the evidence and replay fabric:
- Symthaea: propose and reason, with uncertainty and abstention;
- Symtropy: physical twin, scenario replay, constraints and VVUQ;
- Mycelix: identity, custody, resource/process provenance and audit trail;
- human/site/equipment systems: retain operational authority and stop rights;
- Sol-Atlas: connect technical references, source versions, standards and decisions.

Benchmark not just prediction accuracy but also context completeness, uncertainty, human review burden, action traceability, safe refusal, drift detection and rollback/reconciliation.

## 3. Standards: map to semiconductor digital-twin standardization

SEMI's February 10, 2026 Standards New Activity Report Form #7465 proposes a new *Specification for Digital Twins in Semiconductor Manufacturing* under its Information & Control Committee / Digital Twins in Manufacturing Task Force:

https://downloads.semi.org/web/wstdsbal.nsf/0e0afa4c4969bea688256efd0062a27c/61c2bf6034c0eed988258dcd00669238%21OpenDocument

The stated scope includes:
- definitions and framework evaluation;
- baseline capability requirements and application classes;
- outward-facing and security-related interfaces;
- internal interfaces and lifecycle specifications;
- prediction-quality metrics.

This is a standards activity proposal, not evidence that a final standard has been published or that any Luminous adapter conforms.

**Luminous implication:** maintain versioned mappings and compatibility tests to SEMI terminology and interfaces as the standard evolves. Do not create a competing universal twin ontology. Bind exact model, tool, dataset, configuration, lifecycle and prediction-quality evidence to the existing canonical owners.

## 4. New paradigm shifts to benchmark

1. **Proxy-fidelity-aware optimization:** every surrogate must prove held-out correspondence to the final physical outcome.
2. **Decision-value benchmarking:** charge each method for data acquisition, compute, operator review, retries and full-flow duration.
3. **Evidence-context completeness:** missing calibration, timebase, lot, tool state or PDK identity must lower the claim ceiling.
4. **Human-teamed autonomy:** evaluate correct escalation, abstention, explanation and safe handoff—not just autonomous completion.
5. **Twin assurance:** quantify model discrepancy, uncertainty propagation, drift and OOD behavior; use shadow-mode and measured validation before actuation.
6. **Cross-layer Pareto engineering:** optimize device, circuit, package, thermal, power, testability and lifecycle under explicit constraints.
7. **Standards-compatible openness:** publish exact schemas, adapters, provenance receipts and conformance tests, while respecting proprietary designs/data.

## 5. Competitive strategy

The field does not need another AI system that produces plausible placement coordinates or another dashboard that predicts defects. Our opportunity is to connect the design recommendation to the exact input context, physical evidence, downstream outcome and independent verification.

The strongest near-term claims to pursue are:
- better prediction of final routed PPA on unseen designs;
- lower decision/runtime cost at matched final-quality constraints;
- better calibrated wafer/process risk under time/group shift;
- more reliable virtual metrology with explicit uncertainty and calibration lineage;
- easier independent reproduction and audit of manufacturing decisions.

No physical fab-yield improvement, fabricated-silicon superiority or equipment qualification follows from these benchmark results alone.


## 6. Wafer-map graph learning and unseen-lot evaluation

Awais et al. (4 July 2026) report WM-811K GNN results using 8-connectivity, GIN and production-context contrastive pairs: ARI 0.89/NMI 0.87 for unsupervised clustering, 98.6% fine-tuned accuracy, macro-F1 0.975, 95.7% accuracy with 25% labels and LOCO OOD detection 89.3% at 95% ID retention (AUROC 0.927). The benchmark sampled 30% of 811,457 maps and used stratified sample-level 60/20/20 splits. This does not establish unseen-lot transfer. Source: https://doi.org/10.1007/s10845-026-02906-3.

The released implementation uses lotName/production proximity in pair construction but stratified splitting. Its OOD script appears to calibrate the threshold from test_in although the article describes validation calibration. Treat this as a paper/repository reproduction mismatch until resolved. Do not claim the paper itself used test calibration.

New benchmark: [SEMI-WM-LOT-001 #13122](https://github.com/Luminous-Dynamics/symthaea/issues/13122), which separates exact sample-level reproduction, lot-disjoint evaluation, time-forward evaluation with embargo, split-local contrastive pair construction and validation-only OOD calibration.

## 7. Industrial CP/DFC multi-view wafer maps

Wu et al. (1 July 2026), https://doi.org/10.1016/j.eswa.2026.131994, describe item-wise CP Map and DFC Map semantics with M3Net known-class recognition and Parallel SVM novel-class review. Public raw data access, sample provenance and data rights remain unverified. New benchmark: [SEMI-MULTIVIEW-001 #13123](https://github.com/Luminous-Dynamics/symthaea/issues/13123). It benchmarks CP-only, DFC-only, multi-view fusion, missing-view robustness and open-world disagreement without treating absent views as zero-valued measurements.

## 8. PPAPlace-5000 is an actionable reference corpus, subject to rights review

The PPAPlace-5000 release describes 5,000 complete placements on 10 ChiPBench circuits with post-global-routing WNS/TNS, power and area labels; the compressed archives total 8.47 GiB. Its Hugging Face dataset card exposes a license field of other, so exact terms must be reviewed before copying, caching or redistribution. The PPAPlace source docs pin ChiPBench and OpenROAD commits and distinguish within-circuit from leave-one-circuit-out evaluation. Dataset: https://huggingface.co/datasets/ValleyC/PPAPlace-5000.
