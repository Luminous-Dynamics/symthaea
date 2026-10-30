# QARCH-013 — Cross-suite application benchmark interoperability

**Issue:** [#9099](https://github.com/Luminous-Dynamics/symthaea/issues/9099)  
**Schema:** `docs/research/quantum/schemas/QARCH-013-RESULT-LEDGER.schema.json`  
**Fixtures:** `docs/research/quantum/benchmarks/QARCH-013-LEDGER-FIXTURES.json`

## Design principle

QARCH-013 is a thin result projection over existing QARCH-001 architecture estimates, QARCH-002 stateful fusion, QARCH-003 end-to-end quantum utility, QARCH-004 workload definitions, and the existing provenance/evidence systems. It does not create a second simulator or resource estimator.

The ledger deliberately stores a vector of results instead of one universal “quantum score”: task quality, conditional component performance, end-to-end accepted-answer performance, resource cost, physical/system evidence and the baseline identity must remain independently inspectable.

## External-suite crosswalk

| Suite / reference | What it contributes | Use when | Non-comparability guard |
|---|---|---|---|
| [QOBLIB 2026](https://zib-aopt.github.io/QOBLIB/) | Model-independent optimization tasks, shared instances and strong classical solver baselines | The Luminous problem/objective/constraints match a QOBLIB instance exactly | Do not map photonic sampling or chemistry to an unrelated QUBO score |
| [JUNIQ](https://www.fz-juelich.de/en/juniq/juniq-benchmarks) | Vendor-agnostic application-level tests across QPU technologies and generations | Workload semantics and acceptance rule match | Retain JUNIQ's output/quality protocol; document any adapter |
| [IonQ application-level benchmark framework](https://www.ionq.com/resources/benchmarking-white-paper) | Application workloads with output-quality and time-to-solution reporting | A workload and quality gate are comparable | Treat its aggregate as framework-specific, not a universal quantum metric |
| [QUOPS](https://www.quantinuum.com/blog/introducing-quops-a-new-way-to-measure-the-performance-of-quantum-computers) | Architecture-agnostic system-level capacity/throughput concepts, including correction, decoding and compilation | Reporting system capacity alongside task performance | QUOPS is complementary; it does not replace a task's correctness oracle |
| Published photonic component references | Conditional source/readout, HOM, fusion and interconnect figures | Comparing the corresponding optical component and conditioning event | Conditional-on-detection fidelity is never an unconditional verified-answer rate |

The first crosswalk should use QOBLIB for a genuinely matched optimization instance, and a photonic component reference as a component-level record. The synthetic fixture is only a schema/semantics test; it does not represent an actual QOBLIB submission or hardware run.

## Layered result contract

1. **Task:** fingerprint, instance/objective digest, output-quality predicate, threshold and verifier.
2. **Platform:** architecture family, configuration fingerprint, hardware/compiler/software identity.
3. **Conditional components:** metric, value, uncertainty, conditioning event and denominator description. These are diagnostic parameters, not end-to-end success.
4. **End-to-end result:** attempted instances, accepted verified answers, wall time, energy, cost, tail latency, resource terms included and terms still missing.
5. **Quality:** lower/upper quality interval against the same threshold for quantum and classical.
6. **Evidence:** measured, calibrated simulation, independent simulation, paper-reported, analytic or synthetic, tied to immutable source/artifact identity.
7. **Baseline and mapping:** exact comparator and explicit comparable / adapter-required / not-comparable state.

The existing QARCH-003 utility estimator remains the owner of interval cost per verified acceptable answer. QARCH-013's validator should check that the numerator/denominator and resource scope match the underlying profile; it should not recalculate or replace that estimator.

## Two frozen examples

### Paper-reported conditional photonic reference

The first fixture records representative metrics from the published integrated-silicon-photonics report: state preparation/readout, HOM visibility, fusion and chip-interconnect results. They are marked `paper_reported` and `conditional_component_only`, with conditioning/denominator text. End-to-end attempts, accepted verified answers, loss and total resources are unavailable, so the result is marked `not_comparable`. It cannot be used to infer an unconditional logical operation or application speedup.

### Synthetic matched quantum/classical ledger

The second fixture exercises a full comparison record using 100 synthetic instances, a common quality threshold, independent verifier identity, end-to-end resource totals, and a classical baseline. Its numbers are intentionally illustrative. They are not measurements or an external suite result.

## Semantic validation beyond JSON Schema

A verifier must reject or mark incomplete:
- mismatched workload fingerprint, objective digest, instance or quality predicate;
- conditional component metric copied into an end-to-end accepted-answer probability;
- accepted-answer count greater than attempted-instance count;
- absent verifier, baseline, or unmatched quality gate for a comparison marked matched;
- any reported end-to-end per-answer metric whose denominator is not verified acceptable answers;
- missing resource terms represented as zero rather than unknown;
- evidence class promoted beyond the source artifacts' authority;
- claims of quantum advantage from synthetic, model-only or component-conditional evidence.

## Proposed acceptance gates (not achievements)

- 100% unit/conditioning/denominator consistency in frozen fixtures.
- Zero false promotion of conditional fidelity to unconditional system success.
- Zero “comparable” status if workload/quality/baseline contracts differ.
- Every external mapping records its version, instance identity, and adapter status.
- All application comparisons report the strongest applicable classical reference and Pareto tradeoffs over wall time, energy and cost.
- Any claimed improvement is repeated on frozen instances with independently verified outputs and exact environment provenance.

No single weighted score should hide an energy, cost, tail-latency, fidelity, or verified-output regression.
