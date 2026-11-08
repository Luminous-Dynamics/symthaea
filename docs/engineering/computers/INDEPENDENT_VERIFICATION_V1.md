# Independent verification for physical computing — research contract v1

**Owner:** RELIABILITY-COMPUTE-001 #10993, composed with adaptive mathematics #10933 and FPGA workload qualification #10901.  
**Status:** research design note; no implementation or hardware result is claimed.  
**Purpose:** compare independent verification against full redundant recomputation for detecting incorrect outputs from analog, photonic, FPGA and other imperfect execution substrates.

## 1. Research hypothesis

The cost of establishing correctness may be lower than the cost of repeating the entire computation. We should test algorithm-specific verifiers that exploit mathematical structure, while treating the verifier as a separate component with its own failure modes and physical dependencies.

The verifier is not trusted merely because it is digital. It must have an independent implementation/oracle, an explicit input-provenance boundary, a distinct resource/failure model, and its own measured error/availability.

## 2. Verification families

### V0 — independent exact oracle

Use for exact HDC retrieval, bit identity, protocol behavior and deterministic state transitions. Compare the complete output against a separately implemented scalar oracle. Include ties, malformed input, resets, overflow, incomplete blocks and backpressure.

This is the exact baseline. No probabilistic acceptance is allowed in this lane.

### V1 — algorithm-based fault tolerance (ABFT)

Exploit algebraic invariants, row/column checksums, checksummed data encodings and operation-specific conservation laws. Measure both detection and localization, and distinguish checks that detect an error from checks that can reconstruct/correct it.

Use established systolic-array ABFT as a baseline rather than claiming checksum verification as new. A 2023 IEEE Transactions on Computers study reported under 1% arithmetic overhead for its evaluated CNN applications; the result is workload- and architecture-specific, not a universal overhead guarantee.

### V2 — randomized matrix-product verification

For a candidate product C = AB, a Freivalds-style check samples random vector r and compares A(Br) with Cr. Over a finite field with the standard random-vector construction, one independent check has false-acceptance probability at most 1/2 for a fixed incorrect product; k independent checks reduce that bound to at most 2^-k.

For real-valued / analog arithmetic, do not reuse that finite-field bound. Gaussian projection variants have a measure-zero false-positive result under exact arithmetic, but floating-point roundoff, analog noise, quantization, nonzero acceptance thresholds and correlated faults change the operational false-acceptance probability. Establish it for the actual arithmetic and fault model; otherwise report it as unknown.

For a repeated MVM workload y = Ax with fixed A, investigate precomputing a trusted projection qᵀ = rᵀA and checking qᵀx against rᵀy. The one-time preprocessing cost, input immutability, verifier storage and projection refresh cost must be counted. The checker must not derive q from the same faulty execution path as y.

### V3 — residual, backward-error and condition-aware verification

For Ax = b, measure at least:
- residual norm ||Ax-b||;
- relative residual ||Ax-b|| / (||A||||x|| + ||b||);
- backward error;
- forward error against a high-precision reference where available;
- condition estimate and its uncertainty.

A small residual alone is not proof of a small forward error when A is ill-conditioned. The verification contract must carry the conditioning assumptions and uncertainty of the independent A/b inputs and norm estimates.

### V4 — dynamical defect verification

For a candidate ODE trajectory x(t), evaluate the defect d(t) = dx/dt - f(x,t) using an independent path. Report integrated defect, trajectory error against a reference, event timing error, stability and the declared horizon.

A small local defect is not automatically a global trajectory-error guarantee. A certified or empirically justified stability/Lipschitz envelope is needed to translate defect into a trajectory bound.

### V5 — independent fallback / redundant execution

Run the same task on an independent path and compare outputs. This remains a necessary high-cost baseline, especially where an operation has no cheap independent mathematical invariant.

Measure whether the paths truly have separate power, clock, reset, controller, firmware, input acquisition and failure-analysis domains. Two copies under one common cause are not independent redundancy.

## 3. Independence and trust boundary

A verifier is eligible for an independence claim only when the manifest records:

- separate implementation identity and source digest;
- separate execution resources and fault-domain membership;
- independent power/clock/reset/controller and data path, or an explicit list of shared dependencies;
- the provenance and digest of the input before it enters the candidate path;
- verifier calibration, numerical precision and evidence age;
- random challenge generation and timing, where randomized checks are used;
- evaluator-only ground truth withheld from both candidate and verifier.

A check performed after the input has already been corrupted in a shared sensor/host path cannot establish that the original input was correct. Bind ingress integrity and input provenance to the workload contract.

For security against adaptive or malicious outputs, generate the verification challenge independently and unpredictably after the candidate output is committed. For non-adversarial stochastic hardware faults, fixed-seed replay remains useful for reproducibility, but does not itself provide adversarial unpredictability.

## 4. Benchmark matrix

Compare the same workload and fault scenarios under:

A. No verifier.  
B. Full redundant recomputation and comparison.  
C. ABFT/checksum verifier.  
D. Randomized Freivalds/Gaussian projection verifier.  
E. Residual/backward-error/condition-aware verifier.  
F. Verifier plus fail-closed rejection.  
G. Verifier plus independent fallback/recovery.  
H. Combined topology-aware mitigation and the cheapest qualified verifier.

Keep the algorithm-track and system-track results separate. Keep same-algorithm substrate comparisons separate from best co-designed architecture comparisons.

## 5. Required measurements

Report, separately:

- correct result accepted by deadline;
- silent incorrect result accepted;
- detected/rejected result;
- verifier false-acceptance and false-rejection rates with confidence intervals;
- fault detection coverage and localization resolution;
- verifier unavailability and verifier self-test failure;
- candidate and verifier resource/failure-domain overlap;
- verification latency p50/p95/p99 and energy per scheduled / accepted task;
- input movement, projection preprocessing, challenge generation and calibration costs;
- fallback activation, recovery probability, recovery time, recovery energy and state loss;
- exact workload, source, scenario and measurement digests.

A verification method wins only if it reduces the full cost of a correct accepted result under the same task contract and fault distribution, without silently increasing undetected-corruption risk.

## 6. Required negative controls

- Candidate and verifier share a failed power rail, clock or reset.
- Candidate and verifier share a controller, host, input sensor or preprocessor.
- The verifier itself is faulted or its calibration is stale.
- The same software library / algorithm bug exists in candidate and checker.
- Input corruption occurs before both candidate and verifier ingest the data.
- Random challenge is predictable to an adaptive adversary (security evaluation only).
- A fault lies in the null space / blind spot of a deterministic checksum.
- High noise / quantization causes a false rejection or hides a small but task-critical error.
- Ill-conditioned matrix: residual passes but forward error violates the task contract.

## 7. External anchors

- Mascagni, Ji & Li, *Gaussian Variant of Freivalds' Algorithm for Efficient and Reliable Matrix Product Verification*, SIAM Journal on Computing; NIST record: https://www.nist.gov/publications/gaussian-variant-freivalds-algorithm-efficient-and-reliable-matrix-product-verification
- *Efficient Error Detection for Matrix Multiplication With Systolic Arrays on FPGAs*, IEEE Transactions on Computers (2023): https://doi.org/10.1109/TC.2023.3248282
- Cora et al., *Fast SEU Detection and Recovery in FPGA-Based AI Accelerators*, ACM Transactions on Reconfigurable Technology and Systems (2026), DOI: https://doi.org/10.1145/3806052
- Mannocci et al., *Achieving High Precision in Analog In-Memory Computing Systems*, npj Unconventional Computing (2026): https://www.nature.com/articles/s44335-025-00044-2

## 8. Claim ceiling

A mathematically valid checker under exact arithmetic is not automatically reliable under finite precision. A low measured verifier error on synthetic faults is not a physical probability bound. A verifier that shares an upstream input or infrastructure failure is not independent. Randomized checking is not exact checking. A detected and rejected output is safer than silent corruption, but is not a completed task. All physical reliability claims require identified hardware, measured fault distributions, raw receipts and independent reproduction.
