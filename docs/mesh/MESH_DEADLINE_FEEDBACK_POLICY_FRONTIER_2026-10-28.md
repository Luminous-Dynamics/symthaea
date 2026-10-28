# Conservative deadline-aware recovery policy frontier — 28 October 2026

**Evidence class:** Deterministic post-processing of a saved synthetic event-model CSV. This is not a new simulation, Rust test, production policy, or radio measurement.

## Decision rule

For each channel model, spreading factor, ACK-loss mode and deadline, a strategy is eligible only if its **lower Wilson 95% confidence bound** for delivery by that deadline is at least the requested reliability target. Among eligible strategies, select minimum mean analytical PHY airtime across all trials, including failed deliveries. Tie-break by higher point-estimate delivery, then lower mean PHY bytes. If nothing qualifies, report `infeasible` and preserve the best-observed strategy for diagnosis rather than relaxing the target.

This is a multi-objective decision policy: correctness and deadline-qualified delivery are gates; airtime is optimized only after a strategy satisfies the target. It avoids maximizing a headline average at the expense of reliability assurance.

## Feasibility across the synthetic scenario matrix

The matrix includes four channel models, three spreading factors (SF7/SF9/SF12), four ACK-loss modes and four deadlines (5/15/30/90 seconds), producing 192 scenario/deadline cases per target. Each underlying configuration uses 2,000 synthetic trials.

| Required lower 95% delivery bound | Feasible cases | Total | Interpretation |
|---:|---:|---:|---|
| 90.0% | 84 | 192 | Only 43.8% of tested scenario/deadline combinations clear the target |
| 95.0% | 76 | 192 | 39.6% clear the target |
| 99.0% | 42 | 192 | 21.9% clear the target |
| 99.9% | 0 | 192 | No tested case is statistically qualified to this target |

These are model-feasibility counts, not evidence that the radio/network is inherently incapable of 99.9% delivery. A finite synthetic run with these strategies and this model cannot qualify that target. More trials alone may narrow intervals, but they cannot make a low point estimate satisfy an ambitious target; the model or protocol may need to improve.

## Focus: IID 10% data loss, SF7, 90-second deadline

| ACK-loss setting | Target | Selected policy | Delivery estimate | 95% lower bound | Mean PHY airtime |
|---|---:|---|---:|---:|---:|
| 0% | 90% | Hybrid FEC + retry | 98.25% | 97.58% | 3.983 s |
| 0% | 95% | Hybrid FEC + retry | 98.25% | 97.58% | 3.983 s |
| 0% | 99% | Infeasible; best observed is hybrid | 98.25% | 97.58% | — |
| 0% | 99.9% | Infeasible; best observed is hybrid | 98.25% | 97.58% | — |
| 10% | 90% | Hybrid FEC + retry | 98.20% | 97.52% | 4.330 s |
| 10% | 95% | Hybrid FEC + retry | 98.20% | 97.52% | 4.330 s |
| 25% | 90% | Hybrid FEC + retry | 98.20% | 97.52% | 4.765 s |
| 25% | 95% | Hybrid FEC + retry | 98.20% | 97.52% | 4.765 s |
| 25% | 99% | Infeasible; best observed is hybrid | 98.20% | 97.52% | — |
| 25% | 99.9% | Infeasible; best observed is hybrid | 98.20% | 97.52% | — |

For this case, choosing on point estimate alone would overstate assurance. The lower confidence bound remains below 99%, so the policy correctly refuses to label the 99% target achieved despite the roughly 98.2% point estimate.

## Physical deadline floor

For a 2,048-byte payload split by the present fragmenter, the frozen Semtech SX127x model (125 kHz, CR 4/5, 8-symbol preamble, explicit header and CRC) estimates the ten data frames require 3.351 s at SF7, 10.660 s at SF9, and 77.251 s at SF12. The data plus XOR-parity baseline is 3.699 s, 11.767 s, and 85.287 s respectively.

A 30-second deadline is therefore infeasible at SF12 for this payload/profile before ACK airtime, inter-frame gaps, retries or other overhead. No recovery policy can beat the serialization floor without changing profile, payload representation, or delivery contract. These are analytical PHY sums, not completion-time or hardware measurements. The equation follows the [Semtech SX1276/77/78/79 datasheet, section 4.1.1.7](https://cdn.sparkfun.com/assets/7/7/3/2/2/SX1276_Datasheet.pdf).

## Model limits and next gates

The event model assumes a hypothetical 24-byte ACK, 100-ms inter-frame gap, 100-ms ACK turnaround, 1-second lost-ACK timeout and one bounded recovery round. A lost ACK triggers one blind resend of all data. The channel trace is exogenous and packet error is approximated by airtime-weighted good/bad-state loss probability. This is **not** a calibrated LoRa SNR/PER model.

Still missing: qualified authenticated ACK wire format and replay semantics; actual radio adapter/region profile; duty-cycle scheduling; carrier-sense/collision/hidden-terminal behavior; queueing and multiple concurrent senders; multi-hop; TX/RX energy; board/radio measurements and matched field trials.

## Interpretation

1. Require lower-confidence-bound delivery by deadline, not only the point estimate or eventual completion.
2. Optimize measured airtime/energy only among policies that pass the reliability gate.
3. Report `infeasible` when no candidate passes. Do not silently reduce the target or treat a skipped workflow as a pass.
4. Compare against standards such as [RFC 8931 selective fragment recovery](https://www.rfc-editor.org/rfc/rfc8931.html) for design principles, but do not imply wire interoperability with this custom raw-LoRa protocol.

## Qualification

The derived policy table was generated from the saved 768-row event-model CSV. Its pure post-processor was run twice with byte-identical output and report hashes. This report does not certify the Python model as calibrated, and does not claim a Rust pass: at PR head `90530ba9af4e1ebbb47e4673c4ff9fbf974f511c`, CI, PR Governance, Showroom Integrity and Symthaea Benchmark Suite all concluded `skipped`. The source code remains a draft until exact-head Rust tests, formatting, Clippy and independent review produce valid receipts.