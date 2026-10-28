# LoRa PHY airtime baseline for Mycelial Mesh

**Status:** Analytical reference model, not a radio bench result and not a certification of actual device latency, reliability, energy, or regulatory compliance.

## Frozen assumptions

The benchmark extension uses the Semtech SX1276/77/78/79 datasheet Rev. 7, section 4.1.1.7 time-on-air equation:

- PHY payload length includes the mesh's 8-byte fragment header; valid payload sizes are 1..=255 bytes.
- Bandwidth: 125 kHz.
- Coding rate: 4/5.
- Preamble: 8 symbols.
- Explicit header enabled; PHY CRC enabled.
- Spreading factors swept from SF7 through SF12.
- Low-data-rate optimization is enabled when symbol duration exceeds 16 ms.
- The current fragmenter is exercised at its actual 214-byte fragment payload ceiling, so a full frame is 222 bytes including the custom 8-byte header.

Reference: [Semtech SX1276/77/78/79 datasheet, section 4.1.1.7](https://cdn.sparkfun.com/assets/7/7/3/2/2/SX1276_Datasheet.pdf).

## 2,048-byte payload baseline

The current fragmenter produces 10 data frames and one XOR-parity frame. For this payload size that means nine 222-byte PHY payloads plus one 130-byte final data payload; XOR adds another 222-byte PHY payload.

| Spreading factor | Data only (10 frames) | Data + XOR (11 frames) | Increment for XOR |
|---:|---:|---:|---:|
| SF7 | 3.351040 s | 3.699456 s | 0.348416 s |
| SF8 | 5.913600 s | 6.528512 s | 0.614912 s |
| SF9 | 10.659840 s | 11.766784 s | 1.106944 s |
| SF10 | 19.353600 s | 21.362688 s | 2.009088 s |
| SF11 | 42.557440 s | 46.985216 s | 4.427776 s |
| SF12 | 77.250560 s | 85.286912 s | 8.036352 s |

These are sums of physical-layer transmission times for data frames only. The table is not message-completion latency. It excludes feedback/ACK frames and their airtime, turnaround windows, inter-frame delays, carrier sensing, collisions, duty-cycle or dwell-time scheduling, oscillator/radio startup, receive windows, retries and multihop forwarding. SF-specific propagation and receive sensitivity also affect observed delivery and must not be inferred from airtime alone.

The SF7 values are a useful arithmetic check on the model: 222 bytes take 348,416 microseconds and 130 bytes take 215,296 microseconds under this exact PHY profile. The automated Rust test asserts both values and verifies that data-only and XOR airtime rise monotonically from SF7 to SF12.

## Why this changes the optimization objective

Frame count and wire bytes are insufficient cost metrics for low-rate radio. The correct optimization target is a measured Pareto surface over:

1. Delivery probability and exact reconstruction.
2. Total PHY airtime, including ACKs and retries.
3. End-to-end completion latency (median, p95 and p99).
4. TX/RX energy per successfully delivered byte or message.
5. Memory residency and reassembly timeout cost.
6. Congestion/fairness under concurrent senders and multiple hops.

The selective retry and hybrid results in the current synthetic channel model assume perfect aggregated feedback and no ACK loss. Their airtime figures must therefore be labelled optimistic lower bounds until actual feedback frames, radio scheduling, and device measurements are implemented.

A relevant standards comparator is [IETF RFC 8931, Selective Fragment Recovery](https://www.rfc-editor.org/rfc/rfc8931.html), which includes selective recovery and congestion-control concerns for 6LoWPAN meshes. It is not a wire-compatible replacement for this raw-LoRa protocol; it is a useful systems-design comparator.

## South African deployment gate

The code's historical 868 MHz comment is not, by itself, evidence that every channel, power level, duty cycle or waveform configuration is legal for deployment in South Africa. The released build should select an explicit, reviewed region/radio profile and validate the current ICASA frequency allocation and exemption conditions before transmitting. Start at the [ICASA final radio-frequency spectrum plans](https://www.icasa.org.za/legislation-and-regulations/radio-frequency-spectrum-plans/final-radio-frequency-spectrum-plans) and applicable radio-frequency regulations; do not silently apply EU868 regulatory assumptions to South African devices.

## Required evidence before a performance claim

- Exact-head test, formatting and feature-on/feature-off build receipts.
- Reproducible bench capture with radio/board model, firmware, antenna, power, SF/BW/CR, and region documented.
- Measured ACK, retries, airtime, completion latency, and TX/RX energy.
- IID and burst-loss sweeps, plus multi-sender collision/congestion experiments.
- Confidence intervals over independent seeds and hardware repetitions.
- Matched comparison against the same payload, MTU, PHY profile, ACK semantics, and loss trace.
- Explicit statement of which results are synthetic, analytical, simulation-derived, or measured.

No claim of state-of-the-art performance is supported by this analytical table alone.
