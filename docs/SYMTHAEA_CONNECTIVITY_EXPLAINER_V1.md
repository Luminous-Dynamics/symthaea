# Symthaea Connectivity Explainer v1

**Status:** deterministic explanation layer and synthetic fixture. It is not a generative chatbot and has not been evaluated on end-user comprehension yet.

## Purpose

Connectivity measurement is multi-disciplinary: Wi-Fi performance, Internet/WAN availability, queue management, application task completion, energy/cost accounting, privacy and statistical uncertainty all use different vocabulary. This crate takes a structured recovery-simulation report and produces a plain-language summary while preserving its source evidence class and important limits.

The central design rule is: **simplify the language, never simplify away the evidence boundary.** A synthetic result must not turn into a measured claim; missing counters must not become zero; a fairness index must not hide missed essential-task deadlines.

## Levels of explanation

1. **Household:** explain what the scenario represents and whether the result is measured or simulated.
2. **Operator:** describe outcomes by scenario, task class and baseline/candidate counts.
3. **Researcher:** preserve denominators, source digest, priority-specific missed deadlines, and limitations on what can be inferred.
4. **Audit:** keep a deterministic JSON record whose claims can be traced to fields in the input report.

No prose is invented from unconstrained generation in v1. The output uses reviewed templates over structured, validated report fields. This makes the wording predictable, testable, translatable and suitable for a future HDC similarity/retrieval layer. Hyperdimensional computing may help retrieve a relevant explanation pattern from a reviewed library, but HDC similarity is not proof that a diagnosis is true and must not alter the underlying claims.

When paired task-level receipts are supplied, the explainer goes beyond aggregate counts: it compares the same task in both arms, checks that demand/deadline metadata matches, and explains the difference using requested bytes, delivered bytes, deadline/expiry window, terminal window and failure reason. It shows up to 50 changed tasks in the explanation and records the total/truncation flag; malformed or mismatched task pairs fail closed. Without task rows, it explicitly says task-level causality is unavailable rather than inventing a reason.

## Evidence-aware language

The explainer recognizes these declared evidence classes:

- `synthetic_simulation`: result from a mathematical/software model; never a radio or field measurement.
- `bench_observed_snapshot`: observed counters; this alone does not prove successful application tasks.
- `lab_measurement`, `field_pilot`, `reliability_qualification`: these are labels supplied by the input report, not independently authenticated by this crate.
- Unknown or missing value: performance claims remain unverified.

The function refuses absent required fields, invalid completion counts, and mismatched baseline/candidate denominators. If priority-specific rows are missing, it calls that out explicitly instead of assuming no deadline misses. Every mode is narrated separately because the current multi-window simulator runs each mode as an independent episode.

## Example

Given the checked-in synthetic fixture, the explainer says the candidate completes more tasks in the workload shown and separately flags FIFO's essential and interactive deadline misses. It also says this is not a physical Wi-Fi result, that a byte grant is not measured 802.11 airtime fairness, and that real application success, TCP/QUIC behavior, radio retries, latency, calibrated energy, prices and provider permissions remain unproven.

Run inside the Symthaea workspace. The default command prints a plain-language report that has sections for evidence class, outcomes by scenario, per-priority completion/deadline results, changed task outcomes, limitations, next step and provenance:

```sh
cargo run -p symthaea-connectivity-explainer -- \
  crates/domains/symthaea-connectivity-explainer/examples/connectivity-report-synthetic-v1.json
```

The default command prints a plain-language report. For machine-readable output on stdout:

```sh
cargo run -p symthaea-connectivity-explainer -- \
  crates/domains/symthaea-connectivity-explainer/examples/connectivity-report-synthetic-v1.json \
  --json
```

Or write JSON for another tool to consume:

```sh
cargo run -p symthaea-connectivity-explainer -- \
  crates/domains/symthaea-connectivity-explainer/examples/connectivity-report-synthetic-v1.json \
  --json /tmp/connectivity-explanation.json
```

## Industry comparison and research standards

The project should complement recognized device/network qualification rather than inventing a replacement for it. The Broadband Forum technical library lists **TR-398 Issue 4: Wi-Fi Residential & SOHO Performance Testing** as in force in September 2026; that work is the AP RF performance/coverage/capacity/stability reference. IETF RFC 6349 provides a structured method for sustained TCP throughput testing that includes path MTU, baseline RTT and bottleneck bandwidth. Neither one alone supplies this project's user-task fairness, outage/recovery, privacy, economic cost and plain-language evidence interface.

- Broadband Forum technical library: https://broadband-forum.org/public/technical-library/
- Broadband Forum BBF.398 program: https://broadband-forum.org/public/certifications/bbf-398-carrier-grade-wi-fi/
- IETF RFC 6349: https://www.rfc-editor.org/rfc/rfc6349.html

Do not state TR-398 compliance, regulatory approval or superiority to SOTA from these explanation tests.

## Next benchmarks

- **Factual fidelity:** every numeric claim in output matches the corresponding input field exactly; zero unsupported numeric claims.
- **Evidence-class calibration:** 100% of synthetic reports explicitly identify themselves as synthetic; missing/unknown evidence class never maps to “measured.”
- **Denominator integrity:** 100% of mismatched paired denominators fail closed; never discard failed or inconclusive trials silently.
- **Critical-failure visibility:** every essential/interactive deadline miss present in the source appears in the explanation.
- **Traceability:** every explanation includes the source scenario digest when available and preserves the evidence class.
- **Comprehension:** blind participant study comparing user understanding against raw technical reports; target at least 90% correct answers on “what happened?”, “what is unproven?” and “what action is next?”, while reducing median time-to-correct-answer. This remains a future human-subjects study and not a current result.
- **Non-regression:** technical reviewers can reconstruct each sentence's input fields; unexplained claims should be a hard failure.

## Explicit limits

This initial crate does not yet call a live Symthaea inference loop, does not use HDC to generate text, does not inspect raw radio counters, and does not diagnose network faults. It is a Symthaea workspace component designed to establish the trusted explanation contract first. Live data adapters, verified evidence graph integration, localization and HDC retrieval should be additive and separately benchmarked.
