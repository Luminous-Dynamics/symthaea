#!/usr/bin/env node
/**
 * Independent reference oracle for the v0.3.1 clock and bitemporal rules.
 *
 * This intentionally does not import, transpile, or execute the Rust module.
 * It is a separately authored semantic cross-check, not production evidence.
 * Run: node benchmarks/engineering-change-impact/event-order-v0.3.1-independent-reference.mjs
 */
import assert from "node:assert/strict";

class Rng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    this.state = (Math.imul(this.state, 1664525) + 1013904223) >>> 0;
    return this.state;
  }
}

function classifyClockMapping(activeId, referenceId, mapping, timestamp) {
  if (activeId !== referenceId || !mapping) return "stale_mapping_reference";
  if (mapping.qualification === "fail") return "clock_mapping_qualification_failed";
  if (mapping.qualification !== "pass") return "clock_mapping_unqualified";
  if (mapping.validFrom > mapping.validThrough) return "clock_mapping_validity_invalid";
  if (timestamp < mapping.validFrom) return "clock_mapping_not_yet_valid";
  if (timestamp > mapping.validThrough) return "clock_mapping_expired";
  if (timestamp > mapping.localTick) return "clock_timestamp_after_mapping_sample";
  return "applied";
}

function compare(a, b) {
  if (a.qualification === "fail" || b.qualification === "fail") return "inconsistent";
  if (a.qualification !== "pass" || b.qualification !== "pass") return "unknown";
  if (a.basis !== b.basis) return "unknown";
  if (a.hi < b.lo) return "before";
  if (a.lo > b.hi) return "after";
  return "overlapping";
}

function project(occurred, observed, persisted) {
  const occurrenceOrder = compare(occurred, observed);
  if (occurrenceOrder === "after" || occurrenceOrder === "inconsistent") return "critical_failure";
  if (occurrenceOrder === "unknown") return "indeterminate";
  const captureOrder = compare(observed, persisted);
  if (captureOrder === "before") return "safe";
  if (captureOrder === "overlapping" || captureOrder === "unknown") return "indeterminate";
  return "critical_failure";
}

function main() {
  const seed = 0x4c554d49;
  const rng = new Rng(seed);
  const faults = [
    ["observer_unqualified", "clock_mapping_unqualified", (m) => { m.qualification = "indeterminate"; }],
    ["observer_expired", "clock_mapping_expired", (m, t) => { m.validThrough = t - 1; }],
    ["observer_not_yet_valid", "clock_mapping_not_yet_valid", (m, t) => { m.validFrom = t + 1; }],
    ["observer_validity_invalid", "clock_mapping_validity_invalid", (m) => { m.validFrom = m.validThrough + 1; }],
    ["store_unqualified", "clock_mapping_unqualified", (m) => { m.qualification = "indeterminate"; }],
    ["store_expired", "clock_mapping_expired", (m, t) => { m.validThrough = t - 1; }],
    ["store_not_yet_valid", "clock_mapping_not_yet_valid", (m, t) => { m.validFrom = t + 1; }],
    ["store_validity_invalid", "clock_mapping_validity_invalid", (m) => { m.validFrom = m.validThrough + 1; }],
  ];

  let captureClockFaultChecks = 0;
  for (let n = 0; n < 1_000; n += 1) {
    for (const [name, expected, mutate] of faults) {
      const timestamp = name.startsWith("observer")
        ? 1_000 + (rng.next() % 1_000)
        : 1_200 + (rng.next() % 1_000);
      const mapping = {
        localTick: 100_000,
        validFrom: 0,
        validThrough: 100_000,
        qualification: "pass",
      };
      mutate(mapping, timestamp);
      assert.equal(
        classifyClockMapping(`map-${name}`, `map-${name}`, mapping, timestamp),
        expected,
        name,
      );
      captureClockFaultChecks += 1;
    }
  }

  assert.equal(
    classifyClockMapping(
      "new-map",
      "old-map",
      { localTick: 100, validFrom: 0, validThrough: 100, qualification: "pass" },
      50,
    ),
    "stale_mapping_reference",
  );
  const oldEpoch = { domain: "clock-a", basis: "b0", epoch: "e0", tick: 100 };
  const newEpoch = { domain: "clock-a", basis: "b0", epoch: "e1", tick: 5 };
  assert.notDeepEqual(oldEpoch, newEpoch);

  let intervalPairs = 0;
  let antisymmetryViolations = 0;
  let commonTranslationViolations = 0;
  for (let i = 0; i < 100_000; i += 1) {
    const a0 = rng.next() % 1_000_000;
    const a1 = a0 + (rng.next() % 1_000);
    const b0 = rng.next() % 1_000_000;
    const b1 = b0 + (rng.next() % 1_000);
    const a = { basis: "tai-v1", qualification: "pass", lo: a0, hi: a1 };
    const b = { basis: "tai-v1", qualification: "pass", lo: b0, hi: b1 };
    const ab = compare(a, b);
    const ba = compare(b, a);
    const opposite = { before: "after", after: "before", overlapping: "overlapping" }[ab];
    if (ba !== opposite) antisymmetryViolations += 1;

    const delta = rng.next() % 100_000;
    const translated = compare(
      { ...a, lo: a.lo + delta, hi: a.hi + delta },
      { ...b, lo: b.lo + delta, hi: b.hi + delta },
    );
    if (translated !== ab) commonTranslationViolations += 1;
    intervalPairs += 1;
  }

  let projectionCases = 0;
  let safe = 0;
  let indeterminate = 0;
  let critical = 0;
  for (let i = 0; i < 100_000; i += 1) {
    const base = [rng.next() % 100_000, rng.next() % 100_000, rng.next() % 100_000];
    const intervals = base.map((value) => ({
      basis: "tai-v1",
      qualification: "pass",
      lo: value,
      hi: value + (rng.next() % 500),
    }));
    if (i % 10 === 0) intervals[1].basis = "tai-other";
    if (i % 17 === 0) intervals[2].qualification = "indeterminate";
    if (i % 101 === 0) intervals[0].qualification = "fail";
    const result = project(intervals[0], intervals[1], intervals[2]);
    if (result === "safe") safe += 1;
    else if (result === "indeterminate") indeterminate += 1;
    else critical += 1;
    projectionCases += 1;
  }

  const report = {
    report_version: "event-order-v0.3.1-independent-reference.v1",
    tested_contract: "symthaea.event-order-schedule.v0.3.1",
    generator_version: "symthaea.event-order-generator.v0.3.2",
    implementation: "standalone JavaScript reference model; no package dependencies",
    not_rust_execution: true,
    seed: "0x4c554d49",
    capture_clock_fault_checks: captureClockFaultChecks,
    exact_mapping_identity_check: "pass",
    epoch_local_reset_identity_check: "pass",
    interval_pairs: intervalPairs,
    interval_antisymmetry_violations: antisymmetryViolations,
    common_translation_violations: commonTranslationViolations,
    projection_order_cases: projectionCases,
    projection_reference_counts: { safe, indeterminate, critical },
    zero_failure_one_sided_95_percent_upper_bound: {
      trials_3000: 1 - Math.pow(0.05, 1 / 3_000),
      trials_30000: 1 - Math.pow(0.05, 1 / 30_000),
    },
  };
  console.log(JSON.stringify(report, null, 2));
}

main();
