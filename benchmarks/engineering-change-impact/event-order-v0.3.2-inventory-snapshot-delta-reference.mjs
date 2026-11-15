#!/usr/bin/env node
/**
 * Independent snapshot-plus-delta reference checks for Symthaea event-order v0.3.2.
 * This is deliberately separate from repository Rust code.
 * Run: node event-order-v0.3.2-inventory-snapshot-delta-reference.mjs
 */
function run() {
  let state = 0x4c554d49 >>> 0;
  const next = () => { state ^= state << 13; state ^= state >>> 17; state ^= state << 5; state >>>= 0; return state; };
  const compare = (a, b) => {
    if (a.qualification !== "pass" || b.qualification !== "pass" || a.basis !== b.basis) return "unknown";
    if (a.hi < b.lo) return "before";
    if (a.lo > b.hi) return "after";
    return "overlapping";
  };
  function snapshotAtCut(attempts, frontier) {
    const snapshot = [];
    let indeterminate = 0, critical = 0, futureTerminalExcluded = 0, futureDischargeExcluded = 0, priorDischargeIncluded = 0;
    for (const attempt of attempts) {
      const startOrder = compare(attempt.start, frontier);
      if (startOrder === "before") {
        const copy = { ...attempt, stopRelations: [...attempt.stopRelations] };
        if (attempt.terminal) {
          const terminalOrder = compare(attempt.terminal, frontier);
          if (terminalOrder === "after") { copy.terminal = null; futureTerminalExcluded++; }
          else if (terminalOrder !== "before") { copy.terminal = null; indeterminate++; }
        }
        const relations = [];
        for (const relation of attempt.stopRelations) {
          const order = compare(relation.discharge, frontier);
          if (order === "before") { relations.push(relation); priorDischargeIncluded++; }
          else if (order === "after") futureDischargeExcluded++;
          else if (order === "inconsistent") critical++;
          else indeterminate++;
        }
        copy.stopRelations = relations;
        snapshot.push(copy);
      } else if (startOrder !== "after") indeterminate++;
    }
    return { snapshot, indeterminate, critical, futureTerminalExcluded, futureDischargeExcluded, priorDischargeIncluded };
  }
  function classifyDelta(attempt, discharge, frontier, targetLogical) {
    const startOrder = compare(attempt.start, frontier);
    if (startOrder !== "after") return "not_delta";
    const dischargeOrder = compare(discharge, attempt.start);
    if (dischargeOrder === "before") return attempt.logicalId === targetLogical && !attempt.terminal ? "indeterminate" : "safe";
    if (dischargeOrder === "after") return "critical";
    return "indeterminate";
  }
  const families = {
    post_discharge_pending_sibling: { n: 0, violations: 0 },
    omitted_pre_discharge_attempt: { n: 0, violations: 0 },
    pending_requested_operation: { n: 0, violations: 0 },
    ambiguous_inventory_cut: { n: 0, violations: 0 },
    post_cut_terminal_excluded_from_snapshot: { n: 0, violations: 0 },
    prior_discharge_relation_retained: { n: 0, violations: 0 },
  };
  let total = 0, futureTerminals = 0, futureRelations = 0, priorRelations = 0;
  for (let i = 0; i < 100000; i++) {
    const family = i % 5, j = next() % 10000;
    const frontier = { basis: "tai", qualification: "pass", lo: 1000, hi: 1000 };
    if (family === 0 || family === 1 || family === 2) {
      const startNs = (family === 1 ? 1050 : 1400) + j / 100;
      const attempt = {
        logicalId: family === 2 ? "robot-1" : "robot-2",
        start: { basis: "tai", qualification: "pass", lo: startNs, hi: startNs },
        terminal: null,
        stopRelations: [{ discharge: { basis: "tai", qualification: "pass", lo: 1200, hi: 1200 } }],
      };
      const cut = snapshotAtCut([attempt], frontier);
      const result = classifyDelta(attempt, attempt.stopRelations[0].discharge, frontier, "robot-1");
      const name = family === 0 ? "post_discharge_pending_sibling" : family === 1 ? "omitted_pre_discharge_attempt" : "pending_requested_operation";
      const expected = family === 0 ? "safe" : family === 1 ? "critical" : "indeterminate";
      families[name].n++;
      if (cut.snapshot.length !== 0 || result !== expected) families[name].violations++;
      futureRelations += cut.futureDischargeExcluded;
    } else if (family === 3) {
      const startNs = 995 + j % 10;
      const attempt = { logicalId: "robot-2", start: { basis: "tai", qualification: "pass", lo: startNs, hi: startNs + 10 }, terminal: null, stopRelations: [] };
      const cut = snapshotAtCut([attempt], frontier);
      families.ambiguous_inventory_cut.n++;
      if (cut.snapshot.length !== 0 || cut.indeterminate !== 1) families.ambiguous_inventory_cut.violations++;
    } else {
      const attempt = {
        logicalId: "robot-1",
        start: { basis: "tai", qualification: "pass", lo: 800, hi: 800 },
        terminal: { basis: "tai", qualification: "pass", lo: 1100 + j / 100, hi: 1100 + j / 100 },
        stopRelations: [
          { discharge: { basis: "tai", qualification: "pass", lo: 1200, hi: 1200 } },
          { discharge: { basis: "tai", qualification: "pass", lo: 900, hi: 900 } },
        ],
      };
      const cut = snapshotAtCut([attempt], frontier);
      families.post_cut_terminal_excluded_from_snapshot.n++;
      families.prior_discharge_relation_retained.n++;
      if (cut.snapshot.length !== 1 || cut.snapshot[0].terminal !== null || cut.snapshot[0].stopRelations.length !== 1 || cut.futureTerminalExcluded !== 1 || cut.futureDischargeExcluded !== 1 || cut.priorDischargeIncluded !== 1) {
        families.post_cut_terminal_excluded_from_snapshot.violations++;
        families.prior_discharge_relation_retained.violations++;
      }
      futureTerminals += cut.futureTerminalExcluded;
      futureRelations += cut.futureDischargeExcluded;
      priorRelations += cut.priorDischargeIncluded;
    }
    total++;
  }
  const violations = Object.values(families).reduce((sum, family) => sum + family.violations, 0);
  return {
    report_version: "event-order-v0.3.2-inventory-snapshot-delta-reference.v1",
    tested_contract: "symthaea.event-order-schedule.v0.3.2",
    implementation: "independently authored deterministic JavaScript reference model; no repository imports",
    runtime: typeof process !== "undefined" ? process.version : "JavaScript runtime",
    seed: "0x4c554d49",
    total_cases: total,
    case_families: families,
    snapshot_cut_semantics: {
      future_terminal_reconciliations_excluded: futureTerminals,
      future_stop_relations_excluded: futureRelations,
      prior_stop_relations_retained: priorRelations,
    },
    violations,
    not_rust_execution: true,
    interpretation: "Reference-model semantic check only. It does not execute Rust, verify real inventory signatures, or establish production reliability.",
  };
}
const report = run();
if (report.violations !== 0) throw new Error("snapshot/delta reference mismatch");
console.log(JSON.stringify(report, null, 2));
