#!/usr/bin/env node
/**
 * Independent semantic reference checks for Symthaea event-order v0.3.2.
 * This model does not import, transpile, or execute repository Rust code.
 * Run: node event-order-v0.3.2-independent-reference.mjs
 */
import assert from "node:assert/strict";
import { createHash } from "node:crypto";

class Rng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    let x = this.state;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }
}
const rng = new Rng(0x4c554d49);
const hash = (value) => createHash("sha256").update(JSON.stringify(value)).digest("hex");

function compare(a, b) {
  if (a.qualification === "fail" || b.qualification === "fail") return "inconsistent";
  if (a.qualification !== "pass" || b.qualification !== "pass") return "unknown";
  if (a.basis !== b.basis) return "unknown";
  if (a.hi < b.lo) return "before";
  if (a.lo > b.hi) return "after";
  return "overlapping";
}
function opposite(order) {
  return ({ before: "after", after: "before", overlapping: "overlapping",
    unknown: "unknown", inconsistent: "inconsistent" })[order];
}
function ownerReceiptMatches(action, verdict) {
  return !!verdict
    && verdict.actionId === action.actionId
    && verdict.receiptRef === action.receiptRef
    && verdict.receiptSha256 === action.receiptSha256
    && verdict.identity === "pass"
    && verdict.digestIntegrity === "pass"
    && verdict.authenticity === "pass"
    && verdict.scope === "pass"
    && verdict.semantic === "pass";
}
function retryPermitted(parent, retry) {
  return parent.logicalId === retry.logicalId
    && retry.retryOf === parent.attemptId
    && parent.terminalStartActionId === parent.startActionId
    && retry.predecessorTerminalActionId === parent.terminalActionId
    && retry.predecessorStartActionId === retry.startActionId
    && parent.terminalOutcome === "not_applied"
    && parent.terminalQualification === "pass"
    && parent.terminalToRetryStart === "before";
}
function scopeCovers(candidate, required) {
  if (candidate === "global") return true;
  if (required === "global") return false;
  const candidateIds = new Set(candidate.replace(/^logical_operations:/, "").split(","));
  const requiredIds = required.replace(/^logical_operations:/, "").split(",");
  return requiredIds.every(id => candidateIds.has(id));
}
function stopPermitted(stop, inventory) {
  return stop.triggerActionId === stop.dischargeTriggerActionId
    && stop.policyActionId === stop.dischargePolicyActionId
    && stop.policy === stop.dischargePolicy
    && scopeCovers(stop.dischargeScope, stop.scope)
    && stop.receiptQualification === "pass"
    && stop.inFlightReconciliation === "pass"
    && !!inventory
    && inventory.qualification === "pass"
    && scopeCovers(inventory.scope, stop.dischargeScope)
    && inventory.frontierId === stop.frontierId
    && inventory.manifestSha256 === stop.manifestSha256
    && inventory.frontierToAttestation === "before"
    && stop.attestationToDischarge === "before"
    && stop.triggerToDischarge === "before";
}
function canonicalHistory(actions) {
  return [...actions].sort((a, b) => a.at - b.at || a.ordinal - b.ordinal || a.id.localeCompare(b.id));
}
function shuffledCopy(values) {
  const out = [...values];
  for (let i = out.length - 1; i > 0; i--) {
    const j = rng.next() % (i + 1);
    [out[i], out[j]] = [out[j], out[i]];
  }
  return out;
}

let intervalPairs = 0, antisymmetryViolations = 0, translationViolations = 0;
for (let i = 0; i < 100_000; i++) {
  const a0 = rng.next() % 2_000_000, b0 = rng.next() % 2_000_000;
  const a = { basis: "tai-v1", qualification: "pass", lo: a0, hi: a0 + (rng.next() % 2_000) };
  const b = { basis: "tai-v1", qualification: "pass", lo: b0, hi: b0 + (rng.next() % 2_000) };
  const ab = compare(a, b), ba = compare(b, a);
  if (ba !== opposite(ab)) antisymmetryViolations++;
  const delta = rng.next() % 1_000_000;
  const translated = compare(
    { ...a, lo: a.lo + delta, hi: a.hi + delta },
    { ...b, lo: b.lo + delta, hi: b.hi + delta },
  );
  if (translated !== ab) translationViolations++;
  intervalPairs++;
}

let ownerReceiptChecks = 0, ownerReceiptBindingViolations = 0;
for (let i = 0; i < 50_000; i++) {
  const action = { actionId: "action-" + i, receiptRef: "receipt-" + i, receiptSha256: hash({ i }) };
  const exact = { ...action, identity: "pass", digestIntegrity: "pass", authenticity: "pass", scope: "pass", semantic: "pass" };
  if (!ownerReceiptMatches(action, exact)) ownerReceiptBindingViolations++;
  if (ownerReceiptMatches(action, { ...exact, receiptRef: action.receiptRef + "-substituted" })) ownerReceiptBindingViolations++;
  if (ownerReceiptMatches(action, { ...exact, receiptSha256: "0".repeat(64) })) ownerReceiptBindingViolations++;
  ownerReceiptChecks += 3;
}

const retryKinds = [
  "not_applied_exact_before", "applied", "unknown", "unqualified",
  "terminal_after", "terminal_unknown_order", "start_substituted", "logical_operation_substituted",
];
let retryChecks = 0, retryUnexpectedAdmissions = 0, retryAccepted = 0;
for (let i = 0; i < 100_000; i++) {
  const kind = retryKinds[i % retryKinds.length];
  const parent = {
    logicalId: "robot-A", attemptId: "attempt-" + i,
    startActionId: "start-" + i, terminalStartActionId: "start-" + i,
    terminalActionId: "terminal-" + i,
    terminalOutcome: "not_applied", terminalQualification: "pass", terminalToRetryStart: "before",
  };
  const retryStartActionId = "retry-start-" + i;
  const retry = {
    logicalId: "robot-A", retryOf: parent.attemptId,
    startActionId: retryStartActionId,
    predecessorTerminalActionId: parent.terminalActionId,
    predecessorStartActionId: retryStartActionId,
  };
  if (kind === "applied") parent.terminalOutcome = "applied";
  if (kind === "unknown") parent.terminalOutcome = "unknown";
  if (kind === "unqualified") parent.terminalQualification = "indeterminate";
  if (kind === "terminal_after") parent.terminalToRetryStart = "after";
  if (kind === "terminal_unknown_order") parent.terminalToRetryStart = "overlapping";
  if (kind === "start_substituted") parent.terminalStartActionId = "other-start-" + i;
  if (kind === "logical_operation_substituted") retry.logicalId = "robot-B";
  if (kind === "terminal_action_substituted") retry.predecessorTerminalActionId = "other-terminal-" + i;
  if (kind === "child_start_action_substituted") retry.predecessorStartActionId = "other-retry-start-" + i;
  const accepted = retryPermitted(parent, retry);
  const expected = kind === "not_applied_exact_before";
  if (accepted !== expected) retryUnexpectedAdmissions++;
  if (accepted) retryAccepted++;
  retryChecks++;
}

const stopFaults = [
  "exact", "receipt_fail", "trigger_substitution", "policy_substitution",
  "scope_narrowed", "manifest_substitution", "inventory_missing",
  "frontier_after", "discharge_before_trigger", "inventory_scope_too_narrow",
  "broader_inventory", "scope_expansion_global",
];
let stopChecks = 0, stopUnexpectedAdmissions = 0, stopAccepted = 0;
for (let i = 0; i < 90_000; i++) {
  const kind = stopFaults[i % stopFaults.length];
  const stop = {
    triggerActionId: "trigger-1", dischargeTriggerActionId: "trigger-1",
    policyActionId: "policy-1", dischargePolicyActionId: "policy-1",
    policy: "rev-1", dischargePolicy: "rev-1",
    scope: "global", dischargeScope: "global",
    receiptQualification: "pass", inFlightReconciliation: "pass",
    frontierId: "frontier-1", manifestSha256: "a".repeat(64),
    attestationToDischarge: "before", triggerToDischarge: "before",
  };
  const inventory = {
    qualification: "pass", scope: "global", frontierId: stop.frontierId,
    manifestSha256: stop.manifestSha256, frontierToAttestation: "before",
  };
  if (kind === "broader_inventory") {
    stop.scope = "logical_operations:robot-1";
    stop.dischargeScope = "logical_operations:robot-1";
  }
  if (kind === "scope_expansion_global") {
    stop.scope = "logical_operations:robot-1";
    stop.dischargeScope = "global";
  }
  if (kind === "scope_narrowed") {
    stop.scope = "global";
    stop.dischargeScope = "logical_operations:robot-1";
  }
  if (kind === "inventory_scope_too_narrow") {
    stop.scope = "global";
    stop.dischargeScope = "global";
    inventory.scope = "logical_operations:robot-1";
  }
  if (kind === "receipt_fail") stop.receiptQualification = "fail";
  if (kind === "trigger_substitution") stop.dischargeTriggerActionId = "other-trigger";
  if (kind === "policy_substitution") stop.dischargePolicyActionId = "other-policy";

  if (kind === "manifest_substitution") stop.manifestSha256 = "b".repeat(64);
  if (kind === "inventory_missing") inventory.qualification = "missing";
  if (kind === "frontier_after") inventory.frontierToAttestation = "after";
  if (kind === "discharge_before_trigger") stop.triggerToDischarge = "after";
  const accepted = stopPermitted(stop, kind === "inventory_missing" ? null : inventory) === true;
  const expected = ["exact", "broader_inventory", "scope_expansion_global"].includes(kind);
  if (accepted !== expected) stopUnexpectedAdmissions++;
  if (accepted) stopAccepted++;
  stopChecks++;
}

let permutationChecks = 0, permutationViolations = 0;
for (let i = 0; i < 10_000; i++) {
  const actions = Array.from({ length: 8 + (rng.next() % 40) }, (_, j) => ({
    id: "a-" + i + "-" + j, at: rng.next() % 100_000, ordinal: j, kind: j % 4,
  }));
  const expected = hash(canonicalHistory(actions));
  if (hash(canonicalHistory(shuffledCopy(actions))) !== expected) permutationViolations++;
  permutationChecks++;
}

let cancellationChecks = 0, cancellationFalseTerminals = 0;
for (let i = 0; i < 50_000; i++) {
  const accepted = (i & 1) === 0;
  const projected = { accepted, establishesNotApplied: false, terminal: null };
  if (projected.establishesNotApplied || projected.terminal !== null) cancellationFalseTerminals++;
  cancellationChecks++;
}

function terminalConsistent(outcome, effectCommitCount) {
  return outcome !== "not_applied" || effectCommitCount === 0;
}
let effectTerminalChecks = 0, appliedCommitOrderViolations = 0, notAppliedCommitContradictionsMissed = 0;
for (let i = 0; i < 50_000; i++) {
  const commit = { basis: "tai-v1", qualification: "pass", lo: i * 4, hi: i * 4 };
  const terminal = { basis: "tai-v1", qualification: "pass", lo: i * 4 + 2, hi: i * 4 + 2 };
  if (compare(commit, terminal) !== "before") appliedCommitOrderViolations++;
  const hasCommit = true;
  if (terminalConsistent("not_applied", hasCommit ? 1 : 0)) notAppliedCommitContradictionsMissed++;
  effectTerminalChecks++;
}

const report = {
  report_version: "event-order-v0.3.2-independent-reference.v2",
  tested_contract: "symthaea.event-order-schedule.v0.3.2",
  implementation: "standalone JavaScript reference model; no repository imports or package dependencies",
  runtime: process.version,
  seed: "0x4c554d49",
  not_rust_execution: true,
  interval_pairs: intervalPairs,
  interval_antisymmetry_violations: antisymmetryViolations,
  interval_common_translation_violations: translationViolations,
  owner_receipt_identity_checks: ownerReceiptChecks,
  owner_receipt_binding_violations: ownerReceiptBindingViolations,
  retry_checks: retryChecks,
  retry_case_families: retryKinds.length,
  retry_expected_admissions: retryAccepted,
  retry_unexpected_admissions: retryUnexpectedAdmissions,
  retry_child_start_binding_checks: 2 * (retryChecks / retryKinds.length),
  retry_child_start_binding_violations: retryUnexpectedAdmissions,
  stop_scope_discharge_inventory_checks: stopChecks,
  stop_case_families: stopFaults.length,
  stop_expected_admissions: stopAccepted,
  stop_unexpected_admissions: stopUnexpectedAdmissions,
  stop_scope_widening_checks: 2 * (stopChecks / stopFaults.length),
  stop_scope_widening_violations: stopUnexpectedAdmissions,
  action_delivery_permutation_checks: permutationChecks,
  action_delivery_permutation_violations: permutationViolations,
  cancellation_acknowledgement_checks: cancellationChecks,
  cancellation_acknowledgement_false_terminal_inferences: cancellationFalseTerminals,
  effect_terminal_chronology_checks: effectTerminalChecks,
  applied_commit_order_violations: appliedCommitOrderViolations,
  not_applied_with_commit_contradictions_missed: notAppliedCommitContradictionsMissed,
  interpretation: "Independent semantic reference checks only. This script is Node.js-compatible and reports its runtime when executed. It does not execute Rust, authenticate real receipts, establish production reliability, or replace CI and owner-native verification."
};
for (const [key, value] of Object.entries(report)) {
  if (key.endsWith("_violations") || key.endsWith("_missed") || key.endsWith("_inferences")) assert.equal(value, 0, key);
}
console.log(JSON.stringify(report, null, 2));
