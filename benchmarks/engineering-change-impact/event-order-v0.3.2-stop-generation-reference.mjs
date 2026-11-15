#!/usr/bin/env node
/**
 * Independent reference for policy-generation-aware stop discharge projection.
 * This is a small semantic model, not an import or execution of Symthaea Rust.
 * Run: node event-order-v0.3.2-stop-generation-reference.mjs
 */
import assert from "node:assert/strict";

class Rng {
  constructor(seed) { this.state = seed >>> 0; }
  next() {
    let x = this.state;
    x ^= x << 13; x ^= x >>> 17; x ^= x << 5;
    this.state = x >>> 0;
    return this.state;
  }
}

function scopeCovers(candidate, required) {
  if (candidate.kind === "global") return true;
  if (required.kind === "global") return false;
  return required.ids.every(id => candidate.ids.includes(id));
}

function projectHistory(actions, originalScope) {
  const ordered = [...actions].sort((a, b) => a.at - b.at || a.ordinal - b.ordinal);
  let trigger = null;
  let currentPolicy = null;
  let latestAcceptedDischarge = null;
  const historical = [];

  for (const action of ordered) {
    if (action.kind === "trigger") {
      trigger = action;
      currentPolicy = action;
    } else if (action.kind === "revision") {
      if (!scopeCovers(action.scope, originalScope)) {
        historical.push({ actionId: action.actionId, accepted: false, reason: "scope_narrowing" });
      } else {
        currentPolicy = action;
      }
    } else if (action.kind === "discharge") {
      const accepted = !!trigger && !!currentPolicy
        && action.triggerActionId === trigger.actionId
        && action.policyActionId === currentPolicy.actionId
        && action.policyRevision === currentPolicy.revision
        && JSON.stringify(action.scope) === JSON.stringify(currentPolicy.scope)
        && scopeCovers(action.inventoryScope, action.scope)
        && action.frontierBeforeAttestation
        && action.attestationBeforeDischarge
        && action.policyBeforeDischarge;
      historical.push({ actionId: action.actionId, accepted, policyActionId: currentPolicy?.actionId ?? null });
      if (accepted) latestAcceptedDischarge = { actionId: action.actionId, policyActionId: currentPolicy.actionId };
    }
  }

  const finalDischarge = latestAcceptedDischarge
    && currentPolicy
    && latestAcceptedDischarge.policyActionId === currentPolicy.actionId
      ? latestAcceptedDischarge
      : null;
  return {
    currentPolicyActionId: currentPolicy?.actionId ?? null,
    historicalDischarges: historical,
    finalDischarge,
    currentObligationDischarged: finalDischarge !== null,
  };
}

const rng = new Rng(0x4c554d49);
const baseScope = { kind: "logical", ids: ["op-a"] };
const widenedScope = { kind: "logical", ids: ["op-a", "op-b"] };
let cases = 0;
let historicalDischargeFalseRejects = 0;
let validHistoricalDischarges = 0;
let staleDischargeFalseAccepts = 0;
let finalPolicyFalseSafes = 0;
let scopeNarrowingFalseAccepts = 0;
let validWidenedFinalDischarges = 0;

for (let i = 0; i < 100_000; i += 1) {
  const actions = [{
    kind: "trigger", actionId: "trigger", at: 1, ordinal: 1,
    revision: "v1", scope: baseScope,
  }];
  let current = actions[0];
  const revisionCount = rng.next() % 4;
  let ordinal = 2;
  for (let j = 0; j < revisionCount; j += 1) {
    const scope = (rng.next() & 3) === 0 ? baseScope : widenedScope;
    const revision = {
      kind: "revision", actionId: `revision-${j}`, at: 2 + j * 4,
      ordinal: ordinal++, revision: `v${j + 2}`, scope,
    };
    actions.push(revision);
    if (scopeCovers(scope, baseScope)) current = revision;

    if ((rng.next() & 1) === 1) {
      const stale = rng.next() % 5 === 0;
      actions.push({
        kind: "discharge", actionId: `discharge-${i}-${j}`,
        at: revision.at + 1, ordinal: ordinal++,
        triggerActionId: "trigger",
        policyActionId: stale ? "substituted-policy" : current.actionId,
        policyRevision: stale ? "stale" : current.revision,
        scope: current.scope,
        inventoryScope: { kind: "global" },
        frontierBeforeAttestation: true,
        attestationBeforeDischarge: true,
        policyBeforeDischarge: true,
      });
    }
  }

  if ((rng.next() & 1) === 1) {
    actions.push({
      kind: "discharge", actionId: `final-${i}`,
      at: 30, ordinal: ordinal++,
      triggerActionId: "trigger",
      policyActionId: current.actionId,
      policyRevision: current.revision,
      scope: current.scope,
      inventoryScope: { kind: "global" },
      frontierBeforeAttestation: true,
      attestationBeforeDischarge: true,
      policyBeforeDischarge: true,
    });
  }

  const actual = projectHistory(actions, baseScope);
  const ordered = [...actions].sort((a, b) => a.at - b.at || a.ordinal - b.ordinal);
  let expectedCurrent = actions[0];
  let expectedFinal = null;
  for (const action of ordered) {
    if (action.kind === "revision") {
      if (scopeCovers(action.scope, baseScope)) {
        expectedCurrent = action;
      } else {
        const seenRevision = actual.historicalDischarges.find(item => item.actionId === action.actionId);
        if (seenRevision?.accepted || actual.currentPolicyActionId === action.actionId) {
          scopeNarrowingFalseAccepts += 1;
        }
      }
    } else if (action.kind === "discharge") {
      const valid = action.triggerActionId === "trigger"
        && action.policyActionId === expectedCurrent.actionId
        && action.policyRevision === expectedCurrent.revision
        && JSON.stringify(action.scope) === JSON.stringify(expectedCurrent.scope)
        && scopeCovers(action.inventoryScope, action.scope)
        && action.frontierBeforeAttestation
        && action.attestationBeforeDischarge
        && action.policyBeforeDischarge;
      const seen = actual.historicalDischarges.find(item => item.actionId === action.actionId);
      if (valid) validHistoricalDischarges += 1;
      if (valid && !seen?.accepted) historicalDischargeFalseRejects += 1;
      if (!valid && seen?.accepted) staleDischargeFalseAccepts += 1;
      if (valid) expectedFinal = { actionId: action.actionId, policyActionId: expectedCurrent.actionId };
    }
  }
  const expectedCurrentDischarge = expectedFinal
    && expectedFinal.policyActionId === expectedCurrent.actionId
      ? expectedFinal
      : null;
  if (JSON.stringify(actual.finalDischarge) !== JSON.stringify(expectedCurrentDischarge)) {
    finalPolicyFalseSafes += 1;
  }
  if (actual.currentObligationDischarged && !expectedCurrentDischarge) finalPolicyFalseSafes += 1;
  if (expectedCurrent.scope === widenedScope && expectedCurrentDischarge) validWidenedFinalDischarges += 1;
  cases += 1;
}

assert.ok(validHistoricalDischarges > 0);
assert.equal(historicalDischargeFalseRejects, 0);
assert.equal(staleDischargeFalseAccepts, 0);
assert.equal(finalPolicyFalseSafes, 0);
assert.equal(scopeNarrowingFalseAccepts, 0);

console.log(JSON.stringify({
  report_version: "event-order-v0.3.2-stop-generation-reference.v1",
  model: "independent finite-state policy-generation reference",
  runtime: "Node.js",
  seed: "0x4c554d49",
  policy_history_cases: cases,
  valid_historical_discharges: validHistoricalDischarges,
  historical_discharge_false_rejects: historicalDischargeFalseRejects,
  stale_or_substituted_discharge_false_accepts: staleDischargeFalseAccepts,
  final_policy_false_safes: finalPolicyFalseSafes,
  scope_narrowing_false_accepts: scopeNarrowingFalseAccepts,
  valid_widened_final_discharges: validWidenedFinalDischarges,
  interpretation: "Reference-model execution only. Does not execute the Rust projector, authenticate owner receipts, or estimate production reliability."
}, null, 2));
