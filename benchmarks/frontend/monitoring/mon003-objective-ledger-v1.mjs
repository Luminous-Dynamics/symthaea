#!/usr/bin/env node
// SFEB-MON-003: objective ledger transition reference model.
// In-memory deterministic tests only; this is not a durable storage service.
import assert from 'node:assert/strict';

const TRANSITIONS = Object.freeze({
  registered: ['waiting', 'not_triggered', 'expired', 'revoked'],
  waiting: ['wake_pending', 'not_triggered', 'expired', 'revoked'],
  wake_pending: ['reconciling', 'expired', 'revoked'],
  reconciling: ['eligible', 'waiting', 'not_authorized', 'expired', 'revoked', 'inconclusive'],
  eligible: ['executing', 'waiting', 'not_authorized', 'expired', 'revoked', 'inconclusive'],
  executing: ['succeeded', 'outcome_unknown', 'inconclusive', 'quarantined'],
  outcome_unknown: ['succeeded', 'executing', 'inconclusive', 'quarantined'],
  succeeded: [], not_triggered: [], expired: [], revoked: [],
  not_authorized: [], inconclusive: [], quarantined: [],
});

function fail(code, message) {
  const error = new Error(message);
  error.code = code;
  throw error;
}

class ObjectiveLedger {
  constructor(objectiveId) {
    if (typeof objectiveId !== 'string' || !objectiveId) fail('INVALID_OBJECTIVE_ID', 'objective ID is required');
    this.objectiveId = objectiveId;
    this.version = 0;
    this.state = 'registered';
    this.pendingOperationId = null;
  }

  transition({ expectedVersion, nextState, evidence = {} }) {
    if (expectedVersion !== this.version) fail('STALE_OBJECTIVE_WRITE', 'objective version compare-and-swap failed');
    if (!Object.hasOwn(TRANSITIONS, this.state) || !TRANSITIONS[this.state].includes(nextState)) {
      fail('ILLEGAL_OBJECTIVE_TRANSITION', this.state + ' cannot transition to ' + nextState);
    }
    if (nextState === 'eligible' && (!evidence.sourceReceiptId || !Number.isSafeInteger(evidence.permissionRevision))) {
      fail('MISSING_TRANSITION_EVIDENCE', 'eligibility requires a source receipt and current permission revision');
    }
    if (nextState === 'executing' && (!evidence.authorizationGrantId || !evidence.idempotencyKey)) {
      fail('MISSING_TRANSITION_EVIDENCE', 'execution requires a current authorization grant and stable idempotency key');
    }
    if (nextState === 'outcome_unknown') {
      if (!evidence.operationId) fail('MISSING_TRANSITION_EVIDENCE', 'unknown outcome must retain the operation ID');
      this.pendingOperationId = evidence.operationId;
    }
    if (this.state === 'outcome_unknown' && nextState === 'executing' && evidence.operationId !== this.pendingOperationId) {
      fail('OPERATION_ID_MISMATCH', 'recovery must retain the original operation identity');
    }
    if (nextState === 'succeeded') {
      if (!evidence.executionReceiptId || !evidence.postconditionProof) {
        fail('MISSING_TRANSITION_EVIDENCE', 'success requires an execution receipt and independent postcondition proof');
      }
      if (this.pendingOperationId && evidence.operationId !== this.pendingOperationId) {
        fail('OPERATION_ID_MISMATCH', 'success proof does not reconcile the pending operation');
      }
    }
    this.state = nextState;
    this.version += 1;
    return Object.freeze({ objectiveId: this.objectiveId, state: this.state, version: this.version, pendingOperationId: this.pendingOperationId });
  }
}

const cases = [];
function test(name, fn) { fn(); cases.push({ name, result: 'pass' }); }
function expectCode(code, fn) {
  assert.throws(fn, error => error && error.code === code, 'expected ' + code);
}

test('stale objective writer rejected by compare-and-swap version', () => {
  const ledger = new ObjectiveLedger('objective-cas-1');
  const waiting = ledger.transition({ expectedVersion: 0, nextState: 'waiting' });
  assert.equal(waiting.version, 1);
  expectCode('STALE_OBJECTIVE_WRITE', () => ledger.transition({ expectedVersion: 0, nextState: 'wake_pending' }));
  assert.equal(ledger.state, 'waiting');
});

test('illegal jump and success without independent proof are rejected', () => {
  const ledger = new ObjectiveLedger('objective-proof-1');
  expectCode('ILLEGAL_OBJECTIVE_TRANSITION', () => ledger.transition({ expectedVersion: 0, nextState: 'succeeded' }));
  ledger.transition({ expectedVersion: 0, nextState: 'waiting' });
  ledger.transition({ expectedVersion: 1, nextState: 'wake_pending' });
  ledger.transition({ expectedVersion: 2, nextState: 'reconciling' });
  ledger.transition({ expectedVersion: 3, nextState: 'eligible', evidence: { sourceReceiptId: 'source#event-1', permissionRevision: 7 } });
  ledger.transition({ expectedVersion: 4, nextState: 'executing', evidence: { authorizationGrantId: 'grant-1', idempotencyKey: 'idem-1' } });
  expectCode('MISSING_TRANSITION_EVIDENCE', () => ledger.transition({ expectedVersion: 5, nextState: 'succeeded' }));
  assert.equal(ledger.state, 'executing');
});

test('unknown outcome can only be completed against original operation identity and proof', () => {
  const ledger = new ObjectiveLedger('objective-recovery-1');
  ledger.transition({ expectedVersion: 0, nextState: 'waiting' });
  ledger.transition({ expectedVersion: 1, nextState: 'wake_pending' });
  ledger.transition({ expectedVersion: 2, nextState: 'reconciling' });
  ledger.transition({ expectedVersion: 3, nextState: 'eligible', evidence: { sourceReceiptId: 'source#event-2', permissionRevision: 9 } });
  ledger.transition({ expectedVersion: 4, nextState: 'executing', evidence: { authorizationGrantId: 'grant-2', idempotencyKey: 'idem-2' } });
  ledger.transition({ expectedVersion: 5, nextState: 'outcome_unknown', evidence: { operationId: 'op-original' } });
  expectCode('OPERATION_ID_MISMATCH', () => ledger.transition({ expectedVersion: 6, nextState: 'executing', evidence: { operationId: 'op-new', authorizationGrantId: 'grant-3', idempotencyKey: 'idem-2' } }));
  const done = ledger.transition({ expectedVersion: 6, nextState: 'succeeded', evidence: { operationId: 'op-original', executionReceiptId: 'receipt-exec-1', postconditionProof: 'readback-sha256' } });
  assert.equal(done.state, 'succeeded');
  assert.equal(done.pendingOperationId, 'op-original');
});

console.log(JSON.stringify({
  schema: 'sfeb-mon003-objective-ledger-report-v1',
  status: 'passed',
  deterministic_cases: cases.length,
  cases,
  limitations: [
    'Pure in-memory state-machine reference; it does not persist to a database or distributed log.',
    'The state version is a model of compare-and-swap, not proof of a production transaction.',
    'Postcondition proof is a non-empty fixture token; independent cryptographic verification remains unimplemented.'
  ]
}, null, 2));
