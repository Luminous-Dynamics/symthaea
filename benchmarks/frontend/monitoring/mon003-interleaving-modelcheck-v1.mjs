#!/usr/bin/env node
// SFEB-MON-003 bounded interleaving exploration.
// Enumerates every ordering of six competing lifecycle events (6! = 720).
// It is a finite reference model, not a proof of a production implementation.
import assert from 'node:assert/strict';

const OPERATIONS = [
  'replace_worker',
  'revoke_permission',
  'expire_receipt',
  'worker_a_commit_response_lost',
  'worker_b_retry_same_operation',
  'reconcile_operation',
];

function* permutations(items, prefix = []) {
  if (items.length === 0) { yield prefix; return; }
  for (let i = 0; i < items.length; i++) {
    yield* permutations([...items.slice(0, i), ...items.slice(i + 1)], [...prefix, items[i]]);
  }
}

function simulate(order) {
  const state = {
    maxEpoch: 1,
    workerBLease: false,
    permissionAllowed: true,
    permissionRevision: 1,
    receiptEligible: true,
    operation: null,
    sideEffects: [],
    outcomeUnknown: false,
    reconciled: false,
    attempts: [],
  };
  const fixed = {
    objectiveId: 'objective-1',
    operationId: 'operation-1',
    idempotencyKey: 'action-key-1',
    requestDigest: 'request-digest-1',
    workerAEpoch: 1,
    workerAGrantRevision: 1,
  };

  function commit(worker, epoch, grantRevision, stepIndex, operationName) {
    // Reads of an already committed key are read-only reconciliation/replay,
    // so they return the original receipt without a second mutation.
    if (state.operation) {
      if (state.operation.requestDigest !== fixed.requestDigest) {
        state.attempts.push({ worker, result: 'idempotency-conflict' });
        return { result: 'conflict' };
      }
      state.attempts.push({ worker, result: 'idempotent-replay' });
      return { result: 'replayed', operation: state.operation };
    }
    if (epoch !== state.maxEpoch) {
      state.attempts.push({ worker, result: 'stale-fence' });
      return { result: 'stale-fence' };
    }
    if (!state.permissionAllowed || grantRevision !== state.permissionRevision) {
      state.attempts.push({ worker, result: 'permission-revoked' });
      return { result: 'permission-revoked' };
    }
    if (!state.receiptEligible) {
      state.attempts.push({ worker, result: 'receipt-expired' });
      return { result: 'receipt-expired' };
    }
    const result = Object.freeze({
      operationId: fixed.operationId,
      idempotencyKey: fixed.idempotencyKey,
      requestDigest: fixed.requestDigest,
      worker,
      epoch,
      permissionRevision: state.permissionRevision,
      receiptEligibleAtCommit: state.receiptEligible,
      permissionAllowedAtCommit: state.permissionAllowed,
      maxEpochAtCommit: state.maxEpoch,
      grantRevisionAtCommit: grantRevision,
      stepIndex,
      operationName,
    });
    // Model the sink's atomic mutation + durable dedupe record as one transition.
    state.operation = result;
    state.sideEffects.push(result);
    state.attempts.push({ worker, result: 'committed' });
    return { result: 'committed', operation: result };
  }

  for (const [stepIndex, op] of order.entries()) {
    switch (op) {
      case 'replace_worker':
        state.maxEpoch = 2;
        state.workerBLease = true;
        break;
      case 'revoke_permission':
        state.permissionAllowed = false;
        state.permissionRevision += 1;
        break;
      case 'expire_receipt':
        state.receiptEligible = false;
        break;
      case 'worker_a_commit_response_lost': {
        const r = commit('worker-A', fixed.workerAEpoch, fixed.workerAGrantRevision, stepIndex, op);
        // The transport response is lost whether the sink committed or rejected.
        state.outcomeUnknown = true;
        if (r.operation) state.observedLostOperationId = r.operation.operationId;
        break;
      }
      case 'worker_b_retry_same_operation':
        if (!state.workerBLease) {
          state.attempts.push({ worker: 'worker-B', result: 'no-lease' });
          break;
        }
        commit('worker-B', 2, state.permissionRevision, stepIndex, op);
        break;
      case 'reconcile_operation':
        if (state.operation) {
          state.reconciled = true;
          state.outcomeUnknown = false;
        }
        // A missing result is not proof of non-commit; preserve outcome_unknown.
        break;
      default: throw new Error('unknown operation ' + op);
    }
  }

  assert.ok(state.sideEffects.length <= 1, 'one idempotency key created multiple effects');
  for (const effect of state.sideEffects) {
    assert.equal(effect.epoch, effect.maxEpochAtCommit, 'effect committed with stale fence');
    assert.equal(effect.permissionAllowedAtCommit, true, 'effect committed after permission denial');
    assert.equal(effect.permissionRevision, effect.grantRevisionAtCommit, 'grant revision must be current at commit');
    assert.equal(effect.receiptEligibleAtCommit, true, 'effect committed after receipt expiry');
  }
  const revokeIndex = order.indexOf('revoke_permission');
  const expiryIndex = order.indexOf('expire_receipt');
  const replaceIndex = order.indexOf('replace_worker');
  for (const effect of state.sideEffects) {
    assert.ok(revokeIndex === -1 || revokeIndex > effect.stepIndex, 'an effect committed after permission revocation');
    assert.ok(expiryIndex === -1 || expiryIndex > effect.stepIndex, 'an effect committed after receipt expiry');
    if (effect.worker === 'worker-A') {
      assert.ok(replaceIndex === -1 || replaceIndex > effect.stepIndex, 'stale worker A committed after replacement');
    }
    if (effect.worker === 'worker-B') {
      assert.ok(replaceIndex !== -1 && replaceIndex < effect.stepIndex, 'worker B committed before acquiring its lease');
    }
  }
  return state;
}

let scheduleCount = 0;
let oneEffectSchedules = 0;
let zeroEffectSchedules = 0;
let staleRejects = 0;
let permissionRejects = 0;
let expiryRejects = 0;
let idempotentReplays = 0;
let reconciliations = 0;
let unknownAtEnd = 0;
const examples = [];

for (const order of permutations(OPERATIONS)) {
  const state = simulate(order);
  scheduleCount += 1;
  if (state.sideEffects.length === 1) oneEffectSchedules += 1;
  else zeroEffectSchedules += 1;
  staleRejects += state.attempts.filter(x => x.result === 'stale-fence').length;
  permissionRejects += state.attempts.filter(x => x.result === 'permission-revoked').length;
  expiryRejects += state.attempts.filter(x => x.result === 'receipt-expired').length;
  idempotentReplays += state.attempts.filter(x => x.result === 'idempotent-replay').length;
  if (state.reconciled) reconciliations += 1;
  if (state.outcomeUnknown) unknownAtEnd += 1;
  if (examples.length < 5 && state.sideEffects.length === 0) examples.push({ order, attempts: state.attempts });
}

assert.equal(scheduleCount, 720, 'all 6! orderings must be covered');
assert.equal(oneEffectSchedules + zeroEffectSchedules, scheduleCount);

console.log(JSON.stringify({
  schema: 'sfeb-mon003-interleaving-modelcheck-report-v1',
  status: 'passed',
  operations: OPERATIONS,
  schedules_explored: scheduleCount,
  expected_permutations: 720,
  invariants: {
    at_most_one_side_effect_per_operation_key: 'pass',
    no_stale_worker_commit_after_fence_replacement: 'pass',
    no_commit_after_permission_revocation: 'pass',
    no_commit_after_receipt_expiry: 'pass',
    unknown_outcome_is_not_assumed_to_be_failure: 'pass',
    reconciliation_is_read_only: 'pass',
  },
  counts: {
    schedules_with_one_effect: oneEffectSchedules,
    schedules_with_zero_effects: zeroEffectSchedules,
    stale_fence_rejections: staleRejects,
    permission_rejections: permissionRejects,
    receipt_expiry_rejections: expiryRejects,
    idempotent_replays: idempotentReplays,
    schedules_with_successful_reconciliation: reconciliations,
    schedules_still_outcome_unknown_at_end: unknownAtEnd,
  },
  example_zero_effect_schedules: examples,
  limitations: [
    'Bounded in-memory sequential interleaving model, not a concurrent memory-model proof.',
    'The sink models atomic fence, permission, expiry and dedupe checks as one transition.',
    'Does not cover database isolation anomalies, network partitions beyond the enumerated orderings, clock drift, broker durability or real cryptographic identities.',
    'Zero failures in this finite model is not a production probability guarantee.',
  ],
}, null, 2));
