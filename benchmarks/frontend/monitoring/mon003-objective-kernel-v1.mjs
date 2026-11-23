#!/usr/bin/env node
// SFEB-MON-003: deterministic, non-destructive reference kernel.
// This models sink-side fencing, source-event evidence, current authorization,
// idempotency and ambiguous-outcome reconciliation. It is not a production service.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

const clone = value => JSON.parse(JSON.stringify(value));
function canonical(value) {
  if (value === null || typeof value !== 'object') return JSON.stringify(value);
  if (Array.isArray(value)) return '[' + value.map(canonical).join(',') + ']';
  return '{' + Object.keys(value).sort().map(key => JSON.stringify(key) + ':' + canonical(value[key])).join(',') + '}';
}
const digest = value => createHash('sha256').update(canonical(value)).digest('hex');
const fail = (code, message) => {
  const error = new Error(message);
  error.code = code;
  throw error;
};

class SourceOfRecord {
  constructor({ source, threshold }) {
    this.source = source;
    this.threshold = threshold;
    this.sequence = 0;
    this.revision = 0;
    this.capacity = 0;
    this.providerReady = false;
    this.receipts = new Map();
    this.eventIds = new Set();
  }

  append({ eventId, capacity, providerReady, now, validUntil, resourceId, actionType }) {
    if (!eventId || this.sequence < 0 || !Number.isFinite(capacity) || !Number.isFinite(now) || validUntil <= now) {
      fail('INVALID_SOURCE_EVENT', 'source event fields are invalid');
    }
    if (this.eventIds.has(eventId)) fail('DUPLICATE_SOURCE_EVENT', 'source event identity already exists');
    this.eventIds.add(eventId);
    const previousCapacity = this.capacity;
    this.sequence += 1;
    this.revision += 1;
    this.capacity = capacity;
    this.providerReady = providerReady;
    const crossed = previousCapacity < this.threshold && capacity >= this.threshold && providerReady === true;
    const qualified = crossed;
    if (!qualified) return null;

    const receipt = {
      receiptId: this.source + '#' + eventId,
      source: this.source,
      eventId,
      sequence: this.sequence,
      revision: this.revision,
      occurredAt: now,
      expiresAt: validUntil,
      predicateDigest: digest({ kind: 'capacity-crossing', threshold: this.threshold, providerReady: true }),
      resourceId,
      actionType,
      revoked: false,
      payloadDigest: digest({ previousCapacity, capacity, providerReady, eventId, resourceId, actionType }),
    };
    this.receipts.set(receipt.receiptId, receipt);
    return clone(receipt);
  }

  revoke(receiptId) {
    const receipt = this.receipts.get(receiptId);
    if (!receipt) fail('UNKNOWN_RECEIPT', 'cannot revoke unknown source receipt');
    receipt.revoked = true;
    receipt.revokedAtRevision = ++this.revision;
  }

  get(receiptId) {
    const receipt = this.receipts.get(receiptId);
    return receipt ? clone(receipt) : null;
  }
}

class LeaseCoordinator {
  constructor({ objectiveId, sink }) {
    this.objectiveId = objectiveId;
    this.sink = sink;
    this.epoch = 0;
    this.active = null;
  }

  acquire(workerId, { now, ttl }) {
    if (!workerId || !Number.isFinite(now) || !Number.isFinite(ttl) || ttl <= 0) {
      fail('INVALID_LEASE', 'lease fields are invalid');
    }
    if (this.active && this.active.expiresAt > now) {
      fail('LEASE_HELD', 'current worker lease has not expired');
    }
    this.epoch += 1;
    const lease = Object.freeze({
      objectiveId: this.objectiveId,
      workerId,
      epoch: this.epoch,
      issuedAt: now,
      expiresAt: now + ttl,
    });
    this.active = lease;
    // The mutation sink observes the new fencing epoch before this lease is used.
    this.sink.installFence(this.objectiveId, lease.epoch);
    return lease;
  }
}

class PolicyPlane {
  constructor() {
    this.revision = 1;
    this.allowed = true;
  }

  revoke() {
    this.allowed = false;
    this.revision += 1;
  }

  issueGrant({ lease, objective, source, receiptId, action, now, ttl }) {
    if (!this.allowed) fail('NOT_AUTHORIZED', 'current permission is denied');
    const receipt = source.get(receiptId);
    if (!receipt) fail('UNKNOWN_SOURCE_RECEIPT', 'authority plane could not resolve the receipt in the source of record');
    if (lease.objectiveId !== objective.id || receipt.resourceId !== action.resourceId || receipt.actionType !== action.type) {
      fail('GRANT_SCOPE_MISMATCH', 'event receipt, objective and action are not aligned');
    }
    const actionDigest = digest(action);
    return Object.freeze({
      grantId: 'grant-' + digest({ objective: objective.id, version: objective.version, actor: lease.workerId, epoch: lease.epoch, actionDigest, revision: this.revision }).slice(0, 20),
      actor: lease.workerId,
      objectiveId: objective.id,
      objectiveVersion: objective.version,
      eventReceiptId: receipt.receiptId,
      actionDigest,
      resourceId: action.resourceId,
      permissionRevision: this.revision,
      fencingEpoch: lease.epoch,
      issuedAt: now,
      expiresAt: Math.min(now + ttl, receipt.expiresAt),
    });
  }
}

class DurableMockSink {
  constructor() {
    this.maxEpoch = new Map();
    this.results = new Map();
    this.sideEffects = [];
    this.loseNextResponse = false;
  }

  installFence(objectiveId, epoch) {
    const current = this.maxEpoch.get(objectiveId) ?? 0;
    if (!Number.isSafeInteger(epoch) || epoch <= current) fail('NON_MONOTONIC_FENCE', 'fencing epoch must increase');
    this.maxEpoch.set(objectiveId, epoch);
  }

  execute({ objective, lease, grant, source, receiptId, action, actionKey, requestDigest, policy, now }) {
    const prior = this.results.get(actionKey);
    if (prior) {
      if (prior.requestDigest !== requestDigest) fail('IDEMPOTENCY_CONFLICT', 'same idempotency key used with a different request digest');
      return { ...clone(prior), replayed: true };
    }

    const receipt = source.get(receiptId);
    if (!receipt) fail('UNKNOWN_SOURCE_RECEIPT', 'sink could not resolve an authoritative source receipt');
    const expectedActionKey = deriveActionKey({ objective, receipt, action });
    const expectedRequestDigest = deriveRequestDigest({ objective, receipt, action });
    if (actionKey !== expectedActionKey) fail('ACTION_KEY_MISMATCH', 'idempotency key is not derived from the objective, event and action');
    if (requestDigest !== expectedRequestDigest) fail('REQUEST_DIGEST_MISMATCH', 'request digest does not match the canonical action envelope');

    const maxEpoch = this.maxEpoch.get(objective.id) ?? 0;
    if (lease.objectiveId !== objective.id || lease.epoch !== maxEpoch || lease.expiresAt <= now) {
      fail('STALE_FENCE', 'sink rejected stale, foreign or expired worker lease');
    }
    if (grant.actor !== lease.workerId || grant.fencingEpoch !== lease.epoch || grant.objectiveId !== objective.id || grant.objectiveVersion !== objective.version) {
      fail('GRANT_BINDING_MISMATCH', 'authorization grant is not bound to this actor, fence and objective version');
    }
    if (grant.eventReceiptId !== receipt.receiptId || grant.actionDigest !== digest(action) || grant.resourceId !== action.resourceId) {
      fail('GRANT_SCOPE_MISMATCH', 'authorization grant does not cover the requested event/action/target');
    }
    if (!policy.allowed || grant.permissionRevision !== policy.revision) {
      fail('PERMISSION_REVISION_STALE', 'permission was revoked or changed after grant issuance');
    }
    if (now < grant.issuedAt || now >= grant.expiresAt || now >= receipt.expiresAt || receipt.revoked) {
      fail('RECEIPT_OR_GRANT_INELIGIBLE', 'source receipt or authorization grant is expired/revoked');
    }
    if (receipt.resourceId !== action.resourceId || receipt.actionType !== action.type) {
      fail('TARGET_SCOPE_MISMATCH', 'action differs from the source receipt scope');
    }

    const result = {
      objectiveId: objective.id,
      objectiveVersion: objective.version,
      eventReceiptId: receipt.receiptId,
      actionKey,
      requestDigest,
      operationId: 'op-' + digest({ actionKey, requestDigest }).slice(0, 24),
      workerId: lease.workerId,
      fencingEpoch: lease.epoch,
      authorizationGrantId: grant.grantId,
      permissionRevision: policy.revision,
      committedAt: now,
      sinkStatus: 'committed',
      postcondition: { resourceId: action.resourceId, state: 'reserved' },
      replayed: false,
    };
    // This Map update and side-effect append model one atomic sink transaction.
    this.results.set(actionKey, clone(result));
    this.sideEffects.push(clone(result));

    if (this.loseNextResponse) {
      this.loseNextResponse = false;
      const error = new Error('sink committed, but the response was lost');
      error.code = 'OUTCOME_UNKNOWN';
      error.actionKey = actionKey;
      throw error;
    }
    return clone(result);
  }

  reconcile(actionKey) {
    const result = this.results.get(actionKey);
    return result ? clone(result) : null;
  }
}

function deriveActionKey({ objective, receipt, action }) {
  return 'sfeb:' + digest({ domain: 'sfeb.mon003.action.v1', objectiveId: objective.id, objectiveVersion: objective.version, source: receipt.source, eventId: receipt.eventId, actionDigest: digest(action) });
}
function deriveRequestDigest({ objective, receipt, action }) {
  return digest({ tenant: objective.tenant, objectiveId: objective.id, objectiveVersion: objective.version, source: receipt.source, eventId: receipt.eventId, action });
}

function makeWorld() {
  const objective = Object.freeze({ id: 'objective-reservation-1', version: 1, tenant: 'tenant-test' });
  const sink = new DurableMockSink();
  const leases = new LeaseCoordinator({ objectiveId: objective.id, sink });
  const source = new SourceOfRecord({ source: 'urn:fixture:capacity-service', threshold: 3 });
  const policy = new PolicyPlane();
  source.append({ eventId: 'baseline-ready', capacity: 1, providerReady: true, now: 0, validUntil: 100, resourceId: 'slot-17', actionType: 'reserve' });
  const receipt = source.append({ eventId: 'capacity-crossing-1', capacity: 3, providerReady: true, now: 10, validUntil: 30, resourceId: 'slot-17', actionType: 'reserve' });
  source.append({ eventId: 'capacity-falls', capacity: 1, providerReady: true, now: 12, validUntil: 100, resourceId: 'slot-17', actionType: 'reserve' });
  assert.ok(receipt, 'the threshold crossing should issue a source event receipt');
  const action = Object.freeze({ type: 'reserve', resourceId: 'slot-17', amount: 1 });
  return { objective, sink, leases, source, policy, receipt, action };
}

const cases = [];
function test(name, fn) {
  fn();
  cases.push({ name, result: 'pass' });
}
function expectCode(code, fn) {
  assert.throws(fn, error => error && error.code === code, 'expected ' + code);
}

// 1. A durable edge remains available after current level state has fallen false.
test('edge receipt survives level falling false but remains event evidence', () => {
  const world = makeWorld();
  assert.equal(world.source.capacity, 1);
  assert.equal(world.source.get(world.receipt.receiptId).revoked, false);
  assert.equal(world.source.get(world.receipt.receiptId).expiresAt, 30);
});

// 2. A replaced worker cannot mutate using an earlier fencing epoch.
test('stale worker A is fenced after worker B acquires a higher epoch', () => {
  const world = makeWorld();
  const leaseA = world.leases.acquire('worker-A', { now: 1, ttl: 5 });
  const grantA = world.policy.issueGrant({ lease: leaseA, objective: world.objective, source: world.source, receiptId: world.receipt.receiptId, action: world.action, now: 2, ttl: 10 });
  const leaseB = world.leases.acquire('worker-B', { now: 7, ttl: 10 });
  assert.equal(leaseA.epoch, 1);
  assert.equal(leaseB.epoch, 2);
  expectCode('STALE_FENCE', () => world.sink.execute({
    objective: world.objective, lease: leaseA, grant: grantA, source: world.source, receiptId: world.receipt.receiptId,
    action: world.action, actionKey: deriveActionKey({ objective: world.objective, receipt: world.receipt, action: world.action }),
    requestDigest: deriveRequestDigest({ objective: world.objective, receipt: world.receipt, action: world.action }), policy: world.policy, now: 8,
  }));
  assert.equal(world.sink.sideEffects.length, 0);
});

// 3. A grant issued before revocation is invalid at commit time.
test('permission revocation between grant issuance and sink commit blocks the action', () => {
  const world = makeWorld();
  const lease = world.leases.acquire('worker-B', { now: 13, ttl: 10 });
  const grant = world.policy.issueGrant({ lease, objective: world.objective, source: world.source, receiptId: world.receipt.receiptId, action: world.action, now: 14, ttl: 10 });
  world.policy.revoke();
  expectCode('PERMISSION_REVISION_STALE', () => world.sink.execute({
    objective: world.objective, lease, grant, source: world.source, receiptId: world.receipt.receiptId, action: world.action,
    actionKey: deriveActionKey({ objective: world.objective, receipt: world.receipt, action: world.action }),
    requestDigest: deriveRequestDigest({ objective: world.objective, receipt: world.receipt, action: world.action }), policy: world.policy, now: 15,
  }));
  assert.equal(world.sink.sideEffects.length, 0);
});

// 4. Expired and revoked source receipts are rejected even if the worker holds a lease.
test('expired and revoked event receipts cannot authorize a side effect', () => {
  const expired = makeWorld();
  const lease1 = expired.leases.acquire('worker-X', { now: 13, ttl: 20 });
  const grant1 = expired.policy.issueGrant({ lease: lease1, objective: expired.objective, source: expired.source, receiptId: expired.receipt.receiptId, action: expired.action, now: 14, ttl: 30 });
  expectCode('RECEIPT_OR_GRANT_INELIGIBLE', () => expired.sink.execute({
    objective: expired.objective, lease: lease1, grant: grant1, source: expired.source, receiptId: expired.receipt.receiptId, action: expired.action,
    actionKey: deriveActionKey({ objective: expired.objective, receipt: expired.receipt, action: expired.action }),
    requestDigest: deriveRequestDigest({ objective: expired.objective, receipt: expired.receipt, action: expired.action }), policy: expired.policy, now: 30,
  }));

  const revoked = makeWorld();
  const lease2 = revoked.leases.acquire('worker-Y', { now: 13, ttl: 20 });
  const grant2 = revoked.policy.issueGrant({ lease: lease2, objective: revoked.objective, source: revoked.source, receiptId: revoked.receipt.receiptId, action: revoked.action, now: 14, ttl: 10 });
  revoked.source.revoke(revoked.receipt.receiptId);
  expectCode('RECEIPT_OR_GRANT_INELIGIBLE', () => revoked.sink.execute({
    objective: revoked.objective, lease: lease2, grant: grant2, source: revoked.source, receiptId: revoked.receipt.receiptId, action: revoked.action,
    actionKey: deriveActionKey({ objective: revoked.objective, receipt: revoked.receipt, action: revoked.action }),
    requestDigest: deriveRequestDigest({ objective: revoked.objective, receipt: revoked.receipt, action: revoked.action }), policy: revoked.policy, now: 15,
  }));
  assert.equal(revoked.sink.sideEffects.length, 0);
});

// 5. Lost response: the operation committed, and retry/reconciliation uses the same key.
test('lost sink response reconciles to one durable execution receipt', () => {
  const world = makeWorld();
  const lease = world.leases.acquire('worker-recovery', { now: 13, ttl: 20 });
  const grant = world.policy.issueGrant({ lease, objective: world.objective, source: world.source, receiptId: world.receipt.receiptId, action: world.action, now: 14, ttl: 10 });
  const actionKey = deriveActionKey({ objective: world.objective, receipt: world.receipt, action: world.action });
  world.sink.loseNextResponse = true;
  expectCode('OUTCOME_UNKNOWN', () => world.sink.execute({
    objective: world.objective, lease, grant, source: world.source, receiptId: world.receipt.receiptId, action: world.action,
    actionKey, requestDigest: deriveRequestDigest({ objective: world.objective, receipt: world.receipt, action: world.action }), policy: world.policy, now: 15,
  }));
  const reconciled = world.sink.reconcile(actionKey);
  assert.ok(reconciled, 'sink must expose the committed operation by the same key');
  const replay = world.sink.execute({
    objective: world.objective, lease, grant, source: world.source, receiptId: world.receipt.receiptId, action: world.action,
    actionKey, requestDigest: deriveRequestDigest({ objective: world.objective, receipt: world.receipt, action: world.action }), policy: world.policy, now: 16,
  });
  assert.equal(replay.operationId, reconciled.operationId);
  assert.equal(replay.replayed, true);
  assert.equal(world.sink.sideEffects.length, 1);
});

// 6. Key reuse with a different payload is a conflict, not a second operation.
test('same idempotency key with a different request digest is rejected', () => {
  const world = makeWorld();
  const lease = world.leases.acquire('worker-idem', { now: 13, ttl: 20 });
  const grant = world.policy.issueGrant({ lease, objective: world.objective, source: world.source, receiptId: world.receipt.receiptId, action: world.action, now: 14, ttl: 10 });
  const actionKey = deriveActionKey({ objective: world.objective, receipt: world.receipt, action: world.action });
  world.sink.execute({ objective: world.objective, lease, grant, source: world.source, receiptId: world.receipt.receiptId, action: world.action,
    actionKey, requestDigest: deriveRequestDigest({ objective: world.objective, receipt: world.receipt, action: world.action }), policy: world.policy, now: 15 });
  expectCode('IDEMPOTENCY_CONFLICT', () => world.sink.execute({
    objective: world.objective, lease, grant, source: world.source, receiptId: world.receipt.receiptId,
    action: { ...world.action, resourceId: 'slot-attacker' }, actionKey,
    requestDigest: deriveRequestDigest({ objective: world.objective, receipt: world.receipt, action: { ...world.action, resourceId: 'slot-attacker' } }), policy: world.policy, now: 16,
  }));
  assert.equal(world.sink.sideEffects.length, 1);
});

// 7. Wrong-target grant substitution fails at the sink.
test('authorization grant cannot be rebound to another target', () => {
  const world = makeWorld();
  const lease = world.leases.acquire('worker-scope', { now: 13, ttl: 20 });
  const grant = world.policy.issueGrant({ lease, objective: world.objective, source: world.source, receiptId: world.receipt.receiptId, action: world.action, now: 14, ttl: 10 });
  expectCode('GRANT_SCOPE_MISMATCH', () => world.sink.execute({
    objective: world.objective, lease, grant, source: world.source, receiptId: world.receipt.receiptId,
    action: { ...world.action, resourceId: 'slot-other' },
    actionKey: deriveActionKey({ objective: world.objective, receipt: world.receipt, action: { ...world.action, resourceId: 'slot-other' } }), requestDigest: deriveRequestDigest({ objective: world.objective, receipt: world.receipt, action: { ...world.action, resourceId: 'slot-other' } }),
    policy: world.policy, now: 15,
  }));
  assert.equal(world.sink.sideEffects.length, 0);
});

test('worker-supplied fabricated receipt is not accepted by the authority plane', () => {
  const world = makeWorld();
  const lease = world.leases.acquire('worker-forgery', { now: 13, ttl: 20 });
  expectCode('UNKNOWN_SOURCE_RECEIPT', () => world.policy.issueGrant({
    lease, objective: world.objective, source: world.source, receiptId: 'urn:attacker#fabricated',
    action: world.action, now: 14, ttl: 10,
  }));
  assert.equal(world.sink.sideEffects.length, 0);
});

console.log(JSON.stringify({
  schema: 'sfeb-mon003-reference-kernel-report-v1',
  status: 'passed',
  deterministic_cases: cases.length,
  cases,
  side_effect_claim: 'All negative controls produced zero side effects; the lost-response recovery case produced exactly one side effect.',
  limitations: [
    'Pure in-memory reference kernel; no database, broker, cryptographic signer, external authorization service or browser is exercised.',
    'The mock sink models fence, policy-revision and idempotency checks atomically; production atomicity must be independently implemented and qualified.',
    'The source receipt is a trusted in-memory source-of-record object, not a signed production receipt.',
    'This does not demonstrate exactly-once delivery for arbitrary external sinks.'
  ]
}, null, 2));
