#!/usr/bin/env node
// SFEB-MON-003 tenant-scope and authority-boundary reference tests.
// In-memory negative controls only; no production cryptographic/persistence claim.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

function canon(v) {
  if (v === null || typeof v !== 'object') return JSON.stringify(v);
  if (Array.isArray(v)) return '[' + v.map(canon).join(',') + ']';
  return '{' + Object.keys(v).sort().map(k => JSON.stringify(k) + ':' + canon(v[k])).join(',') + '}';
}
const hash = v => createHash('sha256').update(canon(v)).digest('hex');
const clone = v => JSON.parse(JSON.stringify(v));
function fail(code, message) { const e = new Error(message); e.code = code; throw e; }

class Source {
  constructor(tenantId) { this.tenantId = tenantId; this.events = new Map(); }
  append({ source, id, sequence, at, expiresAt, resourceId, actionType }) {
    if (!source || !id || !Number.isSafeInteger(sequence) || sequence < 1 || expiresAt <= at) fail('INVALID_EVENT', 'invalid source event');
    const key = source + '\n' + id;
    if (this.events.has(key)) fail('DUPLICATE_EVENT', 'CloudEvents source+id identity already exists');
    const receipt = Object.freeze({ tenantId: this.tenantId, source, id, sequence, at, expiresAt, resourceId, actionType, revoked: false,
      digest: hash({ tenantId: this.tenantId, source, id, sequence, at, expiresAt, resourceId, actionType }) });
    this.events.set(key, receipt); return receipt;
  }
}

class Authority {
  constructor(tenants = ['tenant-A']) { this.revision = 1; this.tenants = new Set(tenants); }
  revoke(tenant) { this.tenants.delete(tenant); this.revision += 1; }
  grant({ objective, receipt, actor, epoch, action, now, ttl }) {
    if (!receipt || receipt.tenantId !== objective.tenantId) fail('TENANT_SCOPE_MISMATCH', 'receipt/objective tenant differs');
    if (!this.tenants.has(objective.tenantId)) fail('NOT_AUTHORIZED', 'tenant permission denied');
    if (receipt.resourceId !== action.resourceId || receipt.actionType !== action.type) fail('ACTION_SCOPE_MISMATCH', 'receipt/action scope differs');
    if (receipt.revoked || now >= receipt.expiresAt) fail('RECEIPT_INELIGIBLE', 'receipt revoked or expired');
    return Object.freeze({ tenantId: objective.tenantId, objectiveId: objective.id, version: objective.version, actor, epoch,
      event: receipt.source + '\n' + receipt.id, receiptDigest: receipt.digest, actionDigest: hash(action), resourceId: action.resourceId,
      revision: this.revision, expiresAt: Math.min(now + ttl, receipt.expiresAt), issuedAt: now });
  }
}

class Sink {
  constructor() { this.fences = new Map(); this.results = new Map(); this.effects = []; }
  fence(objectiveId, epoch) {
    const prior = this.fences.get(objectiveId) || 0;
    if (!Number.isSafeInteger(epoch) || epoch <= prior) fail('BAD_FENCE', 'fence must increase');
    this.fences.set(objectiveId, epoch);
  }
  commit({ objective, receipt, grant, actor, epoch, action, key, request, authority, now }) {
    if (receipt.tenantId !== objective.tenantId || grant.tenantId !== objective.tenantId) fail('TENANT_SCOPE_MISMATCH', 'objective, receipt and grant tenant mismatch');
    const expectedRequest = hash({ tenantId: objective.tenantId, objectiveId: objective.id, version: objective.version, source: receipt.source, eventId: receipt.id, action });
    if (request !== expectedRequest) fail('REQUEST_DIGEST_MISMATCH', 'request digest mismatch');
    const prior = this.results.get(key);
    if (prior) {
      if (prior.request !== expectedRequest) fail('IDEMPOTENCY_CONFLICT', 'key reused with different request');
      return { ...clone(prior), replayed: true };
    }
    const expectedKey = 'sfeb:' + hash({ tenantId: objective.tenantId, objectiveId: objective.id, version: objective.version,
      source: receipt.source, eventId: receipt.id, actionDigest: hash(action) });
    if (key !== expectedKey) fail('ACTION_KEY_MISMATCH', 'idempotency key is not content bound');
    if (epoch !== (this.fences.get(objective.id) || 0)) fail('STALE_FENCE', 'sink rejected stale worker epoch');
    if (grant.actor !== actor || grant.epoch !== epoch || grant.objectiveId !== objective.id || grant.version !== objective.version) fail('GRANT_BINDING_MISMATCH', 'grant actor/objective/fence mismatch');
    if (grant.event !== receipt.source + '\n' + receipt.id || grant.receiptDigest !== receipt.digest || grant.actionDigest !== hash(action) || grant.resourceId !== action.resourceId) fail('GRANT_SCOPE_MISMATCH', 'grant not bound to receipt/action');
    if (!authority.tenants.has(objective.tenantId) || grant.revision !== authority.revision) fail('PERMISSION_REVISION_STALE', 'permission revoked/changed before commit');
    if (now < grant.issuedAt || now >= grant.expiresAt || now >= receipt.expiresAt || receipt.revoked) fail('RECEIPT_INELIGIBLE', 'receipt or grant expired/revoked');
    const result = { objectiveId: objective.id, tenantId: objective.tenantId, request: expectedRequest,
      operationId: 'op-' + hash({ key, request: expectedRequest }).slice(0, 20), actor, epoch, status: 'committed' };
    this.results.set(key, clone(result)); this.effects.push(clone(result)); return clone(result);
  }
}

function setup(tenantId = 'tenant-A') {
  const objective = Object.freeze({ id: 'objective-1', version: 1, tenantId });
  const source = new Source(tenantId);
  const receipt = source.append({ source: 'urn:capacity:' + tenantId, id: 'event-1', sequence: 1, at: 10, expiresAt: 30, resourceId: 'slot-17', actionType: 'reserve' });
  const authority = new Authority([tenantId]); const sink = new Sink(); sink.fence(objective.id, 1);
  const actor = 'worker-1', epoch = 1, action = Object.freeze({ type: 'reserve', resourceId: 'slot-17', amount: 1 });
  const grant = authority.grant({ objective, receipt, actor, epoch, action, now: 11, ttl: 10 });
  const key = 'sfeb:' + hash({ tenantId, objectiveId: objective.id, version: objective.version, source: receipt.source, eventId: receipt.id, actionDigest: hash(action) });
  const request = hash({ tenantId, objectiveId: objective.id, version: objective.version, source: receipt.source, eventId: receipt.id, action });
  return { objective, source, receipt, authority, sink, actor, epoch, action, grant, key, request };
}

const cases = [];
function test(name, fn) { fn(); cases.push({ name, result: 'pass' }); }
function expect(code, fn) { assert.throws(fn, e => e && e.code === code, 'expected ' + code); }

test('source receipt from tenant A cannot authorize objective in tenant B', () => {
  const w = setup(); const other = { ...w.objective, id: 'objective-B', tenantId: 'tenant-B' };
  expect('TENANT_SCOPE_MISMATCH', () => w.authority.grant({ objective: other, receipt: w.receipt, actor: 'worker-B', epoch: 1, action: w.action, now: 12, ttl: 5 }));
  assert.equal(w.sink.effects.length, 0);
});

test('sink rejects cross-tenant receipt/grant transplant', () => {
  const w = setup(); const other = { ...w.objective, tenantId: 'tenant-B' };
  expect('TENANT_SCOPE_MISMATCH', () => w.sink.commit({ objective: other, receipt: w.receipt, grant: w.grant, actor: w.actor, epoch: w.epoch,
    action: w.action, key: w.key, request: hash({ tenantId: other.tenantId, objectiveId: other.id, version: other.version, source: w.receipt.source, eventId: w.receipt.id, action: w.action }), authority: w.authority, now: 12 }));
  assert.equal(w.sink.effects.length, 0);
});

test('CloudEvents source plus id deduplicates within source without cross-source collisions', () => {
  const s = new Source('tenant-A'); const base = { id: 'same-id', sequence: 1, at: 1, expiresAt: 5, resourceId: 'slot-1', actionType: 'reserve' };
  const a = s.append({ source: 'urn:source:A', ...base }); const b = s.append({ source: 'urn:source:B', ...base });
  assert.notEqual(a.digest, b.digest); expect('DUPLICATE_EVENT', () => s.append({ source: 'urn:source:A', ...base }));
});

test('same key and same request replays the original operation once', () => {
  const w = setup(); const args = { objective: w.objective, receipt: w.receipt, grant: w.grant, actor: w.actor, epoch: w.epoch,
    action: w.action, key: w.key, request: w.request, authority: w.authority, now: 12 };
  const a = w.sink.commit(args), b = w.sink.commit(args); assert.equal(a.operationId, b.operationId); assert.equal(b.replayed, true); assert.equal(w.sink.effects.length, 1);
});

test('same key with changed request is a conflict and cannot add an effect', () => {
  const w = setup(); w.sink.commit({ objective: w.objective, receipt: w.receipt, grant: w.grant, actor: w.actor, epoch: w.epoch, action: w.action, key: w.key, request: w.request, authority: w.authority, now: 12 });
  const changed = { ...w.action, amount: 2 }; const request = hash({ tenantId: w.objective.tenantId, objectiveId: w.objective.id, version: w.objective.version, source: w.receipt.source, eventId: w.receipt.id, action: changed });
  expect('IDEMPOTENCY_CONFLICT', () => w.sink.commit({ objective: w.objective, receipt: w.receipt, grant: w.grant, actor: w.actor, epoch: w.epoch, action: changed, key: w.key, request, authority: w.authority, now: 13 }));
  assert.equal(w.sink.effects.length, 1);
});

test('current tenant permission revocation after grant issuance blocks commit', () => {
  const w = setup(); w.authority.revoke(w.objective.tenantId);
  expect('PERMISSION_REVISION_STALE', () => w.sink.commit({ objective: w.objective, receipt: w.receipt, grant: w.grant, actor: w.actor, epoch: w.epoch, action: w.action, key: w.key, request: w.request, authority: w.authority, now: 12 }));
  assert.equal(w.sink.effects.length, 0);
});

test('sink rejects stale worker after a replacement epoch is installed', () => {
  const w = setup(); w.sink.fence(w.objective.id, 2);
  expect('STALE_FENCE', () => w.sink.commit({ objective: w.objective, receipt: w.receipt, grant: w.grant, actor: w.actor, epoch: 1, action: w.action, key: w.key, request: w.request, authority: w.authority, now: 12 }));
  assert.equal(w.sink.effects.length, 0);
});

test('grant cannot be rebound to an attacker-controlled resource', () => {
  const w = setup(); const changed = { ...w.action, resourceId: 'slot-attacker' };
  const key = 'sfeb:' + hash({ tenantId: w.objective.tenantId, objectiveId: w.objective.id, version: w.objective.version, source: w.receipt.source, eventId: w.receipt.id, actionDigest: hash(changed) });
  const request = hash({ tenantId: w.objective.tenantId, objectiveId: w.objective.id, version: w.objective.version, source: w.receipt.source, eventId: w.receipt.id, action: changed });
  expect('GRANT_SCOPE_MISMATCH', () => w.sink.commit({ objective: w.objective, receipt: w.receipt, grant: w.grant, actor: w.actor, epoch: w.epoch, action: changed, key, request, authority: w.authority, now: 12 }));
  assert.equal(w.sink.effects.length, 0);
});

console.log(JSON.stringify({ schema: 'sfeb-mon003-tenant-boundary-report-v1', status: 'passed', deterministic_cases: cases.length, cases,
  total_side_effects_in_negative_controls: 0, limitations: [ 'Pure in-memory reference model; no cryptographic issuer validation, persistent policy store or production sink is exercised.',
    'Tenant identity and event provenance are trusted model inputs; signatures and key revocation remain unimplemented.', 'This does not qualify cross-tenant isolation in a real database or service mesh.' ] }, null, 2));
