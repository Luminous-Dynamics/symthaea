#!/usr/bin/env node
// SFEB-MON-003 capability leases: runtime-bounded source guarantees.
// In-memory reference only; it does not issue/verify production signatures.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';

const canonical = v => v === null || typeof v !== 'object' ? JSON.stringify(v) : Array.isArray(v) ? '[' + v.map(canonical).join(',') + ']' : '{' + Object.keys(v).sort().map(k => JSON.stringify(k) + ':' + canonical(v[k])).join(',') + '}';
const digest = v => createHash('sha256').update(canonical(v)).digest('hex');
const fail = (code, message) => { const e = new Error(message); e.code = code; throw e; };

function capabilityDigest(profile) {
  return digest({ sourceId: profile.sourceId, connectorVersion: profile.connectorVersion, capabilities: profile.capabilities });
}

class CapabilityLeaseRegistry {
  constructor() { this.entries = new Map(); }

  register({ profile, attestation, now }) {
    if (!profile || !attestation || !Number.isFinite(now)) fail('INVALID_INPUT', 'profile, attestation and current time required');
    if (!profile.sourceId || !profile.connectorVersion || !profile.capabilities) fail('INVALID_PROFILE', 'source identity, connector version and capabilities required');
    const expectedDigest = capabilityDigest(profile);
    if (attestation.sourceId !== profile.sourceId) fail('SOURCE_BINDING_MISMATCH', 'attestation source differs from profile');
    if (attestation.connectorVersion !== profile.connectorVersion) fail('CONNECTOR_VERSION_MISMATCH', 'attestation connector version differs');
    if (attestation.profileDigest !== expectedDigest) fail('PROFILE_DIGEST_MISMATCH', 'attestation is not bound to current capability content');
    if (!attestation.signatureVerified || !attestation.issuerScopeValid || !attestation.keyNotRevoked) fail('ATTESTATION_UNTRUSTED', 'signature, issuer scope or key status invalid');
    if (!attestation.conformanceProbePassed) fail('CONFORMANCE_FAILED', 'source capability conformance probe did not pass');
    if (!Number.isFinite(attestation.issuedAt) || !Number.isFinite(attestation.expiresAt) || attestation.issuedAt > now || attestation.expiresAt <= now || attestation.expiresAt <= attestation.issuedAt) fail('ATTESTATION_TIME_INVALID', 'attestation is not currently valid');
    this.entries.set(profile.sourceId, { profile: structuredClone(profile), attestation: structuredClone(attestation), state: 'active' });
    return this.evaluate(profile.sourceId, { now, observations: {} });
  }

  updateProfile(profile) {
    const entry = this.entries.get(profile.sourceId);
    if (!entry) fail('UNKNOWN_SOURCE', 'source is not registered');
    entry.profile = structuredClone(profile);
  }

  evaluate(sourceId, { now, observations = {} }) {
    const entry = this.entries.get(sourceId);
    if (!entry) fail('UNKNOWN_SOURCE', 'source is not registered');
    const { profile, attestation } = entry;
    const reasons = [];
    if (attestation.sourceId !== sourceId) reasons.push('source_binding_mismatch');
    if (attestation.connectorVersion !== profile.connectorVersion) reasons.push('connector_version_changed');
    if (attestation.profileDigest !== capabilityDigest(profile)) reasons.push('capability_profile_changed');
    if (!attestation.signatureVerified || !attestation.issuerScopeValid || !attestation.keyNotRevoked) reasons.push('attestation_trust_revoked');
    if (!Number.isFinite(now) || now < attestation.issuedAt || now >= attestation.expiresAt) reasons.push('attestation_expired');
    if (observations.replayProbePassed === false) reasons.push('replay_probe_failed');
    if (observations.highWatermarkMonotonic === false) reasons.push('source_cursor_rollback');
    if (observations.sinkPermissionRevisionEnforced === false) reasons.push('sink_authority_probe_failed');
    const caps = profile.capabilities;
    const replayContract = caps.delivery === 'push_durable' && caps.durableLog && caps.replayFromCursor && caps.partitionedSequence && caps.highWatermark && caps.stableEventIdentity;
    const retentionAdequate = replayContract && caps.replayRetentionSeconds >= caps.maxExpectedDisconnectSeconds;
    const trustedEvent = caps.signedEvents && attestation.signatureVerified && attestation.issuerScopeValid && attestation.keyNotRevoked;
    const liveAuthority = caps.permissionRevisionAtSink && observations.sinkPermissionRevisionEnforced !== false;
    if (caps.delivery === 'poll') reasons.push('poll_only_source');
    else if (!replayContract) reasons.push('no_durable_replay_contract');
    else if (!retentionAdequate) reasons.push('replay_retention_insufficient');
    if (!trustedEvent) reasons.push('event_provenance_untrusted');
    if (!liveAuthority) reasons.push('current_sink_authority_unverified');

    const blocked = reasons.some(x => ['source_binding_mismatch','connector_version_changed','capability_profile_changed','attestation_trust_revoked','attestation_expired','replay_probe_failed','source_cursor_rollback','sink_authority_probe_failed'].includes(x));
    const edgeRecovery = caps.delivery === 'poll' ? 'not_supported' : !replayContract ? 'best_effort_wakeup_only' : !retentionAdequate ? 'replayable_but_retention_insufficient' : !trustedEvent ? 'bounded_replay_untrusted_provenance' : 'bounded_replay_with_authenticated_provenance';
    const status = blocked ? 'inconclusive' : reasons.length ? 'degraded' : 'active';
    const strongest = status !== 'active' ? 'no_unqualified_edge_recall_claim' : edgeRecovery === 'bounded_replay_with_authenticated_provenance' && liveAuthority ? 'bounded_replay_with_authenticated_provenance_and_live_authority' : edgeRecovery;
    entry.state = status;
    return Object.freeze({ sourceId, status, edgeRecovery, strongestSupportedClaim: strongest, attestationId: attestation.attestationId, attestationExpiresAt: attestation.expiresAt, reasons, capabilityDigest: capabilityDigest(profile) });
  }
}

function makeProfile(overrides = {}) {
  return {
    sourceId: 'urn:service:calendar-A',
    connectorVersion: '3.4.1',
    capabilities: {
      delivery: 'push_durable', durableLog: true, replayFromCursor: true, partitionedSequence: true,
      highWatermark: true, stableEventIdentity: true, signedEvents: true,
      replayRetentionSeconds: 86400, maxExpectedDisconnectSeconds: 3600,
      permissionRevisionAtSink: true,
    },
    ...overrides,
  };
}
function attest(profile, overrides = {}) {
  return {
    attestationId: 'attest-' + profile.connectorVersion,
    sourceId: profile.sourceId,
    connectorVersion: profile.connectorVersion,
    profileDigest: capabilityDigest(profile),
    issuedAt: 100,
    expiresAt: 200,
    signatureVerified: true,
    issuerScopeValid: true,
    keyNotRevoked: true,
    conformanceProbePassed: true,
    ...overrides,
  };
}

const cases = [];
function test(name, fn) { fn(); cases.push({ name, result: 'pass' }); }
function expect(code, fn) { assert.throws(fn, e => e && e.code === code, 'expected ' + code); }

test('fresh trusted attestation establishes active bounded replay capability', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  const result = registry.register({ profile: p, attestation: attest(p), now: 110 });
  assert.equal(result.status, 'active');
  assert.equal(result.strongestSupportedClaim, 'bounded_replay_with_authenticated_provenance_and_live_authority');
});

test('attestation expires at the boundary and stops unqualified edge claims', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  registry.register({ profile: p, attestation: attest(p), now: 110 });
  const result = registry.evaluate(p.sourceId, { now: 200 });
  assert.equal(result.status, 'inconclusive');
  assert.ok(result.reasons.includes('attestation_expired'));
  assert.equal(result.strongestSupportedClaim, 'no_unqualified_edge_recall_claim');
});

test('connector version change invalidates old capability lease', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  registry.register({ profile: p, attestation: attest(p), now: 110 });
  registry.updateProfile({ ...p, connectorVersion: '3.5.0' });
  const result = registry.evaluate(p.sourceId, { now: 120 });
  assert.ok(result.reasons.includes('connector_version_changed'));
  assert.equal(result.status, 'inconclusive');
});

test('capability profile change invalidates old attestation digest', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  registry.register({ profile: p, attestation: attest(p), now: 110 });
  registry.updateProfile({ ...p, capabilities: { ...p.capabilities, replayRetentionSeconds: 60 } });
  const result = registry.evaluate(p.sourceId, { now: 120 });
  assert.ok(result.reasons.includes('capability_profile_changed'));
  assert.equal(result.status, 'inconclusive');
});

test('failed replay conformance probe downgrades the objective to inconclusive', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  registry.register({ profile: p, attestation: attest(p), now: 110 });
  const result = registry.evaluate(p.sourceId, { now: 120, observations: { replayProbePassed: false } });
  assert.ok(result.reasons.includes('replay_probe_failed'));
  assert.equal(result.status, 'inconclusive');
});

test('source high-watermark rollback is a blocking capability failure', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  registry.register({ profile: p, attestation: attest(p), now: 110 });
  const result = registry.evaluate(p.sourceId, { now: 120, observations: { highWatermarkMonotonic: false } });
  assert.ok(result.reasons.includes('source_cursor_rollback'));
  assert.equal(result.status, 'inconclusive');
});

test('sink permission probe failure blocks active authority guarantee', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  registry.register({ profile: p, attestation: attest(p), now: 110 });
  const result = registry.evaluate(p.sourceId, { now: 120, observations: { sinkPermissionRevisionEnforced: false } });
  assert.ok(result.reasons.includes('sink_authority_probe_failed'));
  assert.equal(result.status, 'inconclusive');
});

test('short source retention is degraded rather than misrepresented as full coverage', () => {
  const p = makeProfile({ capabilities: { ...makeProfile().capabilities, replayRetentionSeconds: 300 } }), registry = new CapabilityLeaseRegistry();
  const result = registry.register({ profile: p, attestation: attest(p), now: 110 });
  assert.equal(result.status, 'degraded');
  assert.equal(result.edgeRecovery, 'replayable_but_retention_insufficient');
  assert.equal(result.strongestSupportedClaim, 'no_unqualified_edge_recall_claim');
});

test('poll-only source is an explicit degraded capability, not event wake', () => {
  const base = makeProfile();
  const p = { ...base, capabilities: { ...base.capabilities, delivery: 'poll', durableLog: false, replayFromCursor: false, partitionedSequence: false, highWatermark: false, signedEvents: false } };
  const registry = new CapabilityLeaseRegistry();
  const result = registry.register({ profile: p, attestation: attest(p), now: 110 });
  assert.equal(result.status, 'degraded');
  assert.equal(result.edgeRecovery, 'not_supported');
});

test('signature/key trust revocation blocks capability lease', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  const a = attest(p, { keyNotRevoked: false });
  expect('ATTESTATION_UNTRUSTED', () => registry.register({ profile: p, attestation: a, now: 110 }));
});

test('unverified event signature cannot establish authenticated source capability', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  expect('ATTESTATION_UNTRUSTED', () => registry.register({ profile: p, attestation: attest(p, { signatureVerified: false }), now: 110 }));
});

test('attestation for another source cannot be transplanted', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  expect('SOURCE_BINDING_MISMATCH', () => registry.register({ profile: p, attestation: attest(p, { sourceId: 'urn:attacker:source' }), now: 110 }));
});

test('attestation for different connector version cannot be transplanted', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  expect('CONNECTOR_VERSION_MISMATCH', () => registry.register({ profile: p, attestation: attest(p, { connectorVersion: '1.0.0' }), now: 110 }));
});

test('new valid attestation restores capability after connector update', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  registry.register({ profile: p, attestation: attest(p), now: 110 });
  const upgraded = { ...p, connectorVersion: '3.5.0' };
  registry.updateProfile(upgraded);
  const degraded = registry.evaluate(p.sourceId, { now: 120 });
  assert.equal(degraded.status, 'inconclusive');
  const restored = registry.register({ profile: upgraded, attestation: attest(upgraded, { attestationId: 'attest-3.5.0', issuedAt: 121, expiresAt: 221 }), now: 122 });
  assert.equal(restored.status, 'active');
});

test('attestation cannot be registered before issue time or after expiry', () => {
  const p = makeProfile(), registry = new CapabilityLeaseRegistry();
  expect('ATTESTATION_TIME_INVALID', () => registry.register({ profile: p, attestation: attest(p, { issuedAt: 120, expiresAt: 200 }), now: 110 }));
  expect('ATTESTATION_TIME_INVALID', () => registry.register({ profile: p, attestation: attest(p), now: 200 }));
});

console.log(JSON.stringify({
  schema: 'sfeb-mon003-capability-lease-report-v1',
  status: 'passed',
  deterministic_cases: cases.length,
  cases,
  invariants: {
    source_capability_is_time_bounded: 'pass',
    connector_or_profile_change_invalidates_attestation: 'pass',
    replay_or_cursor_probe_failure_blocks_edge_guarantee: 'pass',
    key_revocation_and_signature_trust_required: 'pass',
    capability_recovery_requires_fresh_attestation: 'pass',
    guarantees_downgrade_instead_of_silent_overclaim: 'pass',
  },
  limitations: [
    'Pure in-memory capability-lease model; no production signature verifier, key registry, source probe or scheduler is exercised.',
    'The booleans representing signature, issuer and conformance checks are trusted test inputs, not cryptographic evidence.',
    'A production adapter must persist and independently verify capability attestations, probe receipts and version changes.',
  ],
}, null, 2));
