#!/usr/bin/env node
// SFEB MON-003 DSSE receipt trust preflight v1.
// Executes real Ed25519 sign/verify over DSSE PAE bytes using Node's built-in crypto.
// Trust-store contents are test fixtures; this is not a production PKI or HSM integration.
import assert from 'node:assert/strict';
import {
  createHash,
  createPublicKey,
  generateKeyPairSync,
  sign as cryptoSign,
  verify as cryptoVerify,
} from 'node:crypto';

const PAYLOAD_TYPE = 'application/vnd.luminous.sfeb.source-receipt.v1+json';
const NOW = 1_800_000_000;
const fail = (code, message) => {
  const error = new Error(message);
  error.code = code;
  throw error;
};
const sha256 = (bytes) => createHash('sha256').update(bytes).digest('hex');

// DSSE v1 PAE: DSSEv1 SP LEN(type) SP type SP LEN(body) SP body.
// Lengths are UTF-8 byte lengths, not JavaScript UTF-16 code-unit counts.
function pae(payloadType, body) {
  if (typeof payloadType !== 'string' || payloadType.length === 0) {
    fail('INVALID_PAYLOAD_TYPE', 'payload type must be a non-empty string');
  }
  if (!Buffer.isBuffer(body)) fail('INVALID_PAYLOAD_BYTES', 'payload must be bytes');
  const typeBytes = Buffer.from(payloadType, 'utf8');
  return Buffer.concat([
    Buffer.from(`DSSEv1 ${typeBytes.length} `, 'ascii'),
    typeBytes,
    Buffer.from(` ${body.length} `, 'ascii'),
    body,
  ]);
}

function encodeEnvelope(payloadType, payloadBytes, keyId, privateKey) {
  const message = pae(payloadType, payloadBytes);
  const signature = cryptoSign(null, message, privateKey);
  return {
    payloadType,
    payload: payloadBytes.toString('base64'),
    signatures: [{ keyid: keyId, sig: signature.toString('base64') }],
  };
}

function decodeCanonicalBase64(value, code) {
  if (typeof value !== 'string' || value.length === 0) fail(code, 'base64 field missing');
  const bytes = Buffer.from(value, 'base64');
  if (bytes.toString('base64') !== value) fail(code, 'base64 field is malformed or non-canonical');
  return bytes;
}

class TrustStore {
  constructor() {
    this.keys = new Map();
  }

  register({ keyId, publicKeyPem, issuerId, tenantIds, sourceIds, notBefore, notAfter }) {
    if (!keyId || !issuerId || typeof publicKeyPem !== 'string' ||
        !Array.isArray(tenantIds) || tenantIds.length === 0 ||
        !Array.isArray(sourceIds) || sourceIds.length === 0 ||
        !Number.isSafeInteger(notBefore) || !Number.isSafeInteger(notAfter) || notAfter <= notBefore) {
      fail('INVALID_TRUST_RECORD', 'trust record is incomplete or has invalid validity');
    }
    if (this.keys.has(keyId)) fail('DUPLICATE_KEY_ID', 'key id already registered');
    this.keys.set(keyId, {
      keyId,
      publicKeyPem,
      issuerId,
      tenantIds: new Set(tenantIds),
      sourceIds: new Set(sourceIds),
      notBefore,
      notAfter,
      revokedAt: null,
    });
  }

  revoke(keyId, at) {
    const record = this.keys.get(keyId);
    if (!record) fail('UNKNOWN_KEY', 'cannot revoke an unknown key');
    if (!Number.isSafeInteger(at)) fail('INVALID_REVOCATION_TIME', 'revocation time must be an integer epoch');
    record.revokedAt = at;
  }

  get(keyId) {
    return this.keys.get(keyId) ?? null;
  }
}

function validateReceipt(receipt, { expectedTenantId, expectedSource, now, record }) {
  if (!receipt || typeof receipt !== 'object' || Array.isArray(receipt)) {
    fail('INVALID_RECEIPT', 'payload must be a JSON object');
  }
  const requiredStrings = ['tenantId', 'source', 'eventId', 'resourceId', 'actionType', 'issuerId'];
  for (const field of requiredStrings) {
    if (typeof receipt[field] !== 'string' || receipt[field].length === 0) {
      fail('INVALID_RECEIPT', `receipt field ${field} is required`);
    }
  }
  if (!Number.isSafeInteger(receipt.sequence) || receipt.sequence < 1 ||
      !Number.isSafeInteger(receipt.issuedAt) || !Number.isSafeInteger(receipt.expiresAt) ||
      !Number.isSafeInteger(receipt.occurredAt) || receipt.expiresAt <= receipt.issuedAt) {
    fail('INVALID_RECEIPT', 'receipt sequence and validity timestamps must be integer epochs');
  }
  if (receipt.tenantId !== expectedTenantId) fail('TENANT_SCOPE_MISMATCH', 'receipt tenant does not match requested tenant');
  if (receipt.source !== expectedSource) fail('SOURCE_SCOPE_MISMATCH', 'receipt source does not match requested source');
  if (receipt.issuerId !== record.issuerId) fail('ISSUER_IDENTITY_MISMATCH', 'payload issuer does not match trusted key owner');
  if (!record.tenantIds.has(receipt.tenantId)) fail('KEY_TENANT_SCOPE_DENIED', 'trusted key is not authorized for this tenant');
  if (!record.sourceIds.has(receipt.source)) fail('KEY_SOURCE_SCOPE_DENIED', 'trusted key is not authorized for this source');
  if (receipt.issuedAt > now || receipt.occurredAt > now) fail('RECEIPT_FROM_FUTURE', 'receipt issue/occurrence time is in the future');
  if (receipt.expiresAt <= now) fail('RECEIPT_EXPIRED', 'source receipt has expired');
}

function verifyEnvelope(envelope, { trustStore, expectedTenantId, expectedSource, now }) {
  if (!envelope || typeof envelope !== 'object' || Array.isArray(envelope)) {
    fail('INVALID_ENVELOPE', 'DSSE envelope must be an object');
  }
  if (envelope.payloadType !== PAYLOAD_TYPE) fail('UNEXPECTED_PAYLOAD_TYPE', 'payload type is not the registered source-receipt profile');
  const payloadBytes = decodeCanonicalBase64(envelope.payload, 'INVALID_PAYLOAD_BASE64');
  if (!Array.isArray(envelope.signatures) || envelope.signatures.length !== 1) {
    fail('INVALID_SIGNATURE_SET', 'this profile requires exactly one DSSE signature');
  }
  const signatureEntry = envelope.signatures[0];
  if (!signatureEntry || typeof signatureEntry.keyid !== 'string' || signatureEntry.keyid.length === 0) {
    fail('MISSING_KEY_ID', 'this profile requires keyid as a lookup hint');
  }
  const signatureBytes = decodeCanonicalBase64(signatureEntry.sig, 'INVALID_SIGNATURE_BASE64');
  const record = trustStore.get(signatureEntry.keyid);
  if (!record) fail('UNKNOWN_KEY', 'keyid did not resolve to a trusted key record');
  if (!Number.isSafeInteger(now)) fail('INVALID_VERIFIER_TIME', 'verifier time must be integer epoch seconds');
  if (now < record.notBefore || now >= record.notAfter) fail('KEY_OUTSIDE_VALIDITY', 'signing key is outside its trusted validity window');
  if (record.revokedAt !== null && now >= record.revokedAt) fail('KEY_REVOKED', 'signing key has been revoked');

  // keyid is an unauthenticated hint. Authorization comes from the trust record,
  // and integrity comes from verifying the DSSE signature over type + exact bytes.
  let publicKey;
  try {
    publicKey = createPublicKey(record.publicKeyPem);
  } catch {
    fail('INVALID_TRUST_KEY', 'trusted public key could not be parsed');
  }
  if (publicKey.asymmetricKeyType !== 'ed25519') fail('UNSUPPORTED_KEY_TYPE', 'this profile accepts Ed25519 keys only');
  const signatureValid = cryptoVerify(null, pae(envelope.payloadType, payloadBytes), publicKey, signatureBytes);
  if (!signatureValid) fail('INVALID_SIGNATURE', 'DSSE signature verification failed');

  let receipt;
  try {
    receipt = JSON.parse(payloadBytes.toString('utf8'));
  } catch {
    fail('INVALID_RECEIPT_JSON', 'verified payload is not valid UTF-8 JSON');
  }
  validateReceipt(receipt, { expectedTenantId, expectedSource, now, record });
  return {
    receipt,
    proof: {
      payloadType: envelope.payloadType,
      payloadSha256: sha256(payloadBytes),
      keyIdHint: signatureEntry.keyid,
      issuerId: record.issuerId,
      signatureVerified: true,
      keyStatus: 'active-at-verification-time',
    },
  };
}

function makeKeys() {
  const trusted = generateKeyPairSync('ed25519');
  const attacker = generateKeyPairSync('ed25519');
  return { trusted, attacker };
}
function baseReceipt(overrides = {}) {
  return {
    schema: 'sfeb.source-receipt.v1',
    tenantId: 'tenant-A',
    source: 'urn:example:capacity-service',
    eventId: 'evt-000042',
    sequence: 42,
    occurredAt: NOW - 30,
    issuedAt: NOW - 10,
    expiresAt: NOW + 300,
    resourceId: 'slot-17',
    actionType: 'reserve',
    issuerId: 'capacity-service-prod',
    ...overrides,
  };
}
function makeContext() {
  const keys = makeKeys();
  const trustStore = new TrustStore();
  trustStore.register({
    keyId: 'key-prod-2026-01',
    publicKeyPem: keys.trusted.publicKey.export({ type: 'spki', format: 'pem' }).toString(),
    issuerId: 'capacity-service-prod',
    tenantIds: ['tenant-A'],
    sourceIds: ['urn:example:capacity-service'],
    notBefore: NOW - 1000,
    notAfter: NOW + 1000,
  });
  const bytes = Buffer.from(JSON.stringify(baseReceipt()), 'utf8');
  const envelope = encodeEnvelope(PAYLOAD_TYPE, bytes, 'key-prod-2026-01', keys.trusted.privateKey);
  return { keys, trustStore, bytes, envelope, verify: (env = envelope, opts = {}) => verifyEnvelope(env, {
    trustStore,
    expectedTenantId: 'tenant-A',
    expectedSource: 'urn:example:capacity-service',
    now: NOW,
    ...opts,
  }) };
}

const cases = [];
function test(name, fn) { fn(); cases.push({ name, result: 'pass' }); }
function expect(code, fn) { assert.throws(fn, (error) => error && error.code === code, `expected ${code}`); }

// PAE byte-length vector: the type contains a multi-byte character.
test('DSSE PAE encodes UTF-8 byte lengths and exact payload bytes', () => {
  const body = Buffer.from('payload-✓', 'utf8');
  const type = 'application/example+json;profile=✓';
  const bytes = Buffer.from(type, 'utf8');
  const encoded = pae(type, body);
  const expected = Buffer.concat([Buffer.from(`DSSEv1 ${bytes.length} `), bytes, Buffer.from(` ${body.length} `), body]);
  assert.deepEqual(encoded, expected);
});
test('valid Ed25519 DSSE receipt verifies against issuer and scope trust record', () => {
  const c = makeContext();
  const v = c.verify();
  assert.equal(v.proof.signatureVerified, true);
  assert.equal(v.receipt.eventId, 'evt-000042');
});
test('payload-byte tampering fails signature verification', () => {
  const c = makeContext();
  const changed = Buffer.from(c.bytes); changed[0] ^= 0x01;
  const altered = { ...c.envelope, payload: changed.toString('base64') };
  expect('INVALID_SIGNATURE', () => c.verify(altered));
});
test('payload-type substitution is rejected', () => {
  const c = makeContext();
  expect('UNEXPECTED_PAYLOAD_TYPE', () => c.verify({ ...c.envelope, payloadType: 'application/json' }));
});
test('signature-byte tampering fails verification', () => {
  const c = makeContext();
  const sig = Buffer.from(c.envelope.signatures[0].sig, 'base64'); sig[0] ^= 0x01;
  const altered = { ...c.envelope, signatures: [{ ...c.envelope.signatures[0], sig: sig.toString('base64') }] };
  expect('INVALID_SIGNATURE', () => c.verify(altered));
});
test('unknown keyid hint does not resolve to trusted authority', () => {
  const c = makeContext();
  const altered = { ...c.envelope, signatures: [{ ...c.envelope.signatures[0], keyid: 'untrusted-key' }] };
  expect('UNKNOWN_KEY', () => c.verify(altered));
});
test('attacker signature cannot be relabeled with an allowed keyid', () => {
  const c = makeContext();
  const body = Buffer.from(JSON.stringify(baseReceipt()), 'utf8');
  const attack = encodeEnvelope(PAYLOAD_TYPE, body, 'key-prod-2026-01', c.keys.attacker.privateKey);
  expect('INVALID_SIGNATURE', () => c.verify(attack));
});
test('revoked signing key is rejected even when the signature is valid', () => {
  const c = makeContext(); c.trustStore.revoke('key-prod-2026-01', NOW - 1);
  expect('KEY_REVOKED', () => c.verify());
});
test('key outside not-before/not-after interval is rejected', () => {
  const c = makeContext();
  const record = c.trustStore.get('key-prod-2026-01'); record.notBefore = NOW + 1;
  expect('KEY_OUTSIDE_VALIDITY', () => c.verify());
});
test('expired signing key is rejected even for an intact envelope', () => {
  const c = makeContext(); c.trustStore.get('key-prod-2026-01').notAfter = NOW;
  expect('KEY_OUTSIDE_VALIDITY', () => c.verify());
});
test('trusted key cannot sign for a tenant outside its trust scope', () => {
  const c = makeContext();
  const body = Buffer.from(JSON.stringify(baseReceipt({ tenantId: 'tenant-B' })), 'utf8');
  const env = encodeEnvelope(PAYLOAD_TYPE, body, 'key-prod-2026-01', c.keys.trusted.privateKey);
  expect('KEY_TENANT_SCOPE_DENIED', () => c.verify(env, { expectedTenantId: 'tenant-B' }));
});
test('trusted key cannot sign for a source outside its trust scope', () => {
  const c = makeContext();
  const body = Buffer.from(JSON.stringify(baseReceipt({ source: 'urn:attacker:source' })), 'utf8');
  const env = encodeEnvelope(PAYLOAD_TYPE, body, 'key-prod-2026-01', c.keys.trusted.privateKey);
  expect('KEY_SOURCE_SCOPE_DENIED', () => c.verify(env, { expectedSource: 'urn:attacker:source' }));
});
test('payload issuer must equal the registered key issuer', () => {
  const c = makeContext();
  const body = Buffer.from(JSON.stringify(baseReceipt({ issuerId: 'other-issuer' })), 'utf8');
  const env = encodeEnvelope(PAYLOAD_TYPE, body, 'key-prod-2026-01', c.keys.trusted.privateKey);
  expect('ISSUER_IDENTITY_MISMATCH', () => c.verify(env));
});
test('receipt tenant must equal the independently requested tenant', () => {
  const c = makeContext();
  expect('TENANT_SCOPE_MISMATCH', () => c.verify(c.envelope, { expectedTenantId: 'tenant-B' }));
});
test('receipt source must equal the independently requested source', () => {
  const c = makeContext();
  expect('SOURCE_SCOPE_MISMATCH', () => c.verify(c.envelope, { expectedSource: 'urn:other:source' }));
});
test('expired receipt is rejected after successful signature verification', () => {
  const c = makeContext();
  const body = Buffer.from(JSON.stringify(baseReceipt({ expiresAt: NOW })), 'utf8');
  const env = encodeEnvelope(PAYLOAD_TYPE, body, 'key-prod-2026-01', c.keys.trusted.privateKey);
  expect('RECEIPT_EXPIRED', () => c.verify(env));
});
test('future-issued or future-occurred receipt is rejected', () => {
  const c = makeContext();
  const body = Buffer.from(JSON.stringify(baseReceipt({ issuedAt: NOW + 1 })), 'utf8');
  const env = encodeEnvelope(PAYLOAD_TYPE, body, 'key-prod-2026-01', c.keys.trusted.privateKey);
  expect('RECEIPT_FROM_FUTURE', () => c.verify(env));
});
test('malformed payload base64 is rejected before JSON parsing', () => {
  const c = makeContext();
  expect('INVALID_PAYLOAD_BASE64', () => c.verify({ ...c.envelope, payload: '%%%not-base64%%%' }));
});
test('malformed signature base64 is rejected', () => {
  const c = makeContext();
  expect('INVALID_SIGNATURE_BASE64', () => c.verify({ ...c.envelope, signatures: [{ keyid: 'key-prod-2026-01', sig: '%%%' }] }));
});
test('multiple signatures fail the single-signer profile closed', () => {
  const c = makeContext();
  expect('INVALID_SIGNATURE_SET', () => c.verify({ ...c.envelope, signatures: [...c.envelope.signatures, ...c.envelope.signatures] }));
});
test('valid signature over non-JSON bytes is rejected before receipt use', () => {
  const c = makeContext();
  const env = encodeEnvelope(PAYLOAD_TYPE, Buffer.from('not-json', 'utf8'), 'key-prod-2026-01', c.keys.trusted.privateKey);
  expect('INVALID_RECEIPT_JSON', () => c.verify(env));
});
test('revocation after the supplied verification time does not backdate key invalidity', () => {
  const c = makeContext(); c.trustStore.revoke('key-prod-2026-01', NOW + 5);
  assert.equal(c.verify().proof.signatureVerified, true);
});

console.log(JSON.stringify({
  schema: 'sfeb-mon003-dsse-receipt-report-v1',
  status: 'passed',
  deterministic_cases: cases.length,
  cases,
  cryptography: { algorithm: 'Ed25519', envelope: 'DSSE v1', payload_type: PAYLOAD_TYPE, implementation: 'Node.js built-in crypto; actual sign/verify' },
  limitations: [
    'Ephemeral test keypairs and an in-memory trust store; no root-of-trust provisioning, HSM, certificate transparency, production key rotation service or durable revocation feed.',
    'Single-signer profile only. Multi-signer thresholds and cross-issuer consensus are not implemented.',
    'This verifies payload integrity and configured issuer/tenant/source authorization at a supplied time; it does not prove the source event itself is truthful.',
    'The test profile signs exact serialized JSON bytes and does not canonicalize JSON. Use RFC 8785/JCS only where semantic JSON equality across serializers is required.'
  ]
}, null, 2));
