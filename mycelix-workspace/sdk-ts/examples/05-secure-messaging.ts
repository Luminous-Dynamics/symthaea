/**
 * Secure Messaging Example
 *
 * Demonstrates the security module for cryptographic operations,
 * signed messages, rate limiting, and audit logging.
 *
 * Run with: npx ts-node examples/05-secure-messaging.ts
 */

import {
  // Security utilities
  signMessage,
  verifyMessage,
  createRateLimitedOperation,

  // Security primitives
  security,

  // Async utilities
  retry,
} from '../src/index.js';

async function main() {
  console.log('🍄 Secure Messaging Example\n');

  // Generate secure random values
  console.log('=== Secure Random Generation ===');

  const randomBytes = security.secureRandomBytes(32);
  console.log(`Random bytes (32): ${Buffer.from(randomBytes).toString('hex').slice(0, 32)}...`);

  const uuid = security.secureUUID();
  console.log(`Secure UUID: ${uuid}`);

  const randomInt = security.secureRandomInt(1, 100);
  console.log(`Random int (1-100): ${randomInt}`);

  const randomFloat = security.secureRandomFloat();
  console.log(`Random float (0-1): ${randomFloat.toFixed(6)}`);
  console.log();

  // Cryptographic hashing
  console.log('=== Cryptographic Hashing ===');

  const message = 'Hello, Mycelix!';
  const hash256 = await security.hashHex(message, 'SHA-256');
  const hash512 = await security.hashHex(message, 'SHA-512');

  console.log(`Message: "${message}"`);
  console.log(`SHA-256: ${hash256}`);
  console.log(`SHA-512: ${hash512.slice(0, 64)}...`);
  console.log();

  // HMAC for message authentication
  console.log('=== HMAC Authentication ===');

  const secretKey = security.secureRandomBytes(32);
  const hmacResult = await security.hmac(secretKey, message);
  console.log(`HMAC: ${Buffer.from(hmacResult).toString('hex')}`);

  const isValid = await security.verifyHmac(secretKey, message, hmacResult);
  console.log(`HMAC valid: ${isValid}`);

  // Tampered message
  const tamperedValid = await security.verifyHmac(secretKey, message + '!', hmacResult);
  console.log(`Tampered HMAC valid: ${tamperedValid}`);
  console.log();

  // Signed messages (utility)
  console.log('=== Signed Messages ===');

  const payload = {
    action: 'transfer',
    amount: 100,
    from: 'alice',
    to: 'bob',
    timestamp: Date.now(),
  };

  const signingKey = security.secureRandomBytes(32);
  const signedMsg = await signMessage(payload, signingKey);

  console.log('Signed message:');
  console.log(`  Payload: ${JSON.stringify(signedMsg.payload)}`);
  console.log(`  Signature: ${signedMsg.signature.slice(0, 32)}...`);
  console.log(`  Timestamp: ${new Date(signedMsg.timestamp).toISOString()}`);
  console.log();

  // Verify the signed message
  const verification = await verifyMessage(signedMsg, signingKey);
  console.log('Verification:');
  console.log(`  Valid: ${verification.valid}`);
  console.log(`  Expired: ${verification.expired}`);
  console.log(`  Tampered: ${verification.tampered}`);
  console.log();

  // Tamper detection
  console.log('=== Tamper Detection ===');

  const tamperedMsg = {
    ...signedMsg,
    payload: { ...signedMsg.payload, amount: 1000000 }, // Modified amount
  };

  const tamperedVerification = await verifyMessage(tamperedMsg, signingKey);
  console.log('Tampered message verification:');
  console.log(`  Valid: ${tamperedVerification.valid}`);
  console.log(`  Tampered: ${tamperedVerification.tampered}`);
  console.log();

  // Message expiration
  console.log('=== Message Expiration ===');

  const oldMessage = await signMessage({ data: 'old' }, signingKey);
  // Manually backdate
  oldMessage.timestamp = Date.now() - 60000; // 1 minute ago

  const expiredVerification = await verifyMessage(oldMessage, signingKey, 30000); // 30s max age
  console.log('Expired message (30s max age, 60s old):');
  console.log(`  Valid: ${expiredVerification.valid}`);
  console.log(`  Expired: ${expiredVerification.expired}`);
  console.log();

  // Rate limiting
  console.log('=== Rate Limiting ===');

  let apiCallCount = 0;
  const rateLimitedApi = createRateLimitedOperation(
    () => {
      apiCallCount++;
      return `API response #${apiCallCount}`;
    },
    3, // max 3 requests
    5000 // per 5 seconds
  );

  console.log('Rate limit: 3 requests per 5 seconds');
  console.log('Making 5 requests:');

  for (let i = 1; i <= 5; i++) {
    const result = await rateLimitedApi.execute();
    if (result) {
      console.log(`  Request ${i}: ${result}`);
    } else {
      console.log(`  Request ${i}: RATE LIMITED`);
    }
  }

  console.log('\nResetting rate limiter...');
  rateLimitedApi.reset();

  const afterReset = await rateLimitedApi.execute();
  console.log(`After reset: ${afterReset}`);
  console.log();

  // Timing-safe comparison
  console.log('=== Timing-Safe Comparison ===');

  const token1 = security.secureRandomBytes(32);
  const token2 = new Uint8Array(token1); // Copy
  const token3 = security.secureRandomBytes(32); // Different

  console.log('Comparing tokens (constant-time):');
  console.log(`  token1 === token2: ${security.constantTimeEqual(token1, token2)}`);
  console.log(`  token1 === token3: ${security.constantTimeEqual(token1, token3)}`);
  console.log();

  // Secure secrets
  console.log('=== Secure Secrets ===');

  const secret = new security.SecureSecret(
    new TextEncoder().encode('my-secret-api-key')
  );

  console.log('Secret operations:');
  console.log(`  Is destroyed: ${secret.isDestroyed}`);
  console.log(`  Use count: ${secret.useCount}`);

  secret.use((value) => {
    console.log(`  Value length: ${value.length} bytes`);
  });

  console.log(`  Use count after use: ${secret.useCount}`);

  secret.destroy();
  console.log(`  Is destroyed after destroy(): ${secret.isDestroyed}`);

  try {
    secret.use(() => {});
  } catch (e) {
    console.log(`  Use after destroy: Error - ${(e as Error).message}`);
  }
  console.log();

  // BFT validation
  console.log('=== BFT Validation ===');

  const scenarios = [
    { total: 4, f: 1 }, // 4 nodes, 1 Byzantine
    { total: 7, f: 2 }, // 7 nodes, 2 Byzantine
    { total: 10, f: 3 }, // 10 nodes, 3 Byzantine
    { total: 100, f: 33 }, // 100 nodes, 33 Byzantine
  ];

  console.log('BFT scenarios (n = 3f + 1):');
  for (const { total, f } of scenarios) {
    const maxF = security.maxByzantineFailures(total);
    const quorumSize = Math.ceil((total + maxF + 1) / 2);

    console.log(`  n=${total}: max Byzantine=${maxF}, quorum=${quorumSize}`);

    // Check if we have quorum
    const hasQuorum = security.hasQuorum(total, quorumSize, maxF);
    console.log(`    Has quorum with ${quorumSize} responses: ${hasQuorum}`);
  }
  console.log();

  // Retry with exponential backoff
  console.log('=== Retry with Exponential Backoff ===');

  let attempts = 0;
  const unreliableOperation = async () => {
    attempts++;
    console.log(`  Attempt ${attempts}...`);
    if (attempts < 3) {
      throw new Error('Temporary failure');
    }
    return 'Success!';
  };

  try {
    const result = await retry(unreliableOperation, {
      maxAttempts: 5,
      initialDelay: 100,
      backoffFactor: 2,
    });
    console.log(`  Result: ${result}`);
  } catch (e) {
    console.log(`  Failed after max attempts: ${(e as Error).message}`);
  }
  console.log();

  // Security audit logging
  console.log('=== Security Audit Logging ===');

  security.securityAudit.log({
    type: security.SecurityEventType.AUTH_SUCCESS,
    message: 'User authenticated successfully',
    metadata: { userId: 'alice', method: '2FA' },
  });

  security.securityAudit.log({
    type: security.SecurityEventType.RATE_LIMIT_EXCEEDED,
    message: 'Rate limit exceeded for API endpoint',
    severity: 'warning',
    metadata: { endpoint: '/api/transfer', ip: '192.168.1.100' },
  });

  security.securityAudit.log({
    type: security.SecurityEventType.CRYPTO_OPERATION,
    message: 'Message signed and sent',
    metadata: { recipient: 'bob' },
  });

  const recentEvents = security.securityAudit.getEvents();
  console.log('Recent security events:');
  for (const event of recentEvents.slice(-3)) {
    console.log(`  [${event.type}] ${event.message}`);
  }

  const authEvents = security.securityAudit.getEventsByType(security.SecurityEventType.AUTH_SUCCESS);
  console.log(`\nAuth success events: ${authEvents.length}`);
}

main().catch(console.error);
