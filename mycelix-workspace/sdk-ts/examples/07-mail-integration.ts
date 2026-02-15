/**
 * @mycelix/sdk Mail Integration Example
 *
 * Demonstrates email sender trust tracking, verification, and epistemic claims.
 *
 * Run with: npx tsx examples/07-mail-integration.ts
 */

import {
  getMailTrustService,
  MailTrustService,
  type SenderTrust,
  type EmailClaim,
} from '../src/integrations/mail/index.js';

// === Basic Sender Trust Tracking ===

console.log('=== Mail Trust Integration ===\n');

// Get the singleton service instance
const mailService = getMailTrustService();

// Simulate email interactions over time
const senders = {
  trusted: 'newsletter@trustedcompany.com',
  spam: 'prince@nigerian-lottery.com',
  unknown: 'first-time@newcontact.com',
};

// Build trust through positive interactions
console.log('Recording interactions...');
for (let i = 0; i < 10; i++) {
  mailService.recordInteraction(senders.trusted, true); // Opened, read, replied
}

// Record negative interactions for spam
for (let i = 0; i < 5; i++) {
  mailService.recordInteraction(senders.spam, false); // Marked as spam
}

// Check trust levels
console.log('\n--- Sender Trust Levels ---');
for (const [label, email] of Object.entries(senders)) {
  const trust = mailService.getSenderTrust(email);
  console.log(`${label}: ${email}`);
  console.log(`  Level: ${trust.level}`);
  console.log(`  Score: ${trust.score.toFixed(3)}`);
  console.log(`  Trustworthy: ${trust.trustworthy}`);
  console.log(`  Confidence: ${(trust.confidence * 100).toFixed(0)}%`);
  console.log();
}

// === Email Verification and Claims ===

console.log('--- Email Claims with Verification ---');

// Verified email with DKIM + SPF
const verifiedEmail = mailService.createEmailClaim({
  id: 'email-001',
  subject: 'Contract Signed - Project Alpha',
  from: senders.trusted,
  to: ['recipient@mycompany.com'],
  body: 'Please find attached the signed contract...',
  timestamp: Date.now(),
  verification: {
    dkimVerified: true,
    spfPassed: true,
    dmarcPassed: true,
  },
});

console.log('Verified Email Claim:');
console.log(`  Email ID: ${verifiedEmail.emailId}`);
console.log(`  Verification Type: ${verifiedEmail.verificationType}`);
console.log(`  Claim Content: ${verifiedEmail.claim.content}`);
console.log();

// Email with credential verification (higher trust)
const credentialEmail = mailService.createEmailClaim({
  id: 'email-002',
  subject: 'Identity Verified - KYC Complete',
  from: 'verification@identity-provider.com',
  to: ['user@example.com'],
  body: 'Your identity has been verified.',
  timestamp: Date.now(),
  verification: {
    dkimVerified: true,
    spfPassed: true,
    dmarcPassed: true,
    credentialVerified: true,
    senderDid: 'did:mycelix:identity-provider',
  },
});

console.log('Credential-Verified Email Claim:');
console.log(`  Verification Type: ${credentialEmail.verificationType}`);
console.log();

// Unverified email (spam-like)
const unverifiedEmail = mailService.createEmailClaim({
  id: 'email-003',
  subject: 'You Won $1,000,000!!!',
  from: senders.spam,
  to: ['victim@example.com'],
  body: 'Click here to claim your prize...',
  timestamp: Date.now(),
  verification: {
    dkimVerified: false,
    spfPassed: false,
    dmarcPassed: false,
  },
});

console.log('Unverified Email Claim:');
console.log(`  Verification Type: ${unverifiedEmail.verificationType}`);
console.log();

// === Quick Trust Checks ===

console.log('--- Quick Trust Checks ---');
console.log(`Trusted sender is trusted: ${mailService.isTrusted(senders.trusted)}`);
console.log(`Spam sender is trusted: ${mailService.isTrusted(senders.spam)}`);
console.log(`Unknown sender is trusted: ${mailService.isTrusted(senders.unknown)}`);
console.log();

// Custom threshold
console.log(`Trusted sender at 0.8 threshold: ${mailService.isTrusted(senders.trusted, 0.8)}`);

// === Real-World Pattern: Email Filtering ===

console.log('\n--- Email Filtering Example ---');

interface IncomingEmail {
  from: string;
  subject: string;
  verified: boolean;
}

const incomingEmails: IncomingEmail[] = [
  { from: senders.trusted, subject: 'Weekly Report', verified: true },
  { from: senders.spam, subject: 'FREE MONEY', verified: false },
  { from: senders.unknown, subject: 'Meeting Request', verified: true },
];

for (const email of incomingEmails) {
  const trust = mailService.getSenderTrust(email.from);

  let action: string;
  if (trust.level === 'verified' || trust.level === 'high') {
    action = '✅ INBOX';
  } else if (trust.level === 'unknown' && email.verified) {
    action = '📋 REVIEW';
  } else if (trust.level === 'low' || !email.verified) {
    action = '🚫 SPAM';
  } else {
    action = '📋 REVIEW';
  }

  console.log(`${action} | ${email.subject} (from: ${email.from})`);
}

console.log('\n=== Mail Integration Complete ===');
