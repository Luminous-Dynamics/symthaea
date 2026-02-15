/**
 * Example 19: Mobile Wallet SDK
 *
 * This example demonstrates how to use the Mobile module for:
 * - Creating and managing mobile wallets
 * - Biometric authentication
 * - QR code generation and scanning
 * - Credential requests and selective disclosure
 */

import { mobile } from '../src/index.js';

async function main() {
  console.log('=== Mycelix Mobile Wallet SDK Example ===\n');

  // =========================================================================
  // Part 1: Wallet Creation and Management
  // =========================================================================
  console.log('--- Part 1: Wallet Creation ---\n');

  const walletManager = mobile.createWalletManager();

  // Create a new wallet with biometric protection
  const walletMetadata = await walletManager.createWallet(
    'My Mycelix Wallet',
    '123456', // PIN
    {
      enableBiometrics: true,
      autoLockTimeout: 300000, // 5 minutes
      backupEnabled: true,
    }
  );

  console.log('Wallet created successfully!');
  console.log(`  Wallet ID: ${walletMetadata.walletId}`);
  console.log(`  Name: ${walletMetadata.name}`);
  console.log(`  Accounts: ${walletMetadata.accounts.length}`);
  console.log(`  Biometrics: ${walletMetadata.biometricsEnabled ? 'Enabled' : 'Disabled'}`);
  console.log(`  Status: ${walletMetadata.status}`);

  // =========================================================================
  // Part 2: Unlock and Account Management
  // =========================================================================
  console.log('\n--- Part 2: Account Management ---\n');

  // Unlock wallet with PIN
  console.log('Unlocking wallet with PIN...');
  await walletManager.unlock('123456');
  console.log('Wallet unlocked!\n');

  // Get default account
  const defaultAccount = walletManager.getDefaultAccount();
  console.log('Default account:');
  console.log(`  Account ID: ${defaultAccount.accountId}`);
  console.log(`  Agent ID: ${defaultAccount.agentId}`);
  console.log(`  Name: ${defaultAccount.name}`);
  console.log(`  Created: ${new Date(defaultAccount.createdAt).toLocaleDateString()}`);

  // Create additional accounts
  console.log('\nCreating additional accounts...');

  const tradingAccount = await walletManager.createAccount('Trading');
  console.log(`  Created: ${tradingAccount.name} (${tradingAccount.accountId})`);

  const savingsAccount = await walletManager.createAccount('Savings');
  console.log(`  Created: ${savingsAccount.name} (${savingsAccount.accountId})`);

  // List all accounts
  const accounts = walletManager.listAccounts();
  console.log(`\nTotal accounts: ${accounts.length}`);

  // =========================================================================
  // Part 3: Signing Transactions
  // =========================================================================
  console.log('\n--- Part 3: Transaction Signing ---\n');

  // Prepare transaction data
  const transaction = {
    action: 'transfer',
    from: defaultAccount.agentId,
    to: 'did:mycelix:recipient-alice-123',
    amount: 100,
    currency: 'MCX',
    memo: 'Payment for services',
    timestamp: Date.now(),
  };

  const txData = new TextEncoder().encode(JSON.stringify(transaction));
  console.log('Transaction to sign:');
  console.log(`  Action: ${transaction.action}`);
  console.log(`  Amount: ${transaction.amount} ${transaction.currency}`);
  console.log(`  To: ${transaction.to}`);

  // Sign the transaction
  console.log('\nSigning transaction...');
  const signature = await walletManager.sign(defaultAccount.accountId, txData);
  console.log(`  Signature: ${Buffer.from(signature.slice(0, 32)).toString('hex')}...`);
  console.log(`  Signature length: ${signature.length} bytes`);

  // Verify the signature
  const publicKey = walletManager.getPublicKey(defaultAccount.accountId);
  const isValid = await walletManager.verify(publicKey!, txData, signature);
  console.log(`  Verification: ${isValid ? 'VALID' : 'INVALID'}`);

  // =========================================================================
  // Part 4: Biometric Authentication
  // =========================================================================
  console.log('\n--- Part 4: Biometric Authentication ---\n');

  const biometricManager = mobile.createBiometricManager();

  // Check biometric availability
  const isAvailable = await biometricManager.isAvailable();
  console.log(`Biometrics available: ${isAvailable}`);

  if (isAvailable) {
    const capabilities = await biometricManager.getCapabilities();
    console.log('Capabilities:');
    console.log(`  Hardware supported: ${capabilities.hardwareSupported}`);
    console.log(`  Enrolled: ${capabilities.isEnrolled}`);
    console.log(`  Available types: ${capabilities.availableTypes.join(', ')}`);
    console.log(`  Security level: ${capabilities.securityLevel}`);

    // Authenticate
    console.log('\nAuthenticating...');
    const authResult = await biometricManager.authenticate('Confirm transaction');

    if (authResult.success) {
      console.log(`  Success! Authenticated via: ${authResult.biometricType}`);
    } else {
      console.log(`  Failed: ${mobile.biometricErrorToMessage(authResult.error!)}`);
    }
  }

  // =========================================================================
  // Part 5: QR Code Generation - Wallet Connect
  // =========================================================================
  console.log('\n--- Part 5: QR Code Generation ---\n');

  // Create wallet connect QR for dApp pairing
  console.log('Creating Wallet Connect QR...');
  const walletConnectPayload = mobile.createWalletConnectPayload(
    'marketplace-happ',
    'Mycelix Marketplace',
    ['read_profile', 'view_balance', 'sign_transactions'],
    'https://marketplace.mycelix.net/callback'
  );

  const walletConnectQR = mobile.encodeQRPayload(walletConnectPayload);
  console.log('Wallet Connect QR generated:');
  console.log(`  Type: ${walletConnectPayload.type}`);
  console.log(`  App: ${walletConnectPayload.appName}`);
  console.log(`  Permissions: ${walletConnectPayload.permissions?.join(', ')}`);
  console.log(`  QR Data length: ${walletConnectQR.length} chars`);
  console.log(`  QR Preview: ${walletConnectQR.slice(0, 50)}...`);

  // =========================================================================
  // Part 6: QR Code Generation - Payment Request
  // =========================================================================
  console.log('\n--- Part 6: Payment Request QR ---\n');

  const paymentPayload = mobile.createPaymentRequestPayload(
    'did:mycelix:merchant-coffee-shop',
    4.50,
    'USD',
    'Coffee Shop - Latte',
    'order-2026-01-18-001'
  );

  const paymentQR = mobile.encodeQRPayload(paymentPayload);
  console.log('Payment Request QR generated:');
  console.log(`  Merchant: ${paymentPayload.recipientId}`);
  console.log(`  Amount: $${paymentPayload.amount} ${paymentPayload.currency}`);
  console.log(`  Description: ${paymentPayload.description}`);
  console.log(`  Reference: ${paymentPayload.reference}`);
  console.log(`  QR Data length: ${paymentQR.length} chars`);

  // =========================================================================
  // Part 7: Credential Request Builder
  // =========================================================================
  console.log('\n--- Part 7: Credential Request ---\n');

  // Build a credential request for job application
  console.log('Building credential request for job application...\n');

  const credentialRequest = new mobile.CredentialRequestBuilder()
    // Required: Identity verification
    .request('IdentityCredential', true)
    // Required: Education with specific claims
    .requestClaims('EducationCredential', ['degree', 'institution', 'graduationYear'], true)
    // Optional: Work history
    .requestClaims('EmploymentCredential', ['employer', 'position', 'duration'], false)
    // Optional: Reputation with minimum threshold
    .requestWithPredicate('ReputationCredential', [
      { claim: 'trustScore', operator: '>=', value: 0.7 },
      { claim: 'totalInteractions', operator: '>=', value: 10 },
    ], false)
    .build();

  console.log('Credential request built:');
  console.log(`  Total credentials requested: ${credentialRequest.length}`);
  credentialRequest.forEach((cred, i) => {
    console.log(`\n  ${i + 1}. ${cred.type} (${cred.required ? 'Required' : 'Optional'})`);
    if (cred.claims) {
      console.log(`     Claims: ${cred.claims.join(', ')}`);
    }
    if (cred.predicates) {
      cred.predicates.forEach(p => {
        console.log(`     Predicate: ${p.claim} ${p.operator} ${p.value}`);
      });
    }
  });

  // Create QR payload for credential request
  const credRequestPayload = mobile.createCredentialRequestPayload(
    'did:mycelix:employer-tech-corp',
    'Tech Corp Job Application',
    credentialRequest
  );

  const credRequestQR = mobile.encodeQRPayload(credRequestPayload);
  console.log(`\nCredential Request QR generated (${credRequestQR.length} chars)`);

  // =========================================================================
  // Part 8: Processing Scanned QR Codes
  // =========================================================================
  console.log('\n--- Part 8: QR Code Processing ---\n');

  // Simulate scanning the payment QR
  console.log('Processing scanned QR code...');
  const scanResult = mobile.processScannedQR(paymentQR);

  if (scanResult.success) {
    console.log('QR scan successful!');
    console.log(`  Type: ${scanResult.payload.type}`);
    console.log(`  Action: ${mobile.getQRPayloadAction(scanResult.payload.type)}`);

    // Validate the payload
    const validation = mobile.validateQRPayload(scanResult.payload);
    console.log(`  Valid: ${validation.valid}`);

    if (!validation.valid) {
      console.log(`  Errors: ${validation.errors.join(', ')}`);
    }

    // Check expiration
    const isExpired = mobile.isPayloadExpired(scanResult.payload);
    console.log(`  Expired: ${isExpired}`);

    // Handle based on type
    switch (scanResult.payload.type) {
      case 'payment_request':
        console.log('\n  → Showing payment confirmation dialog...');
        console.log(`     Pay $${scanResult.payload.amount} to ${scanResult.payload.recipientId}?`);
        break;
      case 'wallet_connect':
        console.log('\n  → Showing permission request dialog...');
        console.log(`     ${scanResult.payload.appName} wants to connect`);
        break;
      case 'credential_request':
        console.log('\n  → Showing credential selection dialog...');
        console.log(`     Select credentials to share with ${scanResult.payload.verifierName}`);
        break;
    }
  } else {
    console.log(`QR scan failed: ${scanResult.error}`);
  }

  // =========================================================================
  // Part 9: Contact Exchange
  // =========================================================================
  console.log('\n--- Part 9: Contact Exchange ---\n');

  const contactPayload = mobile.createContactExchangePayload({
    agentId: defaultAccount.agentId,
    displayName: 'Alice Developer',
    publicKey: Buffer.from(publicKey || new Uint8Array(32)).toString('hex'),
    metadata: {
      organization: 'Mycelix Labs',
      role: 'Software Engineer',
      location: 'San Francisco',
    },
  });

  const contactQR = mobile.encodeQRPayload(contactPayload);
  console.log('Contact Exchange QR generated:');
  console.log(`  Name: ${contactPayload.contact?.displayName}`);
  console.log(`  Agent ID: ${contactPayload.contact?.agentId?.slice(0, 20)}...`);
  console.log(`  Organization: ${contactPayload.contact?.metadata?.organization}`);
  console.log(`  QR Data length: ${contactQR.length} chars`);

  // =========================================================================
  // Part 10: Wallet Export/Import (Backup)
  // =========================================================================
  console.log('\n--- Part 10: Wallet Backup ---\n');

  // Export wallet for backup
  console.log('Exporting wallet backup...');
  const backupPassword = 'secure-backup-password-2026';
  const backup = await walletManager.export(backupPassword);

  console.log('Wallet backup created:');
  console.log(`  Backup size: ${backup.length} bytes`);
  console.log(`  Encrypted: Yes (AES-GCM)`);
  console.log(`  Preview: ${Buffer.from(backup.slice(0, 20)).toString('hex')}...`);

  // Lock wallet
  console.log('\nLocking wallet...');
  walletManager.lock();
  console.log('Wallet locked.');

  // Simulate restore on new device
  console.log('\nRestoring wallet from backup...');
  const newWalletManager = mobile.createWalletManager();
  await newWalletManager.import(backup, backupPassword);
  await newWalletManager.unlock('123456');

  const restoredAccounts = newWalletManager.listAccounts();
  console.log(`Wallet restored! ${restoredAccounts.length} accounts recovered.`);

  // Clean up
  newWalletManager.lock();

  console.log('\n=== Mobile Wallet SDK Example Complete ===');
}

main().catch(console.error);
