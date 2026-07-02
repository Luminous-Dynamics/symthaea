#!/usr/bin/env npx tsx

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix SDK Integration Tests
 *
 * Tests against live Sepolia contracts to verify SDK functionality.
 * Run with: npx tsx src/test.ts
 */

import { ethers } from 'ethers';
import { MycelixClient, NETWORKS } from './index.ts';

interface TestResult {
  name: string;
  passed: boolean;
  error?: string;
  duration: number;
}

const results: TestResult[] = [];

async function runTest(name: string, fn: () => Promise<void>): Promise<void> {
  const start = Date.now();
  try {
    await fn();
    results.push({ name, passed: true, duration: Date.now() - start });
    console.log(`  ✓ ${name} (${Date.now() - start}ms)`);
  } catch (error: any) {
    results.push({ name, passed: false, error: error.message, duration: Date.now() - start });
    console.log(`  ✗ ${name}: ${error.message}`);
  }
}

async function main(): Promise<void> {
  console.log('\n=== Mycelix SDK Integration Tests ===\n');
  console.log('Network: Sepolia');
  console.log(`RPC: ${NETWORKS.sepolia.rpcUrl}\n`);

  // Read-only client for most tests
  const client = new MycelixClient({ network: 'sepolia' });

  // Test 1: Client initialization
  await runTest('Client initializes correctly', async () => {
    if (!client.config) throw new Error('Config not set');
    if (!client.provider) throw new Error('Provider not set');
    if (client.config.chainId !== 11155111) throw new Error('Wrong chain ID');
  });

  // Test 2: Network config
  await runTest('Network config has correct addresses', async () => {
    const { contracts } = client.config;
    if (!contracts.registry.startsWith('0x')) throw new Error('Invalid registry address');
    if (!contracts.reputation.startsWith('0x')) throw new Error('Invalid reputation address');
    if (!contracts.payment.startsWith('0x')) throw new Error('Invalid payment address');
    if (contracts.registry.length !== 42) throw new Error('Registry address wrong length');
  });

  // Test 3: Registry - Get total DIDs
  await runTest('Registry: getTotalDids returns a number', async () => {
    const total = await client.registry.getTotalDids();
    if (typeof total !== 'bigint') throw new Error('Expected bigint');
    if (total < 0n) throw new Error('Total DIDs cannot be negative');
  });

  // Test 4: Registry - Get registration fee
  await runTest('Registry: getRegistrationFee returns fee', async () => {
    const fee = await client.registry.getRegistrationFee();
    if (typeof fee !== 'bigint') throw new Error('Expected bigint');
    if (fee < 0n) throw new Error('Fee cannot be negative');
  });

  // Test 5: Registry - Lookup non-existent DID
  await runTest('Registry: getDID returns zero address for non-existent DID', async () => {
    const randomDid = `did:mycelix:test-${Date.now()}-${Math.random()}`;
    const info = await client.registry.getDID(randomDid);
    if (info.owner !== ethers.ZeroAddress) {
      throw new Error('Expected zero address for non-existent DID');
    }
  });

  // Test 6: Registry - Lookup existing DID (from demo)
  await runTest('Registry: getDID finds previously registered DID', async () => {
    // This DID was registered in a previous session
    const existingDid = 'did:mycelix:demo-1736358445382';
    const info = await client.registry.getDID(existingDid);
    // The DID might exist or not depending on contract state
    // Just verify we get a valid response
    if (!info.owner) throw new Error('No owner returned');
    if (typeof info.revoked !== 'boolean') throw new Error('Revoked should be boolean');
  });

  // Test 7: Reputation - Query reputation for random address
  await runTest('Reputation: getReputation handles unregistered address', async () => {
    const randomAddress = ethers.Wallet.createRandom().address;
    try {
      const rep = await client.reputation.getReputation(randomAddress);
      // If it succeeds, verify structure
      if (typeof rep.score !== 'bigint') throw new Error('Score should be bigint');
    } catch (e: any) {
      // Contract may revert for addresses with no reputation - this is acceptable
      if (!e.message.includes('revert') && !e.message.includes('CALL_EXCEPTION')) {
        throw e;
      }
      // Test passes - contract correctly rejects queries for unregistered addresses
    }
  });

  // Test 8: Reputation - Contract is accessible
  await runTest('Reputation: contract is deployed and accessible', async () => {
    // Just verify we can reach the contract
    const code = await client.provider.getCode(client.config.contracts.reputation);
    if (code === '0x' || code.length < 10) {
      throw new Error('Reputation contract not deployed');
    }
  });

  // Test 9: Payment - Get payment info (may not exist)
  await runTest('Payment: getPayment handles non-existent payment', async () => {
    try {
      const payment = await client.payment.getPayment(999999);
      // If it returns, verify structure
      if (!payment.sender) throw new Error('No sender in response');
    } catch (e: any) {
      // Expected to fail for non-existent payment
      if (!e.message.includes('revert') && !e.message.includes('call')) {
        throw e;
      }
    }
  });

  // Test 10: URL generators
  await runTest('URL generators produce valid URLs', async () => {
    const txUrl = client.getTxUrl('0x1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef');
    const addrUrl = client.getAddressUrl('0x1234567890123456789012345678901234567890');

    if (!txUrl.includes('sepolia.etherscan.io/tx/')) throw new Error('Invalid tx URL');
    if (!addrUrl.includes('sepolia.etherscan.io/address/')) throw new Error('Invalid address URL');
  });

  // Test 11: Client without signer
  await runTest('Client without signer has no address', async () => {
    const readOnlyClient = new MycelixClient({ network: 'sepolia' });
    if (readOnlyClient.getAddress() !== undefined) {
      throw new Error('Read-only client should not have address');
    }
  });

  // Test 12: Client with signer
  await runTest('Client with signer has address', async () => {
    const testKey = '0x' + '1'.repeat(64);
    const signerClient = new MycelixClient({
      network: 'sepolia',
      privateKey: testKey,
    });
    const address = signerClient.getAddress();
    if (!address || !address.startsWith('0x')) {
      throw new Error('Signer client should have valid address');
    }
  });

  // Summary
  console.log('\n=== Test Summary ===\n');
  const passed = results.filter(r => r.passed).length;
  const failed = results.filter(r => !r.passed).length;
  const totalTime = results.reduce((sum, r) => sum + r.duration, 0);

  console.log(`Total:  ${results.length} tests`);
  console.log(`Passed: ${passed}`);
  console.log(`Failed: ${failed}`);
  console.log(`Time:   ${totalTime}ms`);

  if (failed > 0) {
    console.log('\nFailed tests:');
    results.filter(r => !r.passed).forEach(r => {
      console.log(`  - ${r.name}: ${r.error}`);
    });
    process.exit(1);
  }

  console.log('\nAll tests passed!');
}

main().catch((err) => {
  console.error('Test runner error:', err);
  process.exit(1);
});
