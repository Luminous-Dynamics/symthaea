#!/usr/bin/env npx ts-node --esm

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix SDK Demo
 *
 * Demonstrates live interaction with deployed Mycelix contracts on Sepolia.
 *
 * Usage:
 *   npx ts-node --esm src/demo.ts [command]
 *
 * Commands:
 *   info       - Show contract addresses and wallet info
 *   register   - Register a test DID
 *   reputation - Query reputation for an address
 *   payment    - Create a test payment
 */

import { ethers } from 'ethers';
import { MycelixClient, NETWORKS } from './index.ts';

// Load private key from environment or use the deployment wallet
const PRIVATE_KEY = process.env.PRIVATE_KEY ||
  '0xfeba3c7cef586df86dabb303e597ce451f0cdde0b54dc9772bde9d29aba18fa4';

async function main() {
  const command = process.argv[2] || 'info';

  console.log('\n========================================');
  console.log('  Mycelix SDK Demo - Sepolia Testnet');
  console.log('========================================\n');

  // Initialize client
  const client = new MycelixClient({
    network: 'sepolia',
    privateKey: PRIVATE_KEY,
  });

  const address = client.getAddress();
  console.log(`Wallet: ${address}`);

  const balance = await client.getBalance();
  console.log(`Balance: ${ethers.formatEther(balance)} ETH\n`);

  switch (command) {
    case 'info':
      await showInfo(client);
      break;
    case 'register':
      await registerDID(client);
      break;
    case 'reputation':
      await queryReputation(client);
      break;
    case 'payment':
      await createPayment(client);
      break;
    default:
      console.log(`Unknown command: ${command}`);
      console.log('Available commands: info, register, reputation, payment');
  }
}

async function showInfo(client: MycelixClient) {
  console.log('Contract Addresses:');
  console.log('-------------------');
  console.log(`Registry:   ${client.config.contracts.registry}`);
  console.log(`Reputation: ${client.config.contracts.reputation}`);
  console.log(`Payment:    ${client.config.contracts.payment}`);
  console.log('');
  console.log('Block Explorer:');
  console.log('---------------');
  console.log(`Registry:   ${client.getAddressUrl(client.config.contracts.registry)}`);
  console.log(`Reputation: ${client.getAddressUrl(client.config.contracts.reputation)}`);
  console.log(`Payment:    ${client.getAddressUrl(client.config.contracts.payment)}`);
  console.log('');

  // Check registration fee
  try {
    const fee = await client.registry.getRegistrationFee();
    console.log(`Registration Fee: ${ethers.formatEther(fee)} ETH`);
  } catch (e) {
    console.log('Registration Fee: (unable to fetch)');
  }
}

async function registerDID(client: MycelixClient) {
  const timestamp = Date.now();
  const didId = `did:mycelix:demo-${timestamp}`;
  const metadataUri = `ipfs://QmDemo${timestamp}`;

  console.log('Registering DID...');
  console.log(`  DID ID: ${didId}`);
  console.log(`  Metadata: ${metadataUri}`);
  console.log('');

  try {
    const tx = await client.registry.registerDID(didId, metadataUri);
    console.log(`Transaction submitted: ${tx.hash}`);
    console.log(`View on Etherscan: ${client.getTxUrl(tx.hash)}`);
    console.log('');
    console.log('Waiting for confirmation...');

    const receipt = await tx.wait();
    console.log(`Confirmed in block: ${receipt?.blockNumber}`);
    console.log(`Gas used: ${receipt?.gasUsed.toString()}`);
    console.log('');
    console.log('DID registered successfully!');

    // Verify by reading back
    const didInfo = await client.registry.getDID(didId);
    console.log('');
    console.log('Verified DID Info:');
    console.log(`  Owner: ${didInfo.owner}`);
    console.log(`  Metadata Hash: ${didInfo.metadataHash}`);
    console.log(`  Revoked: ${didInfo.revoked}`);
    console.log(`  Registered At: ${new Date(Number(didInfo.registeredAt) * 1000).toISOString()}`);
  } catch (e: any) {
    console.error('Error registering DID:', e.message || e);
  }
}

async function queryReputation(client: MycelixClient) {
  const address = client.getAddress()!;

  console.log(`Querying reputation for: ${address}`);
  console.log('');

  try {
    const rep = await client.reputation.getReputation(address);
    console.log('Reputation Info:');
    console.log(`  Score: ${rep.score.toString()}`);
    console.log(`  Submissions: ${rep.submissions.toString()}`);
    console.log(`  Last Update: ${rep.lastUpdate.toString()}`);

    if (rep.submissions === 0n) {
      console.log('');
      console.log('No reputation data yet. This address has not received any reputation submissions.');
    }
  } catch (e: any) {
    console.error('Error querying reputation:', e.message || e);
  }
}

async function createPayment(client: MycelixClient) {
  const recipient = client.getAddress()!; // Send to self for demo
  const amount = ethers.parseEther('0.0001'); // 0.0001 ETH
  const releaseTime = Math.floor(Date.now() / 1000) + 60; // 1 minute from now

  console.log('Creating escrow payment...');
  console.log(`  Recipient: ${recipient}`);
  console.log(`  Amount: ${ethers.formatEther(amount)} ETH`);
  console.log(`  Release Time: ${new Date(releaseTime * 1000).toISOString()}`);
  console.log('');

  try {
    const tx = await client.payment.createPayment(
      recipient,
      amount,
      releaseTime,
      ethers.ZeroHash
    );
    console.log(`Transaction submitted: ${tx.hash}`);
    console.log(`View on Etherscan: ${client.getTxUrl(tx.hash)}`);
    console.log('');
    console.log('Waiting for confirmation...');

    const receipt = await tx.wait();
    console.log(`Confirmed in block: ${receipt?.blockNumber}`);
    console.log('');
    console.log('Payment created successfully!');
  } catch (e: any) {
    console.error('Error creating payment:', e.message || e);
  }
}

main().catch(console.error);
