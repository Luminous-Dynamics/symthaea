#!/usr/bin/env npx tsx

// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Mycelix CLI
 *
 * Command-line interface for interacting with Mycelix smart contracts.
 *
 * Usage:
 *   npx tsx src/cli.ts <command> [options]
 *
 * Commands:
 *   info                    - Show network and contract information
 *   balance                 - Check wallet balance
 *   register <did> <uri>    - Register a new DID
 *   lookup <did>            - Look up DID information
 *   reputation <address>    - Query reputation for an address
 *   payment create <to> <amount> - Create an escrow payment
 *   payment release <id>    - Release a payment
 *   payment info <id>       - Get payment details
 *
 * Environment:
 *   PRIVATE_KEY            - Wallet private key (required for write operations)
 *   MYCELIX_NETWORK        - Network to use (default: sepolia)
 */

import { ethers } from 'ethers';
import { MycelixClient, NETWORKS } from './index.ts';

const HELP_TEXT = `
Mycelix CLI - Interact with Mycelix smart contracts

USAGE:
  mycelix <command> [arguments]

COMMANDS:
  info                         Show network and contract information
  balance                      Check wallet balance
  register <did> <metadata>    Register a new DID
  lookup <did>                 Look up DID information
  reputation <address>         Query reputation for an address
  payment create <to> <eth>    Create escrow payment
  payment release <id>         Release a payment
  payment refund <id>          Refund a payment
  payment info <id>            Get payment details

ENVIRONMENT:
  PRIVATE_KEY                  Wallet private key (required for writes)
  MYCELIX_NETWORK              Network: sepolia, local (default: sepolia)

EXAMPLES:
  mycelix info
  mycelix register did:mycelix:alice ipfs://QmMetadata
  mycelix lookup did:mycelix:alice
  mycelix reputation 0x1234...
  mycelix payment create 0x1234... 0.01
`;

function getClient(): MycelixClient {
  const network = process.env.MYCELIX_NETWORK || 'sepolia';
  const privateKey = process.env.PRIVATE_KEY;

  return new MycelixClient({
    network: network as keyof typeof NETWORKS,
    privateKey,
  });
}

function requireSigner(client: MycelixClient): void {
  if (!client.signer) {
    console.error('Error: PRIVATE_KEY environment variable required for this operation');
    process.exit(1);
  }
}

async function cmdInfo(): Promise<void> {
  const client = getClient();
  const network = process.env.MYCELIX_NETWORK || 'sepolia';

  console.log('\n=== Mycelix Network Info ===\n');
  console.log(`Network:     ${network}`);
  console.log(`Chain ID:    ${client.config.chainId}`);
  console.log(`RPC URL:     ${client.config.rpcUrl}`);
  console.log(`Explorer:    ${client.config.blockExplorer}`);
  console.log('');
  console.log('Contracts:');
  console.log(`  Registry:   ${client.config.contracts.registry}`);
  console.log(`  Reputation: ${client.config.contracts.reputation}`);
  console.log(`  Payment:    ${client.config.contracts.payment}`);

  if (client.signer) {
    console.log('');
    console.log(`Wallet:      ${client.getAddress()}`);
    try {
      const balance = await client.getBalance();
      console.log(`Balance:     ${ethers.formatEther(balance)} ETH`);
    } catch {
      console.log('Balance:     (unable to fetch)');
    }
  }

  try {
    const totalDids = await client.registry.getTotalDids();
    const fee = await client.registry.getRegistrationFee();
    console.log('');
    console.log('Registry Stats:');
    console.log(`  Total DIDs: ${totalDids.toString()}`);
    console.log(`  Reg. Fee:   ${ethers.formatEther(fee)} ETH`);
  } catch {
    // Contract might not be deployed on this network
  }
}

async function cmdBalance(): Promise<void> {
  const client = getClient();
  requireSigner(client);

  const balance = await client.getBalance();
  console.log(`${ethers.formatEther(balance)} ETH`);
}

async function cmdRegister(did: string, metadata: string): Promise<void> {
  const client = getClient();
  requireSigner(client);

  console.log(`Registering DID: ${did}`);
  console.log(`Metadata URI:    ${metadata}`);
  console.log('');

  try {
    const tx = await client.registry.registerDID(did, metadata);
    console.log(`Transaction:     ${tx.hash}`);
    console.log(`Explorer:        ${client.getTxUrl(tx.hash)}`);
    console.log('');
    console.log('Waiting for confirmation...');

    const receipt = await tx.wait();
    console.log(`Confirmed:       Block ${receipt?.blockNumber}`);
    console.log(`Gas Used:        ${receipt?.gasUsed.toString()}`);
    console.log('');
    console.log('DID registered successfully!');
  } catch (err: any) {
    console.error('Error:', err.message || err);
    process.exit(1);
  }
}

async function cmdLookup(did: string): Promise<void> {
  const client = getClient();

  console.log(`Looking up: ${did}`);
  console.log('');

  try {
    const info = await client.registry.getDID(did);

    if (info.owner === ethers.ZeroAddress) {
      console.log('DID not found');
      return;
    }

    console.log(`Owner:       ${info.owner}`);
    console.log(`Registered:  ${new Date(Number(info.registeredAt) * 1000).toISOString()}`);
    console.log(`Updated:     ${new Date(Number(info.updatedAt) * 1000).toISOString()}`);
    console.log(`Metadata:    ${info.metadataHash}`);
    console.log(`Revoked:     ${info.revoked}`);
  } catch (err: any) {
    console.error('Error:', err.message || err);
    process.exit(1);
  }
}

async function cmdReputation(address: string): Promise<void> {
  const client = getClient();

  console.log(`Querying reputation for: ${address}`);
  console.log('');

  try {
    const rep = await client.reputation.getReputation(address);
    console.log(`Score:       ${rep.score.toString()}`);
    console.log(`Submissions: ${rep.submissions.toString()}`);
    console.log(`Last Update: ${rep.lastUpdate === 0n ? 'Never' : new Date(Number(rep.lastUpdate) * 1000).toISOString()}`);
  } catch (err: any) {
    console.error('Error:', err.message || err);
    process.exit(1);
  }
}

async function cmdPaymentCreate(to: string, amountEth: string): Promise<void> {
  const client = getClient();
  requireSigner(client);

  const amount = ethers.parseEther(amountEth);
  const releaseTime = Math.floor(Date.now() / 1000) + 3600; // 1 hour from now

  console.log(`Creating escrow payment:`);
  console.log(`  To:          ${to}`);
  console.log(`  Amount:      ${amountEth} ETH`);
  console.log(`  Release:     ${new Date(releaseTime * 1000).toISOString()}`);
  console.log('');

  try {
    const tx = await client.payment.createPayment(to, amount, releaseTime);
    console.log(`Transaction:   ${tx.hash}`);
    console.log(`Explorer:      ${client.getTxUrl(tx.hash)}`);
    console.log('');
    console.log('Waiting for confirmation...');

    const receipt = await tx.wait();
    console.log(`Confirmed:     Block ${receipt?.blockNumber}`);
    console.log('');
    console.log('Payment created successfully!');
  } catch (err: any) {
    console.error('Error:', err.message || err);
    process.exit(1);
  }
}

async function cmdPaymentRelease(paymentId: string): Promise<void> {
  const client = getClient();
  requireSigner(client);

  console.log(`Releasing payment: ${paymentId}`);
  console.log('');

  try {
    const tx = await client.payment.releasePayment(parseInt(paymentId));
    console.log(`Transaction: ${tx.hash}`);
    await tx.wait();
    console.log('Payment released successfully!');
  } catch (err: any) {
    console.error('Error:', err.message || err);
    process.exit(1);
  }
}

async function cmdPaymentRefund(paymentId: string): Promise<void> {
  const client = getClient();
  requireSigner(client);

  console.log(`Refunding payment: ${paymentId}`);
  console.log('');

  try {
    const tx = await client.payment.refundPayment(parseInt(paymentId));
    console.log(`Transaction: ${tx.hash}`);
    await tx.wait();
    console.log('Payment refunded successfully!');
  } catch (err: any) {
    console.error('Error:', err.message || err);
    process.exit(1);
  }
}

async function cmdPaymentInfo(paymentId: string): Promise<void> {
  const client = getClient();

  console.log(`Payment ID: ${paymentId}`);
  console.log('');

  try {
    const info = await client.payment.getPayment(parseInt(paymentId));
    console.log(`Sender:      ${info.sender}`);
    console.log(`Recipient:   ${info.recipient}`);
    console.log(`Amount:      ${ethers.formatEther(info.amount)} ETH`);
    console.log(`Release:     ${new Date(Number(info.releaseTime) * 1000).toISOString()}`);
    console.log(`Released:    ${info.released}`);
    console.log(`Refunded:    ${info.refunded}`);
  } catch (err: any) {
    console.error('Error:', err.message || err);
    process.exit(1);
  }
}

async function main(): Promise<void> {
  const args = process.argv.slice(2);
  const command = args[0]?.toLowerCase();

  if (!command || command === 'help' || command === '-h' || command === '--help') {
    console.log(HELP_TEXT);
    return;
  }

  switch (command) {
    case 'info':
      await cmdInfo();
      break;

    case 'balance':
      await cmdBalance();
      break;

    case 'register':
      if (args.length < 3) {
        console.error('Usage: mycelix register <did> <metadata-uri>');
        process.exit(1);
      }
      await cmdRegister(args[1], args[2]);
      break;

    case 'lookup':
      if (args.length < 2) {
        console.error('Usage: mycelix lookup <did>');
        process.exit(1);
      }
      await cmdLookup(args[1]);
      break;

    case 'reputation':
      if (args.length < 2) {
        console.error('Usage: mycelix reputation <address>');
        process.exit(1);
      }
      await cmdReputation(args[1]);
      break;

    case 'payment':
      const subCmd = args[1]?.toLowerCase();
      switch (subCmd) {
        case 'create':
          if (args.length < 4) {
            console.error('Usage: mycelix payment create <to-address> <amount-eth>');
            process.exit(1);
          }
          await cmdPaymentCreate(args[2], args[3]);
          break;

        case 'release':
          if (args.length < 3) {
            console.error('Usage: mycelix payment release <payment-id>');
            process.exit(1);
          }
          await cmdPaymentRelease(args[2]);
          break;

        case 'refund':
          if (args.length < 3) {
            console.error('Usage: mycelix payment refund <payment-id>');
            process.exit(1);
          }
          await cmdPaymentRefund(args[2]);
          break;

        case 'info':
          if (args.length < 3) {
            console.error('Usage: mycelix payment info <payment-id>');
            process.exit(1);
          }
          await cmdPaymentInfo(args[2]);
          break;

        default:
          console.error('Unknown payment command. Use: create, release, refund, info');
          process.exit(1);
      }
      break;

    default:
      console.error(`Unknown command: ${command}`);
      console.log('Run "mycelix help" for usage information');
      process.exit(1);
  }
}

main().catch((err) => {
  console.error('Fatal error:', err);
  process.exit(1);
});
