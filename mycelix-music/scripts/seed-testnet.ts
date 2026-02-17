#!/usr/bin/env tsx
/**
 * Seed script for testnet environments
 *
 * - If contract addresses + RPC are configured, mirrors seed-local on-chain actions
 * - Always attempts to index a minimal set of songs in the API for discoverability
 *
 * Safe to run multiple times (API upserts records by id)
 */

import * as dotenv from 'dotenv';
import { ethers } from 'ethers';
dotenv.config();

const apiBase = (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:3100').replace(/\/$/, '');
const rpcUrl = process.env.RPC_URL;

const SONGS = [
  { title: 'Testnet Drift', artist: 'DJ Nova', genre: 'Electronic', ipfsHash: 'QmTestHashA', description: 'Testnet only track' },
  { title: 'Testnet Lights', artist: 'The Echoes', genre: 'Rock', ipfsHash: 'QmTestHashB', description: 'Testnet only track' },
];

async function indexApi() {
  for (const s of SONGS) {
    const id = `${s.artist}-${s.title}`;
    const payload = {
      id,
      title: s.title,
      artist: s.artist,
      artistAddress: process.env.TEST_ARTIST_ADDRESS || '0x0000000000000000000000000000000000000000',
      genre: s.genre,
      description: s.description,
      ipfsHash: s.ipfsHash,
      paymentModel: 'pay_per_stream',
      audioUrl: `${process.env.IPFS_GATEWAY || 'https://w3s.link/ipfs'}/${s.ipfsHash}`,
    };
    try {
      const res = await fetch(`${apiBase}/api/songs`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(payload) });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      console.log(`  ✓ Indexed in API: ${id}`);
    } catch (e: any) {
      console.warn(`  ⚠ Failed API indexing for ${id}: ${e.message}`);
    }
  }
}

async function main() {
  console.log('🌱 Seeding testnet...');
  await indexApi();

  // Optional on-chain registration (requires RPC_URL and private key)
  const router = process.env.NEXT_PUBLIC_ROUTER_ADDRESS;
  if (!rpcUrl || !router || !process.env.PRIVATE_KEY) {
    console.log('ℹ Skipping on-chain registration (RPC_URL / PRIVATE_KEY / ROUTER not set)');
    console.log('✅ Done');
    return;
  }

  try {
    const provider = new ethers.JsonRpcProvider(rpcUrl);
    const wallet = new ethers.Wallet(process.env.PRIVATE_KEY!, provider);
    const ROUTER_ABI = ['function registerSong(bytes32 songId, bytes32 strategyId)', 'function songStrategy(bytes32) view returns (address)'];
    const routerC = new ethers.Contract(router, ROUTER_ABI, wallet);
    const PAY_PER_STREAM_ID = ethers.id('pay-per-stream-v1');
    for (const s of SONGS) {
      const id = ethers.id(`${s.artist}-${s.title}`);
      const tx = await routerC.registerSong(id, PAY_PER_STREAM_ID);
      await tx.wait();
      console.log(`  ✓ Registered on-chain: ${s.artist}-${s.title}`);
    }
  } catch (e: any) {
    console.warn('⚠ On-chain seeding failed:', e.message);
  }

  console.log('✅ Done');
}

main();

