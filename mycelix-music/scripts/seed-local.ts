#!/usr/bin/env tsx

/**
 * @title Seed Local Database
 * @notice Populates local database with test data for development
 * @dev Creates 3 artists, 10 songs, and 5 listeners with various economic models
 */

import { ethers } from 'ethers';
import * as dotenv from 'dotenv';

dotenv.config();

// Contract ABIs (minimal for testing)
const ERC20_ABI = [
  'function approve(address spender, uint256 amount) external returns (bool)',
  'function balanceOf(address account) external view returns (uint256)',
  'function transfer(address to, uint256 amount) external returns (bool)',
];

const ROUTER_ABI = [
  'function registerSong(bytes32 songId, bytes32 strategyId, address artist) external',
  'function songStrategy(bytes32 songId) external view returns (bytes32)',
];

const PAY_PER_STREAM_ABI = [
  'function configureRoyaltySplit(bytes32 songId, address[] recipients, uint256[] basisPoints, string[] roles) external',
];

const GIFT_ECONOMY_ABI = [
  'function configureGiftEconomy(bytes32 songId, address artist, uint256 cgcPerListen, uint256 earlyListenerBonus, uint256 earlyListenerThreshold, uint256 repeatListenerMultiplier) external',
];

// Test data
const ARTISTS = [
  {
    name: 'DJ Nova',
    address: '0x70997970C51812dc3A010C7d01b50e0d17dc79C8', // Anvil account #1
    privateKey: '0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d',
    bio: 'Electronic music producer focused on consciousness-expanding soundscapes',
    genre: 'Electronic',
    model: 'gift-economy',
  },
  {
    name: 'The Echoes',
    address: '0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC', // Anvil account #2
    privateKey: '0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a',
    bio: 'Four-piece indie rock band from Portland',
    genre: 'Rock',
    model: 'pay-per-stream',
    members: [
      { role: 'Lead Singer', bps: 3000 },
      { role: 'Guitarist', bps: 2500 },
      { role: 'Bassist', bps: 2000 },
      { role: 'Drummer', bps: 2500 },
    ],
  },
  {
    name: 'Symphony Orchestra',
    address: '0x90F79bf6EB2c4f870365E785982E1f101E93b906', // Anvil account #3
    privateKey: '0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6',
    bio: 'Community orchestra bringing classical music to the blockchain',
    genre: 'Classical',
    model: 'pay-per-stream',
    members: [
      { role: 'Conductor', bps: 2000 },
      { role: 'Orchestra', bps: 8000 },
    ],
  },
];

const SONGS = [
  // DJ Nova songs (Gift Economy)
  {
    title: 'Cosmic Drift',
    artist: 'DJ Nova',
    duration: 245,
    ipfsHash: 'QmTestHash1',
    genre: 'Electronic',
    description: 'A journey through infinite space',
  },
  {
    title: 'Neural Pathways',
    artist: 'DJ Nova',
    duration: 312,
    ipfsHash: 'QmTestHash2',
    genre: 'Electronic',
    description: 'Exploring consciousness through sound',
  },
  {
    title: 'Quantum Resonance',
    artist: 'DJ Nova',
    duration: 298,
    ipfsHash: 'QmTestHash3',
    genre: 'Electronic',
    description: 'When particles dance',
  },
  // The Echoes songs (Pay Per Stream)
  {
    title: 'Mountain Echo',
    artist: 'The Echoes',
    duration: 223,
    ipfsHash: 'QmTestHash4',
    genre: 'Rock',
    description: 'Our debut single about finding yourself',
  },
  {
    title: 'City Lights',
    artist: 'The Echoes',
    duration: 267,
    ipfsHash: 'QmTestHash5',
    genre: 'Rock',
    description: 'Late night drives through urban landscapes',
  },
  {
    title: 'Thunder Road',
    artist: 'The Echoes',
    duration: 301,
    ipfsHash: 'QmTestHash6',
    genre: 'Rock',
    description: 'High energy anthem',
  },
  // Symphony Orchestra songs (Pay Per Stream)
  {
    title: 'Beethoven Symphony No. 9',
    artist: 'Symphony Orchestra',
    duration: 4200,
    ipfsHash: 'QmTestHash7',
    genre: 'Classical',
    description: 'Timeless masterpiece performed live',
  },
  {
    title: 'Mozart Requiem',
    artist: 'Symphony Orchestra',
    duration: 3600,
    ipfsHash: 'QmTestHash8',
    genre: 'Classical',
    description: 'Haunting beauty in every note',
  },
  {
    title: 'Vivaldi Four Seasons',
    artist: 'Symphony Orchestra',
    duration: 2400,
    ipfsHash: 'QmTestHash9',
    genre: 'Classical',
    description: 'Spring, Summer, Autumn, Winter',
  },
  {
    title: 'Bach Brandenburg Concertos',
    artist: 'Symphony Orchestra',
    duration: 3000,
    ipfsHash: 'QmTestHash10',
    genre: 'Classical',
    description: 'Baroque brilliance',
  },
];

async function main() {
  console.log('🌱 Starting local database seeding...\n');

  // Connect to local Anvil RPC
  const provider = new ethers.JsonRpcProvider(process.env.RPC_URL || 'http://localhost:8545');

  // Get contract addresses from environment
  const routerAddress = process.env.NEXT_PUBLIC_ROUTER_ADDRESS;
  const flowTokenAddress = process.env.NEXT_PUBLIC_FLOW_TOKEN_ADDRESS;
  const payPerStreamAddress = process.env.NEXT_PUBLIC_PAY_PER_STREAM_ADDRESS;
  const giftEconomyAddress = process.env.NEXT_PUBLIC_GIFT_ECONOMY_ADDRESS;

  if (!routerAddress || !flowTokenAddress || !payPerStreamAddress || !giftEconomyAddress) {
    throw new Error('Contract addresses not found in .env file. Run deployment first!');
  }

  console.log('📋 Contract Addresses:');
  console.log('  Router:', routerAddress);
  console.log('  FLOW Token:', flowTokenAddress);
  console.log('  Pay Per Stream:', payPerStreamAddress);
  console.log('  Gift Economy:', giftEconomyAddress);
  console.log('');

  // Strategy IDs (must match deployment)
  const PAY_PER_STREAM_ID = ethers.id('pay-per-stream-v1');
  const GIFT_ECONOMY_ID = ethers.id('gift-economy-v1');

  // Create contract instances
  const router = new ethers.Contract(routerAddress, ROUTER_ABI, provider);
  const payPerStream = new ethers.Contract(payPerStreamAddress, PAY_PER_STREAM_ABI, provider);
  const giftEconomy = new ethers.Contract(giftEconomyAddress, GIFT_ECONOMY_ABI, provider);

  // Register songs
  console.log('🎵 Registering songs on-chain...\n');

  for (const song of SONGS) {
    const artist = ARTISTS.find((a) => a.name === song.artist)!;
    const wallet = new ethers.Wallet(artist.privateKey, provider);

    const songId = ethers.id(`${song.artist}-${song.title}`);
    const strategyId = artist.model === 'gift-economy' ? GIFT_ECONOMY_ID : PAY_PER_STREAM_ID;

    try {
      // Register song with router
      console.log(`  Registering: "${song.title}" by ${song.artist}`);
      const tx = await router.connect(wallet).registerSong(songId, strategyId, artist.address);
      await tx.wait();

      // Configure strategy
      if (artist.model === 'gift-economy') {
        // Gift Economy: Free listening + CGC rewards
        const tx2 = await giftEconomy.connect(wallet).configureGiftEconomy(
          songId,
          artist.address,
          ethers.parseEther('1'), // 1 CGC per listen
          ethers.parseEther('10'), // 10 CGC early listener bonus
          100, // First 100 listeners get bonus
          15000 // 1.5x multiplier for repeat listeners
        );
        await tx2.wait();
        console.log(`    ✓ Configured gift economy (1 CGC per listen, 10 CGC early bonus)`);
      } else {
        // Pay Per Stream: Configure royalty splits
        const recipients = artist.members!.map(() => artist.address); // TODO: Real addresses
        const basisPoints = artist.members!.map((m) => m.bps);
        const roles = artist.members!.map((m) => m.role);

        const tx2 = await payPerStream
          .connect(wallet)
          .configureRoyaltySplit(songId, recipients, basisPoints, roles);
        await tx2.wait();
        console.log(`    ✓ Configured royalty splits (${artist.members!.length} recipients)`);
      }

      console.log(`    ✓ Song ID: ${songId.slice(0, 10)}...`);
      console.log('');
    } catch (error: any) {
      console.error(`    ✗ Failed to register song: ${error.message}\n`);
    }
  }

  console.log('✅ Seeding complete!\n');
  console.log('📊 Summary:');
  console.log(`  Artists: ${ARTISTS.length}`);
  console.log(`  Songs: ${SONGS.length}`);
  console.log(`  Gift Economy: ${SONGS.filter((s) => ARTISTS.find((a) => a.name === s.artist)?.model === 'gift-economy').length}`);
  console.log(`  Pay Per Stream: ${SONGS.filter((s) => ARTISTS.find((a) => a.name === s.artist)?.model === 'pay-per-stream').length}`);
  console.log('');
  console.log('🎉 Ready to test! Visit http://localhost:3000');
}

main()
  .then(() => process.exit(0))
  .catch((error) => {
    console.error('❌ Error seeding database:', error);
    process.exit(1);
  });
