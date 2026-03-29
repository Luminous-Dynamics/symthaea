// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Development Seed Data
 *
 * Provides realistic test data for local development and testing.
 * Run with: npm run db:seed
 */

import { Pool } from 'pg';
import { randomUUID } from 'crypto';

// Sample genres for music
const GENRES = [
  'electronic',
  'ambient',
  'downtempo',
  'techno',
  'house',
  'experimental',
  'drone',
  'idm',
  'chillout',
  'breaks',
];

// Sample artists with Ethereum-like addresses
const SAMPLE_ARTISTS = [
  { name: 'Aurora Synthesis', address: '0x1234567890abcdef1234567890abcdef12345678' },
  { name: 'Mycelial Dreams', address: '0x2345678901bcdef12345678901bcdef123456789' },
  { name: 'Quantum Resonance', address: '0x3456789012cdef123456789012cdef1234567890' },
  { name: 'Ethereal Waves', address: '0x4567890123def1234567890123def12345678901' },
  { name: 'Digital Moss', address: '0x5678901234ef12345678901234ef123456789012' },
  { name: 'Fractal Bloom', address: '0x6789012345f123456789012345f1234567890123' },
  { name: 'Neural Garden', address: '0x7890123456123456789012345612345678901234' },
  { name: 'Spore Network', address: '0x8901234567234567890123456723456789012345' },
];

// Song title components for random generation
const TITLE_PREFIXES = [
  'The', 'A', 'Into', 'Beyond', 'Through', 'Within', 'Beneath', 'Above',
];
const TITLE_NOUNS = [
  'Void', 'Light', 'Signal', 'Forest', 'Network', 'Dream', 'Wave', 'Pattern',
  'Rhythm', 'Pulse', 'Flow', 'Cycle', 'Spiral', 'Horizon', 'Echo', 'Fragment',
];
const TITLE_MODIFIERS = [
  'Infinite', 'Cosmic', 'Digital', 'Organic', 'Sacred', 'Hidden', 'Ancient',
  'Emergent', 'Resonant', 'Luminous', 'Ethereal', 'Quantum', 'Harmonic',
];

// Generate a random song title
function generateTitle(): string {
  const usePrefix = Math.random() > 0.5;
  const useModifier = Math.random() > 0.3;

  const parts: string[] = [];
  if (usePrefix) parts.push(TITLE_PREFIXES[Math.floor(Math.random() * TITLE_PREFIXES.length)]);
  if (useModifier) parts.push(TITLE_MODIFIERS[Math.floor(Math.random() * TITLE_MODIFIERS.length)]);
  parts.push(TITLE_NOUNS[Math.floor(Math.random() * TITLE_NOUNS.length)]);

  return parts.join(' ');
}

// Generate random IPFS-like hash
function generateIpfsHash(): string {
  const chars = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789';
  let hash = 'Qm';
  for (let i = 0; i < 44; i++) {
    hash += chars[Math.floor(Math.random() * chars.length)];
  }
  return hash;
}

// Generate random plays (power law distribution - most songs have few plays)
function generatePlays(): number {
  const rand = Math.random();
  if (rand > 0.95) return Math.floor(Math.random() * 10000) + 1000; // Top 5%: 1000-11000
  if (rand > 0.8) return Math.floor(Math.random() * 1000) + 100;   // Next 15%: 100-1100
  if (rand > 0.5) return Math.floor(Math.random() * 100) + 10;     // Next 30%: 10-110
  return Math.floor(Math.random() * 10);                            // Bottom 50%: 0-10
}

// Generate earnings based on plays
function generateEarnings(plays: number): string {
  const baseRate = 0.001; // Base earnings per play
  const variance = 0.5 + Math.random(); // 0.5x to 1.5x variance
  const earnings = plays * baseRate * variance;
  return earnings.toFixed(6);
}

// Generate a random date in the past year
function generateDate(): Date {
  const now = Date.now();
  const oneYearAgo = now - (365 * 24 * 60 * 60 * 1000);
  return new Date(oneYearAgo + Math.random() * (now - oneYearAgo));
}

/**
 * Generate seed songs
 */
export function generateSongs(count: number): Array<{
  id: string;
  title: string;
  artist: string;
  artist_address: string;
  genre: string;
  description: string;
  ipfs_hash: string;
  payment_model: string;
  plays: number;
  earnings: string;
  created_at: Date;
}> {
  const songs = [];

  for (let i = 0; i < count; i++) {
    const artist = SAMPLE_ARTISTS[Math.floor(Math.random() * SAMPLE_ARTISTS.length)];
    const plays = generatePlays();

    songs.push({
      id: randomUUID(),
      title: generateTitle(),
      artist: artist.name,
      artist_address: artist.address,
      genre: GENRES[Math.floor(Math.random() * GENRES.length)],
      description: `A ${TITLE_MODIFIERS[Math.floor(Math.random() * TITLE_MODIFIERS.length)].toLowerCase()} piece exploring sonic landscapes.`,
      ipfs_hash: generateIpfsHash(),
      payment_model: Math.random() > 0.7 ? 'subscription' : 'per_play',
      plays,
      earnings: generateEarnings(plays),
      created_at: generateDate(),
    });
  }

  return songs.sort((a, b) => a.created_at.getTime() - b.created_at.getTime());
}

/**
 * Generate seed plays
 */
export function generatePlays(
  songs: Array<{ id: string; artist_address: string }>,
  count: number
): Array<{
  id: string;
  song_id: string;
  listener_address: string;
  amount: string;
  status: string;
  created_at: Date;
}> {
  const plays = [];
  const listenerAddresses = [
    '0xabc1234567890abcdef1234567890abcdef12340',
    '0xbcd2345678901bcdef12345678901bcdef123450',
    '0xcde3456789012cdef123456789012cdef1234560',
    '0xdef4567890123def1234567890123def12345670',
    '0xef15678901234ef12345678901234ef123456780',
    '0xf126789012345f123456789012345f1234567890',
  ];

  for (let i = 0; i < count; i++) {
    const song = songs[Math.floor(Math.random() * songs.length)];
    const amount = (Math.random() * 0.01).toFixed(6);

    plays.push({
      id: randomUUID(),
      song_id: song.id,
      listener_address: listenerAddresses[Math.floor(Math.random() * listenerAddresses.length)],
      amount,
      status: Math.random() > 0.02 ? 'confirmed' : 'pending',
      created_at: generateDate(),
    });
  }

  return plays.sort((a, b) => a.created_at.getTime() - b.created_at.getTime());
}

/**
 * Seed the database
 */
export async function seed(pool: Pool): Promise<void> {
  console.log('Starting database seed...');

  // Clear existing data
  console.log('Clearing existing data...');
  await pool.query('TRUNCATE TABLE plays CASCADE');
  await pool.query('TRUNCATE TABLE songs CASCADE');

  // Generate and insert songs
  const songCount = 100;
  console.log(`Generating ${songCount} songs...`);
  const songs = generateSongs(songCount);

  for (const song of songs) {
    await pool.query(
      `INSERT INTO songs (id, title, artist, artist_address, genre, description, ipfs_hash, payment_model, plays, earnings, created_at)
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)`,
      [
        song.id,
        song.title,
        song.artist,
        song.artist_address,
        song.genre,
        song.description,
        song.ipfs_hash,
        song.payment_model,
        song.plays,
        song.earnings,
        song.created_at,
      ]
    );
  }
  console.log(`Inserted ${songs.length} songs`);

  // Generate and insert plays
  const playCount = 500;
  console.log(`Generating ${playCount} plays...`);
  const plays = generatePlays(songs, playCount);

  for (const play of plays) {
    await pool.query(
      `INSERT INTO plays (id, song_id, listener_address, amount, status, created_at)
       VALUES ($1, $2, $3, $4, $5, $6)`,
      [
        play.id,
        play.song_id,
        play.listener_address,
        play.amount,
        play.status,
        play.created_at,
      ]
    );
  }
  console.log(`Inserted ${plays.length} plays`);

  console.log('Seed completed successfully!');
}

/**
 * CLI entry point
 */
if (require.main === module) {
  const pool = new Pool({
    connectionString: process.env.DATABASE_URL || 'postgresql://localhost:5432/mycelix',
  });

  seed(pool)
    .then(() => pool.end())
    .catch((err) => {
      console.error('Seed failed:', err);
      pool.end();
      process.exit(1);
    });
}
