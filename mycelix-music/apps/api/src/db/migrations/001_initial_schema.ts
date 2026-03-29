// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Migration: Initial Schema
 * Version: 1
 * Created: 2024-01-01
 *
 * Creates the initial database schema for Mycelix Music.
 */

import { PoolClient } from 'pg';
import { Migration } from './migrator';

const migration: Migration = {
  version: 1,
  name: 'initial_schema',

  async up(client: PoolClient): Promise<void> {
    // Create extension for UUID generation
    await client.query(`
      CREATE EXTENSION IF NOT EXISTS "uuid-ossp"
    `);

    // Create songs table
    await client.query(`
      CREATE TABLE IF NOT EXISTS songs (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        title VARCHAR(200) NOT NULL,
        artist VARCHAR(100) NOT NULL,
        artist_address VARCHAR(42) NOT NULL,
        genre VARCHAR(50) NOT NULL,
        description TEXT,
        ipfs_hash VARCHAR(64) NOT NULL UNIQUE,
        song_hash VARCHAR(66),
        cover_art VARCHAR(500),
        audio_url VARCHAR(500),
        payment_model VARCHAR(20) NOT NULL DEFAULT 'per_play',
        claim_stream_id VARCHAR(200),
        plays INTEGER NOT NULL DEFAULT 0,
        earnings NUMERIC(20, 6) NOT NULL DEFAULT 0,
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        updated_at TIMESTAMP WITH TIME ZONE,

        CONSTRAINT valid_payment_model CHECK (payment_model IN ('per_play', 'subscription', 'tip')),
        CONSTRAINT valid_artist_address CHECK (artist_address ~ '^0x[a-fA-F0-9]{40}$'),
        CONSTRAINT non_negative_plays CHECK (plays >= 0),
        CONSTRAINT non_negative_earnings CHECK (earnings >= 0)
      )
    `);

    // Create plays table
    await client.query(`
      CREATE TABLE IF NOT EXISTS plays (
        id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
        song_id UUID NOT NULL REFERENCES songs(id) ON DELETE CASCADE,
        listener_address VARCHAR(42) NOT NULL,
        amount NUMERIC(20, 6) NOT NULL,
        transaction_hash VARCHAR(66),
        status VARCHAR(20) NOT NULL DEFAULT 'pending',
        created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
        confirmed_at TIMESTAMP WITH TIME ZONE,

        CONSTRAINT valid_status CHECK (status IN ('pending', 'confirmed', 'failed')),
        CONSTRAINT valid_listener_address CHECK (listener_address ~ '^0x[a-fA-F0-9]{40}$'),
        CONSTRAINT non_negative_amount CHECK (amount >= 0)
      )
    `);

    // Create indexes for songs
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_songs_artist_address ON songs(artist_address);
      CREATE INDEX IF NOT EXISTS idx_songs_genre ON songs(genre);
      CREATE INDEX IF NOT EXISTS idx_songs_plays ON songs(plays DESC);
      CREATE INDEX IF NOT EXISTS idx_songs_earnings ON songs(earnings DESC);
      CREATE INDEX IF NOT EXISTS idx_songs_created_at ON songs(created_at DESC);
      CREATE INDEX IF NOT EXISTS idx_songs_payment_model ON songs(payment_model);
    `);

    // Create indexes for plays
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_plays_song_id ON plays(song_id);
      CREATE INDEX IF NOT EXISTS idx_plays_listener_address ON plays(listener_address);
      CREATE INDEX IF NOT EXISTS idx_plays_status ON plays(status);
      CREATE INDEX IF NOT EXISTS idx_plays_created_at ON plays(created_at DESC);
    `);

    // Create full-text search index for songs
    await client.query(`
      CREATE INDEX IF NOT EXISTS idx_songs_search ON songs
        USING gin(to_tsvector('english', title || ' ' || artist || ' ' || COALESCE(description, '')))
    `);

    // Create function for updating updated_at
    await client.query(`
      CREATE OR REPLACE FUNCTION update_updated_at_column()
      RETURNS TRIGGER AS $$
      BEGIN
        NEW.updated_at = NOW();
        RETURN NEW;
      END;
      $$ language 'plpgsql'
    `);

    // Create trigger for songs updated_at
    await client.query(`
      DROP TRIGGER IF EXISTS update_songs_updated_at ON songs;
      CREATE TRIGGER update_songs_updated_at
        BEFORE UPDATE ON songs
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column()
    `);
  },

  async down(client: PoolClient): Promise<void> {
    // Drop triggers
    await client.query(`DROP TRIGGER IF EXISTS update_songs_updated_at ON songs`);

    // Drop function
    await client.query(`DROP FUNCTION IF EXISTS update_updated_at_column`);

    // Drop tables (plays first due to foreign key)
    await client.query(`DROP TABLE IF EXISTS plays CASCADE`);
    await client.query(`DROP TABLE IF EXISTS songs CASCADE`);
  },
};

export default migration;
