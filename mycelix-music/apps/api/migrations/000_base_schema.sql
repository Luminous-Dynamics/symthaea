-- Core tables for Mycelix Music API

CREATE TABLE IF NOT EXISTS songs (
  id TEXT PRIMARY KEY,
  title TEXT NOT NULL,
  artist TEXT NOT NULL,
  artist_address TEXT NOT NULL,
  genre TEXT NOT NULL,
  description TEXT,
  ipfs_hash TEXT NOT NULL,
  payment_model TEXT NOT NULL,
  cover_art TEXT,
  audio_url TEXT,
  claim_stream_id TEXT,
  song_hash TEXT,
  created_at TIMESTAMP DEFAULT NOW(),
  plays INTEGER DEFAULT 0,
  earnings NUMERIC DEFAULT 0
);

CREATE TABLE IF NOT EXISTS plays (
  id SERIAL PRIMARY KEY,
  song_id TEXT REFERENCES songs(id),
  listener_address TEXT NOT NULL,
  amount NUMERIC DEFAULT 0,
  payment_type TEXT NOT NULL,
  song_hash TEXT,
  tx_hash TEXT,
  log_index INTEGER,
  block_number INTEGER,
  protocol_fee NUMERIC DEFAULT 0,
  net_amount NUMERIC DEFAULT 0,
  timestamp TIMESTAMP DEFAULT NOW()
);

-- Useful indexes for filtering
CREATE INDEX IF NOT EXISTS idx_songs_created_at ON songs (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_songs_genre ON songs (genre);
CREATE INDEX IF NOT EXISTS idx_songs_payment_model ON songs (payment_model);
CREATE INDEX IF NOT EXISTS idx_songs_title_lower ON songs ((LOWER(title)));
CREATE INDEX IF NOT EXISTS idx_songs_artist_lower ON songs ((LOWER(artist)));
CREATE INDEX IF NOT EXISTS idx_songs_plays ON songs (plays);
CREATE INDEX IF NOT EXISTS idx_songs_earnings ON songs (earnings);
CREATE INDEX IF NOT EXISTS idx_plays_timestamp ON plays (timestamp);
