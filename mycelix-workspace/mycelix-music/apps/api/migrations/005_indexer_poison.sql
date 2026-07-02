CREATE TABLE IF NOT EXISTS indexer_poison (
  id SERIAL PRIMARY KEY,
  tx_hash TEXT NOT NULL,
  log_index INTEGER NOT NULL,
  song_hash TEXT,
  reason TEXT,
  attempts INTEGER DEFAULT 0,
  created_at TIMESTAMP DEFAULT NOW(),
  UNIQUE (tx_hash, log_index)
);
