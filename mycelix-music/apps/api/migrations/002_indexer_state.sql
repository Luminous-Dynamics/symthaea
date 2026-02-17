CREATE TABLE IF NOT EXISTS indexer_state (
  name TEXT PRIMARY KEY,
  last_block INTEGER NOT NULL,
  updated_at TIMESTAMP DEFAULT NOW()
);
