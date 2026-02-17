-- Adds hash and indexing fields for chain-sourced plays
ALTER TABLE songs ADD COLUMN IF NOT EXISTS song_hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_songs_song_hash ON songs (song_hash);

ALTER TABLE plays ADD COLUMN IF NOT EXISTS song_hash TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS tx_hash TEXT;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS log_index INTEGER;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS block_number INTEGER;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS protocol_fee NUMERIC DEFAULT 0;
ALTER TABLE plays ADD COLUMN IF NOT EXISTS net_amount NUMERIC DEFAULT 0;
CREATE UNIQUE INDEX IF NOT EXISTS idx_plays_tx_log ON plays (tx_hash, log_index);
