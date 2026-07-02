-- Add FK constraint from plays to songs and supporting index

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'fk_plays_song') THEN
    ALTER TABLE plays
      ADD CONSTRAINT fk_plays_song
      FOREIGN KEY (song_id) REFERENCES songs(id)
      ON DELETE SET NULL;
  END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_plays_song_id ON plays (song_id);
