-- Clamp negative values and add non-negative checks

UPDATE songs SET earnings = 0 WHERE earnings < 0;
UPDATE songs SET plays = 0 WHERE plays < 0;
UPDATE plays SET amount = 0 WHERE amount < 0;
UPDATE plays SET net_amount = 0 WHERE net_amount < 0;
UPDATE plays SET protocol_fee = 0 WHERE protocol_fee < 0;

DO $$
BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_songs_earnings_nonneg') THEN
    ALTER TABLE songs ADD CONSTRAINT ck_songs_earnings_nonneg CHECK (earnings >= 0);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_songs_plays_nonneg') THEN
    ALTER TABLE songs ADD CONSTRAINT ck_songs_plays_nonneg CHECK (plays >= 0);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_plays_amount_nonneg') THEN
    ALTER TABLE plays ADD CONSTRAINT ck_plays_amount_nonneg CHECK (amount >= 0);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_plays_net_nonneg') THEN
    ALTER TABLE plays ADD CONSTRAINT ck_plays_net_nonneg CHECK (net_amount >= 0);
  END IF;
  IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'ck_plays_protocol_fee_nonneg') THEN
    ALTER TABLE plays ADD CONSTRAINT ck_plays_protocol_fee_nonneg CHECK (protocol_fee >= 0);
  END IF;
END $$;
