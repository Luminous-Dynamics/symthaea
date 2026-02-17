ALTER TABLE strategy_configs ADD COLUMN IF NOT EXISTS hash TEXT;
CREATE UNIQUE INDEX IF NOT EXISTS idx_strategy_configs_hash ON strategy_configs (hash);
