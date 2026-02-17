Run migrations against your DATABASE_URL before starting the API/indexer:

```bash
psql "$DATABASE_URL" -f apps/api/migrations/000_base_schema.sql
psql "$DATABASE_URL" -f apps/api/migrations/001_add_hashes.sql
psql "$DATABASE_URL" -f apps/api/migrations/002_indexer_state.sql
psql "$DATABASE_URL" -f apps/api/migrations/003_strategy_configs.sql
psql "$DATABASE_URL" -f apps/api/migrations/004_strategy_config_hash.sql
psql "$DATABASE_URL" -f apps/api/migrations/005_indexer_poison.sql
psql "$DATABASE_URL" -f apps/api/migrations/006_indexer_poison_block.sql
psql "$DATABASE_URL" -f apps/api/migrations/007_strategy_admin_sig.sql
psql "$DATABASE_URL" -f apps/api/migrations/008_nonnegative_constraints.sql
psql "$DATABASE_URL" -f apps/api/migrations/009_fk_plays_song.sql
```

Add new files under `apps/api/migrations/NNN_description.sql` and apply in order. Prefer idempotent statements (`IF NOT EXISTS`) to make re-runs safe in shared dev environments.
