CREATE TABLE IF NOT EXISTS strategy_configs (
  id SERIAL PRIMARY KEY,
  name TEXT NOT NULL,
  payload JSONB NOT NULL,
  published BOOLEAN DEFAULT FALSE,
  created_at TIMESTAMP DEFAULT NOW(),
  published_at TIMESTAMP
);
