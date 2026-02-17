-- Event Indexer Tables
-- Tracks blockchain events and payment records synced from smart contracts

-- Payments table - stores all on-chain payment events
CREATE TABLE IF NOT EXISTS payments (
    id SERIAL PRIMARY KEY,
    tx_hash VARCHAR(66) NOT NULL UNIQUE,
    block_number BIGINT NOT NULL,
    song_id VARCHAR(66) NOT NULL,
    listener_address VARCHAR(42) NOT NULL,
    amount_wei VARCHAR(78) NOT NULL, -- Store as string to handle large numbers
    payment_type SMALLINT NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),

    -- Indexes for common queries
    CONSTRAINT valid_payment_type CHECK (payment_type >= 0 AND payment_type <= 4)
);

CREATE INDEX IF NOT EXISTS idx_payments_song_id ON payments(song_id);
CREATE INDEX IF NOT EXISTS idx_payments_listener ON payments(listener_address);
CREATE INDEX IF NOT EXISTS idx_payments_block ON payments(block_number);
CREATE INDEX IF NOT EXISTS idx_payments_timestamp ON payments(timestamp);

-- Indexed events table - tracks indexer progress
CREATE TABLE IF NOT EXISTS indexed_events (
    id SERIAL PRIMARY KEY,
    event_type VARCHAR(50) NOT NULL,
    block_number BIGINT NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_indexed_events_block ON indexed_events(block_number DESC);

-- Add on-chain registration tracking to songs table (if not exists)
-- This assumes a songs table exists - adjust as needed for your schema
DO $$
BEGIN
    IF EXISTS (SELECT FROM information_schema.tables WHERE table_name = 'songs') THEN
        IF NOT EXISTS (SELECT FROM information_schema.columns
                       WHERE table_name = 'songs' AND column_name = 'registered_on_chain') THEN
            ALTER TABLE songs ADD COLUMN registered_on_chain BOOLEAN DEFAULT FALSE;
        END IF;

        IF NOT EXISTS (SELECT FROM information_schema.columns
                       WHERE table_name = 'songs' AND column_name = 'registration_tx') THEN
            ALTER TABLE songs ADD COLUMN registration_tx VARCHAR(66);
        END IF;

        IF NOT EXISTS (SELECT FROM information_schema.columns
                       WHERE table_name = 'songs' AND column_name = 'registration_block') THEN
            ALTER TABLE songs ADD COLUMN registration_block BIGINT;
        END IF;

        IF NOT EXISTS (SELECT FROM information_schema.columns
                       WHERE table_name = 'songs' AND column_name = 'strategy_id') THEN
            ALTER TABLE songs ADD COLUMN strategy_id VARCHAR(66);
        END IF;
    END IF;
END $$;

-- Payment type enum reference (for documentation)
COMMENT ON TABLE payments IS 'On-chain payment events from EconomicStrategyRouter.
payment_type: 0=Stream, 1=Download, 2=Tip, 3=Patronage, 4=NFTAccess';

-- View for payment analytics
CREATE OR REPLACE VIEW payment_analytics AS
SELECT
    song_id,
    listener_address,
    payment_type,
    COUNT(*) as payment_count,
    SUM(CAST(amount_wei AS NUMERIC) / 1e18) as total_amount_eth,
    MIN(timestamp) as first_payment,
    MAX(timestamp) as last_payment
FROM payments
GROUP BY song_id, listener_address, payment_type;

-- View for artist earnings
CREATE OR REPLACE VIEW artist_earnings AS
SELECT
    s.artist_address,
    p.song_id,
    p.payment_type,
    COUNT(*) as stream_count,
    SUM(CAST(p.amount_wei AS NUMERIC) / 1e18) as total_earnings_eth,
    DATE_TRUNC('day', p.timestamp) as day
FROM payments p
JOIN songs s ON p.song_id = s.song_id
GROUP BY s.artist_address, p.song_id, p.payment_type, DATE_TRUNC('day', p.timestamp);
