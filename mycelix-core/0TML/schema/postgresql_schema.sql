-- ZeroTrustML PostgreSQL Schema
-- Phase 10 Multi-Backend Implementation
-- Created: October 2, 2025

-- Drop existing tables if they exist (for clean schema migration)
DROP TABLE IF EXISTS byzantine_events CASCADE;
DROP TABLE IF EXISTS credits CASCADE;
DROP TABLE IF EXISTS gradients CASCADE;

-- Gradients table: Stores federated learning gradients
-- Note: gradient_data may contain AES-256-GCM encrypted data (base64 encoded)
-- when encryption is enabled for HIPAA/GDPR compliance
CREATE TABLE gradients (
    id VARCHAR(64) PRIMARY KEY,
    node_id VARCHAR(64) NOT NULL,
    round_num INTEGER NOT NULL,
    gradient_data TEXT NOT NULL,  -- JSON-encoded gradient array (may be encrypted base64)
    gradient_hash VARCHAR(64) NOT NULL,  -- SHA-256 hash of plaintext gradient for integrity
    pogq_score INTEGER,
    zkpoc_verified BOOLEAN DEFAULT FALSE,
    validation_passed BOOLEAN DEFAULT TRUE,
    reputation_score REAL DEFAULT 0.5,
    submitted_at TIMESTAMP NOT NULL DEFAULT NOW(),
    encrypted BOOLEAN DEFAULT FALSE  -- Indicates if gradient_data is AES-256-GCM encrypted
);

-- Indexes for gradients table
CREATE INDEX idx_gradients_round ON gradients(round_num);
CREATE INDEX idx_gradients_node ON gradients(node_id);
CREATE INDEX idx_gradients_submitted ON gradients(submitted_at);

-- Credits table: Tracks credit issuance and balances
CREATE TABLE credits (
    id SERIAL PRIMARY KEY,
    holder VARCHAR(64) NOT NULL,
    amount INTEGER NOT NULL,
    earned_from VARCHAR(255) NOT NULL,
    issued_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for credits table
CREATE INDEX idx_credits_holder ON credits(holder);
CREATE INDEX idx_credits_issued ON credits(issued_at);

-- Byzantine events table: Logs Byzantine fault detections
CREATE TABLE byzantine_events (
    id SERIAL PRIMARY KEY,
    node_id VARCHAR(64) NOT NULL,
    round_num INTEGER NOT NULL,
    detection_method VARCHAR(50) NOT NULL,
    severity VARCHAR(20) NOT NULL,
    details TEXT,  -- JSON-encoded event details
    detected_at TIMESTAMP NOT NULL DEFAULT NOW()
);

-- Indexes for byzantine_events table
CREATE INDEX idx_byzantine_node ON byzantine_events(node_id);
CREATE INDEX idx_byzantine_round ON byzantine_events(round_num);
CREATE INDEX idx_byzantine_detected ON byzantine_events(detected_at);

-- Views for common queries
CREATE OR REPLACE VIEW gradient_summary AS
SELECT
    node_id,
    COUNT(*) as total_submitted,
    SUM(CASE WHEN validation_passed THEN 1 ELSE 0 END) as accepted,
    AVG(pogq_score) as avg_pogq_score,
    MAX(round_num) as latest_round
FROM gradients
GROUP BY node_id;

CREATE OR REPLACE VIEW credit_balances AS
SELECT
    holder,
    SUM(amount) as total_balance,
    COUNT(*) as transaction_count,
    MAX(issued_at) as last_credit
FROM credits
GROUP BY holder;

-- Grant permissions (adjust user as needed)
-- GRANT ALL ON ALL TABLES IN SCHEMA public TO zerotrustml;
-- GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO zerotrustml;

-- Insert test data (optional, for verification)
-- INSERT INTO gradients (id, node_id, round_num, gradient_data, gradient_hash, pogq_score)
-- VALUES ('test123', 'node456', 1, '[1.0, 2.0, 3.0]', 'abc123', 850);

COMMENT ON TABLE gradients IS 'Federated learning gradient submissions';
COMMENT ON TABLE credits IS 'ZeroTrustML credit issuance and balances';
COMMENT ON TABLE byzantine_events IS 'Byzantine fault detection events';
