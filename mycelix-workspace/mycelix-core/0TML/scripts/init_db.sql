-- ZeroTrustML Phase 9 - Database Initialization Script
-- PostgreSQL schema for credits system

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ============================================
-- Credits Table
-- ============================================
CREATE TABLE IF NOT EXISTS credits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    holder TEXT NOT NULL,
    amount BIGINT NOT NULL CHECK (amount > 0),
    earned_from TEXT NOT NULL,
    timestamp BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes for performance
    INDEX idx_credits_holder (holder),
    INDEX idx_credits_timestamp (timestamp),
    INDEX idx_credits_earned_from (earned_from)
);

-- ============================================
-- Transactions Table
-- ============================================
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    from_holder TEXT,
    to_holder TEXT NOT NULL,
    amount BIGINT NOT NULL CHECK (amount > 0),
    transaction_type TEXT NOT NULL,
    status TEXT DEFAULT 'pending',
    metadata JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    
    -- Constraints
    CHECK (status IN ('pending', 'completed', 'failed')),
    CHECK (transaction_type IN ('issuance', 'transfer', 'burn')),
    
    -- Indexes
    INDEX idx_transactions_from (from_holder),
    INDEX idx_transactions_to (to_holder),
    INDEX idx_transactions_status (status),
    INDEX idx_transactions_type (transaction_type),
    INDEX idx_transactions_created (created_at)
);

-- ============================================
-- Reputation Table
-- ============================================
CREATE TABLE IF NOT EXISTS reputation (
    node_id TEXT PRIMARY KEY,
    score REAL NOT NULL DEFAULT 0.5 CHECK (score >= 0 AND score <= 1),
    total_submissions BIGINT DEFAULT 0,
    accepted_submissions BIGINT DEFAULT 0,
    rejected_submissions BIGINT DEFAULT 0,
    byzantine_detections BIGINT DEFAULT 0,
    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Indexes
    INDEX idx_reputation_score (score),
    INDEX idx_reputation_updated (last_updated)
);

-- ============================================
-- Gradients Table (audit trail)
-- ============================================
CREATE TABLE IF NOT EXISTS gradients (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id TEXT NOT NULL,
    round_num INTEGER NOT NULL,
    gradient_hash TEXT NOT NULL,
    pogq_score REAL,
    validation_status TEXT DEFAULT 'pending',
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    validated_at TIMESTAMP,
    
    -- Constraints
    CHECK (validation_status IN ('pending', 'accepted', 'rejected', 'byzantine')),
    CHECK (pogq_score IS NULL OR (pogq_score >= 0 AND pogq_score <= 1)),
    
    -- Indexes
    INDEX idx_gradients_node (node_id),
    INDEX idx_gradients_round (round_num),
    INDEX idx_gradients_status (validation_status),
    INDEX idx_gradients_submitted (submitted_at)
);

-- ============================================
-- Byzantine Events Table
-- ============================================
CREATE TABLE IF NOT EXISTS byzantine_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id TEXT NOT NULL,
    round_num INTEGER NOT NULL,
    detection_method TEXT NOT NULL,
    severity TEXT NOT NULL,
    details JSONB,
    detected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    
    -- Constraints
    CHECK (detection_method IN ('pogq', 'reputation', 'anomaly', 'manual')),
    CHECK (severity IN ('low', 'medium', 'high', 'critical')),
    
    -- Indexes
    INDEX idx_byzantine_node (node_id),
    INDEX idx_byzantine_round (round_num),
    INDEX idx_byzantine_method (detection_method),
    INDEX idx_byzantine_severity (severity),
    INDEX idx_byzantine_detected (detected_at)
);

-- ============================================
-- Materialized Views for Performance
-- ============================================

-- Balance view (aggregated credits per holder)
CREATE MATERIALIZED VIEW IF NOT EXISTS balances AS
SELECT 
    holder,
    SUM(amount) as total_balance,
    COUNT(*) as credit_count,
    MAX(timestamp) as last_credit_timestamp
FROM credits
GROUP BY holder;

CREATE UNIQUE INDEX ON balances (holder);

-- Node statistics view
CREATE MATERIALIZED VIEW IF NOT EXISTS node_stats AS
SELECT 
    r.node_id,
    r.score as reputation_score,
    r.total_submissions,
    r.accepted_submissions,
    r.byzantine_detections,
    COUNT(DISTINCT g.round_num) as rounds_participated,
    AVG(g.pogq_score) as avg_pogq_score,
    COUNT(CASE WHEN g.validation_status = 'byzantine' THEN 1 END) as byzantine_count
FROM reputation r
LEFT JOIN gradients g ON r.node_id = g.node_id
GROUP BY r.node_id, r.score, r.total_submissions, r.accepted_submissions, r.byzantine_detections;

CREATE UNIQUE INDEX ON node_stats (node_id);

-- ============================================
-- Functions
-- ============================================

-- Function to refresh materialized views
CREATE OR REPLACE FUNCTION refresh_views()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY balances;
    REFRESH MATERIALIZED VIEW CONCURRENTLY node_stats;
END;
$$ LANGUAGE plpgsql;

-- Function to get node balance
CREATE OR REPLACE FUNCTION get_balance(holder_id TEXT)
RETURNS BIGINT AS $$
    SELECT COALESCE(SUM(amount), 0) FROM credits WHERE holder = holder_id;
$$ LANGUAGE sql STABLE;

-- Function to issue credit
CREATE OR REPLACE FUNCTION issue_credit(
    p_holder TEXT,
    p_amount BIGINT,
    p_earned_from TEXT,
    p_timestamp BIGINT DEFAULT NULL
)
RETURNS UUID AS $$
DECLARE
    v_credit_id UUID;
BEGIN
    INSERT INTO credits (holder, amount, earned_from, timestamp)
    VALUES (
        p_holder,
        p_amount,
        p_earned_from,
        COALESCE(p_timestamp, EXTRACT(EPOCH FROM CURRENT_TIMESTAMP)::BIGINT * 1000000)
    )
    RETURNING id INTO v_credit_id;
    
    -- Record transaction
    INSERT INTO transactions (to_holder, amount, transaction_type, status, completed_at)
    VALUES (p_holder, p_amount, 'issuance', 'completed', CURRENT_TIMESTAMP);
    
    RETURN v_credit_id;
END;
$$ LANGUAGE plpgsql;

-- Function to update reputation
CREATE OR REPLACE FUNCTION update_reputation(
    p_node_id TEXT,
    p_score_delta REAL,
    p_submission_accepted BOOLEAN DEFAULT NULL
)
RETURNS void AS $$
BEGIN
    INSERT INTO reputation (node_id, score, total_submissions, accepted_submissions, rejected_submissions)
    VALUES (
        p_node_id,
        GREATEST(0.0, LEAST(1.0, 0.5 + p_score_delta)),
        CASE WHEN p_submission_accepted IS NOT NULL THEN 1 ELSE 0 END,
        CASE WHEN p_submission_accepted = TRUE THEN 1 ELSE 0 END,
        CASE WHEN p_submission_accepted = FALSE THEN 1 ELSE 0 END
    )
    ON CONFLICT (node_id) DO UPDATE SET
        score = GREATEST(0.0, LEAST(1.0, reputation.score + p_score_delta)),
        total_submissions = reputation.total_submissions + CASE WHEN p_submission_accepted IS NOT NULL THEN 1 ELSE 0 END,
        accepted_submissions = reputation.accepted_submissions + CASE WHEN p_submission_accepted = TRUE THEN 1 ELSE 0 END,
        rejected_submissions = reputation.rejected_submissions + CASE WHEN p_submission_accepted = FALSE THEN 1 ELSE 0 END,
        last_updated = CURRENT_TIMESTAMP;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- Triggers
-- ============================================

-- Auto-refresh views after credits insert
CREATE OR REPLACE FUNCTION trigger_refresh_balances()
RETURNS TRIGGER AS $$
BEGIN
    -- Refresh every 100 credits to avoid overhead
    IF (SELECT COUNT(*) FROM credits) % 100 = 0 THEN
        PERFORM refresh_views();
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER refresh_balances_trigger
AFTER INSERT ON credits
FOR EACH STATEMENT
EXECUTE FUNCTION trigger_refresh_balances();

-- ============================================
-- Initial Data
-- ============================================

-- Create default admin account (optional)
-- INSERT INTO reputation (node_id, score, total_submissions)
-- VALUES ('admin', 1.0, 0);

-- ============================================
-- Grants
-- ============================================

GRANT SELECT, INSERT, UPDATE ON ALL TABLES IN SCHEMA public TO zerotrustml;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO zerotrustml;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO zerotrustml;

-- ============================================
-- Completion
-- ============================================

SELECT 'ZeroTrustML database initialized successfully!' as status;
