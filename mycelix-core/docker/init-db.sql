-- =============================================================================
-- Mycelix-Core PostgreSQL Database Initialization
-- =============================================================================

-- Enable extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- =============================================================================
-- Node Management
-- =============================================================================

CREATE TABLE IF NOT EXISTS nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id VARCHAR(255) UNIQUE NOT NULL,
    public_key TEXT,
    capabilities JSONB DEFAULT '{}',
    trust_score DECIMAL(5,4) DEFAULT 1.0,
    status VARCHAR(50) DEFAULT 'active',
    rounds_participated INTEGER DEFAULT 0,
    last_seen TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_nodes_status ON nodes(status);
CREATE INDEX idx_nodes_trust_score ON nodes(trust_score);
CREATE INDEX idx_nodes_last_seen ON nodes(last_seen);

-- =============================================================================
-- Training Rounds
-- =============================================================================

CREATE TABLE IF NOT EXISTS training_rounds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    round_number INTEGER UNIQUE NOT NULL,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    participating_nodes TEXT[],
    byzantine_nodes TEXT[],
    convergence_score DECIMAL(10,8),
    aggregated_weights_hash VARCHAR(128),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_rounds_status ON training_rounds(status);
CREATE INDEX idx_rounds_number ON training_rounds(round_number);

-- =============================================================================
-- Model Updates
-- =============================================================================

CREATE TABLE IF NOT EXISTS model_updates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    round_id UUID REFERENCES training_rounds(id),
    node_id VARCHAR(255) NOT NULL,
    gradients_hash VARCHAR(128),
    metrics JSONB DEFAULT '{}',
    is_byzantine BOOLEAN DEFAULT FALSE,
    trust_score_at_submission DECIMAL(5,4),
    signature TEXT,
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_updates_round ON model_updates(round_id);
CREATE INDEX idx_updates_node ON model_updates(node_id);
CREATE INDEX idx_updates_byzantine ON model_updates(is_byzantine);

-- =============================================================================
-- Byzantine Detection Events
-- =============================================================================

CREATE TABLE IF NOT EXISTS byzantine_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    round_id UUID REFERENCES training_rounds(id),
    node_id VARCHAR(255) NOT NULL,
    detection_type VARCHAR(100),
    deviation_score DECIMAL(10,6),
    previous_trust_score DECIMAL(5,4),
    new_trust_score DECIMAL(5,4),
    evidence JSONB DEFAULT '{}',
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_byzantine_node ON byzantine_events(node_id);
CREATE INDEX idx_byzantine_round ON byzantine_events(round_id);
CREATE INDEX idx_byzantine_type ON byzantine_events(detection_type);

-- =============================================================================
-- Model Versions
-- =============================================================================

CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(50) NOT NULL,
    round_number INTEGER,
    weights_hash VARCHAR(128),
    performance_metrics JSONB DEFAULT '{}',
    is_current BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_model_version ON model_versions(version);
CREATE INDEX idx_model_current ON model_versions(is_current);

-- =============================================================================
-- Audit Log
-- =============================================================================

CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type VARCHAR(100) NOT NULL,
    actor_id VARCHAR(255),
    target_type VARCHAR(100),
    target_id VARCHAR(255),
    details JSONB DEFAULT '{}',
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_audit_event ON audit_log(event_type);
CREATE INDEX idx_audit_actor ON audit_log(actor_id);
CREATE INDEX idx_audit_created ON audit_log(created_at);

-- =============================================================================
-- Functions
-- =============================================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER nodes_updated_at
    BEFORE UPDATE ON nodes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- Calculate average trust score
CREATE OR REPLACE FUNCTION avg_trust_score()
RETURNS DECIMAL AS $$
BEGIN
    RETURN (SELECT AVG(trust_score) FROM nodes WHERE status = 'active');
END;
$$ LANGUAGE plpgsql;

-- Get byzantine rate for recent rounds
CREATE OR REPLACE FUNCTION byzantine_rate(num_rounds INTEGER DEFAULT 100)
RETURNS DECIMAL AS $$
DECLARE
    total_updates INTEGER;
    byzantine_updates INTEGER;
BEGIN
    SELECT COUNT(*) INTO total_updates
    FROM model_updates mu
    JOIN training_rounds tr ON mu.round_id = tr.id
    WHERE tr.round_number > (SELECT MAX(round_number) - num_rounds FROM training_rounds);

    SELECT COUNT(*) INTO byzantine_updates
    FROM model_updates mu
    JOIN training_rounds tr ON mu.round_id = tr.id
    WHERE tr.round_number > (SELECT MAX(round_number) - num_rounds FROM training_rounds)
    AND mu.is_byzantine = TRUE;

    IF total_updates = 0 THEN
        RETURN 0;
    END IF;

    RETURN byzantine_updates::DECIMAL / total_updates::DECIMAL;
END;
$$ LANGUAGE plpgsql;

-- =============================================================================
-- Initial Data
-- =============================================================================

-- Insert initial model version
INSERT INTO model_versions (version, round_number, is_current, performance_metrics)
VALUES ('1.0.0', 0, TRUE, '{"accuracy": 0, "loss": 0}')
ON CONFLICT DO NOTHING;

-- Log database initialization
INSERT INTO audit_log (event_type, actor_id, details)
VALUES ('database_initialized', 'system', '{"version": "1.0.0", "timestamp": "' || NOW() || '"}');

-- =============================================================================
-- Grants (if needed for separate users)
-- =============================================================================

-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mycelix;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mycelix;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO mycelix;

COMMIT;
