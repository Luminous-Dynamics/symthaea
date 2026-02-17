-- Mycelix Database Initialization Script
-- This script runs on first PostgreSQL container startup

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS mycelix;
CREATE SCHEMA IF NOT EXISTS fl;
CREATE SCHEMA IF NOT EXISTS consensus;

-- Set search path
SET search_path TO mycelix, public;

-- Validators table
CREATE TABLE IF NOT EXISTS mycelix.validators (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id VARCHAR(255) UNIQUE NOT NULL,
    public_key BYTEA NOT NULL,
    reputation_score DECIMAL(5,4) DEFAULT 0.5,
    stake_amount BIGINT DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    registered_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_seen_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_validators_node_id ON mycelix.validators(node_id);
CREATE INDEX idx_validators_active ON mycelix.validators(is_active);
CREATE INDEX idx_validators_reputation ON mycelix.validators(reputation_score DESC);

-- Consensus rounds table
CREATE TABLE IF NOT EXISTS consensus.rounds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    round_number BIGINT UNIQUE NOT NULL,
    view_number BIGINT NOT NULL,
    proposer_id UUID REFERENCES mycelix.validators(id),
    block_hash BYTEA,
    status VARCHAR(50) DEFAULT 'pending',
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    finalized_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_rounds_number ON consensus.rounds(round_number DESC);
CREATE INDEX idx_rounds_status ON consensus.rounds(status);

-- FL rounds table
CREATE TABLE IF NOT EXISTS fl.rounds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    round_number BIGINT UNIQUE NOT NULL,
    coordinator_id VARCHAR(255),
    status VARCHAR(50) DEFAULT 'pending',
    participant_count INTEGER DEFAULT 0,
    accepted_count INTEGER DEFAULT 0,
    rejected_count INTEGER DEFAULT 0,
    aggregation_method VARCHAR(50),
    started_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_fl_rounds_number ON fl.rounds(round_number DESC);
CREATE INDEX idx_fl_rounds_status ON fl.rounds(status);

-- FL submissions table
CREATE TABLE IF NOT EXISTS fl.submissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    round_id UUID REFERENCES fl.rounds(id),
    participant_id VARCHAR(255) NOT NULL,
    gradient_hash BYTEA NOT NULL,
    gradient_size BIGINT,
    trust_score DECIMAL(5,4),
    proof_valid BOOLEAN,
    status VARCHAR(50) DEFAULT 'pending',
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    processed_at TIMESTAMP WITH TIME ZONE,
    rejection_reason TEXT,
    metadata JSONB DEFAULT '{}'
);

CREATE INDEX idx_submissions_round ON fl.submissions(round_id);
CREATE INDEX idx_submissions_participant ON fl.submissions(participant_id);
CREATE INDEX idx_submissions_status ON fl.submissions(status);

-- Byzantine detections table
CREATE TABLE IF NOT EXISTS fl.byzantine_detections (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    submission_id UUID REFERENCES fl.submissions(id),
    detector_type VARCHAR(100) NOT NULL,
    confidence DECIMAL(5,4),
    details JSONB,
    detected_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_byzantine_submission ON fl.byzantine_detections(submission_id);
CREATE INDEX idx_byzantine_type ON fl.byzantine_detections(detector_type);

-- Trust scores history
CREATE TABLE IF NOT EXISTS mycelix.trust_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    participant_id VARCHAR(255) NOT NULL,
    score DECIMAL(5,4) NOT NULL,
    pogq_component DECIMAL(5,4),
    tcdm_component DECIMAL(5,4),
    entropy_component DECIMAL(5,4),
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_trust_history_participant ON mycelix.trust_history(participant_id);
CREATE INDEX idx_trust_history_time ON mycelix.trust_history(recorded_at DESC);

-- Metrics table (for quick queries)
CREATE TABLE IF NOT EXISTS mycelix.metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(255) NOT NULL,
    metric_value DECIMAL(20,6),
    labels JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_metrics_name ON mycelix.metrics(metric_name);
CREATE INDEX idx_metrics_time ON mycelix.metrics(recorded_at DESC);

-- Create read-only user for monitoring
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mycelix_readonly') THEN
        CREATE ROLE mycelix_readonly WITH LOGIN PASSWORD 'readonly';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA mycelix, fl, consensus TO mycelix_readonly;
GRANT SELECT ON ALL TABLES IN SCHEMA mycelix, fl, consensus TO mycelix_readonly;
ALTER DEFAULT PRIVILEGES IN SCHEMA mycelix, fl, consensus GRANT SELECT ON TABLES TO mycelix_readonly;

-- Create application user
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_catalog.pg_roles WHERE rolname = 'mycelix_app') THEN
        CREATE ROLE mycelix_app WITH LOGIN PASSWORD 'app_password';
    END IF;
END
$$;

GRANT USAGE ON SCHEMA mycelix, fl, consensus TO mycelix_app;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA mycelix, fl, consensus TO mycelix_app;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA mycelix, fl, consensus TO mycelix_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA mycelix, fl, consensus GRANT ALL ON TABLES TO mycelix_app;
ALTER DEFAULT PRIVILEGES IN SCHEMA mycelix, fl, consensus GRANT ALL ON SEQUENCES TO mycelix_app;

-- Insert initial data for development
INSERT INTO mycelix.validators (node_id, public_key, reputation_score, is_active)
VALUES
    ('validator-dev-1', decode('0102030405060708', 'hex'), 0.85, true),
    ('validator-dev-2', decode('0203040506070809', 'hex'), 0.82, true),
    ('validator-dev-3', decode('030405060708090a', 'hex'), 0.78, true)
ON CONFLICT (node_id) DO NOTHING;

-- Done
SELECT 'Database initialization complete' AS status;
