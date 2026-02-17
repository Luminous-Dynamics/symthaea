-- Mycelix Testnet - PostgreSQL Initialization Script

-- Create extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- ===========================================
-- NODES TABLE
-- ===========================================
CREATE TABLE IF NOT EXISTS nodes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    node_id VARCHAR(255) UNIQUE NOT NULL,
    public_key TEXT NOT NULL,
    node_type VARCHAR(50) NOT NULL CHECK (node_type IN ('bootstrap', 'seed', 'validator', 'participant')),
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'inactive', 'banned')),
    endpoints JSONB NOT NULL DEFAULT '[]',
    metadata JSONB DEFAULT '{}',
    reputation_score DECIMAL(10, 4) DEFAULT 100.0000,
    first_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_seen_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_nodes_node_id ON nodes(node_id);
CREATE INDEX idx_nodes_status ON nodes(status);
CREATE INDEX idx_nodes_node_type ON nodes(node_type);
CREATE INDEX idx_nodes_last_seen ON nodes(last_seen_at);

-- ===========================================
-- BLOCKS TABLE
-- ===========================================
CREATE TABLE IF NOT EXISTS blocks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    block_number BIGINT UNIQUE NOT NULL,
    block_hash VARCHAR(66) UNIQUE NOT NULL,
    parent_hash VARCHAR(66) NOT NULL,
    state_root VARCHAR(66) NOT NULL,
    transactions_root VARCHAR(66) NOT NULL,
    proposer_id VARCHAR(255) NOT NULL,
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    transactions_count INTEGER DEFAULT 0,
    size_bytes INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_blocks_number ON blocks(block_number DESC);
CREATE INDEX idx_blocks_hash ON blocks(block_hash);
CREATE INDEX idx_blocks_timestamp ON blocks(timestamp DESC);
CREATE INDEX idx_blocks_proposer ON blocks(proposer_id);

-- ===========================================
-- FL ROUNDS TABLE
-- ===========================================
CREATE TABLE IF NOT EXISTS fl_rounds (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    round_number BIGINT UNIQUE NOT NULL,
    status VARCHAR(50) NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'active', 'aggregating', 'completed', 'failed')),
    model_version VARCHAR(100) NOT NULL,
    participants_count INTEGER DEFAULT 0,
    gradients_received INTEGER DEFAULT 0,
    aggregation_method VARCHAR(50) DEFAULT 'fedavg',
    started_at TIMESTAMP WITH TIME ZONE,
    completed_at TIMESTAMP WITH TIME ZONE,
    aggregated_gradient_hash VARCHAR(66),
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_fl_rounds_number ON fl_rounds(round_number DESC);
CREATE INDEX idx_fl_rounds_status ON fl_rounds(status);
CREATE INDEX idx_fl_rounds_started ON fl_rounds(started_at DESC);

-- ===========================================
-- GRADIENT SUBMISSIONS TABLE
-- ===========================================
CREATE TABLE IF NOT EXISTS gradient_submissions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    round_id UUID REFERENCES fl_rounds(id) ON DELETE CASCADE,
    node_id VARCHAR(255) NOT NULL,
    gradient_hash VARCHAR(66) NOT NULL,
    gradient_size_bytes INTEGER NOT NULL,
    validation_status VARCHAR(50) DEFAULT 'pending' CHECK (validation_status IN ('pending', 'valid', 'invalid', 'rejected')),
    validation_score DECIMAL(10, 4),
    submitted_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    validated_at TIMESTAMP WITH TIME ZONE,
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_gradient_submissions_round ON gradient_submissions(round_id);
CREATE INDEX idx_gradient_submissions_node ON gradient_submissions(node_id);
CREATE INDEX idx_gradient_submissions_status ON gradient_submissions(validation_status);

-- ===========================================
-- FAUCET REQUESTS TABLE
-- ===========================================
CREATE TABLE IF NOT EXISTS faucet_requests (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    address VARCHAR(255) NOT NULL,
    ip_address INET NOT NULL,
    amount NUMERIC(78, 0) NOT NULL,
    transaction_hash VARCHAR(66),
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'completed', 'failed', 'rejected')),
    rejection_reason TEXT,
    requested_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    processed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_faucet_requests_address ON faucet_requests(address);
CREATE INDEX idx_faucet_requests_ip ON faucet_requests(ip_address);
CREATE INDEX idx_faucet_requests_requested ON faucet_requests(requested_at DESC);
CREATE INDEX idx_faucet_requests_status ON faucet_requests(status);

-- Rate limiting view for faucet
CREATE OR REPLACE VIEW faucet_rate_limits AS
SELECT
    address,
    ip_address,
    COUNT(*) as request_count,
    MAX(requested_at) as last_request,
    NOW() - MAX(requested_at) as time_since_last_request
FROM faucet_requests
WHERE requested_at > NOW() - INTERVAL '24 hours'
  AND status IN ('completed', 'pending')
GROUP BY address, ip_address;

-- ===========================================
-- ACCOUNTS TABLE (for token balances)
-- ===========================================
CREATE TABLE IF NOT EXISTS accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    address VARCHAR(255) UNIQUE NOT NULL,
    balance NUMERIC(78, 0) NOT NULL DEFAULT 0,
    nonce BIGINT NOT NULL DEFAULT 0,
    node_id VARCHAR(255),
    account_type VARCHAR(50) DEFAULT 'user' CHECK (account_type IN ('user', 'node', 'faucet', 'treasury')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_accounts_address ON accounts(address);
CREATE INDEX idx_accounts_node ON accounts(node_id);

-- ===========================================
-- TRANSACTIONS TABLE
-- ===========================================
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    tx_hash VARCHAR(66) UNIQUE NOT NULL,
    block_id UUID REFERENCES blocks(id) ON DELETE SET NULL,
    tx_type VARCHAR(50) NOT NULL,
    from_address VARCHAR(255) NOT NULL,
    to_address VARCHAR(255),
    value NUMERIC(78, 0) DEFAULT 0,
    data BYTEA,
    nonce BIGINT NOT NULL,
    status VARCHAR(50) DEFAULT 'pending' CHECK (status IN ('pending', 'included', 'confirmed', 'failed')),
    gas_used INTEGER,
    error_message TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_transactions_hash ON transactions(tx_hash);
CREATE INDEX idx_transactions_block ON transactions(block_id);
CREATE INDEX idx_transactions_from ON transactions(from_address);
CREATE INDEX idx_transactions_to ON transactions(to_address);
CREATE INDEX idx_transactions_status ON transactions(status);

-- ===========================================
-- RATE LIMITING TABLE
-- ===========================================
CREATE TABLE IF NOT EXISTS rate_limits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    identifier VARCHAR(255) NOT NULL,
    identifier_type VARCHAR(50) NOT NULL CHECK (identifier_type IN ('ip', 'node', 'address', 'api_key')),
    action_type VARCHAR(100) NOT NULL,
    request_count INTEGER NOT NULL DEFAULT 0,
    window_start TIMESTAMP WITH TIME ZONE NOT NULL,
    window_end TIMESTAMP WITH TIME ZONE NOT NULL,
    blocked_until TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE (identifier, identifier_type, action_type, window_start)
);

CREATE INDEX idx_rate_limits_identifier ON rate_limits(identifier, identifier_type);
CREATE INDEX idx_rate_limits_window ON rate_limits(window_start, window_end);
CREATE INDEX idx_rate_limits_blocked ON rate_limits(blocked_until);

-- ===========================================
-- NETWORK METRICS TABLE
-- ===========================================
CREATE TABLE IF NOT EXISTS network_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    metric_name VARCHAR(100) NOT NULL,
    metric_value DECIMAL(20, 6) NOT NULL,
    dimensions JSONB DEFAULT '{}',
    recorded_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX idx_network_metrics_name ON network_metrics(metric_name);
CREATE INDEX idx_network_metrics_recorded ON network_metrics(recorded_at DESC);

-- Partition by time for better performance
CREATE INDEX idx_network_metrics_name_time ON network_metrics(metric_name, recorded_at DESC);

-- ===========================================
-- FUNCTIONS AND TRIGGERS
-- ===========================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply trigger to relevant tables
CREATE TRIGGER update_nodes_updated_at BEFORE UPDATE ON nodes
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_accounts_updated_at BEFORE UPDATE ON accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_rate_limits_updated_at BEFORE UPDATE ON rate_limits
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to check faucet rate limit
CREATE OR REPLACE FUNCTION check_faucet_rate_limit(
    p_address VARCHAR(255),
    p_ip INET,
    p_cooldown_hours INTEGER DEFAULT 24,
    p_max_per_address INTEGER DEFAULT 10
) RETURNS BOOLEAN AS $$
DECLARE
    recent_count INTEGER;
    address_total INTEGER;
BEGIN
    -- Check recent requests from IP
    SELECT COUNT(*) INTO recent_count
    FROM faucet_requests
    WHERE ip_address = p_ip
      AND requested_at > NOW() - (p_cooldown_hours || ' hours')::INTERVAL
      AND status IN ('completed', 'pending');

    IF recent_count > 0 THEN
        RETURN FALSE;
    END IF;

    -- Check total requests from address
    SELECT COUNT(*) INTO address_total
    FROM faucet_requests
    WHERE address = p_address
      AND status = 'completed';

    IF address_total >= p_max_per_address THEN
        RETURN FALSE;
    END IF;

    RETURN TRUE;
END;
$$ LANGUAGE plpgsql;

-- ===========================================
-- INITIAL DATA
-- ===========================================

-- Insert faucet account
INSERT INTO accounts (address, balance, account_type)
VALUES ('0xFAUCET', 1000000000000000000000000000, 'faucet')
ON CONFLICT (address) DO NOTHING;

-- Insert treasury account
INSERT INTO accounts (address, balance, account_type)
VALUES ('0xTREASURY', 0, 'treasury')
ON CONFLICT (address) DO NOTHING;

-- Grant permissions
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO mycelix;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO mycelix;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO mycelix;
