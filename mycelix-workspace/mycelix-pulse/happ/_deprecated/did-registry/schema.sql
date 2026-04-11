-- Mycelix Mail DID Registry Schema
-- Layer 5 (Identity) Integration
-- Maps Decentralized Identifiers (DIDs) to Holochain AgentPubKeys

-- Extension for UUID support
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Main DID registry table
CREATE TABLE did_registry (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    did TEXT NOT NULL UNIQUE,
    agent_pubkey TEXT NOT NULL,
    created_at TIMESTAMP NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    last_seen TIMESTAMP,

    -- DID metadata
    did_method TEXT NOT NULL DEFAULT 'mycelix',
    did_method_specific_id TEXT NOT NULL,

    -- Optional profile information
    display_name TEXT,
    email_alias TEXT,

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT true,

    -- Indexing
    CONSTRAINT did_format_check CHECK (did ~ '^did:[a-z0-9]+:[a-zA-Z0-9._-]+$')
);

-- Index for fast DID lookups
CREATE INDEX idx_did_registry_did ON did_registry(did);
CREATE INDEX idx_did_registry_agent_pubkey ON did_registry(agent_pubkey);
CREATE INDEX idx_did_registry_active ON did_registry(is_active) WHERE is_active = true;

-- DID resolution history (for debugging/auditing)
CREATE TABLE did_resolution_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    did TEXT NOT NULL,
    resolved_pubkey TEXT,
    resolution_time TIMESTAMP NOT NULL DEFAULT NOW(),
    success BOOLEAN NOT NULL,
    error_message TEXT,

    -- Request metadata
    request_source TEXT,
    request_ip TEXT
);

-- Index for log queries
CREATE INDEX idx_resolution_log_did ON did_resolution_log(did);
CREATE INDEX idx_resolution_log_time ON did_resolution_log(resolution_time DESC);

-- DID update history (for tracking changes)
CREATE TABLE did_update_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    did TEXT NOT NULL,
    old_agent_pubkey TEXT,
    new_agent_pubkey TEXT NOT NULL,
    updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
    update_reason TEXT,

    -- For key rotation
    rotation_signature TEXT,

    FOREIGN KEY (did) REFERENCES did_registry(did) ON DELETE CASCADE
);

-- Index for history queries
CREATE INDEX idx_update_history_did ON did_update_history(did);
CREATE INDEX idx_update_history_time ON did_update_history(updated_at DESC);

-- Function to automatically update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to auto-update updated_at
CREATE TRIGGER update_did_registry_updated_at
    BEFORE UPDATE ON did_registry
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Function to log DID updates
CREATE OR REPLACE FUNCTION log_did_update()
RETURNS TRIGGER AS $$
BEGIN
    IF OLD.agent_pubkey != NEW.agent_pubkey THEN
        INSERT INTO did_update_history (did, old_agent_pubkey, new_agent_pubkey, update_reason)
        VALUES (NEW.did, OLD.agent_pubkey, NEW.agent_pubkey, 'Key rotation');
    END IF;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger to log updates
CREATE TRIGGER log_did_registry_updates
    AFTER UPDATE ON did_registry
    FOR EACH ROW
    EXECUTE FUNCTION log_did_update();

-- Seed data for testing (example DIDs)
INSERT INTO did_registry (did, agent_pubkey, did_method, did_method_specific_id, display_name) VALUES
    ('did:mycelix:alice', 'uhCAkRMhKv7C4P3sYwQi3JLR5xZJBkqXh8ZYzT0UqN8VRw_M6dBmK', 'mycelix', 'alice', 'Alice (Test User)'),
    ('did:mycelix:bob', 'uhCAkNP8sT2wV9xK4mQ7jR6pYvH5nL0dFgA3cB1eZ8uI7oE4rS2t', 'mycelix', 'bob', 'Bob (Test User)'),
    ('did:mycelix:carol', 'uhCAkQW1yE6tR5uY8iO0pA3sD4fG5hJ6kL7zX9cV2bN4mM8wT7q', 'mycelix', 'carol', 'Carol (Test User)')
ON CONFLICT (did) DO NOTHING;

-- Views for common queries

-- Active DIDs view
CREATE VIEW active_dids AS
SELECT did, agent_pubkey, display_name, email_alias, last_seen
FROM did_registry
WHERE is_active = true
ORDER BY last_seen DESC NULLS LAST;

-- Recent resolutions view
CREATE VIEW recent_resolutions AS
SELECT
    did,
    resolved_pubkey,
    resolution_time,
    success,
    error_message
FROM did_resolution_log
ORDER BY resolution_time DESC
LIMIT 1000;

-- DID statistics view
CREATE VIEW did_statistics AS
SELECT
    COUNT(*) as total_dids,
    COUNT(*) FILTER (WHERE is_active = true) as active_dids,
    COUNT(*) FILTER (WHERE last_seen > NOW() - INTERVAL '7 days') as active_last_week,
    COUNT(*) FILTER (WHERE last_seen > NOW() - INTERVAL '1 day') as active_last_day
FROM did_registry;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE ON did_registry TO mycelix_mail_app;
-- GRANT SELECT, INSERT ON did_resolution_log TO mycelix_mail_app;
-- GRANT SELECT ON did_update_history TO mycelix_mail_app;
-- GRANT SELECT ON active_dids, recent_resolutions, did_statistics TO mycelix_mail_app;

-- Comments for documentation
COMMENT ON TABLE did_registry IS 'Maps Decentralized Identifiers to Holochain AgentPubKeys';
COMMENT ON TABLE did_resolution_log IS 'Audit log of DID resolution requests';
COMMENT ON TABLE did_update_history IS 'History of DID-to-AgentPubKey mappings changes';
COMMENT ON COLUMN did_registry.did IS 'Fully-qualified DID (e.g., did:mycelix:alice)';
COMMENT ON COLUMN did_registry.agent_pubkey IS 'Base64-encoded Holochain AgentPubKey';
COMMENT ON COLUMN did_registry.last_seen IS 'Last time this DID was successfully used';
