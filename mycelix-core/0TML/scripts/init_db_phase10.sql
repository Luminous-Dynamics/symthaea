-- ZeroTrustML Phase 10 - Database Extensions for Hybrid Mode
-- Adds Holochain hash tracking and ZK-PoC metadata

-- ============================================
-- Phase 10: Add Holochain Hash Columns
-- ============================================

-- Link gradients to Holochain immutable entries
ALTER TABLE gradients ADD COLUMN IF NOT EXISTS
    holochain_hash TEXT;

-- Link credits to Holochain
ALTER TABLE credits ADD COLUMN IF NOT EXISTS
    holochain_hash TEXT;

-- Link transactions to Holochain
ALTER TABLE transactions ADD COLUMN IF NOT EXISTS
    holochain_hash TEXT;

-- Link Byzantine events to Holochain
ALTER TABLE byzantine_events ADD COLUMN IF NOT EXISTS
    holochain_hash TEXT;

-- Indexes for Holochain lookups
CREATE INDEX IF NOT EXISTS idx_gradients_holochain_hash ON gradients(holochain_hash);
CREATE INDEX IF NOT EXISTS idx_credits_holochain_hash ON credits(holochain_hash);
CREATE INDEX IF NOT EXISTS idx_transactions_holochain_hash ON transactions(holochain_hash);
CREATE INDEX IF NOT EXISTS idx_byzantine_events_holochain_hash ON byzantine_events(holochain_hash);

-- ============================================
-- Phase 10: ZK-PoC Metadata
-- ============================================

-- Track ZK-PoC proofs for privacy-preserving validation
CREATE TABLE IF NOT EXISTS zkpoc_proofs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    gradient_id UUID REFERENCES gradients(id),

    -- Proof metadata (NOT the score itself!)
    proof_hash TEXT NOT NULL,
    proof_size INTEGER NOT NULL,
    verification_time_ms INTEGER NOT NULL,

    -- Verification result
    verified BOOLEAN NOT NULL,
    verified_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Privacy compliance
    hipaa_mode BOOLEAN DEFAULT FALSE,
    gdpr_mode BOOLEAN DEFAULT FALSE,

    -- Link to Holochain
    holochain_hash TEXT,

    -- Indexes
    INDEX idx_zkpoc_gradient (gradient_id),
    INDEX idx_zkpoc_verified (verified),
    INDEX idx_zkpoc_verified_at (verified_at)
);

-- ============================================
-- Phase 10: Hybrid Bridge Status
-- ============================================

-- Track synchronization between PostgreSQL and Holochain
CREATE TABLE IF NOT EXISTS hybrid_sync_status (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),

    record_type TEXT NOT NULL,  -- 'gradient', 'credit', 'reputation', 'byzantine_event'
    record_id TEXT NOT NULL,    -- UUID or hash
    priority TEXT NOT NULL,     -- 'critical', 'important', 'operational'

    -- PostgreSQL state
    postgres_hash TEXT NOT NULL,
    postgres_written_at TIMESTAMP NOT NULL,

    -- Holochain state
    holochain_hash TEXT,
    holochain_written_at TIMESTAMP,

    -- Sync state
    sync_status TEXT DEFAULT 'pending',  -- 'pending', 'synced', 'failed'
    sync_attempts INTEGER DEFAULT 0,
    last_sync_attempt TIMESTAMP,
    last_sync_error TEXT,

    -- Integrity
    integrity_verified BOOLEAN DEFAULT FALSE,
    integrity_verified_at TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    -- Constraints
    CHECK (sync_status IN ('pending', 'synced', 'failed')),
    CHECK (priority IN ('critical', 'important', 'operational')),

    -- Indexes
    INDEX idx_hybrid_sync_status (sync_status),
    INDEX idx_hybrid_sync_record (record_type, record_id),
    INDEX idx_hybrid_sync_priority (priority),
    INDEX idx_hybrid_sync_created (created_at)
);

-- ============================================
-- Phase 10: Functions
-- ============================================

-- Update gradient with Holochain hash
CREATE OR REPLACE FUNCTION update_gradient_holochain_hash(
    p_gradient_id UUID,
    p_holochain_hash TEXT
)
RETURNS void AS $$
BEGIN
    UPDATE gradients
    SET holochain_hash = p_holochain_hash
    WHERE id = p_gradient_id;

    -- Update sync status
    UPDATE hybrid_sync_status
    SET
        holochain_hash = p_holochain_hash,
        holochain_written_at = CURRENT_TIMESTAMP,
        sync_status = 'synced'
    WHERE record_type = 'gradient' AND record_id = p_gradient_id::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Update credit with Holochain hash
CREATE OR REPLACE FUNCTION update_credit_holochain_hash(
    p_credit_id UUID,
    p_holochain_hash TEXT
)
RETURNS void AS $$
BEGIN
    UPDATE credits
    SET holochain_hash = p_holochain_hash
    WHERE id = p_credit_id;

    UPDATE hybrid_sync_status
    SET
        holochain_hash = p_holochain_hash,
        holochain_written_at = CURRENT_TIMESTAMP,
        sync_status = 'synced'
    WHERE record_type = 'credit' AND record_id = p_credit_id::TEXT;
END;
$$ LANGUAGE plpgsql;

-- Record ZK-PoC proof verification
CREATE OR REPLACE FUNCTION record_zkpoc_verification(
    p_gradient_id UUID,
    p_proof_hash TEXT,
    p_proof_size INTEGER,
    p_verification_time_ms INTEGER,
    p_verified BOOLEAN,
    p_hipaa_mode BOOLEAN DEFAULT FALSE,
    p_gdpr_mode BOOLEAN DEFAULT FALSE
)
RETURNS UUID AS $$
DECLARE
    v_proof_id UUID;
BEGIN
    INSERT INTO zkpoc_proofs (
        gradient_id,
        proof_hash,
        proof_size,
        verification_time_ms,
        verified,
        hipaa_mode,
        gdpr_mode
    ) VALUES (
        p_gradient_id,
        p_proof_hash,
        p_proof_size,
        p_verification_time_ms,
        p_verified,
        p_hipaa_mode,
        p_gdpr_mode
    ) RETURNING id INTO v_proof_id;

    RETURN v_proof_id;
END;
$$ LANGUAGE plpgsql;

-- Get hybrid sync statistics
CREATE OR REPLACE FUNCTION get_hybrid_sync_stats()
RETURNS TABLE(
    total_records BIGINT,
    synced BIGINT,
    pending BIGINT,
    failed BIGINT,
    integrity_verified BIGINT,
    avg_sync_attempts NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_records,
        COUNT(CASE WHEN sync_status = 'synced' THEN 1 END)::BIGINT as synced,
        COUNT(CASE WHEN sync_status = 'pending' THEN 1 END)::BIGINT as pending,
        COUNT(CASE WHEN sync_status = 'failed' THEN 1 END)::BIGINT as failed,
        COUNT(CASE WHEN integrity_verified = TRUE THEN 1 END)::BIGINT as integrity_verified,
        AVG(sync_attempts) as avg_sync_attempts
    FROM hybrid_sync_status;
END;
$$ LANGUAGE plpgsql;

-- Get ZK-PoC verification statistics
CREATE OR REPLACE FUNCTION get_zkpoc_stats()
RETURNS TABLE(
    total_proofs BIGINT,
    verified_proofs BIGINT,
    failed_proofs BIGINT,
    avg_verification_time_ms NUMERIC,
    avg_proof_size_bytes NUMERIC,
    hipaa_mode_count BIGINT,
    gdpr_mode_count BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT
        COUNT(*)::BIGINT as total_proofs,
        COUNT(CASE WHEN verified = TRUE THEN 1 END)::BIGINT as verified_proofs,
        COUNT(CASE WHEN verified = FALSE THEN 1 END)::BIGINT as failed_proofs,
        AVG(verification_time_ms) as avg_verification_time_ms,
        AVG(proof_size) as avg_proof_size_bytes,
        COUNT(CASE WHEN hipaa_mode = TRUE THEN 1 END)::BIGINT as hipaa_mode_count,
        COUNT(CASE WHEN gdpr_mode = TRUE THEN 1 END)::BIGINT as gdpr_mode_count
    FROM zkpoc_proofs;
END;
$$ LANGUAGE plpgsql;

-- ============================================
-- Phase 10: Materialized Views
-- ============================================

-- Hybrid sync dashboard view
CREATE MATERIALIZED VIEW IF NOT EXISTS hybrid_sync_dashboard AS
SELECT
    record_type,
    priority,
    COUNT(*) as total_records,
    COUNT(CASE WHEN sync_status = 'synced' THEN 1 END) as synced,
    COUNT(CASE WHEN sync_status = 'pending' THEN 1 END) as pending,
    COUNT(CASE WHEN sync_status = 'failed' THEN 1 END) as failed,
    AVG(sync_attempts) as avg_attempts,
    COUNT(CASE WHEN integrity_verified = TRUE THEN 1 END) as integrity_verified
FROM hybrid_sync_status
GROUP BY record_type, priority;

CREATE UNIQUE INDEX ON hybrid_sync_dashboard (record_type, priority);

-- ZK-PoC verification dashboard
CREATE MATERIALIZED VIEW IF NOT EXISTS zkpoc_dashboard AS
SELECT
    DATE_TRUNC('hour', verified_at) as hour,
    COUNT(*) as total_verifications,
    COUNT(CASE WHEN verified = TRUE THEN 1 END) as successful,
    COUNT(CASE WHEN verified = FALSE THEN 1 END) as failed,
    AVG(verification_time_ms) as avg_verification_ms,
    AVG(proof_size) as avg_proof_size,
    COUNT(CASE WHEN hipaa_mode = TRUE THEN 1 END) as hipaa_verifications,
    COUNT(CASE WHEN gdpr_mode = TRUE THEN 1 END) as gdpr_verifications
FROM zkpoc_proofs
WHERE verified_at > NOW() - INTERVAL '7 days'
GROUP BY DATE_TRUNC('hour', verified_at);

CREATE UNIQUE INDEX ON zkpoc_dashboard (hour);

-- ============================================
-- Phase 10: Triggers
-- ============================================

-- Auto-refresh Phase 10 views
CREATE OR REPLACE FUNCTION trigger_refresh_phase10_views()
RETURNS TRIGGER AS $$
BEGIN
    -- Refresh every 100 records to avoid overhead
    IF (SELECT COUNT(*) FROM hybrid_sync_status) % 100 = 0 THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY hybrid_sync_dashboard;
    END IF;

    IF (SELECT COUNT(*) FROM zkpoc_proofs) % 50 = 0 THEN
        REFRESH MATERIALIZED VIEW CONCURRENTLY zkpoc_dashboard;
    END IF;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER refresh_hybrid_sync_views
AFTER INSERT ON hybrid_sync_status
FOR EACH STATEMENT
EXECUTE FUNCTION trigger_refresh_phase10_views();

CREATE TRIGGER refresh_zkpoc_views
AFTER INSERT ON zkpoc_proofs
FOR EACH STATEMENT
EXECUTE FUNCTION trigger_refresh_phase10_views();

-- ============================================
-- Phase 10: Grants
-- ============================================

GRANT SELECT, INSERT, UPDATE ON zkpoc_proofs TO zerotrustml;
GRANT SELECT, INSERT, UPDATE ON hybrid_sync_status TO zerotrustml;
GRANT EXECUTE ON FUNCTION update_gradient_holochain_hash TO zerotrustml;
GRANT EXECUTE ON FUNCTION update_credit_holochain_hash TO zerotrustml;
GRANT EXECUTE ON FUNCTION record_zkpoc_verification TO zerotrustml;
GRANT EXECUTE ON FUNCTION get_hybrid_sync_stats TO zerotrustml;
GRANT EXECUTE ON FUNCTION get_zkpoc_stats TO zerotrustml;

-- ============================================
-- Completion
-- ============================================

SELECT 'ZeroTrustML Phase 10 database extensions installed successfully!' as status;
SELECT 'Hybrid bridge tracking enabled' as hybrid_bridge_status;
SELECT 'ZK-PoC metadata tables created' as zkpoc_status;
