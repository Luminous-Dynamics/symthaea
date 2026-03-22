-- Terra Atlas Database Schema (Without PostGIS)
-- PostgreSQL schema for Terra Atlas

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm"; -- For fuzzy text search

-- =====================================================
-- USER MANAGEMENT & AUTHENTICATION
-- =====================================================

-- Users table for authentication and reputation
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email VARCHAR(255) UNIQUE NOT NULL,
    username VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    full_name VARCHAR(255),
    avatar_url VARCHAR(500),
    bio TEXT,
    
    -- Reputation and validation stats
    reputation_score INTEGER DEFAULT 0,
    validations_count INTEGER DEFAULT 0,
    accurate_validations INTEGER DEFAULT 0,
    validation_accuracy DECIMAL(5,2) DEFAULT 0.00,
    trust_level VARCHAR(50) DEFAULT 'novice', -- novice, contributor, expert, guardian
    
    -- Account status
    email_verified BOOLEAN DEFAULT FALSE,
    is_active BOOLEAN DEFAULT TRUE,
    is_moderator BOOLEAN DEFAULT FALSE,
    is_admin BOOLEAN DEFAULT FALSE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_login_at TIMESTAMP WITH TIME ZONE,
    
    -- Preferences
    preferences JSONB DEFAULT '{}',
    notification_settings JSONB DEFAULT '{
        "email_validations": true,
        "email_updates": true,
        "push_alerts": false
    }'
);

-- Create indexes for users
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_username ON users(username);
CREATE INDEX idx_users_reputation ON users(reputation_score DESC);
CREATE INDEX idx_users_trust_level ON users(trust_level);

-- =====================================================
-- DATA POINTS & VALIDATION
-- =====================================================

-- Data points table (using lat/lng instead of PostGIS geometry)
CREATE TABLE data_points (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    
    -- Location (instead of PostGIS geometry)
    latitude DECIMAL(10, 7) NOT NULL,
    longitude DECIMAL(10, 7) NOT NULL,
    
    -- Data attributes
    data_type VARCHAR(50) NOT NULL, -- fire, earthquake, flood, etc
    source_id VARCHAR(100) NOT NULL,
    source_name VARCHAR(255),
    
    -- Data specifics
    title VARCHAR(500),
    description TEXT,
    properties JSONB DEFAULT '{}',
    severity VARCHAR(50),
    confidence DECIMAL(5,2),
    
    -- Trust and validation
    trust_score DECIMAL(5,2) DEFAULT 50.00,
    quality_score DECIMAL(5,2) DEFAULT 50.00,
    confirms_count INTEGER DEFAULT 0,
    disputes_count INTEGER DEFAULT 0,
    flags_count INTEGER DEFAULT 0,
    
    -- Metadata
    observed_at TIMESTAMP WITH TIME ZONE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Verification status
    is_verified BOOLEAN DEFAULT FALSE,
    verified_by UUID REFERENCES users(id),
    verified_at TIMESTAMP WITH TIME ZONE
);

-- Create indexes for data points
CREATE INDEX idx_data_points_location ON data_points(latitude, longitude);
CREATE INDEX idx_data_points_type ON data_points(data_type);
CREATE INDEX idx_data_points_source ON data_points(source_id);
CREATE INDEX idx_data_points_trust ON data_points(trust_score DESC);
CREATE INDEX idx_data_points_created ON data_points(created_at DESC);

-- =====================================================
-- VALIDATION SYSTEM
-- =====================================================

-- Validations table
CREATE TABLE validations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    data_point_id UUID REFERENCES data_points(id) ON DELETE CASCADE,
    
    -- Validation details
    validation_type VARCHAR(20) NOT NULL CHECK (validation_type IN ('confirm', 'dispute', 'flag')),
    previous_type VARCHAR(20), -- For tracking changes
    comment TEXT,
    evidence_urls TEXT[],
    
    -- Anonymous tracking
    anonymous_id VARCHAR(100), -- For non-logged-in users
    client_fingerprint VARCHAR(255),
    ip_address VARCHAR(45),
    user_agent TEXT,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure one vote per user per data point
    CONSTRAINT unique_user_validation UNIQUE (user_id, data_point_id)
);

-- Create indexes for validations
CREATE INDEX idx_validations_user ON validations(user_id);
CREATE INDEX idx_validations_data_point ON validations(data_point_id);
CREATE INDEX idx_validations_type ON validations(validation_type);
CREATE INDEX idx_validations_created ON validations(created_at DESC);

-- =====================================================
-- SESSION MANAGEMENT
-- =====================================================

-- Sessions table for JWT refresh tokens
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    refresh_token_hash VARCHAR(255) UNIQUE NOT NULL,
    
    -- Session details
    ip_address VARCHAR(45),
    user_agent TEXT,
    device_info JSONB,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    last_used_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    revoked_at TIMESTAMP WITH TIME ZONE,
    revoked_reason VARCHAR(255)
);

-- Create indexes for sessions
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_token ON sessions(refresh_token_hash);
CREATE INDEX idx_sessions_expires ON sessions(expires_at);

-- =====================================================
-- API KEY MANAGEMENT
-- =====================================================

-- API keys table for programmatic access
CREATE TABLE user_api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    -- Key details
    key_hash VARCHAR(255) UNIQUE NOT NULL,
    key_prefix VARCHAR(10) NOT NULL, -- First 8 chars for identification
    name VARCHAR(100) NOT NULL,
    description TEXT,
    
    -- Permissions
    scopes TEXT[] DEFAULT ARRAY['read'],
    rate_limit INTEGER DEFAULT 1000, -- Requests per hour
    
    -- Usage tracking
    last_used_at TIMESTAMP WITH TIME ZONE,
    requests_count INTEGER DEFAULT 0,
    requests_reset_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    revoked_at TIMESTAMP WITH TIME ZONE,
    revoked_reason VARCHAR(255)
);

-- Create indexes for API keys
CREATE INDEX idx_api_keys_user ON user_api_keys(user_id);
CREATE INDEX idx_api_keys_hash ON user_api_keys(key_hash);
CREATE INDEX idx_api_keys_prefix ON user_api_keys(key_prefix);

-- =====================================================
-- ACTIVITY LOGGING
-- =====================================================

-- Activity log for audit trail
CREATE TABLE activity_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    api_key_id UUID REFERENCES user_api_keys(id) ON DELETE SET NULL,
    
    -- Action details
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(50),
    resource_id VARCHAR(255),
    
    -- Request details
    ip_address VARCHAR(45),
    user_agent TEXT,
    request_method VARCHAR(10),
    request_path VARCHAR(500),
    request_body JSONB,
    
    -- Response
    status_code INTEGER,
    response_time_ms INTEGER,
    error_message TEXT,
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for activity log
CREATE INDEX idx_activity_user ON activity_log(user_id);
CREATE INDEX idx_activity_action ON activity_log(action);
CREATE INDEX idx_activity_created ON activity_log(created_at DESC);
CREATE INDEX idx_activity_resource ON activity_log(resource_type, resource_id);

-- =====================================================
-- DATA SOURCES
-- =====================================================

-- Data sources configuration
CREATE TABLE data_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    
    -- API configuration
    base_url VARCHAR(500),
    api_key_encrypted VARCHAR(500),
    headers JSONB DEFAULT '{}',
    
    -- Update schedule
    update_frequency VARCHAR(50), -- hourly, daily, weekly, realtime
    last_updated_at TIMESTAMP WITH TIME ZONE,
    next_update_at TIMESTAMP WITH TIME ZONE,
    
    -- Quality metrics
    reliability_score DECIMAL(5,2) DEFAULT 50.00,
    average_latency_ms INTEGER,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    is_public BOOLEAN DEFAULT TRUE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- HELPER FUNCTIONS
-- =====================================================

-- Function to calculate trust score for data points
CREATE OR REPLACE FUNCTION update_data_point_trust_score(point_id UUID)
RETURNS VOID AS $$
DECLARE
    confirms INT;
    disputes INT;
    flags INT;
    quality DECIMAL;
    trust DECIMAL;
BEGIN
    -- Get validation counts
    SELECT 
        COUNT(*) FILTER (WHERE validation_type = 'confirm'),
        COUNT(*) FILTER (WHERE validation_type = 'dispute'),
        COUNT(*) FILTER (WHERE validation_type = 'flag')
    INTO confirms, disputes, flags
    FROM validations
    WHERE data_point_id = point_id;
    
    -- Get quality score
    SELECT quality_score INTO quality
    FROM data_points
    WHERE id = point_id;
    
    -- Calculate trust score
    -- Base: quality score (weighted 70%)
    -- Community: confirms vs disputes (weighted 30%)
    -- Penalties: flags reduce score
    
    IF (confirms + disputes) > 0 THEN
        trust := (quality * 0.7) + ((confirms::DECIMAL / (confirms + disputes)) * 30);
    ELSE
        trust := quality * 0.7;
    END IF;
    
    -- Apply flag penalties (each flag reduces by 2 points after 5 flags)
    IF flags > 5 THEN
        trust := trust - ((flags - 5) * 2);
    END IF;
    
    -- Ensure trust stays between 0 and 100
    trust := GREATEST(0, LEAST(100, trust));
    
    -- Update the data point
    UPDATE data_points
    SET 
        trust_score = trust,
        confirms_count = confirms,
        disputes_count = disputes,
        flags_count = flags,
        updated_at = CURRENT_TIMESTAMP
    WHERE id = point_id;
END;
$$ LANGUAGE plpgsql;

-- Function to calculate user reputation
CREATE OR REPLACE FUNCTION calculate_user_reputation(user_id UUID)
RETURNS INTEGER AS $$
DECLARE
    rep INTEGER;
    val_count INTEGER;
    confirms INTEGER;
    flags INTEGER;
BEGIN
    -- Count validations
    SELECT 
        COUNT(*),
        COUNT(*) FILTER (WHERE validation_type = 'confirm'),
        COUNT(*) FILTER (WHERE validation_type = 'flag')
    INTO val_count, confirms, flags
    FROM validations
    WHERE validations.user_id = calculate_user_reputation.user_id;
    
    -- Calculate reputation
    -- Base: 10 points per validation
    -- Bonus: 5 points per confirm
    -- Special: 15 points per flag (for finding issues)
    rep := (val_count * 10) + (confirms * 5) + (flags * 15);
    
    RETURN rep;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- TRIGGERS
-- =====================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_points_updated_at BEFORE UPDATE ON data_points
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_validations_updated_at BEFORE UPDATE ON validations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- Top validators view
CREATE VIEW top_validators AS
SELECT 
    u.id,
    u.username,
    u.avatar_url,
    u.reputation_score,
    u.trust_level,
    u.validations_count,
    u.validation_accuracy
FROM users u
WHERE u.validations_count > 0
ORDER BY u.reputation_score DESC
LIMIT 100;

-- Recent high-trust data
CREATE VIEW recent_trusted_data AS
SELECT 
    dp.id,
    dp.latitude,
    dp.longitude,
    dp.data_type,
    dp.title,
    dp.trust_score,
    dp.confirms_count,
    dp.disputes_count,
    dp.created_at
FROM data_points dp
WHERE dp.trust_score >= 70
    AND dp.created_at > (CURRENT_TIMESTAMP - INTERVAL '7 days')
ORDER BY dp.created_at DESC
LIMIT 1000;

-- =====================================================
-- INITIAL DATA
-- =====================================================

-- Insert default data sources
INSERT INTO data_sources (name, description, base_url, update_frequency) VALUES
('NASA FIRMS', 'Fire Information for Resource Management System', 'https://firms.modaps.eosdis.nasa.gov', 'hourly'),
('USGS Earthquakes', 'United States Geological Survey Earthquake Data', 'https://earthquake.usgs.gov', 'realtime'),
('NOAA Weather', 'National Oceanic and Atmospheric Administration', 'https://api.weather.gov', 'hourly'),
('ESA Sentinel', 'European Space Agency Sentinel Hub', 'https://services.sentinel-hub.com', 'daily')
ON CONFLICT (name) DO NOTHING;

-- Create default admin user (password: admin123 - change immediately!)
-- Note: In production, never store passwords like this
INSERT INTO users (
    email, 
    username, 
    password_hash, 
    full_name,
    is_admin,
    email_verified,
    trust_level
) VALUES (
    'admin@terraatlas.app',
    'admin',
    '$2a$10$YourHashedPasswordHere', -- Replace with actual bcrypt hash
    'Terra Atlas Admin',
    true,
    true,
    'guardian'
) ON CONFLICT (email) DO NOTHING;