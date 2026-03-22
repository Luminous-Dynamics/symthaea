-- Terra Atlas Database Schema
-- PostgreSQL with PostGIS for geospatial data

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "postgis";
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
CREATE INDEX idx_users_created_at ON users(created_at);

-- =====================================================
-- DATA SOURCES & LAYERS
-- =====================================================

-- Data sources configuration
CREATE TABLE data_sources (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id VARCHAR(100) UNIQUE NOT NULL, -- e.g., 'nasa-firms', 'usgs-earthquakes'
    name VARCHAR(255) NOT NULL,
    description TEXT,
    provider VARCHAR(255), -- e.g., 'NASA', 'USGS', 'NOAA'
    api_endpoint VARCHAR(500),
    
    -- API Configuration
    requires_api_key BOOLEAN DEFAULT FALSE,
    api_key_encrypted VARCHAR(500), -- Encrypted API key storage
    rate_limit INTEGER, -- Requests per minute
    
    -- Trust metrics
    base_trust_score DECIMAL(5,2) DEFAULT 75.00,
    verification_type VARCHAR(100), -- 'official_government', 'satellite_confirmed', etc.
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    last_fetch_at TIMESTAMP WITH TIME ZONE,
    last_fetch_status VARCHAR(50),
    last_error_message TEXT,
    
    -- Metadata
    metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- =====================================================
-- GEOSPATIAL DATA STORAGE
-- =====================================================

-- Main data points table with PostGIS
CREATE TABLE data_points (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    source_id VARCHAR(100) REFERENCES data_sources(source_id),
    external_id VARCHAR(255), -- ID from the external source
    
    -- Geospatial data
    geometry GEOMETRY(Point, 4326) NOT NULL, -- PostGIS point geometry
    properties JSONB NOT NULL, -- All properties from GeoJSON
    
    -- Data classification
    data_type VARCHAR(100), -- 'fire', 'earthquake', 'weather_alert', etc.
    severity VARCHAR(50), -- 'low', 'moderate', 'high', 'extreme'
    magnitude DECIMAL(5,2), -- For earthquakes, etc.
    
    -- Temporal data
    event_time TIMESTAMP WITH TIME ZONE,
    detected_at TIMESTAMP WITH TIME ZONE,
    expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Trust and verification
    quality_score DECIMAL(5,2) DEFAULT 50.00,
    trust_score DECIMAL(5,2) DEFAULT 50.00,
    verification_status VARCHAR(50) DEFAULT 'unverified',
    verification_method VARCHAR(100),
    
    -- Validation counts (denormalized for performance)
    confirms_count INTEGER DEFAULT 0,
    disputes_count INTEGER DEFAULT 0,
    flags_count INTEGER DEFAULT 0,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Prevent duplicates
    UNIQUE(source_id, external_id)
);

-- Spatial and temporal indexes
CREATE INDEX idx_data_points_geometry ON data_points USING GIST(geometry);
CREATE INDEX idx_data_points_event_time ON data_points(event_time DESC);
CREATE INDEX idx_data_points_data_type ON data_points(data_type);
CREATE INDEX idx_data_points_trust_score ON data_points(trust_score DESC);
CREATE INDEX idx_data_points_created_at ON data_points(created_at DESC);

-- =====================================================
-- USER VALIDATIONS & REPUTATION
-- =====================================================

-- Individual validation records
CREATE TABLE validations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    data_point_id UUID REFERENCES data_points(id) ON DELETE CASCADE,
    
    -- Validation action
    validation_type VARCHAR(20) NOT NULL CHECK (validation_type IN ('confirm', 'dispute', 'flag')),
    previous_type VARCHAR(20), -- If user changed their vote
    
    -- Optional feedback
    comment TEXT,
    evidence_urls TEXT[], -- Links to supporting evidence
    
    -- Validation metadata
    ip_address INET,
    user_agent TEXT,
    client_fingerprint VARCHAR(255), -- For anonymous tracking
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    
    -- Ensure one vote per user per data point
    UNIQUE(user_id, data_point_id)
);

-- Create indexes for validations
CREATE INDEX idx_validations_user_id ON validations(user_id);
CREATE INDEX idx_validations_data_point_id ON validations(data_point_id);
CREATE INDEX idx_validations_type ON validations(validation_type);
CREATE INDEX idx_validations_created_at ON validations(created_at DESC);

-- =====================================================
-- API KEYS MANAGEMENT
-- =====================================================

-- User API keys for programmatic access
CREATE TABLE user_api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    key_hash VARCHAR(255) UNIQUE NOT NULL, -- Hashed API key
    key_prefix VARCHAR(20) NOT NULL, -- First 8 chars for identification
    name VARCHAR(255),
    description TEXT,
    
    -- Permissions
    scopes TEXT[] DEFAULT ARRAY['read'], -- 'read', 'write', 'validate', 'admin'
    
    -- Rate limiting
    rate_limit INTEGER DEFAULT 1000, -- Requests per hour
    requests_count INTEGER DEFAULT 0,
    requests_reset_at TIMESTAMP WITH TIME ZONE,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    expires_at TIMESTAMP WITH TIME ZONE,
    last_used_at TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    revoked_at TIMESTAMP WITH TIME ZONE
);

-- =====================================================
-- AUDIT & ACTIVITY LOGGING
-- =====================================================

-- Activity log for all user actions
CREATE TABLE activity_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    
    -- Action details
    action VARCHAR(100) NOT NULL, -- 'validation.create', 'data.fetch', etc.
    resource_type VARCHAR(50), -- 'data_point', 'validation', etc.
    resource_id UUID,
    
    -- Request metadata
    ip_address INET,
    user_agent TEXT,
    request_method VARCHAR(10),
    request_path VARCHAR(500),
    
    -- Response
    status_code INTEGER,
    error_message TEXT,
    
    -- Performance
    duration_ms INTEGER,
    
    -- Timestamp
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for activity log
CREATE INDEX idx_activity_log_user_id ON activity_log(user_id);
CREATE INDEX idx_activity_log_action ON activity_log(action);
CREATE INDEX idx_activity_log_created_at ON activity_log(created_at DESC);

-- =====================================================
-- SESSIONS
-- =====================================================

-- User sessions for authentication
CREATE TABLE sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    
    token_hash VARCHAR(255) UNIQUE NOT NULL,
    refresh_token_hash VARCHAR(255) UNIQUE,
    
    -- Session metadata
    ip_address INET,
    user_agent TEXT,
    
    -- Expiration
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    refresh_expires_at TIMESTAMP WITH TIME ZONE,
    
    -- Status
    is_active BOOLEAN DEFAULT TRUE,
    revoked_at TIMESTAMP WITH TIME ZONE,
    
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_activity_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for sessions
CREATE INDEX idx_sessions_user_id ON sessions(user_id);
CREATE INDEX idx_sessions_token_hash ON sessions(token_hash);
CREATE INDEX idx_sessions_expires_at ON sessions(expires_at);

-- =====================================================
-- FUNCTIONS & TRIGGERS
-- =====================================================

-- Update timestamp trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Apply update trigger to tables
CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_sources_updated_at BEFORE UPDATE ON data_sources
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_data_points_updated_at BEFORE UPDATE ON data_points
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_validations_updated_at BEFORE UPDATE ON validations
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function to update user reputation
CREATE OR REPLACE FUNCTION calculate_user_reputation(user_uuid UUID)
RETURNS INTEGER AS $$
DECLARE
    reputation INTEGER;
BEGIN
    SELECT 
        COALESCE(COUNT(*) * 10, 0) + -- 10 points per validation
        COALESCE(SUM(CASE WHEN validation_type = 'confirm' THEN 5 ELSE 0 END), 0) + -- Bonus for confirms
        COALESCE(SUM(CASE WHEN validation_type = 'flag' THEN 15 ELSE 0 END), 0) -- Bonus for flags
    INTO reputation
    FROM validations
    WHERE user_id = user_uuid;
    
    RETURN reputation;
END;
$$ LANGUAGE plpgsql;

-- Function to update data point trust score
CREATE OR REPLACE FUNCTION update_data_point_trust_score(point_id UUID)
RETURNS VOID AS $$
DECLARE
    confirms INTEGER;
    disputes INTEGER;
    flags INTEGER;
    total_votes INTEGER;
    new_trust_score DECIMAL(5,2);
BEGIN
    -- Get validation counts
    SELECT 
        COUNT(*) FILTER (WHERE validation_type = 'confirm'),
        COUNT(*) FILTER (WHERE validation_type = 'dispute'),
        COUNT(*) FILTER (WHERE validation_type = 'flag')
    INTO confirms, disputes, flags
    FROM validations
    WHERE data_point_id = point_id;
    
    -- Update denormalized counts
    UPDATE data_points
    SET 
        confirms_count = confirms,
        disputes_count = disputes,
        flags_count = flags
    WHERE id = point_id;
    
    -- Calculate new trust score
    total_votes := confirms + disputes;
    IF total_votes > 0 THEN
        new_trust_score := (confirms::DECIMAL / total_votes) * 100;
        
        -- Apply penalties for flags
        IF flags > 5 THEN
            new_trust_score := GREATEST(0, new_trust_score - (flags * 2));
        END IF;
        
        -- Update trust score
        UPDATE data_points
        SET trust_score = new_trust_score
        WHERE id = point_id;
    END IF;
END;
$$ LANGUAGE plpgsql;

-- =====================================================
-- SAMPLE DATA FOR TESTING
-- =====================================================

-- Insert sample data sources
INSERT INTO data_sources (source_id, name, provider, verification_type, base_trust_score) VALUES
('nasa-firms', 'NASA FIRMS Fire Detection', 'NASA', 'satellite_confirmed', 92.00),
('usgs-earthquakes', 'USGS Earthquake Monitoring', 'USGS', 'official_government', 95.00),
('noaa-alerts', 'NOAA Weather Alerts', 'NOAA', 'official_government', 94.00),
('nasa-eonet', 'NASA Earth Observatory', 'NASA', 'satellite_confirmed', 91.00),
('openaq', 'OpenAQ Air Quality', 'OpenAQ', 'sensor_network', 78.00);

-- =====================================================
-- VIEWS FOR COMMON QUERIES
-- =====================================================

-- Top validators view
CREATE VIEW top_validators AS
SELECT 
    u.id,
    u.username,
    u.reputation_score,
    u.validations_count,
    u.validation_accuracy,
    u.trust_level,
    COUNT(v.id) as recent_validations
FROM users u
LEFT JOIN validations v ON u.id = v.user_id 
    AND v.created_at > CURRENT_TIMESTAMP - INTERVAL '7 days'
GROUP BY u.id
ORDER BY u.reputation_score DESC
LIMIT 100;

-- Recent data points with trust scores
CREATE VIEW recent_trusted_data AS
SELECT 
    dp.*,
    ds.name as source_name,
    ds.provider,
    ds.verification_type,
    ST_AsGeoJSON(dp.geometry) as geojson
FROM data_points dp
JOIN data_sources ds ON dp.source_id = ds.source_id
WHERE dp.created_at > CURRENT_TIMESTAMP - INTERVAL '24 hours'
    AND dp.trust_score > 70
ORDER BY dp.created_at DESC;

-- =====================================================
-- PERMISSIONS (Run as superuser)
-- =====================================================

-- Create application user (replace with your actual username)
-- GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO terra_atlas_user;
-- GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO terra_atlas_user;
-- GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA public TO terra_atlas_user;