-- Terra Atlas Energy Infrastructure Tables Migration
-- Adds 5 core energy tables for investment tracking

-- =====================================================
-- ENERGY PROJECTS TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS energy_projects (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    -- Basic Info
    project_type VARCHAR(50) NOT NULL,
    sub_type VARCHAR(100),
    name VARCHAR(500) NOT NULL,
    description TEXT,
    
    -- Location
    latitude DECIMAL(10, 7) NOT NULL,
    longitude DECIMAL(10, 7) NOT NULL,
    address TEXT,
    city VARCHAR(255),
    state VARCHAR(100),
    country VARCHAR(100),
    zip_code VARCHAR(20),
    
    -- Capacity & Generation
    capacity_mw DECIMAL(10, 2),
    capacity_mwh DECIMAL(10, 2),
    annual_generation_gwh DECIMAL(12, 2),
    capacity_factor DECIMAL(5, 2),
    
    -- Development Status
    status VARCHAR(50) NOT NULL,
    development_phase VARCHAR(100),
    cod_date TIMESTAMP,
    construction_start TIMESTAMP,
    permit_date TIMESTAMP,
    ppa_date TIMESTAMP,
    
    -- Financial
    total_cost_million DECIMAL(12, 2),
    lcoe_per_mwh DECIMAL(8, 2),
    ppa_price DECIMAL(8, 2),
    expected_irr DECIMAL(5, 2),
    
    -- Ownership & Development
    developer VARCHAR(500),
    owner VARCHAR(500),
    operator VARCHAR(500),
    epc_contractor VARCHAR(500),
    
    -- Grid Connection
    interconnection_status VARCHAR(100),
    interconnection_queue VARCHAR(100),
    transmission_owner VARCHAR(255),
    substation_name VARCHAR(255),
    voltage_kv INTEGER,
    
    -- Environmental Impact
    co2_avoided_tons_year DECIMAL(12, 2),
    land_area_acres DECIMAL(10, 2),
    water_usage_gallons_year DECIMAL(15, 0),
    
    -- Employment
    construction_jobs INTEGER,
    permanent_jobs INTEGER,
    
    -- Data Source
    data_source_id VARCHAR(100),
    data_source_name VARCHAR(255),
    external_id VARCHAR(255),
    
    -- Metadata
    properties JSONB DEFAULT '{}',
    tags TEXT[],
    is_public BOOLEAN DEFAULT true,
    is_verified BOOLEAN DEFAULT false,
    verified_by UUID REFERENCES users(id),
    verified_at TIMESTAMP,
    
    -- Trust & Quality
    trust_score DECIMAL(5, 2) DEFAULT 50.00,
    data_quality DECIMAL(5, 2) DEFAULT 50.00,
    
    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_data_update TIMESTAMP
);

-- Indexes for energy_projects
CREATE INDEX idx_energy_projects_type ON energy_projects(project_type);
CREATE INDEX idx_energy_projects_status ON energy_projects(status);
CREATE INDEX idx_energy_projects_location ON energy_projects(latitude, longitude);
CREATE INDEX idx_energy_projects_capacity ON energy_projects(capacity_mw);
CREATE INDEX idx_energy_projects_developer ON energy_projects(developer);
CREATE INDEX idx_energy_projects_state ON energy_projects(state);
CREATE INDEX idx_energy_projects_country ON energy_projects(country);

-- =====================================================
-- RENEWABLE ENERGY CERTIFICATES TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS renewable_certificates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES energy_projects(id),
    
    certificate_type VARCHAR(50) NOT NULL,
    serial_number VARCHAR(255) UNIQUE,
    vintage VARCHAR(20),
    
    mwh_generated DECIMAL(12, 2),
    status VARCHAR(50),
    
    issuer VARCHAR(255),
    owner VARCHAR(500),
    retired_by VARCHAR(500),
    retired_date TIMESTAMP,
    
    price DECIMAL(10, 2),
    currency VARCHAR(10) DEFAULT 'USD',
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_renewable_certificates_project ON renewable_certificates(project_id);
CREATE INDEX idx_renewable_certificates_type ON renewable_certificates(certificate_type);
CREATE INDEX idx_renewable_certificates_status ON renewable_certificates(status);

-- =====================================================
-- TRANSMISSION LINES TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS transmission_lines (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    
    name VARCHAR(500) NOT NULL,
    operator VARCHAR(500),
    owner VARCHAR(500),
    
    voltage_kv INTEGER NOT NULL,
    length_miles DECIMAL(10, 2),
    capacity_mw DECIMAL(10, 2),
    
    start_lat DECIMAL(10, 7),
    start_lng DECIMAL(10, 7),
    end_lat DECIMAL(10, 7),
    end_lng DECIMAL(10, 7),
    waypoints JSONB DEFAULT '[]',
    
    status VARCHAR(50),
    in_service_date TIMESTAMP,
    
    estimated_cost_million DECIMAL(12, 2),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_transmission_voltage ON transmission_lines(voltage_kv);
CREATE INDEX idx_transmission_operator ON transmission_lines(operator);

-- =====================================================
-- POWER PURCHASE AGREEMENTS TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS power_purchase_agreements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES energy_projects(id),
    
    buyer VARCHAR(500) NOT NULL,
    seller VARCHAR(500) NOT NULL,
    
    contract_type VARCHAR(50),
    
    capacity_mw DECIMAL(10, 2),
    annual_gwh DECIMAL(12, 2),
    
    price_per_mwh DECIMAL(8, 2),
    escalation_rate DECIMAL(5, 2),
    currency VARCHAR(10) DEFAULT 'USD',
    
    start_date TIMESTAMP,
    end_date TIMESTAMP,
    term_years INTEGER,
    
    delivery_point VARCHAR(255),
    
    status VARCHAR(50),
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_ppa_project ON power_purchase_agreements(project_id);
CREATE INDEX idx_ppa_buyer ON power_purchase_agreements(buyer);
CREATE INDEX idx_ppa_seller ON power_purchase_agreements(seller);
CREATE INDEX idx_ppa_status ON power_purchase_agreements(status);

-- =====================================================
-- BATTERY STORAGE SYSTEMS TABLE
-- =====================================================

CREATE TABLE IF NOT EXISTS battery_storage (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id UUID REFERENCES energy_projects(id),
    
    technology VARCHAR(100),
    manufacturer VARCHAR(255),
    model VARCHAR(255),
    
    power_mw DECIMAL(10, 2),
    energy_mwh DECIMAL(10, 2),
    duration_hours DECIMAL(5, 2),
    
    round_trip_efficiency DECIMAL(5, 2),
    cycles_per_day DECIMAL(5, 2),
    expected_lifetime_cycles INTEGER,
    
    application VARCHAR(100),
    grid_connection VARCHAR(100),
    
    warranty_years INTEGER,
    degradation_rate_year DECIMAL(5, 2),
    
    installation_date TIMESTAMP,
    
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_battery_project ON battery_storage(project_id);
CREATE INDEX idx_battery_technology ON battery_storage(technology);
CREATE INDEX idx_battery_application ON battery_storage(application);

-- =====================================================
-- UPDATE TRIGGERS
-- =====================================================

-- Auto-update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Add triggers to all energy tables
CREATE TRIGGER update_energy_projects_updated_at BEFORE UPDATE ON energy_projects
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_renewable_certificates_updated_at BEFORE UPDATE ON renewable_certificates
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_transmission_lines_updated_at BEFORE UPDATE ON transmission_lines
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_power_purchase_agreements_updated_at BEFORE UPDATE ON power_purchase_agreements
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_battery_storage_updated_at BEFORE UPDATE ON battery_storage
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- =====================================================
-- SAMPLE DATA (Remove for production)
-- =====================================================

-- Insert a sample solar project
INSERT INTO energy_projects (
    project_type, sub_type, name, description,
    latitude, longitude, city, state, country,
    capacity_mw, annual_generation_gwh, capacity_factor,
    status, developer, owner, operator,
    total_cost_million, lcoe_per_mwh, expected_irr,
    co2_avoided_tons_year, construction_jobs, permanent_jobs,
    data_source_name
) VALUES (
    'solar', 'utility-scale', 'West Texas Solar Ranch', 
    'Large-scale solar installation serving Austin metro area',
    31.9686, -99.9018, 'Sweetwater', 'Texas', 'USA',
    500, 1200, 27.4,
    'construction', 'Luminous Energy Partners', 'Terra Atlas Fund I', 'NextEra Energy',
    450, 35, 12.5,
    600000, 500, 15,
    'Terra Atlas Demo Data'
);

-- Stats query
SELECT 
    'Energy Infrastructure Tables Created' as status,
    (SELECT COUNT(*) FROM information_schema.tables 
     WHERE table_name IN ('energy_projects', 'renewable_certificates', 'transmission_lines', 
                          'power_purchase_agreements', 'battery_storage')) as tables_count;