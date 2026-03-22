-- Create sites table for all energy projects
-- This table stores renewable energy sites, dams, and power projects worldwide

CREATE TABLE IF NOT EXISTS sites (
  id SERIAL PRIMARY KEY,

  -- Basic information
  name TEXT NOT NULL,
  type TEXT NOT NULL CHECK (type IN ('solar', 'wind', 'hydro', 'geothermal', 'nuclear', 'storage', 'biomass', 'hybrid')),
  status TEXT NOT NULL DEFAULT 'operational' CHECK (status IN ('operational', 'construction', 'planned', 'decommissioned', 'proposed')),

  -- Location data
  latitude DECIMAL(10, 7) NOT NULL,
  longitude DECIMAL(11, 7) NOT NULL,
  country TEXT,
  state_province TEXT,
  city TEXT,
  address TEXT,

  -- Capacity and performance
  power_mw DECIMAL(12, 3) NOT NULL, -- Megawatts
  capacity_factor DECIMAL(5, 2), -- Percentage (0-100)
  annual_generation_gwh DECIMAL(12, 3), -- Gigawatt-hours per year

  -- Project details
  owner TEXT,
  operator TEXT,
  developer TEXT,
  commissioning_date DATE,
  decommissioning_date DATE,

  -- Financial information
  estimated_cost_usd DECIMAL(15, 2),
  investment_needed_usd DECIMAL(15, 2),
  expected_roi_percentage DECIMAL(5, 2),
  payback_period_years DECIMAL(5, 2),

  -- Data source and quality
  data_source TEXT NOT NULL DEFAULT 'manual', -- 'usace', 'manual', 'global_solar_atlas', etc.
  data_quality TEXT CHECK (data_quality IN ('high', 'medium', 'low')),
  last_verified_at TIMESTAMPTZ,

  -- Additional metadata
  description TEXT,
  image_url TEXT,
  website_url TEXT,
  metadata JSONB DEFAULT '{}'::jsonb,

  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_sites_type ON sites(type);
CREATE INDEX IF NOT EXISTS idx_sites_status ON sites(status);
CREATE INDEX IF NOT EXISTS idx_sites_location ON sites(latitude, longitude);
CREATE INDEX IF NOT EXISTS idx_sites_power ON sites(power_mw DESC);
CREATE INDEX IF NOT EXISTS idx_sites_data_source ON sites(data_source);
CREATE INDEX IF NOT EXISTS idx_sites_country ON sites(country);

-- Spatial index for location-based queries
CREATE INDEX IF NOT EXISTS idx_sites_geom ON sites USING GIST (
  ll_to_earth(latitude, longitude)
);

-- Full-text search index
CREATE INDEX IF NOT EXISTS idx_sites_search ON sites USING GIN(
  to_tsvector('english',
    COALESCE(name, '') || ' ' ||
    COALESCE(description, '') || ' ' ||
    COALESCE(city, '') || ' ' ||
    COALESCE(country, '')
  )
);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_sites_updated_at()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to automatically update updated_at
CREATE TRIGGER sites_updated_at
  BEFORE UPDATE ON sites
  FOR EACH ROW
  EXECUTE FUNCTION update_sites_updated_at();

-- Enable Row Level Security (RLS)
ALTER TABLE sites ENABLE ROW LEVEL SECURITY;

-- Policy: Anyone can read sites (public data)
CREATE POLICY "Sites are publicly readable" ON sites
  FOR SELECT USING (true);

-- Policy: Only authenticated users can insert (for data contributions)
CREATE POLICY "Authenticated users can insert sites" ON sites
  FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- Policy: Users can update their own contributions
CREATE POLICY "Users can update sites they created" ON sites
  FOR UPDATE USING (
    metadata->>'created_by_user_id' = auth.uid()::text
  );

-- Create materialized view for globe data (performance optimization)
CREATE MATERIALIZED VIEW sites_globe_data AS
SELECT
  id,
  name,
  type,
  latitude,
  longitude,
  power_mw,
  status,
  country,
  CASE type
    WHEN 'solar' THEN '#FCD34D'
    WHEN 'wind' THEN '#60A5FA'
    WHEN 'hydro' THEN '#34D399'
    WHEN 'geothermal' THEN '#F87171'
    WHEN 'nuclear' THEN '#A78BFA'
    WHEN 'storage' THEN '#FB923C'
    WHEN 'biomass' THEN '#84CC16'
    WHEN 'hybrid' THEN '#EC4899'
    ELSE '#10B981'
  END as color,
  LEAST(GREATEST(power_mw / 100, 0.5), 5.0) as size
FROM sites
WHERE latitude IS NOT NULL
  AND longitude IS NOT NULL
  AND status IN ('operational', 'construction');

-- Create index on materialized view
CREATE INDEX ON sites_globe_data(type);
CREATE INDEX ON sites_globe_data(power_mw DESC);

-- Function to refresh the materialized view
CREATE OR REPLACE FUNCTION refresh_sites_globe_data()
RETURNS void AS $$
BEGIN
  REFRESH MATERIALIZED VIEW CONCURRENTLY sites_globe_data;
END;
$$ LANGUAGE plpgsql;

-- Create view for site statistics
CREATE OR REPLACE VIEW site_statistics AS
SELECT
  type,
  COUNT(*) as count,
  SUM(power_mw) as total_capacity_mw,
  AVG(power_mw) as avg_capacity_mw,
  MIN(power_mw) as min_capacity_mw,
  MAX(power_mw) as max_capacity_mw,
  COUNT(DISTINCT country) as countries
FROM sites
WHERE status IN ('operational', 'construction')
GROUP BY type;

-- Insert the demo data from the API
INSERT INTO sites (name, type, latitude, longitude, power_mw, status, country, data_source)
VALUES
  -- Solar
  ('Mojave Solar Park', 'solar', 35.0, -116.5, 280, 'operational', 'United States', 'manual'),
  ('Bhadla Solar Park', 'solar', 27.5, 71.9, 2245, 'operational', 'India', 'manual'),
  ('Fukushima Renewable', 'solar', 37.5, 141.0, 100, 'operational', 'Japan', 'manual'),
  ('Noor Complex', 'solar', 31.0, -6.7, 580, 'operational', 'Morocco', 'manual'),

  -- Wind
  ('London Array', 'wind', 51.6, 1.5, 630, 'operational', 'United Kingdom', 'manual'),
  ('Gansu Wind Farm', 'wind', 40.5, 95.8, 20000, 'operational', 'China', 'manual'),
  ('Hornsea Wind Farm', 'wind', 54.0, 1.5, 1218, 'operational', 'United Kingdom', 'manual'),
  ('Alta Wind Energy', 'wind', 34.6, -118.3, 1548, 'operational', 'United States', 'manual'),

  -- Hydro
  ('Three Gorges Dam', 'hydro', 30.8, 111.0, 22500, 'operational', 'China', 'manual'),
  ('Grand Coulee Dam', 'hydro', 47.9, -119.0, 6809, 'operational', 'United States', 'manual'),
  ('Itaipu Dam', 'hydro', -25.4, -54.6, 14000, 'operational', 'Brazil', 'manual'),

  -- Geothermal
  ('Hellisheiði Station', 'geothermal', 64.0, -21.4, 303, 'operational', 'Iceland', 'manual'),
  ('Geysers Complex', 'geothermal', 38.8, -122.8, 1517, 'operational', 'United States', 'manual'),

  -- Nuclear
  ('Vogtle Nuclear', 'nuclear', 33.1, -81.8, 2234, 'construction', 'United States', 'manual'),
  ('Barakah Nuclear', 'nuclear', 24.5, 52.2, 5600, 'operational', 'United Arab Emirates', 'manual'),

  -- Storage
  ('Hornsdale Battery', 'storage', -32.5, 138.5, 150, 'operational', 'Australia', 'manual'),
  ('Gateway Energy', 'storage', 33.7, -117.8, 230, 'operational', 'United States', 'manual'),

  -- Biomass
  ('Drax Power Station', 'biomass', 53.7, -1.0, 2600, 'operational', 'United Kingdom', 'manual'),
  ('Alholmens Kraft', 'biomass', 63.7, 22.7, 265, 'operational', 'Finland', 'manual'),

  -- Hybrid
  ('Kennedy Energy Park', 'hybrid', -28.8, 143.4, 73, 'operational', 'Australia', 'manual'),
  ('Blythe Solar+Storage', 'hybrid', 33.6, -114.6, 485, 'operational', 'United States', 'manual')
ON CONFLICT DO NOTHING;

-- Refresh the materialized view
REFRESH MATERIALIZED VIEW sites_globe_data;

-- Comments for documentation
COMMENT ON TABLE sites IS 'Global database of renewable energy sites and projects';
COMMENT ON COLUMN sites.power_mw IS 'Installed capacity in megawatts';
COMMENT ON COLUMN sites.capacity_factor IS 'Actual output as percentage of theoretical maximum';
COMMENT ON COLUMN sites.data_source IS 'Original source of the data (usace, manual, etc)';
COMMENT ON MATERIALIZED VIEW sites_globe_data IS 'Optimized view for globe visualization with 500 row limit';
