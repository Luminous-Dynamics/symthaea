-- Add missing columns to existing sites table
-- This migration is safe to run - it only adds columns that don't exist

-- Add data_source column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'data_source'
    ) THEN
        ALTER TABLE sites ADD COLUMN data_source TEXT NOT NULL DEFAULT 'manual';
        CREATE INDEX IF NOT EXISTS idx_sites_data_source ON sites(data_source);
    END IF;
END $$;

-- Add data_quality column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'data_quality'
    ) THEN
        ALTER TABLE sites ADD COLUMN data_quality TEXT CHECK (data_quality IN ('high', 'medium', 'low'));
    END IF;
END $$;

-- Add last_verified_at column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'last_verified_at'
    ) THEN
        ALTER TABLE sites ADD COLUMN last_verified_at TIMESTAMPTZ;
    END IF;
END $$;

-- Add capacity_factor column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'capacity_factor'
    ) THEN
        ALTER TABLE sites ADD COLUMN capacity_factor DECIMAL(5, 2);
    END IF;
END $$;

-- Add annual_generation_gwh column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'annual_generation_gwh'
    ) THEN
        ALTER TABLE sites ADD COLUMN annual_generation_gwh DECIMAL(12, 3);
    END IF;
END $$;

-- Add owner column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'owner'
    ) THEN
        ALTER TABLE sites ADD COLUMN owner TEXT;
    END IF;
END $$;

-- Add operator column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'operator'
    ) THEN
        ALTER TABLE sites ADD COLUMN operator TEXT;
    END IF;
END $$;

-- Add developer column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'developer'
    ) THEN
        ALTER TABLE sites ADD COLUMN developer TEXT;
    END IF;
END $$;

-- Add commissioning_date column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'commissioning_date'
    ) THEN
        ALTER TABLE sites ADD COLUMN commissioning_date DATE;
    END IF;
END $$;

-- Add decommissioning_date column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'decommissioning_date'
    ) THEN
        ALTER TABLE sites ADD COLUMN decommissioning_date DATE;
    END IF;
END $$;

-- Add estimated_cost_usd column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'estimated_cost_usd'
    ) THEN
        ALTER TABLE sites ADD COLUMN estimated_cost_usd DECIMAL(15, 2);
    END IF;
END $$;

-- Add investment_needed_usd column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'investment_needed_usd'
    ) THEN
        ALTER TABLE sites ADD COLUMN investment_needed_usd DECIMAL(15, 2);
    END IF;
END $$;

-- Add expected_roi_percentage column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'expected_roi_percentage'
    ) THEN
        ALTER TABLE sites ADD COLUMN expected_roi_percentage DECIMAL(5, 2);
    END IF;
END $$;

-- Add payback_period_years column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'payback_period_years'
    ) THEN
        ALTER TABLE sites ADD COLUMN payback_period_years DECIMAL(5, 2);
    END IF;
END $$;

-- Add description column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'description'
    ) THEN
        ALTER TABLE sites ADD COLUMN description TEXT;
    END IF;
END $$;

-- Add image_url column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'image_url'
    ) THEN
        ALTER TABLE sites ADD COLUMN image_url TEXT;
    END IF;
END $$;

-- Add website_url column
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'website_url'
    ) THEN
        ALTER TABLE sites ADD COLUMN website_url TEXT;
    END IF;
END $$;

-- Add country column if missing
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'country'
    ) THEN
        ALTER TABLE sites ADD COLUMN country TEXT;
        CREATE INDEX IF NOT EXISTS idx_sites_country ON sites(country);
    END IF;
END $$;

-- Add state_province column if missing
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'state_province'
    ) THEN
        ALTER TABLE sites ADD COLUMN state_province TEXT;
    END IF;
END $$;

-- Add city column if missing
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'city'
    ) THEN
        ALTER TABLE sites ADD COLUMN city TEXT;
    END IF;
END $$;

-- Add address column if missing
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'address'
    ) THEN
        ALTER TABLE sites ADD COLUMN address TEXT;
    END IF;
END $$;

-- Create function to refresh sites materialized view (if needed later)
CREATE OR REPLACE FUNCTION refresh_sites_globe_data()
RETURNS void AS $$
BEGIN
  -- For now, just a placeholder
  -- We'll create the materialized view later once we have more data
  RAISE NOTICE 'Materialized view refresh called';
END;
$$ LANGUAGE plpgsql;

-- Verify the updates
SELECT
  'Column count: ' || COUNT(*)::text as info
FROM information_schema.columns
WHERE table_name = 'sites';

SELECT
  'Required columns present: ' ||
  CASE
    WHEN COUNT(*) >= 5 THEN 'YES ✅'
    ELSE 'NO ❌'
  END as check_result
FROM information_schema.columns
WHERE table_name = 'sites'
  AND column_name IN ('data_source', 'data_quality', 'owner', 'commissioning_date', 'description');
