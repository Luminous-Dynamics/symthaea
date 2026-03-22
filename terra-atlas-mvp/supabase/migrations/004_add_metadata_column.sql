-- Add metadata JSONB column to sites table for storing additional data
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_name = 'sites' AND column_name = 'metadata'
    ) THEN
        ALTER TABLE sites ADD COLUMN metadata JSONB DEFAULT '{}'::jsonb;
        CREATE INDEX IF NOT EXISTS idx_sites_metadata ON sites USING GIN (metadata);
        COMMENT ON COLUMN sites.metadata IS 'Additional metadata about the site (source-specific fields, etc.)';
    END IF;
END $$;
