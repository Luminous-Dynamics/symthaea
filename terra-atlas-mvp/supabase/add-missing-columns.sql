-- Add missing columns to projects table
-- These columns were missing and causing migration failures

-- Add data_source column if it doesn't exist
DO $$ 
BEGIN 
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns 
        WHERE table_name = 'projects' AND column_name = 'data_source'
    ) THEN
        ALTER TABLE projects ADD COLUMN data_source TEXT;
    END IF;
END $$;

-- Update schema to ensure all necessary columns exist
-- The country and commissioning_date are now stored in metadata JSON field
-- So we don't need to add them as separate columns

-- Add index on data_source for faster queries
CREATE INDEX IF NOT EXISTS idx_projects_data_source ON projects(data_source);

-- Add index on metadata for JSON queries
CREATE INDEX IF NOT EXISTS idx_projects_metadata ON projects USING GIN(metadata);

-- Verify the table structure
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'projects'
ORDER BY ordinal_position;