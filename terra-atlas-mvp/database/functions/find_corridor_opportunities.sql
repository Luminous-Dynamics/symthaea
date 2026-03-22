-- Function to find corridor opportunities by clustering nearby projects with high transmission costs
CREATE OR REPLACE FUNCTION find_corridor_opportunities(
  min_projects INT DEFAULT 3,
  max_distance_miles FLOAT DEFAULT 50,
  min_total_cost BIGINT DEFAULT 100000000
)
RETURNS TABLE (
  cluster_center_lat FLOAT,
  cluster_center_lng FLOAT,
  state TEXT,
  county TEXT,
  project_count BIGINT,
  project_ids TEXT[],
  total_capacity_mw FLOAT,
  total_individual_cost BIGINT,
  estimated_corridor_cost BIGINT,
  potential_savings BIGINT,
  savings_percentage INT
) AS $$
BEGIN
  RETURN QUERY
  WITH project_clusters AS (
    -- Find clusters of projects within specified distance
    SELECT 
      p1.id as anchor_id,
      p1.state,
      p1.county,
      p1.latitude,
      p1.longitude,
      ARRAY_AGG(DISTINCT p2.id) as project_ids,
      COUNT(DISTINCT p2.id) as project_count,
      SUM(p2.capacity_mw) as total_capacity_mw,
      SUM(p2.total_interconnection_cost) as total_individual_cost,
      AVG(p2.latitude) as cluster_lat,
      AVG(p2.longitude) as cluster_lng
    FROM energy_projects p1
    INNER JOIN energy_projects p2 
      ON ST_DWithin(
        p1.location::geography, 
        p2.location::geography, 
        max_distance_miles * 1609.34  -- Convert miles to meters
      )
    WHERE 
      p1.status = 'development'
      AND p2.status = 'development'
      AND p1.total_interconnection_cost > 0
      AND p2.total_interconnection_cost > 0
    GROUP BY p1.id, p1.state, p1.county, p1.latitude, p1.longitude
    HAVING 
      COUNT(DISTINCT p2.id) >= min_projects
      AND SUM(p2.total_interconnection_cost) >= min_total_cost
  ),
  distinct_clusters AS (
    -- Remove duplicate clusters (same set of projects)
    SELECT DISTINCT ON (project_ids)
      cluster_lat,
      cluster_lng,
      state,
      county,
      project_count,
      project_ids,
      total_capacity_mw,
      total_individual_cost
    FROM project_clusters
    ORDER BY project_ids, total_individual_cost DESC
  )
  SELECT 
    cluster_lat::FLOAT,
    cluster_lng::FLOAT,
    state,
    county,
    project_count,
    project_ids,
    total_capacity_mw::FLOAT,
    total_individual_cost,
    -- Estimate corridor cost: Base cost + 20% of individual costs (shared infrastructure)
    (50000000 + (total_individual_cost * 0.2))::BIGINT as estimated_corridor_cost,
    -- Calculate potential savings
    (total_individual_cost - (50000000 + (total_individual_cost * 0.2)))::BIGINT as potential_savings,
    -- Calculate savings percentage
    ((total_individual_cost - (50000000 + (total_individual_cost * 0.2))) * 100 / total_individual_cost)::INT as savings_percentage
  FROM distinct_clusters
  WHERE 
    -- Only return clusters with significant savings
    (total_individual_cost - (50000000 + (total_individual_cost * 0.2))) > 0
  ORDER BY potential_savings DESC;
END;
$$ LANGUAGE plpgsql;

-- Grant execute permission
GRANT EXECUTE ON FUNCTION find_corridor_opportunities TO anon, authenticated;

-- Create index to speed up spatial queries if not exists
CREATE INDEX IF NOT EXISTS idx_energy_projects_location_gist 
ON energy_projects USING GIST (location);

CREATE INDEX IF NOT EXISTS idx_energy_projects_status 
ON energy_projects (status);

CREATE INDEX IF NOT EXISTS idx_energy_projects_cost 
ON energy_projects (total_interconnection_cost);

-- Create a view for easy access to corridor opportunities
CREATE OR REPLACE VIEW corridor_opportunities AS
SELECT * FROM find_corridor_opportunities();

-- Grant select on view
GRANT SELECT ON corridor_opportunities TO anon, authenticated;