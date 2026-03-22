-- QOL impact metrics for energy projects
-- Stores real-world impact data synced from Holochain DHT

CREATE TABLE IF NOT EXISTS impact_metrics (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id TEXT NOT NULL,
    period_start TIMESTAMPTZ,
    period_end TIMESTAMPTZ,
    co2_avoided_tonnes DOUBLE PRECISION DEFAULT 0 CHECK (co2_avoided_tonnes >= 0),
    jobs_created INTEGER DEFAULT 0 CHECK (jobs_created >= 0),
    community_trust_delta DOUBLE PRECISION DEFAULT 0 CHECK (community_trust_delta >= -1 AND community_trust_delta <= 1),
    energy_access_households INTEGER DEFAULT 0 CHECK (energy_access_households >= 0),
    biodiversity_index_delta DOUBLE PRECISION DEFAULT 0,
    reporter_did TEXT,
    verified_by TEXT,
    verification_evidence TEXT,
    synced_from_holochain_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_impact_project ON impact_metrics(project_id);
CREATE INDEX IF NOT EXISTS idx_impact_period ON impact_metrics(period_end DESC);

-- Aggregated impact per project
CREATE OR REPLACE VIEW project_impact_summary AS
SELECT
    project_id,
    COUNT(*) as report_count,
    COALESCE(SUM(co2_avoided_tonnes), 0) as total_co2_avoided,
    COALESCE(SUM(jobs_created), 0) as total_jobs_created,
    COALESCE(SUM(community_trust_delta), 0) as total_trust_delta,
    MAX(energy_access_households) as peak_households_served,
    MIN(period_start) as earliest_report,
    MAX(period_end) as latest_report
FROM impact_metrics
GROUP BY project_id;

-- Combined project dashboard: consciousness + impact
CREATE OR REPLACE VIEW project_dashboard AS
SELECT
    cs.project_id,
    cs.phi_score,
    cs.harmony_alignment,
    cs.care_activation,
    cs.assessed_at as consciousness_assessed_at,
    COALESCE(im.total_co2_avoided, 0) as total_co2_avoided,
    COALESCE(im.total_jobs_created, 0) as total_jobs_created,
    COALESCE(im.total_trust_delta, 0) as total_trust_delta,
    COALESCE(im.peak_households_served, 0) as peak_households_served,
    COALESCE(im.report_count, 0) as impact_report_count,
    -- Net Humanity Benefit composite
    (COALESCE(im.total_co2_avoided, 0) / 1000.0) * 0.3
    + (COALESCE(im.total_jobs_created, 0)::float / 100.0) * 0.25
    + LEAST(GREATEST(COALESCE(im.total_trust_delta, 0), -1), 1) * 0.25
    + (COALESCE(im.peak_households_served, 0)::float / 1000.0) * 0.2
    as net_humanity_benefit
FROM latest_consciousness_scores cs
LEFT JOIN project_impact_summary im ON cs.project_id = im.project_id;

-- RLS
ALTER TABLE impact_metrics ENABLE ROW LEVEL SECURITY;

CREATE POLICY "impact_metrics_public_read" ON impact_metrics
    FOR SELECT USING (true);

CREATE POLICY "impact_metrics_authenticated_insert" ON impact_metrics
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');
