-- Consciousness scoring for energy projects
-- Stores Symthaea consciousness evaluations synced from Holochain DHT

CREATE TABLE IF NOT EXISTS consciousness_scores (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    project_id TEXT NOT NULL,
    phi_score DOUBLE PRECISION NOT NULL CHECK (phi_score >= 0 AND phi_score <= 1),
    harmony_alignment DOUBLE PRECISION NOT NULL CHECK (harmony_alignment >= 0 AND harmony_alignment <= 1),
    per_harmony_scores JSONB DEFAULT '{}',
    care_activation DOUBLE PRECISION CHECK (care_activation >= 0 AND care_activation <= 1),
    meta_awareness DOUBLE PRECISION CHECK (meta_awareness >= 0 AND meta_awareness <= 1),
    scorer_did TEXT,
    assessment_cycle BIGINT,
    assessed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    synced_from_holochain_at TIMESTAMPTZ DEFAULT NOW(),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for common queries
CREATE INDEX IF NOT EXISTS idx_consciousness_project ON consciousness_scores(project_id);
CREATE INDEX IF NOT EXISTS idx_consciousness_assessed ON consciousness_scores(assessed_at DESC);
CREATE INDEX IF NOT EXISTS idx_consciousness_phi ON consciousness_scores(phi_score DESC);
CREATE INDEX IF NOT EXISTS idx_consciousness_harmony ON consciousness_scores(harmony_alignment DESC);

-- Latest consciousness score per project (for dashboard)
CREATE OR REPLACE VIEW latest_consciousness_scores AS
SELECT DISTINCT ON (project_id)
    id,
    project_id,
    phi_score,
    harmony_alignment,
    per_harmony_scores,
    care_activation,
    meta_awareness,
    scorer_did,
    assessment_cycle,
    assessed_at
FROM consciousness_scores
ORDER BY project_id, assessed_at DESC;

-- RLS: public read, authenticated write
ALTER TABLE consciousness_scores ENABLE ROW LEVEL SECURITY;

CREATE POLICY "consciousness_scores_public_read" ON consciousness_scores
    FOR SELECT USING (true);

CREATE POLICY "consciousness_scores_authenticated_insert" ON consciousness_scores
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');
