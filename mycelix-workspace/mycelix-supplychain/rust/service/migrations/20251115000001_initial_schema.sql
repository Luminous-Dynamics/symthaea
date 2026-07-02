-- Initial schema for Mycelix Supply Chain

-- Main claims table
CREATE TABLE IF NOT EXISTS claims (
    id TEXT PRIMARY KEY NOT NULL,
    issuer TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    product_id TEXT NOT NULL,
    event_type TEXT NOT NULL,
    vc_jwt TEXT NOT NULL,
    lineage_hash TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    claim_json TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Lineage relationships table
CREATE TABLE IF NOT EXISTS lineage (
    claim_id TEXT NOT NULL,
    parent_claim_id TEXT NOT NULL,
    PRIMARY KEY (claim_id, parent_claim_id),
    FOREIGN KEY (claim_id) REFERENCES claims(id) ON DELETE CASCADE,
    FOREIGN KEY (parent_claim_id) REFERENCES claims(id) ON DELETE CASCADE
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_claims_batch_id ON claims(batch_id);
CREATE INDEX IF NOT EXISTS idx_claims_product_id ON claims(product_id);
CREATE INDEX IF NOT EXISTS idx_claims_event_type ON claims(event_type);
CREATE INDEX IF NOT EXISTS idx_claims_timestamp ON claims(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_claims_issuer ON claims(issuer);
CREATE INDEX IF NOT EXISTS idx_claims_created_at ON claims(created_at DESC);

-- Index for lineage lookups
CREATE INDEX IF NOT EXISTS idx_lineage_claim_id ON lineage(claim_id);
CREATE INDEX IF NOT EXISTS idx_lineage_parent_claim_id ON lineage(parent_claim_id);
