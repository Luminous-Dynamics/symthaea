-- CRM Module Migration
-- Accounts, Contacts, Leads, Opportunities, Activities

-- ============================================================================
-- CRM Accounts (Companies/Organizations)
-- ============================================================================

CREATE TABLE IF NOT EXISTS crm_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    account_type VARCHAR(50) DEFAULT 'PROSPECT', -- PROSPECT, CUSTOMER, PARTNER, VENDOR
    industry VARCHAR(100),
    website VARCHAR(255),
    phone VARCHAR(50),
    email VARCHAR(255),
    -- Address
    billing_address_line1 VARCHAR(255),
    billing_address_line2 VARCHAR(255),
    billing_city VARCHAR(100),
    billing_state VARCHAR(100),
    billing_postal_code VARCHAR(20),
    billing_country VARCHAR(100),
    shipping_address_line1 VARCHAR(255),
    shipping_address_line2 VARCHAR(255),
    shipping_city VARCHAR(100),
    shipping_state VARCHAR(100),
    shipping_postal_code VARCHAR(20),
    shipping_country VARCHAR(100),
    -- Financial
    annual_revenue DECIMAL(20, 2),
    employee_count INTEGER,
    -- Ownership
    owner_id UUID,
    -- Metadata
    description TEXT,
    tags TEXT[] DEFAULT '{}',
    custom_fields JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_crm_accounts_tenant ON crm_accounts(tenant_id);
CREATE INDEX idx_crm_accounts_type ON crm_accounts(tenant_id, account_type);
CREATE INDEX idx_crm_accounts_owner ON crm_accounts(owner_id);
CREATE INDEX idx_crm_accounts_name ON crm_accounts(tenant_id, name);

-- ============================================================================
-- CRM Contacts (People)
-- ============================================================================

CREATE TABLE IF NOT EXISTS crm_contacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    account_id UUID REFERENCES crm_accounts(id) ON DELETE SET NULL,
    -- Name
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    title VARCHAR(100),
    department VARCHAR(100),
    -- Contact info
    email VARCHAR(255),
    phone VARCHAR(50),
    mobile VARCHAR(50),
    -- Address
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(100),
    postal_code VARCHAR(20),
    country VARCHAR(100),
    -- Social
    linkedin_url VARCHAR(255),
    twitter_handle VARCHAR(100),
    -- Ownership
    owner_id UUID,
    -- Status
    is_primary BOOLEAN DEFAULT FALSE,
    do_not_call BOOLEAN DEFAULT FALSE,
    do_not_email BOOLEAN DEFAULT FALSE,
    -- Metadata
    description TEXT,
    tags TEXT[] DEFAULT '{}',
    custom_fields JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_crm_contacts_tenant ON crm_contacts(tenant_id);
CREATE INDEX idx_crm_contacts_account ON crm_contacts(account_id);
CREATE INDEX idx_crm_contacts_owner ON crm_contacts(owner_id);
CREATE INDEX idx_crm_contacts_email ON crm_contacts(tenant_id, email);
CREATE INDEX idx_crm_contacts_name ON crm_contacts(tenant_id, last_name, first_name);

-- ============================================================================
-- CRM Leads
-- ============================================================================

CREATE TABLE IF NOT EXISTS crm_leads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    -- Lead info
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    company VARCHAR(255),
    title VARCHAR(100),
    email VARCHAR(255),
    phone VARCHAR(50),
    mobile VARCHAR(50),
    website VARCHAR(255),
    -- Source & Status
    source VARCHAR(100), -- WEB, REFERRAL, TRADE_SHOW, COLD_CALL, etc.
    status VARCHAR(50) DEFAULT 'NEW', -- NEW, CONTACTED, QUALIFIED, UNQUALIFIED, CONVERTED
    rating VARCHAR(20), -- HOT, WARM, COLD
    -- Scoring
    score INTEGER DEFAULT 0,
    -- Address
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(100),
    postal_code VARCHAR(20),
    country VARCHAR(100),
    -- Ownership
    owner_id UUID,
    -- Conversion tracking
    converted_at TIMESTAMPTZ,
    converted_account_id UUID REFERENCES crm_accounts(id),
    converted_contact_id UUID REFERENCES crm_contacts(id),
    converted_opportunity_id UUID,
    -- Metadata
    description TEXT,
    tags TEXT[] DEFAULT '{}',
    custom_fields JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_crm_leads_tenant ON crm_leads(tenant_id);
CREATE INDEX idx_crm_leads_status ON crm_leads(tenant_id, status);
CREATE INDEX idx_crm_leads_owner ON crm_leads(owner_id);
CREATE INDEX idx_crm_leads_source ON crm_leads(tenant_id, source);
CREATE INDEX idx_crm_leads_score ON crm_leads(tenant_id, score DESC);

-- ============================================================================
-- CRM Opportunities (Sales Pipeline)
-- ============================================================================

CREATE TABLE IF NOT EXISTS crm_opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    account_id UUID REFERENCES crm_accounts(id) ON DELETE SET NULL,
    contact_id UUID REFERENCES crm_contacts(id) ON DELETE SET NULL,
    -- Pipeline
    stage VARCHAR(50) DEFAULT 'PROSPECTING', -- PROSPECTING, QUALIFICATION, PROPOSAL, NEGOTIATION, CLOSED_WON, CLOSED_LOST
    probability INTEGER DEFAULT 10, -- 0-100
    -- Value
    amount DECIMAL(20, 2),
    currency VARCHAR(3) DEFAULT 'USD',
    -- Dates
    close_date DATE,
    -- Ownership
    owner_id UUID,
    -- Outcome
    closed_at TIMESTAMPTZ,
    won BOOLEAN,
    loss_reason VARCHAR(255),
    -- Metadata
    description TEXT,
    next_step TEXT,
    tags TEXT[] DEFAULT '{}',
    custom_fields JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_crm_opportunities_tenant ON crm_opportunities(tenant_id);
CREATE INDEX idx_crm_opportunities_account ON crm_opportunities(account_id);
CREATE INDEX idx_crm_opportunities_stage ON crm_opportunities(tenant_id, stage);
CREATE INDEX idx_crm_opportunities_owner ON crm_opportunities(owner_id);
CREATE INDEX idx_crm_opportunities_close_date ON crm_opportunities(tenant_id, close_date);

-- ============================================================================
-- CRM Activities (Tasks, Events, Calls, Emails)
-- ============================================================================

CREATE TABLE IF NOT EXISTS crm_activities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    -- Type
    activity_type VARCHAR(50) NOT NULL, -- TASK, EVENT, CALL, EMAIL, MEETING, NOTE
    subject VARCHAR(255) NOT NULL,
    description TEXT,
    -- Related records
    account_id UUID REFERENCES crm_accounts(id) ON DELETE SET NULL,
    contact_id UUID REFERENCES crm_contacts(id) ON DELETE SET NULL,
    lead_id UUID REFERENCES crm_leads(id) ON DELETE SET NULL,
    opportunity_id UUID REFERENCES crm_opportunities(id) ON DELETE SET NULL,
    -- Timing
    due_date DATE,
    due_time TIME,
    start_time TIMESTAMPTZ,
    end_time TIMESTAMPTZ,
    duration_minutes INTEGER,
    -- Status
    status VARCHAR(50) DEFAULT 'OPEN', -- OPEN, IN_PROGRESS, COMPLETED, CANCELLED
    priority VARCHAR(20) DEFAULT 'NORMAL', -- LOW, NORMAL, HIGH, URGENT
    -- Ownership
    owner_id UUID,
    assigned_to UUID,
    -- Completion
    completed_at TIMESTAMPTZ,
    -- Call specific
    call_direction VARCHAR(20), -- INBOUND, OUTBOUND
    call_outcome VARCHAR(50), -- CONNECTED, LEFT_VOICEMAIL, NO_ANSWER, BUSY
    -- Email specific
    email_status VARCHAR(50), -- DRAFT, SENT, DELIVERED, OPENED, BOUNCED
    -- Meeting specific
    location VARCHAR(255),
    meeting_url VARCHAR(500),
    -- Metadata
    tags TEXT[] DEFAULT '{}',
    custom_fields JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_crm_activities_tenant ON crm_activities(tenant_id);
CREATE INDEX idx_crm_activities_type ON crm_activities(tenant_id, activity_type);
CREATE INDEX idx_crm_activities_account ON crm_activities(account_id);
CREATE INDEX idx_crm_activities_contact ON crm_activities(contact_id);
CREATE INDEX idx_crm_activities_lead ON crm_activities(lead_id);
CREATE INDEX idx_crm_activities_opportunity ON crm_activities(opportunity_id);
CREATE INDEX idx_crm_activities_owner ON crm_activities(owner_id);
CREATE INDEX idx_crm_activities_due_date ON crm_activities(tenant_id, due_date);
CREATE INDEX idx_crm_activities_status ON crm_activities(tenant_id, status);

-- ============================================================================
-- Update Triggers
-- ============================================================================

CREATE OR REPLACE FUNCTION update_crm_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_crm_accounts_updated
    BEFORE UPDATE ON crm_accounts
    FOR EACH ROW EXECUTE FUNCTION update_crm_updated_at();

CREATE TRIGGER trg_crm_contacts_updated
    BEFORE UPDATE ON crm_contacts
    FOR EACH ROW EXECUTE FUNCTION update_crm_updated_at();

CREATE TRIGGER trg_crm_leads_updated
    BEFORE UPDATE ON crm_leads
    FOR EACH ROW EXECUTE FUNCTION update_crm_updated_at();

CREATE TRIGGER trg_crm_opportunities_updated
    BEFORE UPDATE ON crm_opportunities
    FOR EACH ROW EXECUTE FUNCTION update_crm_updated_at();

CREATE TRIGGER trg_crm_activities_updated
    BEFORE UPDATE ON crm_activities
    FOR EACH ROW EXECUTE FUNCTION update_crm_updated_at();
