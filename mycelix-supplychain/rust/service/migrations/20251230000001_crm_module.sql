-- ============================================================================
-- Mycelix ERP: CRM Module
-- Customer Relationship Management
-- ============================================================================
-- Migration: 005_crm_module.sql
-- Created: 2025-12-31
-- Description: Complete CRM with accounts, contacts, leads, opportunities, and activities
-- ============================================================================

-- ============================================================================
-- ACCOUNTS (Companies/Organizations)
-- ============================================================================

CREATE TABLE IF NOT EXISTS crm_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    -- Basic info
    name VARCHAR(255) NOT NULL,
    account_type VARCHAR(50) NOT NULL DEFAULT 'CUSTOMER', -- CUSTOMER, PROSPECT, PARTNER, VENDOR, COMPETITOR
    industry VARCHAR(100),
    website VARCHAR(255),
    description TEXT,

    -- Billing address
    billing_address_line1 VARCHAR(255),
    billing_address_line2 VARCHAR(255),
    billing_city VARCHAR(100),
    billing_state VARCHAR(100),
    billing_postal_code VARCHAR(20),
    billing_country VARCHAR(100),

    -- Shipping address
    shipping_address_line1 VARCHAR(255),
    shipping_address_line2 VARCHAR(255),
    shipping_city VARCHAR(100),
    shipping_state VARCHAR(100),
    shipping_postal_code VARCHAR(20),
    shipping_country VARCHAR(100),

    -- Contact info
    phone VARCHAR(50),
    fax VARCHAR(50),

    -- Business details
    employee_count INTEGER,
    annual_revenue DECIMAL(15, 2),

    -- Ownership & assignment
    owner_id UUID,
    parent_account_id UUID REFERENCES crm_accounts(id),

    -- Classification
    rating VARCHAR(20), -- HOT, WARM, COLD
    account_source VARCHAR(100),

    -- Metadata
    tags TEXT[] DEFAULT '{}',
    custom_fields JSONB,
    is_active BOOLEAN NOT NULL DEFAULT true,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_crm_accounts_tenant ON crm_accounts(tenant_id);
CREATE INDEX idx_crm_accounts_name ON crm_accounts(tenant_id, name);
CREATE INDEX idx_crm_accounts_type ON crm_accounts(tenant_id, account_type);
CREATE INDEX idx_crm_accounts_owner ON crm_accounts(owner_id);
CREATE INDEX idx_crm_accounts_parent ON crm_accounts(parent_account_id);

-- ============================================================================
-- CONTACTS (People)
-- ============================================================================

CREATE TABLE IF NOT EXISTS crm_contacts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    -- Personal info
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    salutation VARCHAR(20), -- Mr., Ms., Dr., etc.
    title VARCHAR(100), -- Job title
    department VARCHAR(100),

    -- Account linkage
    account_id UUID REFERENCES crm_accounts(id),
    reports_to_id UUID REFERENCES crm_contacts(id),

    -- Contact info
    email VARCHAR(255),
    phone VARCHAR(50),
    mobile VARCHAR(50),
    fax VARCHAR(50),

    -- Mailing address
    mailing_address_line1 VARCHAR(255),
    mailing_address_line2 VARCHAR(255),
    mailing_city VARCHAR(100),
    mailing_state VARCHAR(100),
    mailing_postal_code VARCHAR(20),
    mailing_country VARCHAR(100),

    -- Other address
    other_address_line1 VARCHAR(255),
    other_address_line2 VARCHAR(255),
    other_city VARCHAR(100),
    other_state VARCHAR(100),
    other_postal_code VARCHAR(20),
    other_country VARCHAR(100),

    -- Additional info
    description TEXT,
    birthdate DATE,
    assistant_name VARCHAR(100),
    assistant_phone VARCHAR(50),

    -- Preferences
    email_opt_out BOOLEAN NOT NULL DEFAULT false,
    do_not_call BOOLEAN NOT NULL DEFAULT false,

    -- Lead conversion
    lead_source VARCHAR(100),

    -- Ownership
    owner_id UUID,

    -- Metadata
    tags TEXT[] DEFAULT '{}',
    custom_fields JSONB,
    is_active BOOLEAN NOT NULL DEFAULT true,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_crm_contacts_tenant ON crm_contacts(tenant_id);
CREATE INDEX idx_crm_contacts_account ON crm_contacts(account_id);
CREATE INDEX idx_crm_contacts_email ON crm_contacts(email);
CREATE INDEX idx_crm_contacts_name ON crm_contacts(tenant_id, last_name, first_name);
CREATE INDEX idx_crm_contacts_owner ON crm_contacts(owner_id);

-- ============================================================================
-- LEADS (Sales Leads)
-- ============================================================================

CREATE TABLE IF NOT EXISTS crm_leads (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    -- Personal info
    first_name VARCHAR(100),
    last_name VARCHAR(100) NOT NULL,
    salutation VARCHAR(20),
    title VARCHAR(100),

    -- Company info
    company VARCHAR(255),
    industry VARCHAR(100),
    employee_count INTEGER,
    annual_revenue DECIMAL(15, 2),
    website VARCHAR(255),

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

    -- Lead details
    description TEXT,
    lead_source VARCHAR(100), -- WEB, REFERRAL, TRADE_SHOW, ADVERTISEMENT, etc.
    status VARCHAR(50) NOT NULL DEFAULT 'NEW', -- NEW, CONTACTED, QUALIFIED, UNQUALIFIED, CONVERTED
    rating VARCHAR(20), -- HOT, WARM, COLD

    -- Scoring
    lead_score INTEGER DEFAULT 0,

    -- Ownership
    owner_id UUID,

    -- Conversion tracking
    is_converted BOOLEAN NOT NULL DEFAULT false,
    converted_at TIMESTAMP WITH TIME ZONE,
    converted_account_id UUID REFERENCES crm_accounts(id),
    converted_contact_id UUID REFERENCES crm_contacts(id),
    converted_opportunity_id UUID,

    -- Campaign tracking
    campaign_id UUID,

    -- Preferences
    email_opt_out BOOLEAN NOT NULL DEFAULT false,
    do_not_call BOOLEAN NOT NULL DEFAULT false,

    -- Metadata
    tags TEXT[] DEFAULT '{}',
    custom_fields JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_crm_leads_tenant ON crm_leads(tenant_id);
CREATE INDEX idx_crm_leads_status ON crm_leads(tenant_id, status);
CREATE INDEX idx_crm_leads_owner ON crm_leads(owner_id);
CREATE INDEX idx_crm_leads_email ON crm_leads(email);
CREATE INDEX idx_crm_leads_company ON crm_leads(tenant_id, company);
CREATE INDEX idx_crm_leads_converted ON crm_leads(tenant_id, is_converted);
CREATE INDEX idx_crm_leads_score ON crm_leads(tenant_id, lead_score DESC);

-- ============================================================================
-- PIPELINE STAGES
-- ============================================================================

CREATE TABLE IF NOT EXISTS crm_pipeline_stages (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    name VARCHAR(100) NOT NULL,
    probability INTEGER NOT NULL DEFAULT 0, -- 0-100
    sort_order INTEGER NOT NULL DEFAULT 0,
    forecast_category VARCHAR(50) NOT NULL DEFAULT 'PIPELINE', -- PIPELINE, BEST_CASE, COMMIT, CLOSED, OMITTED

    is_closed BOOLEAN NOT NULL DEFAULT false,
    is_won BOOLEAN NOT NULL DEFAULT false,
    is_active BOOLEAN NOT NULL DEFAULT true,

    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_crm_pipeline_stages_tenant ON crm_pipeline_stages(tenant_id);

-- ============================================================================
-- OPPORTUNITIES (Sales Pipeline)
-- ============================================================================

CREATE TABLE IF NOT EXISTS crm_opportunities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    -- Core info
    account_id UUID NOT NULL REFERENCES crm_accounts(id),
    name VARCHAR(255) NOT NULL,
    description TEXT,

    -- Pipeline position
    stage VARCHAR(50) NOT NULL DEFAULT 'PROSPECTING',
    probability INTEGER NOT NULL DEFAULT 10, -- 0-100
    forecast_category VARCHAR(50) NOT NULL DEFAULT 'PIPELINE',

    -- Value
    amount DECIMAL(15, 2),
    expected_revenue DECIMAL(15, 2),

    -- Timing
    close_date DATE,

    -- Source
    lead_source VARCHAR(100),

    -- Progress
    next_step VARCHAR(255),

    -- Status
    is_closed BOOLEAN NOT NULL DEFAULT false,
    is_won BOOLEAN NOT NULL DEFAULT false,

    -- Relationships
    owner_id UUID,
    primary_contact_id UUID REFERENCES crm_contacts(id),
    campaign_id UUID,

    -- Competition
    competitors TEXT[] DEFAULT '{}',

    -- Metadata
    tags TEXT[] DEFAULT '{}',
    custom_fields JSONB,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_crm_opportunities_tenant ON crm_opportunities(tenant_id);
CREATE INDEX idx_crm_opportunities_account ON crm_opportunities(account_id);
CREATE INDEX idx_crm_opportunities_stage ON crm_opportunities(tenant_id, stage);
CREATE INDEX idx_crm_opportunities_owner ON crm_opportunities(owner_id);
CREATE INDEX idx_crm_opportunities_close_date ON crm_opportunities(tenant_id, close_date);
CREATE INDEX idx_crm_opportunities_closed ON crm_opportunities(tenant_id, is_closed);

-- Add FK now that opportunities table exists
ALTER TABLE crm_leads
    ADD CONSTRAINT fk_leads_converted_opportunity
    FOREIGN KEY (converted_opportunity_id) REFERENCES crm_opportunities(id);

-- ============================================================================
-- ACTIVITIES (Tasks, Calls, Emails, Meetings, Notes)
-- ============================================================================

CREATE TABLE IF NOT EXISTS crm_activities (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    -- Activity type
    activity_type VARCHAR(50) NOT NULL, -- TASK, EVENT, CALL, EMAIL, MEETING, NOTE

    -- Core info
    subject VARCHAR(255) NOT NULL,
    description TEXT,
    status VARCHAR(50) NOT NULL DEFAULT 'NOT_STARTED', -- NOT_STARTED, IN_PROGRESS, COMPLETED, DEFERRED, CANCELLED
    priority VARCHAR(20) NOT NULL DEFAULT 'NORMAL', -- LOW, NORMAL, HIGH, URGENT

    -- Scheduling
    due_date DATE,
    due_time TIME,
    start_date DATE,
    start_time TIME,
    end_date DATE,
    end_time TIME,
    duration_minutes INTEGER,
    is_all_day BOOLEAN NOT NULL DEFAULT false,
    location VARCHAR(255),

    -- Related records
    account_id UUID REFERENCES crm_accounts(id),
    contact_id UUID REFERENCES crm_contacts(id),
    lead_id UUID REFERENCES crm_leads(id),
    opportunity_id UUID REFERENCES crm_opportunities(id),

    -- Assignment
    owner_id UUID,
    assigned_to_id UUID,

    -- Call/Email specific
    call_direction VARCHAR(20), -- INBOUND, OUTBOUND
    call_result VARCHAR(100),
    email_message_id VARCHAR(255),

    -- Completion tracking
    completed_at TIMESTAMP WITH TIME ZONE,
    completed_by UUID,

    -- Reminders
    is_reminder_set BOOLEAN NOT NULL DEFAULT false,
    reminder_datetime TIMESTAMP WITH TIME ZONE,

    -- Metadata
    tags TEXT[] DEFAULT '{}',

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_crm_activities_tenant ON crm_activities(tenant_id);
CREATE INDEX idx_crm_activities_type ON crm_activities(tenant_id, activity_type);
CREATE INDEX idx_crm_activities_status ON crm_activities(tenant_id, status);
CREATE INDEX idx_crm_activities_account ON crm_activities(account_id);
CREATE INDEX idx_crm_activities_contact ON crm_activities(contact_id);
CREATE INDEX idx_crm_activities_lead ON crm_activities(lead_id);
CREATE INDEX idx_crm_activities_opportunity ON crm_activities(opportunity_id);
CREATE INDEX idx_crm_activities_assigned ON crm_activities(assigned_to_id);
CREATE INDEX idx_crm_activities_due ON crm_activities(tenant_id, due_date);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Pipeline summary view
CREATE OR REPLACE VIEW crm_pipeline_summary AS
SELECT
    tenant_id,
    stage,
    COUNT(*) as opportunity_count,
    COALESCE(SUM(amount), 0) as total_value,
    COALESCE(SUM(amount * probability / 100), 0) as weighted_value,
    AVG(probability) as avg_probability
FROM crm_opportunities
WHERE is_closed = false
GROUP BY tenant_id, stage;

-- Account health view
CREATE OR REPLACE VIEW crm_account_health AS
SELECT
    a.id,
    a.tenant_id,
    a.name,
    COUNT(DISTINCT c.id) as contact_count,
    COUNT(DISTINCT o.id) as opportunity_count,
    COALESCE(SUM(CASE WHEN o.is_won THEN o.amount ELSE 0 END), 0) as won_revenue,
    COUNT(DISTINCT act.id) as activity_count,
    MAX(act.created_at) as last_activity_date
FROM crm_accounts a
LEFT JOIN crm_contacts c ON c.account_id = a.id
LEFT JOIN crm_opportunities o ON o.account_id = a.id
LEFT JOIN crm_activities act ON act.account_id = a.id
GROUP BY a.id, a.tenant_id, a.name;

-- ============================================================================
-- DEFAULT PIPELINE STAGES (Demo Data)
-- ============================================================================

-- Demo tenant ID for seed data
DO $$
DECLARE
    demo_tenant UUID := '00000000-0000-0000-0000-000000000001';
BEGIN
    -- Insert default pipeline stages
    INSERT INTO crm_pipeline_stages (tenant_id, name, probability, sort_order, forecast_category, is_closed, is_won)
    VALUES
        (demo_tenant, 'Prospecting', 10, 1, 'PIPELINE', false, false),
        (demo_tenant, 'Qualification', 20, 2, 'PIPELINE', false, false),
        (demo_tenant, 'Needs Analysis', 40, 3, 'PIPELINE', false, false),
        (demo_tenant, 'Value Proposition', 50, 4, 'BEST_CASE', false, false),
        (demo_tenant, 'Proposal', 60, 5, 'BEST_CASE', false, false),
        (demo_tenant, 'Negotiation', 80, 6, 'COMMIT', false, false),
        (demo_tenant, 'Closed Won', 100, 7, 'CLOSED', true, true),
        (demo_tenant, 'Closed Lost', 0, 8, 'OMITTED', true, false)
    ON CONFLICT DO NOTHING;

    -- Insert demo accounts
    INSERT INTO crm_accounts (id, tenant_id, name, account_type, industry, website, phone, rating)
    VALUES
        ('11111111-1111-1111-1111-111111111111', demo_tenant, 'Acme Corporation', 'CUSTOMER', 'Technology', 'https://acme.example.com', '+1-555-0100', 'HOT'),
        ('22222222-2222-2222-2222-222222222222', demo_tenant, 'Global Industries', 'PROSPECT', 'Manufacturing', 'https://global.example.com', '+1-555-0200', 'WARM'),
        ('33333333-3333-3333-3333-333333333333', demo_tenant, 'Tech Startup Inc', 'CUSTOMER', 'Software', 'https://techstartup.example.com', '+1-555-0300', 'HOT')
    ON CONFLICT (id) DO NOTHING;

    -- Insert demo contacts
    INSERT INTO crm_contacts (id, tenant_id, first_name, last_name, title, account_id, email, phone)
    VALUES
        ('aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa', demo_tenant, 'John', 'Smith', 'CEO', '11111111-1111-1111-1111-111111111111', 'john.smith@acme.example.com', '+1-555-0101'),
        ('bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb', demo_tenant, 'Jane', 'Doe', 'CTO', '11111111-1111-1111-1111-111111111111', 'jane.doe@acme.example.com', '+1-555-0102'),
        ('cccccccc-cccc-cccc-cccc-cccccccccccc', demo_tenant, 'Bob', 'Wilson', 'VP Sales', '22222222-2222-2222-2222-222222222222', 'bob.wilson@global.example.com', '+1-555-0201')
    ON CONFLICT (id) DO NOTHING;

    -- Insert demo opportunities
    INSERT INTO crm_opportunities (id, tenant_id, account_id, name, stage, amount, probability, close_date, primary_contact_id)
    VALUES
        ('dddddddd-dddd-dddd-dddd-dddddddddddd', demo_tenant, '11111111-1111-1111-1111-111111111111', 'Enterprise License Deal', 'NEGOTIATION', 150000.00, 80, CURRENT_DATE + INTERVAL '30 days', 'aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa'),
        ('eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee', demo_tenant, '22222222-2222-2222-2222-222222222222', 'Manufacturing Module', 'PROPOSAL', 75000.00, 60, CURRENT_DATE + INTERVAL '45 days', 'cccccccc-cccc-cccc-cccc-cccccccccccc'),
        ('ffffffff-ffff-ffff-ffff-ffffffffffff', demo_tenant, '33333333-3333-3333-3333-333333333333', 'Startup Package', 'VALUE_PROPOSITION', 25000.00, 50, CURRENT_DATE + INTERVAL '60 days', NULL)
    ON CONFLICT (id) DO NOTHING;

    -- Insert demo leads
    INSERT INTO crm_leads (id, tenant_id, first_name, last_name, company, email, phone, status, lead_source, lead_score)
    VALUES
        ('11111111-2222-3333-4444-555555555555', demo_tenant, 'Alice', 'Johnson', 'New Ventures LLC', 'alice@newventures.example.com', '+1-555-0400', 'QUALIFIED', 'WEB', 85),
        ('22222222-3333-4444-5555-666666666666', demo_tenant, 'Charlie', 'Brown', 'Brown Enterprises', 'charlie@brown.example.com', '+1-555-0500', 'CONTACTED', 'REFERRAL', 65),
        ('33333333-4444-5555-6666-777777777777', demo_tenant, 'Diana', 'Prince', 'Wonder Co', 'diana@wonder.example.com', '+1-555-0600', 'NEW', 'TRADE_SHOW', 45)
    ON CONFLICT (id) DO NOTHING;

END $$;
