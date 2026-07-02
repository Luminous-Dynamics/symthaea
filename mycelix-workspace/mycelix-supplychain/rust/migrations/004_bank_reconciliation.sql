-- Bank Reconciliation Module
-- Tracks imported bank transactions and matches them to invoices/bills/payments

-- Bank accounts linked via Plaid
CREATE TABLE IF NOT EXISTS fin_bank_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES auth_tenants(id),

    -- Plaid identifiers
    plaid_account_id TEXT,
    plaid_item_id TEXT,
    access_token_encrypted TEXT,  -- Encrypted Plaid access token

    -- Account details
    institution_name TEXT,
    account_name TEXT NOT NULL,
    account_type TEXT NOT NULL,  -- checking, savings, credit, loan
    account_subtype TEXT,
    mask TEXT,  -- Last 4 digits

    -- Linked GL account for reconciliation
    gl_account_id UUID REFERENCES fin_gl_accounts(id),

    -- Balance tracking
    current_balance DECIMAL(15, 2),
    available_balance DECIMAL(15, 2),
    balance_currency TEXT DEFAULT 'USD',
    last_balance_update TIMESTAMPTZ,

    -- Sync tracking
    last_sync_cursor TEXT,
    last_sync_at TIMESTAMPTZ,
    sync_status TEXT DEFAULT 'ACTIVE',  -- ACTIVE, ERROR, DISCONNECTED
    sync_error TEXT,

    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(tenant_id, plaid_account_id)
);

-- Imported bank transactions
CREATE TABLE IF NOT EXISTS fin_bank_transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES auth_tenants(id),
    bank_account_id UUID NOT NULL REFERENCES fin_bank_accounts(id),

    -- External identifiers
    external_transaction_id TEXT NOT NULL,  -- Plaid transaction_id

    -- Transaction details
    transaction_date DATE NOT NULL,
    posted_date DATE,
    amount DECIMAL(15, 2) NOT NULL,  -- Positive = credit, Negative = debit
    currency TEXT DEFAULT 'USD',

    -- Merchant/payee info
    merchant_name TEXT,
    description TEXT,
    category TEXT,
    category_detailed TEXT,

    -- Location (optional)
    location_city TEXT,
    location_region TEXT,
    location_country TEXT,

    -- Status
    is_pending BOOLEAN DEFAULT false,
    payment_channel TEXT,  -- in_store, online, other

    -- Reconciliation status
    reconciliation_status TEXT DEFAULT 'UNMATCHED',  -- UNMATCHED, MATCHED, IGNORED, MANUAL
    matched_at TIMESTAMPTZ,
    matched_by UUID REFERENCES auth_users(id),
    match_confidence DECIMAL(5, 2),  -- 0-100%

    -- Linked documents
    matched_invoice_id UUID REFERENCES fin_invoices(id),
    matched_bill_id UUID REFERENCES fin_bills(id),
    matched_payment_id UUID REFERENCES fin_payments(id),
    matched_journal_id UUID REFERENCES fin_journal_entries(id),

    -- For ignored transactions
    ignore_reason TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(tenant_id, external_transaction_id)
);

-- Reconciliation sessions
CREATE TABLE IF NOT EXISTS fin_reconciliation_sessions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES auth_tenants(id),
    bank_account_id UUID NOT NULL REFERENCES fin_bank_accounts(id),

    -- Session info
    session_date DATE NOT NULL,
    statement_date DATE,
    statement_ending_balance DECIMAL(15, 2),

    -- Calculated values
    book_balance DECIMAL(15, 2),
    bank_balance DECIMAL(15, 2),
    difference DECIMAL(15, 2),

    -- Progress
    status TEXT DEFAULT 'IN_PROGRESS',  -- IN_PROGRESS, COMPLETED, ABANDONED
    started_at TIMESTAMPTZ DEFAULT NOW(),
    completed_at TIMESTAMPTZ,
    completed_by UUID REFERENCES auth_users(id),

    -- Statistics
    total_transactions INT DEFAULT 0,
    matched_count INT DEFAULT 0,
    unmatched_count INT DEFAULT 0,
    ignored_count INT DEFAULT 0,

    notes TEXT,

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Matching rules for auto-reconciliation
CREATE TABLE IF NOT EXISTS fin_reconciliation_rules (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES auth_tenants(id),

    -- Rule definition
    rule_name TEXT NOT NULL,
    rule_type TEXT NOT NULL,  -- EXACT_AMOUNT, AMOUNT_RANGE, MERCHANT, REGEX
    priority INT DEFAULT 100,  -- Lower = higher priority
    is_active BOOLEAN DEFAULT true,

    -- Matching criteria
    match_field TEXT,  -- merchant_name, description, amount
    match_operator TEXT,  -- equals, contains, regex, range
    match_value TEXT,
    match_value_2 TEXT,  -- For ranges

    -- Action
    action_type TEXT NOT NULL,  -- MATCH_INVOICE, MATCH_BILL, CATEGORIZE, IGNORE
    target_gl_account_id UUID REFERENCES fin_gl_accounts(id),

    -- Stats
    times_applied INT DEFAULT 0,
    last_applied_at TIMESTAMPTZ,

    created_by UUID REFERENCES auth_users(id),
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Suggested matches for review
CREATE TABLE IF NOT EXISTS fin_reconciliation_suggestions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES auth_tenants(id),
    bank_transaction_id UUID NOT NULL REFERENCES fin_bank_transactions(id),

    -- Suggested match
    suggested_type TEXT NOT NULL,  -- INVOICE, BILL, PAYMENT
    suggested_id UUID NOT NULL,  -- ID of invoice/bill/payment

    -- Match quality
    confidence DECIMAL(5, 2) NOT NULL,  -- 0-100%
    match_reasons JSONB,  -- Why this was suggested

    -- User action
    status TEXT DEFAULT 'PENDING',  -- PENDING, ACCEPTED, REJECTED
    reviewed_by UUID REFERENCES auth_users(id),
    reviewed_at TIMESTAMPTZ,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_bank_transactions_tenant ON fin_bank_transactions(tenant_id);
CREATE INDEX IF NOT EXISTS idx_bank_transactions_account ON fin_bank_transactions(bank_account_id);
CREATE INDEX IF NOT EXISTS idx_bank_transactions_date ON fin_bank_transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_bank_transactions_status ON fin_bank_transactions(reconciliation_status);
CREATE INDEX IF NOT EXISTS idx_bank_transactions_unmatched ON fin_bank_transactions(tenant_id, reconciliation_status)
    WHERE reconciliation_status = 'UNMATCHED';
CREATE INDEX IF NOT EXISTS idx_bank_accounts_tenant ON fin_bank_accounts(tenant_id);
CREATE INDEX IF NOT EXISTS idx_reconciliation_rules_tenant ON fin_reconciliation_rules(tenant_id, is_active);
