-- Finance Module Database Schema
-- Creates tables for GL, journal entries, invoices, bills, and payments

-- ============================================================================
-- ENUMS
-- ============================================================================

CREATE TYPE account_type AS ENUM (
    'ASSET',
    'LIABILITY',
    'EQUITY',
    'REVENUE',
    'EXPENSE'
);

CREATE TYPE entry_status AS ENUM (
    'DRAFT',
    'POSTED',
    'REVERSED',
    'VOIDED'
);

CREATE TYPE invoice_status AS ENUM (
    'DRAFT',
    'SENT',
    'PAID',
    'PARTIALLY_PAID',
    'OVERDUE',
    'CANCELLED'
);

CREATE TYPE bill_status AS ENUM (
    'DRAFT',
    'APPROVED',
    'PAID',
    'PARTIALLY_PAID',
    'CANCELLED'
);

CREATE TYPE payment_type AS ENUM (
    'RECEIVABLE',
    'PAYABLE'
);

CREATE TYPE payment_method AS ENUM (
    'CASH',
    'CHECK',
    'BANK_TRANSFER',
    'CREDIT_CARD',
    'CRYPTO',
    'OTHER'
);

-- ============================================================================
-- GL ACCOUNTS
-- ============================================================================

CREATE TABLE gl_accounts (
    id UUID PRIMARY KEY,
    account_number VARCHAR(50) NOT NULL UNIQUE,
    account_name VARCHAR(255) NOT NULL,
    account_type account_type NOT NULL,
    parent_account_id UUID REFERENCES gl_accounts(id),
    is_active BOOLEAN NOT NULL DEFAULT true,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_gl_accounts_type ON gl_accounts(account_type);
CREATE INDEX idx_gl_accounts_parent ON gl_accounts(parent_account_id);
CREATE INDEX idx_gl_accounts_active ON gl_accounts(is_active);

-- ============================================================================
-- JOURNAL ENTRIES
-- ============================================================================

CREATE TABLE journal_entries (
    id UUID PRIMARY KEY,
    entry_number VARCHAR(50) NOT NULL UNIQUE,
    entry_date TIMESTAMPTZ NOT NULL,
    description TEXT NOT NULL,
    reference VARCHAR(255),
    status entry_status NOT NULL DEFAULT 'DRAFT',
    lines_hash VARCHAR(64) NOT NULL,  -- SHA-256 hash of all line items
    claim_id VARCHAR(255),  -- Link to DKG claim
    created_by UUID NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    posted_at TIMESTAMPTZ
);

CREATE INDEX idx_journal_entries_date ON journal_entries(entry_date);
CREATE INDEX idx_journal_entries_status ON journal_entries(status);
CREATE INDEX idx_journal_entries_claim ON journal_entries(claim_id);

-- ============================================================================
-- JOURNAL LINES
-- ============================================================================

CREATE TABLE journal_lines (
    id UUID PRIMARY KEY,
    entry_id UUID NOT NULL REFERENCES journal_entries(id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL,
    account_id UUID NOT NULL REFERENCES gl_accounts(id),
    debit_amount DECIMAL(19,4),
    credit_amount DECIMAL(19,4),
    description TEXT,
    dimension1 VARCHAR(100),  -- For cost center, department, etc.
    dimension2 VARCHAR(100),
    dimension3 VARCHAR(100),
    CONSTRAINT check_debit_or_credit CHECK (
        (debit_amount IS NOT NULL AND credit_amount IS NULL) OR
        (debit_amount IS NULL AND credit_amount IS NOT NULL)
    ),
    CONSTRAINT check_amounts_positive CHECK (
        (debit_amount IS NULL OR debit_amount >= 0) AND
        (credit_amount IS NULL OR credit_amount >= 0)
    )
);

CREATE INDEX idx_journal_lines_entry ON journal_lines(entry_id);
CREATE INDEX idx_journal_lines_account ON journal_lines(account_id);
CREATE INDEX idx_journal_lines_dimensions ON journal_lines(dimension1, dimension2, dimension3);

-- ============================================================================
-- INVOICES (Accounts Receivable)
-- ============================================================================

CREATE TABLE invoices (
    id UUID PRIMARY KEY,
    invoice_number VARCHAR(50) NOT NULL UNIQUE,
    customer_id UUID NOT NULL,  -- Will reference CRM customer when CRM module exists
    invoice_date TIMESTAMPTZ NOT NULL,
    due_date TIMESTAMPTZ NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    subtotal DECIMAL(19,4) NOT NULL,
    tax_amount DECIMAL(19,4) NOT NULL DEFAULT 0,
    total_amount DECIMAL(19,4) NOT NULL,
    status invoice_status NOT NULL DEFAULT 'DRAFT',
    journal_entry_id UUID REFERENCES journal_entries(id),
    claim_id VARCHAR(255),  -- Link to DKG claim
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_invoices_customer ON invoices(customer_id);
CREATE INDEX idx_invoices_date ON invoices(invoice_date);
CREATE INDEX idx_invoices_due ON invoices(due_date);
CREATE INDEX idx_invoices_status ON invoices(status);
CREATE INDEX idx_invoices_claim ON invoices(claim_id);

-- ============================================================================
-- INVOICE LINES
-- ============================================================================

CREATE TABLE invoice_lines (
    id UUID PRIMARY KEY,
    invoice_id UUID NOT NULL REFERENCES invoices(id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL,
    description TEXT NOT NULL,
    quantity DECIMAL(19,4) NOT NULL,
    unit_price DECIMAL(19,4) NOT NULL,
    line_total DECIMAL(19,4) NOT NULL,
    tax_rate DECIMAL(5,4),
    tax_amount DECIMAL(19,4),
    item_id UUID  -- Will reference inventory item when MRP module exists
);

CREATE INDEX idx_invoice_lines_invoice ON invoice_lines(invoice_id);
CREATE INDEX idx_invoice_lines_item ON invoice_lines(item_id);

-- ============================================================================
-- BILLS (Accounts Payable)
-- ============================================================================

CREATE TABLE bills (
    id UUID PRIMARY KEY,
    bill_number VARCHAR(50) NOT NULL UNIQUE,
    vendor_id UUID NOT NULL,  -- Will reference vendor when vendor table exists
    bill_date TIMESTAMPTZ NOT NULL,
    due_date TIMESTAMPTZ NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    subtotal DECIMAL(19,4) NOT NULL,
    tax_amount DECIMAL(19,4) NOT NULL DEFAULT 0,
    total_amount DECIMAL(19,4) NOT NULL,
    status bill_status NOT NULL DEFAULT 'DRAFT',
    journal_entry_id UUID REFERENCES journal_entries(id),
    claim_id VARCHAR(255),  -- Link to DKG claim
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_bills_vendor ON bills(vendor_id);
CREATE INDEX idx_bills_date ON bills(bill_date);
CREATE INDEX idx_bills_due ON bills(due_date);
CREATE INDEX idx_bills_status ON bills(status);
CREATE INDEX idx_bills_claim ON bills(claim_id);

-- ============================================================================
-- BILL LINES
-- ============================================================================

CREATE TABLE bill_lines (
    id UUID PRIMARY KEY,
    bill_id UUID NOT NULL REFERENCES bills(id) ON DELETE CASCADE,
    line_number INTEGER NOT NULL,
    description TEXT NOT NULL,
    quantity DECIMAL(19,4) NOT NULL,
    unit_price DECIMAL(19,4) NOT NULL,
    line_total DECIMAL(19,4) NOT NULL,
    tax_rate DECIMAL(5,4),
    tax_amount DECIMAL(19,4),
    expense_account_id UUID REFERENCES gl_accounts(id)
);

CREATE INDEX idx_bill_lines_bill ON bill_lines(bill_id);
CREATE INDEX idx_bill_lines_account ON bill_lines(expense_account_id);

-- ============================================================================
-- PAYMENTS
-- ============================================================================

CREATE TABLE payments (
    id UUID PRIMARY KEY,
    payment_number VARCHAR(50) NOT NULL UNIQUE,
    payment_type payment_type NOT NULL,
    payment_date TIMESTAMPTZ NOT NULL,
    amount DECIMAL(19,4) NOT NULL,
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    payment_method payment_method NOT NULL,
    reference VARCHAR(255),
    invoice_id UUID REFERENCES invoices(id),
    bill_id UUID REFERENCES bills(id),
    journal_entry_id UUID REFERENCES journal_entries(id),
    claim_id VARCHAR(255),  -- Link to DKG claim
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    CONSTRAINT check_payment_target CHECK (
        (payment_type = 'RECEIVABLE' AND invoice_id IS NOT NULL AND bill_id IS NULL) OR
        (payment_type = 'PAYABLE' AND bill_id IS NOT NULL AND invoice_id IS NULL)
    )
);

CREATE INDEX idx_payments_invoice ON payments(invoice_id);
CREATE INDEX idx_payments_bill ON payments(bill_id);
CREATE INDEX idx_payments_date ON payments(payment_date);
CREATE INDEX idx_payments_type ON payments(payment_type);
CREATE INDEX idx_payments_claim ON payments(claim_id);

-- ============================================================================
-- FUNCTIONS & TRIGGERS
-- ============================================================================

-- Function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at
CREATE TRIGGER update_gl_accounts_updated_at BEFORE UPDATE ON gl_accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_invoices_updated_at BEFORE UPDATE ON invoices
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_bills_updated_at BEFORE UPDATE ON bills
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- ============================================================================
-- SEED DATA: Standard Chart of Accounts
-- ============================================================================

-- Assets
INSERT INTO gl_accounts (id, account_number, account_name, account_type, currency) VALUES
    (gen_random_uuid(), '1000', 'Cash', 'ASSET', 'USD'),
    (gen_random_uuid(), '1100', 'Accounts Receivable', 'ASSET', 'USD'),
    (gen_random_uuid(), '1200', 'Inventory', 'ASSET', 'USD'),
    (gen_random_uuid(), '1500', 'Equipment', 'ASSET', 'USD'),
    (gen_random_uuid(), '1600', 'Accumulated Depreciation', 'ASSET', 'USD');

-- Liabilities
INSERT INTO gl_accounts (id, account_number, account_name, account_type, currency) VALUES
    (gen_random_uuid(), '2000', 'Accounts Payable', 'LIABILITY', 'USD'),
    (gen_random_uuid(), '2100', 'Accrued Expenses', 'LIABILITY', 'USD'),
    (gen_random_uuid(), '2500', 'Long-term Debt', 'LIABILITY', 'USD');

-- Equity
INSERT INTO gl_accounts (id, account_number, account_name, account_type, currency) VALUES
    (gen_random_uuid(), '3000', 'Owner Equity', 'EQUITY', 'USD'),
    (gen_random_uuid(), '3100', 'Retained Earnings', 'EQUITY', 'USD');

-- Revenue
INSERT INTO gl_accounts (id, account_number, account_name, account_type, currency) VALUES
    (gen_random_uuid(), '4000', 'Product Sales', 'REVENUE', 'USD'),
    (gen_random_uuid(), '4100', 'Service Revenue', 'REVENUE', 'USD'),
    (gen_random_uuid(), '4200', 'Interest Income', 'REVENUE', 'USD');

-- Expenses
INSERT INTO gl_accounts (id, account_number, account_name, account_type, currency) VALUES
    (gen_random_uuid(), '5000', 'Cost of Goods Sold', 'EXPENSE', 'USD'),
    (gen_random_uuid(), '6000', 'Salaries & Wages', 'EXPENSE', 'USD'),
    (gen_random_uuid(), '6100', 'Rent Expense', 'EXPENSE', 'USD'),
    (gen_random_uuid(), '6200', 'Utilities', 'EXPENSE', 'USD'),
    (gen_random_uuid(), '6300', 'Office Supplies', 'EXPENSE', 'USD'),
    (gen_random_uuid(), '6400', 'Marketing & Advertising', 'EXPENSE', 'USD'),
    (gen_random_uuid(), '6500', 'Travel & Entertainment', 'EXPENSE', 'USD'),
    (gen_random_uuid(), '6600', 'Professional Fees', 'EXPENSE', 'USD'),
    (gen_random_uuid(), '6700', 'Insurance', 'EXPENSE', 'USD'),
    (gen_random_uuid(), '6800', 'Depreciation', 'EXPENSE', 'USD');
