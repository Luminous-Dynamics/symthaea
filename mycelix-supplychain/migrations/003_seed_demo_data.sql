-- ============================================================================
-- Mycelix ERP Demo Seed Data
-- Creates a realistic demo environment with sample data
-- ============================================================================

-- Create demo tenant
INSERT INTO tenants (id, name, slug, plan, settings, created_at, updated_at)
VALUES (
    '11111111-1111-1111-1111-111111111111',
    'Acme Supply Co',
    'acme-supply',
    'PROFESSIONAL',
    '{"currency": "USD", "fiscal_year_start": "01-01", "timezone": "America/New_York"}',
    NOW(),
    NOW()
) ON CONFLICT (id) DO NOTHING;

-- Create demo users (password: 'demo123' hashed with argon2id)
-- Hash: $argon2id$v=19$m=19456,t=2,p=1$YWJjZGVmZ2hpamtsbW5vcA$demopasswordhash
INSERT INTO users (id, tenant_id, email, password_hash, name, role, is_active, created_at, updated_at)
VALUES
    ('22222222-2222-2222-2222-222222222222', '11111111-1111-1111-1111-111111111111',
     'admin@acme-demo.com', '$argon2id$v=19$m=19456,t=2,p=1$c2FsdHNhbHRzYWx0$K8rV8nqYvJ1yvGqvGG5xKQ',
     'Alice Admin', 'ADMIN', true, NOW(), NOW()),
    ('33333333-3333-3333-3333-333333333333', '11111111-1111-1111-1111-111111111111',
     'accountant@acme-demo.com', '$argon2id$v=19$m=19456,t=2,p=1$c2FsdHNhbHRzYWx0$K8rV8nqYvJ1yvGqvGG5xKQ',
     'Bob Bookkeeper', 'ACCOUNTANT', true, NOW(), NOW()),
    ('44444444-4444-4444-4444-444444444444', '11111111-1111-1111-1111-111111111111',
     'sales@acme-demo.com', '$argon2id$v=19$m=19456,t=2,p=1$c2FsdHNhbHRzYWx0$K8rV8nqYvJ1yvGqvGG5xKQ',
     'Carol Sales', 'SALES', true, NOW(), NOW())
ON CONFLICT (email) DO NOTHING;

-- ============================================================================
-- CHART OF ACCOUNTS (Standard US GAAP structure)
-- ============================================================================

-- Assets (1xxx)
INSERT INTO gl_accounts (id, account_number, account_name, account_type, parent_account_id, is_active, currency, created_at, updated_at)
VALUES
    ('a1000000-0000-0000-0000-000000000001', '1000', 'Cash and Cash Equivalents', 'ASSET', NULL, true, 'USD', NOW(), NOW()),
    ('a1010000-0000-0000-0000-000000000001', '1010', 'Operating Checking Account', 'ASSET', 'a1000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a1020000-0000-0000-0000-000000000001', '1020', 'Payroll Account', 'ASSET', 'a1000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a1030000-0000-0000-0000-000000000001', '1030', 'Petty Cash', 'ASSET', 'a1000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a1100000-0000-0000-0000-000000000001', '1100', 'Accounts Receivable', 'ASSET', NULL, true, 'USD', NOW(), NOW()),
    ('a1200000-0000-0000-0000-000000000001', '1200', 'Inventory', 'ASSET', NULL, true, 'USD', NOW(), NOW()),
    ('a1210000-0000-0000-0000-000000000001', '1210', 'Raw Materials', 'ASSET', 'a1200000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a1220000-0000-0000-0000-000000000001', '1220', 'Work in Progress', 'ASSET', 'a1200000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a1230000-0000-0000-0000-000000000001', '1230', 'Finished Goods', 'ASSET', 'a1200000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a1500000-0000-0000-0000-000000000001', '1500', 'Fixed Assets', 'ASSET', NULL, true, 'USD', NOW(), NOW()),
    ('a1510000-0000-0000-0000-000000000001', '1510', 'Equipment', 'ASSET', 'a1500000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a1520000-0000-0000-0000-000000000001', '1520', 'Vehicles', 'ASSET', 'a1500000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a1590000-0000-0000-0000-000000000001', '1590', 'Accumulated Depreciation', 'ASSET', 'a1500000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW())
ON CONFLICT (account_number) DO NOTHING;

-- Liabilities (2xxx)
INSERT INTO gl_accounts (id, account_number, account_name, account_type, parent_account_id, is_active, currency, created_at, updated_at)
VALUES
    ('a2000000-0000-0000-0000-000000000001', '2000', 'Accounts Payable', 'LIABILITY', NULL, true, 'USD', NOW(), NOW()),
    ('a2100000-0000-0000-0000-000000000001', '2100', 'Accrued Expenses', 'LIABILITY', NULL, true, 'USD', NOW(), NOW()),
    ('a2200000-0000-0000-0000-000000000001', '2200', 'Payroll Liabilities', 'LIABILITY', NULL, true, 'USD', NOW(), NOW()),
    ('a2210000-0000-0000-0000-000000000001', '2210', 'Wages Payable', 'LIABILITY', 'a2200000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a2220000-0000-0000-0000-000000000001', '2220', 'Payroll Taxes Payable', 'LIABILITY', 'a2200000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a2300000-0000-0000-0000-000000000001', '2300', 'Sales Tax Payable', 'LIABILITY', NULL, true, 'USD', NOW(), NOW()),
    ('a2500000-0000-0000-0000-000000000001', '2500', 'Long-term Debt', 'LIABILITY', NULL, true, 'USD', NOW(), NOW()),
    ('a2510000-0000-0000-0000-000000000001', '2510', 'Bank Loan', 'LIABILITY', 'a2500000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW())
ON CONFLICT (account_number) DO NOTHING;

-- Equity (3xxx)
INSERT INTO gl_accounts (id, account_number, account_name, account_type, parent_account_id, is_active, currency, created_at, updated_at)
VALUES
    ('a3000000-0000-0000-0000-000000000001', '3000', 'Owner''s Equity', 'EQUITY', NULL, true, 'USD', NOW(), NOW()),
    ('a3100000-0000-0000-0000-000000000001', '3100', 'Common Stock', 'EQUITY', 'a3000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a3200000-0000-0000-0000-000000000001', '3200', 'Retained Earnings', 'EQUITY', 'a3000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a3300000-0000-0000-0000-000000000001', '3300', 'Current Year Earnings', 'EQUITY', 'a3000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW())
ON CONFLICT (account_number) DO NOTHING;

-- Revenue (4xxx)
INSERT INTO gl_accounts (id, account_number, account_name, account_type, parent_account_id, is_active, currency, created_at, updated_at)
VALUES
    ('a4000000-0000-0000-0000-000000000001', '4000', 'Sales Revenue', 'REVENUE', NULL, true, 'USD', NOW(), NOW()),
    ('a4100000-0000-0000-0000-000000000001', '4100', 'Product Sales', 'REVENUE', 'a4000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a4200000-0000-0000-0000-000000000001', '4200', 'Service Revenue', 'REVENUE', 'a4000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a4300000-0000-0000-0000-000000000001', '4300', 'Shipping Revenue', 'REVENUE', 'a4000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a4900000-0000-0000-0000-000000000001', '4900', 'Other Income', 'REVENUE', NULL, true, 'USD', NOW(), NOW())
ON CONFLICT (account_number) DO NOTHING;

-- Expenses (5xxx-6xxx)
INSERT INTO gl_accounts (id, account_number, account_name, account_type, parent_account_id, is_active, currency, created_at, updated_at)
VALUES
    ('a5000000-0000-0000-0000-000000000001', '5000', 'Cost of Goods Sold', 'EXPENSE', NULL, true, 'USD', NOW(), NOW()),
    ('a5100000-0000-0000-0000-000000000001', '5100', 'Materials Cost', 'EXPENSE', 'a5000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a5200000-0000-0000-0000-000000000001', '5200', 'Direct Labor', 'EXPENSE', 'a5000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a5300000-0000-0000-0000-000000000001', '5300', 'Manufacturing Overhead', 'EXPENSE', 'a5000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a6000000-0000-0000-0000-000000000001', '6000', 'Operating Expenses', 'EXPENSE', NULL, true, 'USD', NOW(), NOW()),
    ('a6100000-0000-0000-0000-000000000001', '6100', 'Salaries and Wages', 'EXPENSE', 'a6000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a6200000-0000-0000-0000-000000000001', '6200', 'Rent Expense', 'EXPENSE', 'a6000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a6300000-0000-0000-0000-000000000001', '6300', 'Utilities', 'EXPENSE', 'a6000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a6400000-0000-0000-0000-000000000001', '6400', 'Office Supplies', 'EXPENSE', 'a6000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a6500000-0000-0000-0000-000000000001', '6500', 'Insurance', 'EXPENSE', 'a6000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a6600000-0000-0000-0000-000000000001', '6600', 'Professional Fees', 'EXPENSE', 'a6000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a6700000-0000-0000-0000-000000000001', '6700', 'Marketing and Advertising', 'EXPENSE', 'a6000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a6800000-0000-0000-0000-000000000001', '6800', 'Travel and Entertainment', 'EXPENSE', 'a6000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a6900000-0000-0000-0000-000000000001', '6900', 'Depreciation Expense', 'EXPENSE', 'a6000000-0000-0000-0000-000000000001', true, 'USD', NOW(), NOW()),
    ('a7000000-0000-0000-0000-000000000001', '7000', 'Interest Expense', 'EXPENSE', NULL, true, 'USD', NOW(), NOW())
ON CONFLICT (account_number) DO NOTHING;

-- ============================================================================
-- CUSTOMERS
-- ============================================================================
CREATE TABLE IF NOT EXISTS customers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(50),
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(50),
    postal_code VARCHAR(20),
    country VARCHAR(100) DEFAULT 'USA',
    credit_limit DECIMAL(15,2) DEFAULT 0,
    payment_terms_days INTEGER DEFAULT 30,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO customers (id, tenant_id, name, email, phone, address_line1, city, state, postal_code, credit_limit, payment_terms_days)
VALUES
    ('c1000000-0000-0000-0000-000000000001', '11111111-1111-1111-1111-111111111111',
     'TechCorp Industries', 'ap@techcorp.com', '555-0101', '123 Innovation Blvd', 'Austin', 'TX', '78701', 50000.00, 30),
    ('c2000000-0000-0000-0000-000000000001', '11111111-1111-1111-1111-111111111111',
     'Green Valley Farms', 'billing@greenvalley.com', '555-0102', '456 Rural Route 7', 'Portland', 'OR', '97201', 25000.00, 45),
    ('c3000000-0000-0000-0000-000000000001', '11111111-1111-1111-1111-111111111111',
     'Coastal Restaurants Group', 'accounts@coastalrg.com', '555-0103', '789 Harbor Drive', 'San Diego', 'CA', '92101', 35000.00, 30),
    ('c4000000-0000-0000-0000-000000000001', '11111111-1111-1111-1111-111111111111',
     'Mountain View Hospital', 'procurement@mvh.org', '555-0104', '321 Medical Center Way', 'Denver', 'CO', '80202', 100000.00, 60),
    ('c5000000-0000-0000-0000-000000000001', '11111111-1111-1111-1111-111111111111',
     'City Schools District', 'purchasing@cityschools.edu', '555-0105', '555 Education Lane', 'Chicago', 'IL', '60601', 75000.00, 45)
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- VENDORS
-- ============================================================================
CREATE TABLE IF NOT EXISTS vendors (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id),
    name VARCHAR(255) NOT NULL,
    email VARCHAR(255),
    phone VARCHAR(50),
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(50),
    postal_code VARCHAR(20),
    country VARCHAR(100) DEFAULT 'USA',
    tax_id VARCHAR(50),
    payment_terms_days INTEGER DEFAULT 30,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

INSERT INTO vendors (id, tenant_id, name, email, phone, address_line1, city, state, postal_code, tax_id, payment_terms_days)
VALUES
    ('v1000000-0000-0000-0000-000000000001', '11111111-1111-1111-1111-111111111111',
     'Global Materials Inc', 'ar@globalmaterials.com', '555-0201', '100 Industrial Park', 'Detroit', 'MI', '48201', '12-3456789', 30),
    ('v2000000-0000-0000-0000-000000000001', '11111111-1111-1111-1111-111111111111',
     'Pacific Shipping Co', 'billing@pacificship.com', '555-0202', '200 Port Authority Way', 'Los Angeles', 'CA', '90001', '23-4567890', 15),
    ('v3000000-0000-0000-0000-000000000001', '11111111-1111-1111-1111-111111111111',
     'Tech Components Ltd', 'invoices@techcomp.com', '555-0203', '300 Silicon Avenue', 'San Jose', 'CA', '95101', '34-5678901', 45),
    ('v4000000-0000-0000-0000-000000000001', '11111111-1111-1111-1111-111111111111',
     'Office Solutions Plus', 'accounts@officesolutions.com', '555-0204', '400 Commerce Street', 'Dallas', 'TX', '75201', '45-6789012', 30),
    ('v5000000-0000-0000-0000-000000000001', '11111111-1111-1111-1111-111111111111',
     'Utilities Power Corp', 'billing@utilitiespower.com', '555-0205', '500 Energy Boulevard', 'Houston', 'TX', '77001', '56-7890123', 15)
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- SAMPLE INVOICES (Last 3 months)
-- ============================================================================
INSERT INTO invoices (id, invoice_number, customer_id, invoice_date, due_date, currency, subtotal, tax_amount, total_amount, status, created_at, updated_at)
VALUES
    -- December invoices
    ('i1000000-0000-0000-0000-000000000001', 'INV-2024-0001', 'c1000000-0000-0000-0000-000000000001',
     NOW() - INTERVAL '30 days', NOW(), 'USD', 12500.00, 1031.25, 13531.25, 'PAID', NOW() - INTERVAL '30 days', NOW()),
    ('i2000000-0000-0000-0000-000000000001', 'INV-2024-0002', 'c2000000-0000-0000-0000-000000000001',
     NOW() - INTERVAL '28 days', NOW() + INTERVAL '17 days', 'USD', 8750.00, 721.88, 9471.88, 'SENT', NOW() - INTERVAL '28 days', NOW()),
    ('i3000000-0000-0000-0000-000000000001', 'INV-2024-0003', 'c3000000-0000-0000-0000-000000000001',
     NOW() - INTERVAL '25 days', NOW() + INTERVAL '5 days', 'USD', 15000.00, 1237.50, 16237.50, 'SENT', NOW() - INTERVAL '25 days', NOW()),
    -- Recent invoices
    ('i4000000-0000-0000-0000-000000000001', 'INV-2024-0004', 'c4000000-0000-0000-0000-000000000001',
     NOW() - INTERVAL '15 days', NOW() + INTERVAL '45 days', 'USD', 45000.00, 3712.50, 48712.50, 'SENT', NOW() - INTERVAL '15 days', NOW()),
    ('i5000000-0000-0000-0000-000000000001', 'INV-2024-0005', 'c5000000-0000-0000-0000-000000000001',
     NOW() - INTERVAL '10 days', NOW() + INTERVAL '35 days', 'USD', 22500.00, 1856.25, 24356.25, 'SENT', NOW() - INTERVAL '10 days', NOW()),
    ('i6000000-0000-0000-0000-000000000001', 'INV-2024-0006', 'c1000000-0000-0000-0000-000000000001',
     NOW() - INTERVAL '5 days', NOW() + INTERVAL '25 days', 'USD', 18750.00, 1546.88, 20296.88, 'DRAFT', NOW() - INTERVAL '5 days', NOW())
ON CONFLICT (id) DO NOTHING;

-- Invoice lines
INSERT INTO invoice_lines (id, invoice_id, line_number, description, quantity, unit_price, line_total, tax_rate, tax_amount)
VALUES
    ('il100000-0000-0000-0000-000000000001', 'i1000000-0000-0000-0000-000000000001', 1, 'Industrial Sensors - Model A1', 50, 150.00, 7500.00, 8.25, 618.75),
    ('il100001-0000-0000-0000-000000000001', 'i1000000-0000-0000-0000-000000000001', 2, 'Installation Services', 10, 500.00, 5000.00, 8.25, 412.50),
    ('il200000-0000-0000-0000-000000000001', 'i2000000-0000-0000-0000-000000000001', 1, 'Organic Feed Supplies - Bulk', 175, 50.00, 8750.00, 8.25, 721.88),
    ('il300000-0000-0000-0000-000000000001', 'i3000000-0000-0000-0000-000000000001', 1, 'Commercial Kitchen Equipment', 5, 2500.00, 12500.00, 8.25, 1031.25),
    ('il300001-0000-0000-0000-000000000001', 'i3000000-0000-0000-0000-000000000001', 2, 'Setup and Training', 5, 500.00, 2500.00, 8.25, 206.25),
    ('il400000-0000-0000-0000-000000000001', 'i4000000-0000-0000-0000-000000000001', 1, 'Medical Monitoring Systems', 15, 3000.00, 45000.00, 8.25, 3712.50),
    ('il500000-0000-0000-0000-000000000001', 'i5000000-0000-0000-0000-000000000001', 1, 'Educational Technology Package', 45, 500.00, 22500.00, 8.25, 1856.25),
    ('il600000-0000-0000-0000-000000000001', 'i6000000-0000-0000-0000-000000000001', 1, 'Advanced Sensors - Model B2', 75, 250.00, 18750.00, 8.25, 1546.88)
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- SAMPLE BILLS (Last 3 months)
-- ============================================================================
INSERT INTO bills (id, bill_number, vendor_id, bill_date, due_date, currency, subtotal, tax_amount, total_amount, status, created_at, updated_at)
VALUES
    ('b1000000-0000-0000-0000-000000000001', 'BILL-GM-12001', 'v1000000-0000-0000-0000-000000000001',
     NOW() - INTERVAL '35 days', NOW() - INTERVAL '5 days', 'USD', 25000.00, 2062.50, 27062.50, 'PAID', NOW() - INTERVAL '35 days', NOW()),
    ('b2000000-0000-0000-0000-000000000001', 'BILL-PS-12002', 'v2000000-0000-0000-0000-000000000001',
     NOW() - INTERVAL '20 days', NOW() - INTERVAL '5 days', 'USD', 4500.00, 371.25, 4871.25, 'PAID', NOW() - INTERVAL '20 days', NOW()),
    ('b3000000-0000-0000-0000-000000000001', 'BILL-TC-12003', 'v3000000-0000-0000-0000-000000000001',
     NOW() - INTERVAL '15 days', NOW() + INTERVAL '30 days', 'USD', 18000.00, 1485.00, 19485.00, 'APPROVED', NOW() - INTERVAL '15 days', NOW()),
    ('b4000000-0000-0000-0000-000000000001', 'BILL-OS-12004', 'v4000000-0000-0000-0000-000000000001',
     NOW() - INTERVAL '10 days', NOW() + INTERVAL '20 days', 'USD', 2500.00, 206.25, 2706.25, 'APPROVED', NOW() - INTERVAL '10 days', NOW()),
    ('b5000000-0000-0000-0000-000000000001', 'BILL-UP-12005', 'v5000000-0000-0000-0000-000000000001',
     NOW() - INTERVAL '5 days', NOW() + INTERVAL '10 days', 'USD', 3200.00, 264.00, 3464.00, 'DRAFT', NOW() - INTERVAL '5 days', NOW())
ON CONFLICT (id) DO NOTHING;

-- Bill lines
INSERT INTO bill_lines (id, bill_id, line_number, description, quantity, unit_price, line_total, tax_rate, tax_amount, expense_account_id)
VALUES
    ('bl100000-0000-0000-0000-000000000001', 'b1000000-0000-0000-0000-000000000001', 1, 'Raw Materials - Steel Grade A', 500, 50.00, 25000.00, 8.25, 2062.50, 'a5100000-0000-0000-0000-000000000001'),
    ('bl200000-0000-0000-0000-000000000001', 'b2000000-0000-0000-0000-000000000001', 1, 'Freight Charges - December', 1, 4500.00, 4500.00, 8.25, 371.25, 'a5300000-0000-0000-0000-000000000001'),
    ('bl300000-0000-0000-0000-000000000001', 'b3000000-0000-0000-0000-000000000001', 1, 'Electronic Components Batch', 600, 30.00, 18000.00, 8.25, 1485.00, 'a5100000-0000-0000-0000-000000000001'),
    ('bl400000-0000-0000-0000-000000000001', 'b4000000-0000-0000-0000-000000000001', 1, 'Office Supplies - Q4', 1, 2500.00, 2500.00, 8.25, 206.25, 'a6400000-0000-0000-0000-000000000001'),
    ('bl500000-0000-0000-0000-000000000001', 'b5000000-0000-0000-0000-000000000001', 1, 'Electricity - December', 1, 3200.00, 3200.00, 8.25, 264.00, 'a6300000-0000-0000-0000-000000000001')
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- SAMPLE PAYMENTS
-- ============================================================================
INSERT INTO payments (id, payment_number, payment_type, payment_date, amount, currency, payment_method, reference, invoice_id, bill_id, created_at)
VALUES
    -- Customer payments received
    ('p1000000-0000-0000-0000-000000000001', 'PMT-R-2024-0001', 'RECEIVABLE', NOW() - INTERVAL '10 days', 13531.25, 'USD', 'BANK_TRANSFER', 'Wire Transfer #WT12345', 'i1000000-0000-0000-0000-000000000001', NULL, NOW() - INTERVAL '10 days'),
    -- Vendor payments made
    ('p2000000-0000-0000-0000-000000000001', 'PMT-P-2024-0001', 'PAYABLE', NOW() - INTERVAL '8 days', 27062.50, 'USD', 'CHECK', 'Check #10234', NULL, 'b1000000-0000-0000-0000-000000000001', NOW() - INTERVAL '8 days'),
    ('p3000000-0000-0000-0000-000000000001', 'PMT-P-2024-0002', 'PAYABLE', NOW() - INTERVAL '6 days', 4871.25, 'USD', 'BANK_TRANSFER', 'ACH #ACH98765', NULL, 'b2000000-0000-0000-0000-000000000001', NOW() - INTERVAL '6 days')
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- SAMPLE JOURNAL ENTRIES (Opening balances + recent transactions)
-- ============================================================================

-- Opening balance entry
INSERT INTO journal_entries (id, entry_number, entry_date, description, reference, status, lines_hash, created_by, created_at, posted_at)
VALUES
    ('je100000-0000-0000-0000-000000000001', 'JE-2024-0001', NOW() - INTERVAL '90 days',
     'Opening Balances', 'OPENING', 'POSTED', 'hash_opening_balances',
     '22222222-2222-2222-2222-222222222222', NOW() - INTERVAL '90 days', NOW() - INTERVAL '90 days')
ON CONFLICT (id) DO NOTHING;

INSERT INTO journal_lines (id, entry_id, line_number, account_id, debit_amount, credit_amount, description)
VALUES
    -- Opening balances
    ('jl100001-0000-0000-0000-000000000001', 'je100000-0000-0000-0000-000000000001', 1, 'a1010000-0000-0000-0000-000000000001', 150000.00, NULL, 'Opening cash balance'),
    ('jl100002-0000-0000-0000-000000000001', 'je100000-0000-0000-0000-000000000001', 2, 'a1230000-0000-0000-0000-000000000001', 75000.00, NULL, 'Opening inventory'),
    ('jl100003-0000-0000-0000-000000000001', 'je100000-0000-0000-0000-000000000001', 3, 'a1510000-0000-0000-0000-000000000001', 50000.00, NULL, 'Opening equipment'),
    ('jl100004-0000-0000-0000-000000000001', 'je100000-0000-0000-0000-000000000001', 4, 'a2510000-0000-0000-0000-000000000001', NULL, 25000.00, 'Opening bank loan'),
    ('jl100005-0000-0000-0000-000000000001', 'je100000-0000-0000-0000-000000000001', 5, 'a3100000-0000-0000-0000-000000000001', NULL, 200000.00, 'Opening capital'),
    ('jl100006-0000-0000-0000-000000000001', 'je100000-0000-0000-0000-000000000001', 6, 'a3200000-0000-0000-0000-000000000001', NULL, 50000.00, 'Retained earnings')
ON CONFLICT (id) DO NOTHING;

-- Monthly rent entry
INSERT INTO journal_entries (id, entry_number, entry_date, description, reference, status, lines_hash, created_by, created_at, posted_at)
VALUES
    ('je200000-0000-0000-0000-000000000001', 'JE-2024-0010', NOW() - INTERVAL '15 days',
     'December Rent Payment', 'RENT-DEC', 'POSTED', 'hash_rent_dec',
     '22222222-2222-2222-2222-222222222222', NOW() - INTERVAL '15 days', NOW() - INTERVAL '15 days')
ON CONFLICT (id) DO NOTHING;

INSERT INTO journal_lines (id, entry_id, line_number, account_id, debit_amount, credit_amount, description)
VALUES
    ('jl200001-0000-0000-0000-000000000001', 'je200000-0000-0000-0000-000000000001', 1, 'a6200000-0000-0000-0000-000000000001', 8500.00, NULL, 'Rent expense'),
    ('jl200002-0000-0000-0000-000000000001', 'je200000-0000-0000-0000-000000000001', 2, 'a1010000-0000-0000-0000-000000000001', NULL, 8500.00, 'Cash payment')
ON CONFLICT (id) DO NOTHING;

-- Payroll entry
INSERT INTO journal_entries (id, entry_number, entry_date, description, reference, status, lines_hash, created_by, created_at, posted_at)
VALUES
    ('je300000-0000-0000-0000-000000000001', 'JE-2024-0015', NOW() - INTERVAL '7 days',
     'Bi-weekly Payroll', 'PAYROLL-DEC2', 'POSTED', 'hash_payroll_dec2',
     '22222222-2222-2222-2222-222222222222', NOW() - INTERVAL '7 days', NOW() - INTERVAL '7 days')
ON CONFLICT (id) DO NOTHING;

INSERT INTO journal_lines (id, entry_id, line_number, account_id, debit_amount, credit_amount, description)
VALUES
    ('jl300001-0000-0000-0000-000000000001', 'je300000-0000-0000-0000-000000000001', 1, 'a6100000-0000-0000-0000-000000000001', 25000.00, NULL, 'Gross wages'),
    ('jl300002-0000-0000-0000-000000000001', 'je300000-0000-0000-0000-000000000001', 2, 'a2220000-0000-0000-0000-000000000001', NULL, 3825.00, 'Payroll taxes withheld'),
    ('jl300003-0000-0000-0000-000000000001', 'je300000-0000-0000-0000-000000000001', 3, 'a1020000-0000-0000-0000-000000000001', NULL, 21175.00, 'Net pay from payroll account')
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- SUMMARY VIEW
-- ============================================================================
DO $$
BEGIN
    RAISE NOTICE '============================================================';
    RAISE NOTICE 'Mycelix ERP Demo Data Loaded Successfully!';
    RAISE NOTICE '============================================================';
    RAISE NOTICE '';
    RAISE NOTICE 'Demo Credentials:';
    RAISE NOTICE '  Admin:      admin@acme-demo.com';
    RAISE NOTICE '  Accountant: accountant@acme-demo.com';
    RAISE NOTICE '  Sales:      sales@acme-demo.com';
    RAISE NOTICE '  Password:   (set via API or update hash)';
    RAISE NOTICE '';
    RAISE NOTICE 'Demo Company: Acme Supply Co';
    RAISE NOTICE '';
    RAISE NOTICE 'Data Loaded:';
    RAISE NOTICE '  - 35 GL Accounts (full chart of accounts)';
    RAISE NOTICE '  - 5 Customers';
    RAISE NOTICE '  - 5 Vendors';
    RAISE NOTICE '  - 6 Invoices with line items';
    RAISE NOTICE '  - 5 Bills with line items';
    RAISE NOTICE '  - 3 Payments';
    RAISE NOTICE '  - 3 Journal Entries';
    RAISE NOTICE '============================================================';
END $$;
