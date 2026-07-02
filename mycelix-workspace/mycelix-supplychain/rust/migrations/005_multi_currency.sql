-- Multi-Currency Support Module
-- Enables international business with multiple currencies and exchange rate management

-- Supported currencies
CREATE TABLE IF NOT EXISTS fin_currencies (
    code TEXT PRIMARY KEY,  -- ISO 4217 code (USD, EUR, GBP, etc.)
    name TEXT NOT NULL,
    symbol TEXT NOT NULL,
    decimal_places INT DEFAULT 2,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Exchange rates (base currency is typically USD)
CREATE TABLE IF NOT EXISTS fin_exchange_rates (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES auth_tenants(id),

    from_currency TEXT NOT NULL REFERENCES fin_currencies(code),
    to_currency TEXT NOT NULL REFERENCES fin_currencies(code),
    rate DECIMAL(18, 8) NOT NULL,  -- High precision for exchange rates

    -- Rate metadata
    rate_date DATE NOT NULL,
    rate_type TEXT DEFAULT 'MARKET',  -- MARKET, CUSTOM, BUDGET
    source TEXT,  -- API source or 'MANUAL'

    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    created_by UUID REFERENCES auth_users(id),

    -- One rate per currency pair per day per type
    UNIQUE(tenant_id, from_currency, to_currency, rate_date, rate_type)
);

-- Tenant currency settings
CREATE TABLE IF NOT EXISTS fin_tenant_currencies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES auth_tenants(id),

    -- Primary currency for reporting
    base_currency TEXT NOT NULL REFERENCES fin_currencies(code) DEFAULT 'USD',

    -- Enabled currencies for transactions
    enabled_currencies TEXT[] DEFAULT ARRAY['USD'],

    -- Exchange rate settings
    rate_source TEXT DEFAULT 'MANUAL',  -- MANUAL, OPENEXCHANGE, FIXER, CURRENCYLAYER
    rate_api_key TEXT,  -- Encrypted API key for rate service
    auto_update_rates BOOLEAN DEFAULT false,
    last_rate_update TIMESTAMPTZ,

    -- Rounding preferences
    rounding_mode TEXT DEFAULT 'HALF_UP',  -- HALF_UP, HALF_DOWN, UP, DOWN

    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(tenant_id)
);

-- Add currency fields to existing tables
-- Invoices
ALTER TABLE fin_invoices
    ADD COLUMN IF NOT EXISTS currency TEXT DEFAULT 'USD' REFERENCES fin_currencies(code),
    ADD COLUMN IF NOT EXISTS exchange_rate DECIMAL(18, 8) DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS base_currency_total DECIMAL(15, 2);

-- Bills
ALTER TABLE fin_bills
    ADD COLUMN IF NOT EXISTS currency TEXT DEFAULT 'USD' REFERENCES fin_currencies(code),
    ADD COLUMN IF NOT EXISTS exchange_rate DECIMAL(18, 8) DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS base_currency_total DECIMAL(15, 2);

-- Payments
ALTER TABLE fin_payments
    ADD COLUMN IF NOT EXISTS currency TEXT DEFAULT 'USD' REFERENCES fin_currencies(code),
    ADD COLUMN IF NOT EXISTS exchange_rate DECIMAL(18, 8) DEFAULT 1.0,
    ADD COLUMN IF NOT EXISTS base_currency_amount DECIMAL(15, 2);

-- Journal Entries
ALTER TABLE fin_journal_entries
    ADD COLUMN IF NOT EXISTS currency TEXT DEFAULT 'USD' REFERENCES fin_currencies(code),
    ADD COLUMN IF NOT EXISTS exchange_rate DECIMAL(18, 8) DEFAULT 1.0;

-- GL Accounts - add currency for foreign currency accounts
ALTER TABLE fin_gl_accounts
    ADD COLUMN IF NOT EXISTS currency TEXT DEFAULT 'USD' REFERENCES fin_currencies(code),
    ADD COLUMN IF NOT EXISTS is_foreign_currency BOOLEAN DEFAULT false;

-- Insert common currencies
INSERT INTO fin_currencies (code, name, symbol, decimal_places) VALUES
    ('USD', 'US Dollar', '$', 2),
    ('EUR', 'Euro', '€', 2),
    ('GBP', 'British Pound', '£', 2),
    ('JPY', 'Japanese Yen', '¥', 0),
    ('CHF', 'Swiss Franc', 'CHF', 2),
    ('CAD', 'Canadian Dollar', 'C$', 2),
    ('AUD', 'Australian Dollar', 'A$', 2),
    ('NZD', 'New Zealand Dollar', 'NZ$', 2),
    ('CNY', 'Chinese Yuan', '¥', 2),
    ('INR', 'Indian Rupee', '₹', 2),
    ('MXN', 'Mexican Peso', '$', 2),
    ('BRL', 'Brazilian Real', 'R$', 2),
    ('SGD', 'Singapore Dollar', 'S$', 2),
    ('HKD', 'Hong Kong Dollar', 'HK$', 2),
    ('KRW', 'South Korean Won', '₩', 0),
    ('SEK', 'Swedish Krona', 'kr', 2),
    ('NOK', 'Norwegian Krone', 'kr', 2),
    ('DKK', 'Danish Krone', 'kr', 2),
    ('ZAR', 'South African Rand', 'R', 2),
    ('AED', 'UAE Dirham', 'د.إ', 2)
ON CONFLICT (code) DO NOTHING;

-- Indexes
CREATE INDEX IF NOT EXISTS idx_exchange_rates_tenant ON fin_exchange_rates(tenant_id);
CREATE INDEX IF NOT EXISTS idx_exchange_rates_pair ON fin_exchange_rates(from_currency, to_currency);
CREATE INDEX IF NOT EXISTS idx_exchange_rates_date ON fin_exchange_rates(rate_date DESC);
CREATE INDEX IF NOT EXISTS idx_invoices_currency ON fin_invoices(currency);
CREATE INDEX IF NOT EXISTS idx_bills_currency ON fin_bills(currency);
CREATE INDEX IF NOT EXISTS idx_payments_currency ON fin_payments(currency);

-- Function to get exchange rate
CREATE OR REPLACE FUNCTION get_exchange_rate(
    p_tenant_id UUID,
    p_from_currency TEXT,
    p_to_currency TEXT,
    p_date DATE DEFAULT CURRENT_DATE
) RETURNS DECIMAL(18, 8) AS $$
DECLARE
    v_rate DECIMAL(18, 8);
BEGIN
    -- Same currency = 1.0
    IF p_from_currency = p_to_currency THEN
        RETURN 1.0;
    END IF;

    -- Try direct rate
    SELECT rate INTO v_rate
    FROM fin_exchange_rates
    WHERE tenant_id = p_tenant_id
      AND from_currency = p_from_currency
      AND to_currency = p_to_currency
      AND rate_date <= p_date
      AND is_active = true
    ORDER BY rate_date DESC
    LIMIT 1;

    IF v_rate IS NOT NULL THEN
        RETURN v_rate;
    END IF;

    -- Try inverse rate
    SELECT 1.0 / rate INTO v_rate
    FROM fin_exchange_rates
    WHERE tenant_id = p_tenant_id
      AND from_currency = p_to_currency
      AND to_currency = p_from_currency
      AND rate_date <= p_date
      AND is_active = true
    ORDER BY rate_date DESC
    LIMIT 1;

    IF v_rate IS NOT NULL THEN
        RETURN v_rate;
    END IF;

    -- Triangulate through USD
    IF p_from_currency != 'USD' AND p_to_currency != 'USD' THEN
        DECLARE
            v_from_usd DECIMAL(18, 8);
            v_usd_to DECIMAL(18, 8);
        BEGIN
            v_from_usd := get_exchange_rate(p_tenant_id, p_from_currency, 'USD', p_date);
            v_usd_to := get_exchange_rate(p_tenant_id, 'USD', p_to_currency, p_date);

            IF v_from_usd IS NOT NULL AND v_usd_to IS NOT NULL THEN
                RETURN v_from_usd * v_usd_to;
            END IF;
        END;
    END IF;

    -- No rate found, return NULL
    RETURN NULL;
END;
$$ LANGUAGE plpgsql;

-- Function to convert amount
CREATE OR REPLACE FUNCTION convert_currency(
    p_tenant_id UUID,
    p_amount DECIMAL(15, 2),
    p_from_currency TEXT,
    p_to_currency TEXT,
    p_date DATE DEFAULT CURRENT_DATE
) RETURNS DECIMAL(15, 2) AS $$
DECLARE
    v_rate DECIMAL(18, 8);
    v_decimal_places INT;
BEGIN
    v_rate := get_exchange_rate(p_tenant_id, p_from_currency, p_to_currency, p_date);

    IF v_rate IS NULL THEN
        RAISE EXCEPTION 'No exchange rate found for % to %', p_from_currency, p_to_currency;
    END IF;

    -- Get decimal places for target currency
    SELECT decimal_places INTO v_decimal_places
    FROM fin_currencies WHERE code = p_to_currency;

    RETURN ROUND(p_amount * v_rate, COALESCE(v_decimal_places, 2));
END;
$$ LANGUAGE plpgsql;
