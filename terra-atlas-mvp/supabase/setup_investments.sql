-- Setup Investment Tables for Terra Atlas
-- Run this in Supabase SQL Editor

-- Create investments table if it doesn't exist
CREATE TABLE IF NOT EXISTS investments (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR NOT NULL,
  user_email VARCHAR NOT NULL,
  project_id INTEGER NOT NULL,
  project_name VARCHAR,
  project_type VARCHAR,
  amount DECIMAL(10,2) NOT NULL,
  investment_term_months INTEGER DEFAULT 12,
  payment_method VARCHAR DEFAULT 'stripe',
  payment_status VARCHAR DEFAULT 'pending',
  stripe_payment_intent_id VARCHAR UNIQUE,
  stripe_customer_id VARCHAR,
  stripe_status VARCHAR,
  failure_reason TEXT,
  refund_amount DECIMAL(10,2),
  refunded_at TIMESTAMPTZ,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create index for faster queries
CREATE INDEX IF NOT EXISTS idx_investments_user_id ON investments(user_id);
CREATE INDEX IF NOT EXISTS idx_investments_project_id ON investments(project_id);
CREATE INDEX IF NOT EXISTS idx_investments_payment_intent ON investments(stripe_payment_intent_id);

-- Create failed investment records table for recovery
CREATE TABLE IF NOT EXISTS failed_investment_records (
  id SERIAL PRIMARY KEY,
  payment_intent_id VARCHAR,
  investment_data JSONB,
  error_details JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create subscription table for recurring investments (future feature)
CREATE TABLE IF NOT EXISTS investment_subscriptions (
  id SERIAL PRIMARY KEY,
  stripe_subscription_id VARCHAR UNIQUE,
  stripe_customer_id VARCHAR,
  user_id VARCHAR,
  project_id INTEGER,
  status VARCHAR,
  amount_per_period DECIMAL(10,2),
  current_period_start TIMESTAMPTZ,
  current_period_end TIMESTAMPTZ,
  cancelled_at TIMESTAMPTZ,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Create recurring payments table
CREATE TABLE IF NOT EXISTS recurring_payments (
  id SERIAL PRIMARY KEY,
  stripe_invoice_id VARCHAR UNIQUE,
  stripe_subscription_id VARCHAR,
  amount DECIMAL(10,2),
  paid_at TIMESTAMPTZ,
  created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Add investment tracking columns to projects table if they don't exist
ALTER TABLE projects ADD COLUMN IF NOT EXISTS total_raised DECIMAL(12,2) DEFAULT 0;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS investors_count INTEGER DEFAULT 0;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS target_amount DECIMAL(12,2);
ALTER TABLE projects ADD COLUMN IF NOT EXISTS min_investment DECIMAL(10,2) DEFAULT 10;
ALTER TABLE projects ADD COLUMN IF NOT EXISTS max_investment DECIMAL(10,2) DEFAULT 1000000;

-- Create some test projects if none exist
INSERT INTO projects (
  name, 
  type, 
  capacity_mw, 
  location, 
  state, 
  country, 
  status,
  developer,
  total_raised,
  target_amount,
  min_investment,
  investors_count
) 
SELECT 
  'Desert Solar Farm', 
  'Solar', 
  500, 
  'Mojave Desert', 
  'CA', 
  'USA', 
  'Construction',
  'SunPower Corp',
  1500000,
  5000000,
  10,
  45
WHERE NOT EXISTS (SELECT 1 FROM projects WHERE name = 'Desert Solar Farm');

INSERT INTO projects (
  name, 
  type, 
  capacity_mw, 
  location, 
  state, 
  country, 
  status,
  developer,
  total_raised,
  target_amount,
  min_investment,
  investors_count
) 
SELECT 
  'Atlantic Wind Project', 
  'Wind', 
  800, 
  'Atlantic Coast', 
  'VA', 
  'USA', 
  'Development',
  'Orsted Wind Power',
  3200000,
  10000000,
  10,
  128
WHERE NOT EXISTS (SELECT 1 FROM projects WHERE name = 'Atlantic Wind Project');

INSERT INTO projects (
  name, 
  type, 
  capacity_mw, 
  location, 
  state, 
  country, 
  status,
  developer,
  total_raised,
  target_amount,
  min_investment,
  investors_count
) 
SELECT 
  'Tesla Megapack Storage', 
  'Battery', 
  250, 
  'Austin', 
  'TX', 
  'USA', 
  'Operational',
  'Tesla Energy',
  5500000,
  8000000,
  10,
  312
WHERE NOT EXISTS (SELECT 1 FROM projects WHERE name = 'Tesla Megapack Storage');

-- Grant appropriate permissions
GRANT ALL ON investments TO authenticated;
GRANT ALL ON failed_investment_records TO authenticated;
GRANT ALL ON investment_subscriptions TO authenticated;
GRANT ALL ON recurring_payments TO authenticated;
GRANT SELECT, UPDATE ON projects TO authenticated;

-- Enable Row Level Security (RLS)
ALTER TABLE investments ENABLE ROW LEVEL SECURITY;
ALTER TABLE failed_investment_records ENABLE ROW LEVEL SECURITY;
ALTER TABLE investment_subscriptions ENABLE ROW LEVEL SECURITY;

-- Create RLS policies
CREATE POLICY "Users can view their own investments" ON investments
  FOR SELECT USING (auth.uid()::text = user_id);

CREATE POLICY "Users can create their own investments" ON investments
  FOR INSERT WITH CHECK (auth.uid()::text = user_id);

CREATE POLICY "Service role can manage all investments" ON investments
  FOR ALL USING (auth.role() = 'service_role');

-- Create updated_at trigger
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
  NEW.updated_at = NOW();
  RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_investments_updated_at BEFORE UPDATE ON investments
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_subscriptions_updated_at BEFORE UPDATE ON investment_subscriptions
  FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Success message
SELECT 'Investment tables and test data created successfully!' as message;