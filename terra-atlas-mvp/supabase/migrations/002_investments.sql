-- Create investments table
CREATE TABLE IF NOT EXISTS investments (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  project_id INTEGER NOT NULL,
  project_name TEXT NOT NULL,
  project_type TEXT NOT NULL,
  amount DECIMAL(12, 2) NOT NULL CHECK (amount >= 10),
  status TEXT NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'confirmed', 'processing', 'completed', 'cancelled', 'refunded')),
  payment_method TEXT,
  payment_id TEXT,
  transaction_hash TEXT,
  
  -- Investment details
  share_percentage DECIMAL(10, 6),
  expected_return DECIMAL(10, 2),
  investment_term_months INTEGER,
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  confirmed_at TIMESTAMPTZ,
  completed_at TIMESTAMPTZ,
  cancelled_at TIMESTAMPTZ,
  
  -- Metadata
  metadata JSONB DEFAULT '{}'::jsonb
);

-- Create pledges table (for non-binding interest)
CREATE TABLE IF NOT EXISTS pledges (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  project_id INTEGER NOT NULL,
  project_name TEXT NOT NULL,
  project_type TEXT NOT NULL,
  amount DECIMAL(12, 2) NOT NULL CHECK (amount >= 10),
  
  -- Pledge details
  is_binding BOOLEAN DEFAULT FALSE,
  expires_at TIMESTAMPTZ,
  converted_to_investment_id UUID REFERENCES investments(id),
  
  -- Timestamps
  created_at TIMESTAMPTZ DEFAULT NOW(),
  converted_at TIMESTAMPTZ,
  expired_at TIMESTAMPTZ,
  
  -- Metadata
  notes TEXT,
  metadata JSONB DEFAULT '{}'::jsonb
);

-- Create portfolio_summary view
CREATE OR REPLACE VIEW portfolio_summary AS
SELECT 
  u.id as user_id,
  COUNT(DISTINCT i.id) as total_investments,
  COUNT(DISTINCT i.project_id) as unique_projects,
  SUM(i.amount) as total_invested,
  SUM(i.expected_return) as total_expected_return,
  AVG(i.expected_return / NULLIF(i.amount, 0) * 100) as avg_return_percentage,
  COUNT(DISTINCT p.id) as active_pledges,
  SUM(p.amount) as total_pledged
FROM auth.users u
LEFT JOIN investments i ON u.id = i.user_id AND i.status = 'completed'
LEFT JOIN pledges p ON u.id = p.user_id AND p.converted_to_investment_id IS NULL AND (p.expires_at IS NULL OR p.expires_at > NOW())
GROUP BY u.id;

-- Create investment_transactions table for tracking payment history
CREATE TABLE IF NOT EXISTS investment_transactions (
  id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
  investment_id UUID REFERENCES investments(id) ON DELETE CASCADE,
  user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
  type TEXT NOT NULL CHECK (type IN ('deposit', 'withdrawal', 'dividend', 'fee', 'refund')),
  amount DECIMAL(12, 2) NOT NULL,
  balance_after DECIMAL(12, 2),
  description TEXT,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  metadata JSONB DEFAULT '{}'::jsonb
);

-- Create indexes for performance
CREATE INDEX idx_investments_user_id ON investments(user_id);
CREATE INDEX idx_investments_project_id ON investments(project_id);
CREATE INDEX idx_investments_status ON investments(status);
CREATE INDEX idx_investments_created_at ON investments(created_at DESC);

CREATE INDEX idx_pledges_user_id ON pledges(user_id);
CREATE INDEX idx_pledges_project_id ON pledges(project_id);
CREATE INDEX idx_pledges_created_at ON pledges(created_at DESC);

CREATE INDEX idx_investment_transactions_investment_id ON investment_transactions(investment_id);
CREATE INDEX idx_investment_transactions_user_id ON investment_transactions(user_id);
CREATE INDEX idx_investment_transactions_created_at ON investment_transactions(created_at DESC);

-- Row Level Security
ALTER TABLE investments ENABLE ROW LEVEL SECURITY;
ALTER TABLE pledges ENABLE ROW LEVEL SECURITY;
ALTER TABLE investment_transactions ENABLE ROW LEVEL SECURITY;

-- Policies for investments
CREATE POLICY "Users can view their own investments" ON investments
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own investments" ON investments
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their pending investments" ON investments
  FOR UPDATE USING (auth.uid() = user_id AND status = 'pending');

-- Policies for pledges
CREATE POLICY "Users can view their own pledges" ON pledges
  FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can create their own pledges" ON pledges
  FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own pledges" ON pledges
  FOR UPDATE USING (auth.uid() = user_id AND converted_to_investment_id IS NULL);

CREATE POLICY "Users can delete their unconverted pledges" ON pledges
  FOR DELETE USING (auth.uid() = user_id AND converted_to_investment_id IS NULL);

-- Policies for investment_transactions
CREATE POLICY "Users can view their own transactions" ON investment_transactions
  FOR SELECT USING (auth.uid() = user_id);

-- Function to calculate portfolio performance
CREATE OR REPLACE FUNCTION calculate_portfolio_performance(p_user_id UUID)
RETURNS TABLE (
  total_value DECIMAL,
  total_return DECIMAL,
  return_percentage DECIMAL,
  best_performer_id UUID,
  worst_performer_id UUID
) AS $$
BEGIN
  RETURN QUERY
  SELECT 
    SUM(i.amount + COALESCE(i.expected_return, 0)) as total_value,
    SUM(COALESCE(i.expected_return, 0)) as total_return,
    AVG(COALESCE(i.expected_return, 0) / NULLIF(i.amount, 0) * 100) as return_percentage,
    (SELECT id FROM investments WHERE user_id = p_user_id ORDER BY expected_return / NULLIF(amount, 0) DESC LIMIT 1) as best_performer_id,
    (SELECT id FROM investments WHERE user_id = p_user_id ORDER BY expected_return / NULLIF(amount, 0) ASC LIMIT 1) as worst_performer_id
  FROM investments i
  WHERE i.user_id = p_user_id AND i.status = 'completed';
END;
$$ LANGUAGE plpgsql;