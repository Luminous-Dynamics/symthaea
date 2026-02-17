-- HR Module Migration
-- Employees, Departments, Leave Management, Payroll

-- ============================================================================
-- HR Departments
-- ============================================================================

CREATE TABLE IF NOT EXISTS hr_departments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    name VARCHAR(255) NOT NULL,
    code VARCHAR(50),
    description TEXT,
    parent_department_id UUID REFERENCES hr_departments(id) ON DELETE SET NULL,
    manager_id UUID, -- References hr_employees once created
    cost_center VARCHAR(50),
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_hr_departments_tenant ON hr_departments(tenant_id);
CREATE INDEX idx_hr_departments_parent ON hr_departments(parent_department_id);
CREATE INDEX idx_hr_departments_manager ON hr_departments(manager_id);

-- ============================================================================
-- HR Employees
-- ============================================================================

CREATE TABLE IF NOT EXISTS hr_employees (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    employee_number VARCHAR(50) NOT NULL,
    user_id UUID, -- Link to auth user
    -- Personal info
    first_name VARCHAR(100) NOT NULL,
    last_name VARCHAR(100) NOT NULL,
    preferred_name VARCHAR(100),
    email VARCHAR(255) NOT NULL,
    personal_email VARCHAR(255),
    phone VARCHAR(50),
    mobile VARCHAR(50),
    date_of_birth DATE,
    gender VARCHAR(20),
    -- Address
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(100),
    postal_code VARCHAR(20),
    country VARCHAR(100),
    -- Emergency contact
    emergency_contact_name VARCHAR(255),
    emergency_contact_phone VARCHAR(50),
    emergency_contact_relation VARCHAR(100),
    -- Employment
    department_id UUID REFERENCES hr_departments(id) ON DELETE SET NULL,
    job_title VARCHAR(255),
    manager_id UUID REFERENCES hr_employees(id) ON DELETE SET NULL,
    employment_type VARCHAR(50) DEFAULT 'FULL_TIME', -- FULL_TIME, PART_TIME, CONTRACTOR, INTERN
    employment_status VARCHAR(50) DEFAULT 'ACTIVE', -- ACTIVE, ON_LEAVE, TERMINATED, PENDING
    start_date DATE NOT NULL,
    end_date DATE,
    probation_end_date DATE,
    -- Compensation
    salary_type VARCHAR(50) DEFAULT 'SALARY', -- HOURLY, SALARY, COMMISSION
    base_salary DECIMAL(20, 2),
    salary_currency VARCHAR(3) DEFAULT 'USD',
    pay_frequency VARCHAR(50) DEFAULT 'MONTHLY', -- WEEKLY, BIWEEKLY, MONTHLY
    -- Time off balances
    annual_leave_balance DECIMAL(10, 2) DEFAULT 20.0,
    sick_leave_balance DECIMAL(10, 2) DEFAULT 10.0,
    -- Metadata
    tags TEXT[] DEFAULT '{}',
    custom_fields JSONB,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW(),

    UNIQUE(tenant_id, employee_number),
    UNIQUE(tenant_id, email)
);

CREATE INDEX idx_hr_employees_tenant ON hr_employees(tenant_id);
CREATE INDEX idx_hr_employees_department ON hr_employees(department_id);
CREATE INDEX idx_hr_employees_manager ON hr_employees(manager_id);
CREATE INDEX idx_hr_employees_user ON hr_employees(user_id);
CREATE INDEX idx_hr_employees_status ON hr_employees(tenant_id, employment_status);
CREATE INDEX idx_hr_employees_name ON hr_employees(tenant_id, last_name, first_name);

-- Add foreign key for department manager now that employees table exists
ALTER TABLE hr_departments
    ADD CONSTRAINT fk_hr_departments_manager
    FOREIGN KEY (manager_id) REFERENCES hr_employees(id) ON DELETE SET NULL;

-- ============================================================================
-- HR Leave Requests
-- ============================================================================

CREATE TABLE IF NOT EXISTS hr_leave_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    employee_id UUID NOT NULL REFERENCES hr_employees(id) ON DELETE CASCADE,
    leave_type VARCHAR(50) NOT NULL, -- ANNUAL, SICK, PERSONAL, UNPAID, PARENTAL, BEREAVEMENT
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_days DECIMAL(5, 2) NOT NULL,
    reason TEXT,
    status VARCHAR(50) DEFAULT 'PENDING', -- PENDING, APPROVED, REJECTED, CANCELLED
    approved_by UUID REFERENCES hr_employees(id) ON DELETE SET NULL,
    approved_at TIMESTAMPTZ,
    rejection_reason TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_hr_leave_tenant ON hr_leave_requests(tenant_id);
CREATE INDEX idx_hr_leave_employee ON hr_leave_requests(employee_id);
CREATE INDEX idx_hr_leave_status ON hr_leave_requests(tenant_id, status);
CREATE INDEX idx_hr_leave_dates ON hr_leave_requests(tenant_id, start_date, end_date);
CREATE INDEX idx_hr_leave_approver ON hr_leave_requests(approved_by);

-- ============================================================================
-- HR Pay Runs (Payroll Batches)
-- ============================================================================

CREATE TABLE IF NOT EXISTS hr_pay_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,
    pay_period_start DATE NOT NULL,
    pay_period_end DATE NOT NULL,
    pay_date DATE NOT NULL,
    status VARCHAR(50) DEFAULT 'DRAFT', -- DRAFT, APPROVED, PROCESSING, COMPLETED, CANCELLED
    total_gross DECIMAL(20, 2) DEFAULT 0,
    total_deductions DECIMAL(20, 2) DEFAULT 0,
    total_net DECIMAL(20, 2) DEFAULT 0,
    employee_count INTEGER DEFAULT 0,
    notes TEXT,
    approved_by UUID REFERENCES hr_employees(id) ON DELETE SET NULL,
    approved_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_hr_pay_runs_tenant ON hr_pay_runs(tenant_id);
CREATE INDEX idx_hr_pay_runs_status ON hr_pay_runs(tenant_id, status);
CREATE INDEX idx_hr_pay_runs_date ON hr_pay_runs(tenant_id, pay_date);

-- ============================================================================
-- HR Pay Stubs (Individual Employee Payments)
-- ============================================================================

CREATE TABLE IF NOT EXISTS hr_pay_stubs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pay_run_id UUID NOT NULL REFERENCES hr_pay_runs(id) ON DELETE CASCADE,
    employee_id UUID NOT NULL REFERENCES hr_employees(id) ON DELETE CASCADE,
    pay_period_start DATE NOT NULL,
    pay_period_end DATE NOT NULL,
    -- Earnings
    base_salary DECIMAL(20, 2) DEFAULT 0,
    overtime_hours DECIMAL(10, 2) DEFAULT 0,
    overtime_pay DECIMAL(20, 2) DEFAULT 0,
    bonus DECIMAL(20, 2) DEFAULT 0,
    commission DECIMAL(20, 2) DEFAULT 0,
    other_earnings DECIMAL(20, 2) DEFAULT 0,
    gross_pay DECIMAL(20, 2) DEFAULT 0,
    -- Deductions
    tax_federal DECIMAL(20, 2) DEFAULT 0,
    tax_state DECIMAL(20, 2) DEFAULT 0,
    tax_local DECIMAL(20, 2) DEFAULT 0,
    social_security DECIMAL(20, 2) DEFAULT 0,
    medicare DECIMAL(20, 2) DEFAULT 0,
    health_insurance DECIMAL(20, 2) DEFAULT 0,
    retirement_401k DECIMAL(20, 2) DEFAULT 0,
    other_deductions DECIMAL(20, 2) DEFAULT 0,
    total_deductions DECIMAL(20, 2) DEFAULT 0,
    -- Net
    net_pay DECIMAL(20, 2) DEFAULT 0,
    payment_method VARCHAR(50) DEFAULT 'DIRECT_DEPOSIT', -- DIRECT_DEPOSIT, CHECK
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_hr_pay_stubs_run ON hr_pay_stubs(pay_run_id);
CREATE INDEX idx_hr_pay_stubs_employee ON hr_pay_stubs(employee_id);
CREATE INDEX idx_hr_pay_stubs_period ON hr_pay_stubs(pay_period_start, pay_period_end);

-- ============================================================================
-- Update Triggers
-- ============================================================================

CREATE OR REPLACE FUNCTION update_hr_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trg_hr_departments_updated
    BEFORE UPDATE ON hr_departments
    FOR EACH ROW EXECUTE FUNCTION update_hr_updated_at();

CREATE TRIGGER trg_hr_employees_updated
    BEFORE UPDATE ON hr_employees
    FOR EACH ROW EXECUTE FUNCTION update_hr_updated_at();

CREATE TRIGGER trg_hr_leave_requests_updated
    BEFORE UPDATE ON hr_leave_requests
    FOR EACH ROW EXECUTE FUNCTION update_hr_updated_at();

CREATE TRIGGER trg_hr_pay_runs_updated
    BEFORE UPDATE ON hr_pay_runs
    FOR EACH ROW EXECUTE FUNCTION update_hr_updated_at();

-- ============================================================================
-- Helpful Views
-- ============================================================================

-- Employee directory view
CREATE OR REPLACE VIEW hr_employee_directory AS
SELECT
    e.id,
    e.tenant_id,
    e.employee_number,
    e.first_name,
    e.last_name,
    COALESCE(e.preferred_name, e.first_name) AS display_name,
    e.email,
    e.phone,
    e.job_title,
    e.employment_type,
    e.employment_status,
    d.name AS department_name,
    CONCAT(m.first_name, ' ', m.last_name) AS manager_name,
    e.start_date,
    e.is_active
FROM hr_employees e
LEFT JOIN hr_departments d ON e.department_id = d.id
LEFT JOIN hr_employees m ON e.manager_id = m.id;

-- Who's out today view
CREATE OR REPLACE VIEW hr_whos_out_today AS
SELECT
    lr.id AS leave_request_id,
    e.id AS employee_id,
    e.first_name,
    e.last_name,
    e.job_title,
    d.name AS department_name,
    lr.leave_type,
    lr.start_date,
    lr.end_date,
    lr.total_days
FROM hr_leave_requests lr
JOIN hr_employees e ON lr.employee_id = e.id
LEFT JOIN hr_departments d ON e.department_id = d.id
WHERE lr.status = 'APPROVED'
  AND lr.start_date <= CURRENT_DATE
  AND lr.end_date >= CURRENT_DATE;
