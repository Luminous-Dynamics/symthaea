-- ============================================================================
-- Mycelix ERP: HR Module
-- Human Resources Management
-- ============================================================================
-- Migration: 006_hr_module.sql
-- Created: 2025-12-31
-- Description: Complete HR with departments, employees, leave, and payroll
-- ============================================================================

-- ============================================================================
-- DEPARTMENTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS hr_departments (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    name VARCHAR(100) NOT NULL,
    code VARCHAR(20),
    description TEXT,

    -- Hierarchy
    parent_department_id UUID REFERENCES hr_departments(id),
    manager_id UUID, -- Will be FK to hr_employees after that table exists

    -- Financial
    cost_center VARCHAR(50),

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT true,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_hr_departments_tenant ON hr_departments(tenant_id);
CREATE INDEX idx_hr_departments_parent ON hr_departments(parent_department_id);
CREATE INDEX idx_hr_departments_code ON hr_departments(tenant_id, code);

-- ============================================================================
-- EMPLOYEES
-- ============================================================================

CREATE TABLE IF NOT EXISTS hr_employees (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    employee_number VARCHAR(20) NOT NULL,
    user_id UUID, -- Link to auth users table

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
    emergency_contact_name VARCHAR(100),
    emergency_contact_phone VARCHAR(50),
    emergency_contact_relation VARCHAR(50),

    -- Employment details
    department_id UUID REFERENCES hr_departments(id),
    job_title VARCHAR(100),
    manager_id UUID REFERENCES hr_employees(id),
    employment_type VARCHAR(50) NOT NULL DEFAULT 'FULL_TIME', -- FULL_TIME, PART_TIME, CONTRACTOR, INTERN
    employment_status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE', -- ACTIVE, ON_LEAVE, TERMINATED, PENDING
    start_date DATE NOT NULL,
    end_date DATE,
    probation_end_date DATE,

    -- Compensation
    salary_type VARCHAR(20) NOT NULL DEFAULT 'SALARY', -- HOURLY, SALARY, COMMISSION
    base_salary DECIMAL(12, 2),
    salary_currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    pay_frequency VARCHAR(20) NOT NULL DEFAULT 'MONTHLY', -- WEEKLY, BIWEEKLY, MONTHLY

    -- Leave balances
    annual_leave_balance DECIMAL(5, 2) NOT NULL DEFAULT 20.00,
    sick_leave_balance DECIMAL(5, 2) NOT NULL DEFAULT 10.00,

    -- Metadata
    tags TEXT[] DEFAULT '{}',
    custom_fields JSONB,
    is_active BOOLEAN NOT NULL DEFAULT true,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(tenant_id, employee_number),
    UNIQUE(tenant_id, email)
);

CREATE INDEX idx_hr_employees_tenant ON hr_employees(tenant_id);
CREATE INDEX idx_hr_employees_user ON hr_employees(user_id);
CREATE INDEX idx_hr_employees_department ON hr_employees(department_id);
CREATE INDEX idx_hr_employees_manager ON hr_employees(manager_id);
CREATE INDEX idx_hr_employees_status ON hr_employees(tenant_id, employment_status);
CREATE INDEX idx_hr_employees_type ON hr_employees(tenant_id, employment_type);
CREATE INDEX idx_hr_employees_name ON hr_employees(tenant_id, last_name, first_name);
CREATE INDEX idx_hr_employees_email ON hr_employees(email);

-- Add FK from departments.manager_id to employees
ALTER TABLE hr_departments
    ADD CONSTRAINT fk_department_manager
    FOREIGN KEY (manager_id) REFERENCES hr_employees(id);

-- ============================================================================
-- LEAVE REQUESTS
-- ============================================================================

CREATE TABLE IF NOT EXISTS hr_leave_requests (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    employee_id UUID NOT NULL REFERENCES hr_employees(id),

    -- Leave details
    leave_type VARCHAR(50) NOT NULL, -- ANNUAL, SICK, PERSONAL, UNPAID, PARENTAL, BEREAVEMENT
    start_date DATE NOT NULL,
    end_date DATE NOT NULL,
    total_days DECIMAL(5, 2) NOT NULL,
    reason TEXT,

    -- Approval workflow
    status VARCHAR(50) NOT NULL DEFAULT 'PENDING', -- PENDING, APPROVED, REJECTED, CANCELLED
    approved_by UUID REFERENCES hr_employees(id),
    approved_at TIMESTAMP WITH TIME ZONE,
    rejection_reason TEXT,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Constraints
    CHECK (end_date >= start_date),
    CHECK (total_days > 0)
);

CREATE INDEX idx_hr_leave_tenant ON hr_leave_requests(tenant_id);
CREATE INDEX idx_hr_leave_employee ON hr_leave_requests(employee_id);
CREATE INDEX idx_hr_leave_status ON hr_leave_requests(tenant_id, status);
CREATE INDEX idx_hr_leave_dates ON hr_leave_requests(tenant_id, start_date, end_date);
CREATE INDEX idx_hr_leave_type ON hr_leave_requests(tenant_id, leave_type);

-- ============================================================================
-- PAY RUNS (Payroll Batches)
-- ============================================================================

CREATE TABLE IF NOT EXISTS hr_pay_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    -- Period
    pay_period_start DATE NOT NULL,
    pay_period_end DATE NOT NULL,
    pay_date DATE NOT NULL,

    -- Status
    status VARCHAR(50) NOT NULL DEFAULT 'DRAFT', -- DRAFT, APPROVED, PROCESSING, COMPLETED, CANCELLED

    -- Totals
    total_gross DECIMAL(15, 2) NOT NULL DEFAULT 0,
    total_deductions DECIMAL(15, 2) NOT NULL DEFAULT 0,
    total_net DECIMAL(15, 2) NOT NULL DEFAULT 0,
    employee_count INTEGER NOT NULL DEFAULT 0,

    -- Notes
    notes TEXT,

    -- Approval
    approved_by UUID REFERENCES hr_employees(id),
    approved_at TIMESTAMP WITH TIME ZONE,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Constraints
    CHECK (pay_period_end >= pay_period_start)
);

CREATE INDEX idx_hr_pay_runs_tenant ON hr_pay_runs(tenant_id);
CREATE INDEX idx_hr_pay_runs_status ON hr_pay_runs(tenant_id, status);
CREATE INDEX idx_hr_pay_runs_date ON hr_pay_runs(tenant_id, pay_date);

-- ============================================================================
-- PAY STUBS (Individual Employee Payments)
-- ============================================================================

CREATE TABLE IF NOT EXISTS hr_pay_stubs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    pay_run_id UUID NOT NULL REFERENCES hr_pay_runs(id),
    employee_id UUID NOT NULL REFERENCES hr_employees(id),

    -- Period
    pay_period_start DATE NOT NULL,
    pay_period_end DATE NOT NULL,

    -- Earnings
    base_salary DECIMAL(12, 2) NOT NULL DEFAULT 0,
    overtime_hours DECIMAL(8, 2) NOT NULL DEFAULT 0,
    overtime_pay DECIMAL(12, 2) NOT NULL DEFAULT 0,
    bonus DECIMAL(12, 2) NOT NULL DEFAULT 0,
    commission DECIMAL(12, 2) NOT NULL DEFAULT 0,
    other_earnings DECIMAL(12, 2) NOT NULL DEFAULT 0,
    gross_pay DECIMAL(12, 2) NOT NULL DEFAULT 0,

    -- Deductions - Taxes
    tax_federal DECIMAL(12, 2) NOT NULL DEFAULT 0,
    tax_state DECIMAL(12, 2) NOT NULL DEFAULT 0,
    tax_local DECIMAL(12, 2) NOT NULL DEFAULT 0,
    social_security DECIMAL(12, 2) NOT NULL DEFAULT 0,
    medicare DECIMAL(12, 2) NOT NULL DEFAULT 0,

    -- Deductions - Benefits
    health_insurance DECIMAL(12, 2) NOT NULL DEFAULT 0,
    retirement_401k DECIMAL(12, 2) NOT NULL DEFAULT 0,
    other_deductions DECIMAL(12, 2) NOT NULL DEFAULT 0,
    total_deductions DECIMAL(12, 2) NOT NULL DEFAULT 0,

    -- Net pay
    net_pay DECIMAL(12, 2) NOT NULL DEFAULT 0,

    -- Payment method
    payment_method VARCHAR(50) NOT NULL DEFAULT 'DIRECT_DEPOSIT', -- DIRECT_DEPOSIT, CHECK

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    -- Constraints
    UNIQUE(pay_run_id, employee_id)
);

CREATE INDEX idx_hr_pay_stubs_pay_run ON hr_pay_stubs(pay_run_id);
CREATE INDEX idx_hr_pay_stubs_employee ON hr_pay_stubs(employee_id);
CREATE INDEX idx_hr_pay_stubs_period ON hr_pay_stubs(pay_period_start, pay_period_end);

-- ============================================================================
-- LEAVE POLICIES (For future use)
-- ============================================================================

CREATE TABLE IF NOT EXISTS hr_leave_policies (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    name VARCHAR(100) NOT NULL,
    description TEXT,

    -- Entitlements
    annual_leave_days DECIMAL(5, 2) NOT NULL DEFAULT 20.00,
    sick_leave_days DECIMAL(5, 2) NOT NULL DEFAULT 10.00,
    personal_leave_days DECIMAL(5, 2) NOT NULL DEFAULT 3.00,
    parental_leave_days DECIMAL(5, 2) NOT NULL DEFAULT 0.00,
    bereavement_leave_days DECIMAL(5, 2) NOT NULL DEFAULT 3.00,

    -- Accrual settings
    accrual_frequency VARCHAR(20) NOT NULL DEFAULT 'YEARLY', -- MONTHLY, YEARLY

    -- Carryover
    max_carryover_days DECIMAL(5, 2) NOT NULL DEFAULT 5.00,

    -- Status
    is_default BOOLEAN NOT NULL DEFAULT false,
    is_active BOOLEAN NOT NULL DEFAULT true,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_hr_leave_policies_tenant ON hr_leave_policies(tenant_id);

-- ============================================================================
-- JOB POSITIONS (For future use)
-- ============================================================================

CREATE TABLE IF NOT EXISTS hr_job_positions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL,

    title VARCHAR(100) NOT NULL,
    code VARCHAR(20),
    description TEXT,

    department_id UUID REFERENCES hr_departments(id),

    -- Compensation range
    min_salary DECIMAL(12, 2),
    max_salary DECIMAL(12, 2),
    salary_currency VARCHAR(3) NOT NULL DEFAULT 'USD',

    -- Requirements
    requirements TEXT,
    qualifications TEXT,

    -- Status
    is_active BOOLEAN NOT NULL DEFAULT true,

    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE INDEX idx_hr_job_positions_tenant ON hr_job_positions(tenant_id);
CREATE INDEX idx_hr_job_positions_department ON hr_job_positions(department_id);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Department summary view
CREATE OR REPLACE VIEW hr_department_summary AS
SELECT
    d.id,
    d.tenant_id,
    d.name,
    d.code,
    COUNT(e.id) as employee_count,
    COALESCE(SUM(e.base_salary), 0) as total_salary
FROM hr_departments d
LEFT JOIN hr_employees e ON e.department_id = d.id AND e.is_active = true
GROUP BY d.id, d.tenant_id, d.name, d.code;

-- Employee headcount view
CREATE OR REPLACE VIEW hr_headcount AS
SELECT
    tenant_id,
    COUNT(*) as total_employees,
    COUNT(*) FILTER (WHERE employment_status = 'ACTIVE') as active,
    COUNT(*) FILTER (WHERE employment_status = 'ON_LEAVE') as on_leave,
    COUNT(*) FILTER (WHERE employment_type = 'FULL_TIME') as full_time,
    COUNT(*) FILTER (WHERE employment_type = 'PART_TIME') as part_time,
    COUNT(*) FILTER (WHERE employment_type = 'CONTRACTOR') as contractors,
    COUNT(*) FILTER (WHERE employment_type = 'INTERN') as interns
FROM hr_employees
WHERE is_active = true
GROUP BY tenant_id;

-- Who's out today view
CREATE OR REPLACE VIEW hr_whos_out_today AS
SELECT
    lr.tenant_id,
    lr.employee_id,
    e.first_name,
    e.last_name,
    lr.leave_type,
    lr.start_date,
    lr.end_date
FROM hr_leave_requests lr
JOIN hr_employees e ON lr.employee_id = e.id
WHERE lr.status = 'APPROVED'
  AND CURRENT_DATE BETWEEN lr.start_date AND lr.end_date;

-- Payroll summary view
CREATE OR REPLACE VIEW hr_payroll_summary AS
SELECT
    tenant_id,
    EXTRACT(YEAR FROM pay_date)::int as year,
    EXTRACT(MONTH FROM pay_date)::int as month,
    COUNT(*) as pay_run_count,
    SUM(total_gross) as total_gross,
    SUM(total_deductions) as total_deductions,
    SUM(total_net) as total_net,
    SUM(employee_count) as total_payments
FROM hr_pay_runs
WHERE status = 'COMPLETED'
GROUP BY tenant_id, EXTRACT(YEAR FROM pay_date), EXTRACT(MONTH FROM pay_date);

-- ============================================================================
-- DEMO DATA
-- ============================================================================

DO $$
DECLARE
    demo_tenant UUID := '00000000-0000-0000-0000-000000000001';
    eng_dept_id UUID := '11111111-1111-1111-1111-000000000001';
    sales_dept_id UUID := '11111111-1111-1111-1111-000000000002';
    hr_dept_id UUID := '11111111-1111-1111-1111-000000000003';
    emp1_id UUID := '22222222-2222-2222-2222-000000000001';
    emp2_id UUID := '22222222-2222-2222-2222-000000000002';
    emp3_id UUID := '22222222-2222-2222-2222-000000000003';
    emp4_id UUID := '22222222-2222-2222-2222-000000000004';
    emp5_id UUID := '22222222-2222-2222-2222-000000000005';
BEGIN
    -- Insert departments
    INSERT INTO hr_departments (id, tenant_id, name, code, cost_center)
    VALUES
        (eng_dept_id, demo_tenant, 'Engineering', 'ENG', 'CC-100'),
        (sales_dept_id, demo_tenant, 'Sales', 'SALES', 'CC-200'),
        (hr_dept_id, demo_tenant, 'Human Resources', 'HR', 'CC-300')
    ON CONFLICT (id) DO NOTHING;

    -- Insert employees
    INSERT INTO hr_employees (id, tenant_id, employee_number, first_name, last_name, email, phone, department_id, job_title, employment_type, start_date, salary_type, base_salary)
    VALUES
        (emp1_id, demo_tenant, 'EMP001', 'Alice', 'Johnson', 'alice.johnson@example.com', '+1-555-1001', eng_dept_id, 'Senior Software Engineer', 'FULL_TIME', '2022-01-15', 'SALARY', 120000.00),
        (emp2_id, demo_tenant, 'EMP002', 'Bob', 'Smith', 'bob.smith@example.com', '+1-555-1002', eng_dept_id, 'Software Engineer', 'FULL_TIME', '2023-03-01', 'SALARY', 95000.00),
        (emp3_id, demo_tenant, 'EMP003', 'Carol', 'Williams', 'carol.williams@example.com', '+1-555-1003', sales_dept_id, 'Sales Manager', 'FULL_TIME', '2021-06-01', 'SALARY', 110000.00),
        (emp4_id, demo_tenant, 'EMP004', 'David', 'Brown', 'david.brown@example.com', '+1-555-1004', sales_dept_id, 'Account Executive', 'FULL_TIME', '2023-09-15', 'SALARY', 75000.00),
        (emp5_id, demo_tenant, 'EMP005', 'Eve', 'Davis', 'eve.davis@example.com', '+1-555-1005', hr_dept_id, 'HR Manager', 'FULL_TIME', '2020-11-01', 'SALARY', 90000.00)
    ON CONFLICT (id) DO NOTHING;

    -- Set managers
    UPDATE hr_employees SET manager_id = emp1_id WHERE id = emp2_id;
    UPDATE hr_employees SET manager_id = emp5_id WHERE id IN (emp3_id, emp4_id);

    -- Update department managers
    UPDATE hr_departments SET manager_id = emp1_id WHERE id = eng_dept_id;
    UPDATE hr_departments SET manager_id = emp3_id WHERE id = sales_dept_id;
    UPDATE hr_departments SET manager_id = emp5_id WHERE id = hr_dept_id;

    -- Insert some leave requests
    INSERT INTO hr_leave_requests (tenant_id, employee_id, leave_type, start_date, end_date, total_days, status, reason)
    VALUES
        (demo_tenant, emp2_id, 'ANNUAL', CURRENT_DATE + INTERVAL '7 days', CURRENT_DATE + INTERVAL '11 days', 5, 'PENDING', 'Family vacation'),
        (demo_tenant, emp4_id, 'SICK', CURRENT_DATE - INTERVAL '2 days', CURRENT_DATE, 3, 'APPROVED', 'Not feeling well')
    ON CONFLICT DO NOTHING;

    -- Insert a default leave policy
    INSERT INTO hr_leave_policies (tenant_id, name, description, annual_leave_days, sick_leave_days, is_default)
    VALUES
        (demo_tenant, 'Standard Policy', 'Default leave policy for full-time employees', 20.00, 10.00, true)
    ON CONFLICT DO NOTHING;

END $$;
