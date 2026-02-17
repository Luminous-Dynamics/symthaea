-- Mycelix ERP: Inventory Module
-- Migration 004: Products, Warehouses, Stock Levels, and Movements

-- ============================================================================
-- Product Categories
-- ============================================================================

CREATE TABLE IF NOT EXISTS inv_categories (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    code VARCHAR(50) NOT NULL,
    description TEXT,
    parent_category_id UUID REFERENCES inv_categories(id),
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, code)
);

CREATE INDEX IF NOT EXISTS idx_inv_categories_tenant ON inv_categories(tenant_id);
CREATE INDEX IF NOT EXISTS idx_inv_categories_parent ON inv_categories(parent_category_id);

-- ============================================================================
-- Products
-- ============================================================================

CREATE TABLE IF NOT EXISTS inv_products (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    sku VARCHAR(100) NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    category_id UUID REFERENCES inv_categories(id),
    product_type VARCHAR(50) NOT NULL DEFAULT 'STOCKABLE',
    status VARCHAR(50) NOT NULL DEFAULT 'ACTIVE',
    unit_of_measure VARCHAR(50) NOT NULL DEFAULT 'EACH',
    cost_price DECIMAL(15, 4),
    sale_price DECIMAL(15, 4),
    currency VARCHAR(3) NOT NULL DEFAULT 'USD',
    barcode VARCHAR(100),
    weight DECIMAL(10, 4),
    weight_unit VARCHAR(10),
    dimensions_length DECIMAL(10, 4),
    dimensions_width DECIMAL(10, 4),
    dimensions_height DECIMAL(10, 4),
    dimensions_unit VARCHAR(10),
    min_stock_level DECIMAL(15, 4),
    max_stock_level DECIMAL(15, 4),
    reorder_point DECIMAL(15, 4),
    reorder_quantity DECIMAL(15, 4),
    lead_time_days INTEGER,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, sku)
);

CREATE INDEX IF NOT EXISTS idx_inv_products_tenant ON inv_products(tenant_id);
CREATE INDEX IF NOT EXISTS idx_inv_products_sku ON inv_products(tenant_id, sku);
CREATE INDEX IF NOT EXISTS idx_inv_products_barcode ON inv_products(tenant_id, barcode);
CREATE INDEX IF NOT EXISTS idx_inv_products_category ON inv_products(category_id);
CREATE INDEX IF NOT EXISTS idx_inv_products_status ON inv_products(tenant_id, status);
CREATE INDEX IF NOT EXISTS idx_inv_products_search ON inv_products USING gin(to_tsvector('english', name || ' ' || COALESCE(description, '')));

-- ============================================================================
-- Warehouses
-- ============================================================================

CREATE TABLE IF NOT EXISTS inv_warehouses (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    code VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    warehouse_type VARCHAR(50) NOT NULL DEFAULT 'DISTRIBUTION',
    address_line1 VARCHAR(255),
    address_line2 VARCHAR(255),
    city VARCHAR(100),
    state VARCHAR(100),
    postal_code VARCHAR(20),
    country VARCHAR(100),
    contact_name VARCHAR(255),
    contact_email VARCHAR(255),
    contact_phone VARCHAR(50),
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, code)
);

CREATE INDEX IF NOT EXISTS idx_inv_warehouses_tenant ON inv_warehouses(tenant_id);

-- ============================================================================
-- Storage Locations
-- ============================================================================

CREATE TABLE IF NOT EXISTS inv_locations (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    warehouse_id UUID NOT NULL REFERENCES inv_warehouses(id) ON DELETE CASCADE,
    code VARCHAR(50) NOT NULL,
    name VARCHAR(255) NOT NULL,
    location_type VARCHAR(50) NOT NULL DEFAULT 'RACK',
    zone VARCHAR(50),
    aisle VARCHAR(20),
    rack VARCHAR(20),
    shelf VARCHAR(20),
    bin VARCHAR(20),
    max_weight DECIMAL(10, 4),
    max_volume DECIMAL(10, 4),
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, warehouse_id, code)
);

CREATE INDEX IF NOT EXISTS idx_inv_locations_warehouse ON inv_locations(warehouse_id);
CREATE INDEX IF NOT EXISTS idx_inv_locations_zone ON inv_locations(warehouse_id, zone);

-- ============================================================================
-- Stock Levels
-- ============================================================================

CREATE TABLE IF NOT EXISTS inv_stock_levels (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES inv_products(id) ON DELETE CASCADE,
    warehouse_id UUID NOT NULL REFERENCES inv_warehouses(id) ON DELETE CASCADE,
    location_id UUID REFERENCES inv_locations(id),
    quantity_on_hand DECIMAL(15, 4) NOT NULL DEFAULT 0,
    quantity_reserved DECIMAL(15, 4) NOT NULL DEFAULT 0,
    quantity_available DECIMAL(15, 4) NOT NULL DEFAULT 0,
    quantity_on_order DECIMAL(15, 4) NOT NULL DEFAULT 0,
    unit_cost DECIMAL(15, 4),
    total_value DECIMAL(15, 4),
    last_counted_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, product_id, warehouse_id, COALESCE(location_id, '00000000-0000-0000-0000-000000000000'::uuid))
);

CREATE INDEX IF NOT EXISTS idx_inv_stock_levels_product ON inv_stock_levels(product_id);
CREATE INDEX IF NOT EXISTS idx_inv_stock_levels_warehouse ON inv_stock_levels(warehouse_id);
CREATE INDEX IF NOT EXISTS idx_inv_stock_levels_location ON inv_stock_levels(location_id);
CREATE INDEX IF NOT EXISTS idx_inv_stock_levels_low ON inv_stock_levels(tenant_id) WHERE quantity_on_hand <= 0;

-- ============================================================================
-- Stock Movements
-- ============================================================================

CREATE TABLE IF NOT EXISTS inv_stock_movements (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES inv_products(id) ON DELETE CASCADE,
    warehouse_id UUID NOT NULL REFERENCES inv_warehouses(id) ON DELETE CASCADE,
    location_id UUID REFERENCES inv_locations(id),
    movement_type VARCHAR(50) NOT NULL,
    quantity DECIMAL(15, 4) NOT NULL,
    unit_cost DECIMAL(15, 4),
    reference_type VARCHAR(50),
    reference_id UUID,
    notes TEXT,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inv_movements_product ON inv_stock_movements(product_id);
CREATE INDEX IF NOT EXISTS idx_inv_movements_warehouse ON inv_stock_movements(warehouse_id);
CREATE INDEX IF NOT EXISTS idx_inv_movements_type ON inv_stock_movements(movement_type);
CREATE INDEX IF NOT EXISTS idx_inv_movements_date ON inv_stock_movements(created_at);
CREATE INDEX IF NOT EXISTS idx_inv_movements_reference ON inv_stock_movements(reference_type, reference_id);

-- ============================================================================
-- Lot/Batch Tracking (for FIFO/LIFO)
-- ============================================================================

CREATE TABLE IF NOT EXISTS inv_lots (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES inv_products(id) ON DELETE CASCADE,
    lot_number VARCHAR(100) NOT NULL,
    manufacture_date DATE,
    expiry_date DATE,
    supplier_lot VARCHAR(100),
    notes TEXT,
    is_active BOOLEAN NOT NULL DEFAULT true,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, product_id, lot_number)
);

CREATE INDEX IF NOT EXISTS idx_inv_lots_product ON inv_lots(product_id);
CREATE INDEX IF NOT EXISTS idx_inv_lots_expiry ON inv_lots(expiry_date);

-- ============================================================================
-- Serial Number Tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS inv_serial_numbers (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES inv_products(id) ON DELETE CASCADE,
    serial_number VARCHAR(100) NOT NULL,
    lot_id UUID REFERENCES inv_lots(id),
    warehouse_id UUID REFERENCES inv_warehouses(id),
    location_id UUID REFERENCES inv_locations(id),
    status VARCHAR(50) NOT NULL DEFAULT 'AVAILABLE',
    notes TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(tenant_id, product_id, serial_number)
);

CREATE INDEX IF NOT EXISTS idx_inv_serial_product ON inv_serial_numbers(product_id);
CREATE INDEX IF NOT EXISTS idx_inv_serial_status ON inv_serial_numbers(status);

-- ============================================================================
-- Inventory Counts
-- ============================================================================

CREATE TABLE IF NOT EXISTS inv_counts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    warehouse_id UUID NOT NULL REFERENCES inv_warehouses(id) ON DELETE CASCADE,
    count_type VARCHAR(50) NOT NULL DEFAULT 'FULL',
    status VARCHAR(50) NOT NULL DEFAULT 'DRAFT',
    scheduled_date DATE,
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    notes TEXT,
    created_by UUID REFERENCES users(id),
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_inv_counts_warehouse ON inv_counts(warehouse_id);
CREATE INDEX IF NOT EXISTS idx_inv_counts_status ON inv_counts(status);

CREATE TABLE IF NOT EXISTS inv_count_items (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    count_id UUID NOT NULL REFERENCES inv_counts(id) ON DELETE CASCADE,
    product_id UUID NOT NULL REFERENCES inv_products(id),
    location_id UUID REFERENCES inv_locations(id),
    system_quantity DECIMAL(15, 4) NOT NULL,
    counted_quantity DECIMAL(15, 4),
    variance DECIMAL(15, 4),
    counted_by UUID REFERENCES users(id),
    counted_at TIMESTAMPTZ,
    notes TEXT
);

CREATE INDEX IF NOT EXISTS idx_inv_count_items_count ON inv_count_items(count_id);
CREATE INDEX IF NOT EXISTS idx_inv_count_items_product ON inv_count_items(product_id);

-- ============================================================================
-- Views
-- ============================================================================

-- Product stock summary across all warehouses
CREATE OR REPLACE VIEW v_product_stock_summary AS
SELECT
    p.id as product_id,
    p.tenant_id,
    p.sku,
    p.name,
    COALESCE(SUM(sl.quantity_on_hand), 0) as total_on_hand,
    COALESCE(SUM(sl.quantity_reserved), 0) as total_reserved,
    COALESCE(SUM(sl.quantity_available), 0) as total_available,
    COALESCE(SUM(sl.total_value), 0) as total_value,
    p.reorder_point,
    CASE WHEN COALESCE(SUM(sl.quantity_on_hand), 0) <= COALESCE(p.reorder_point, 0)
         THEN true ELSE false END as below_reorder
FROM inv_products p
LEFT JOIN inv_stock_levels sl ON p.id = sl.product_id
WHERE p.is_active = true
GROUP BY p.id, p.tenant_id, p.sku, p.name, p.reorder_point;

-- Warehouse inventory summary
CREATE OR REPLACE VIEW v_warehouse_inventory_summary AS
SELECT
    w.id as warehouse_id,
    w.tenant_id,
    w.code,
    w.name,
    COUNT(DISTINCT sl.product_id) as unique_products,
    COALESCE(SUM(sl.quantity_on_hand), 0) as total_quantity,
    COALESCE(SUM(sl.total_value), 0) as total_value,
    COUNT(DISTINCT l.id) as location_count
FROM inv_warehouses w
LEFT JOIN inv_stock_levels sl ON w.id = sl.warehouse_id
LEFT JOIN inv_locations l ON w.id = l.warehouse_id AND l.is_active = true
WHERE w.is_active = true
GROUP BY w.id, w.tenant_id, w.code, w.name;

-- Recent movements
CREATE OR REPLACE VIEW v_recent_movements AS
SELECT
    m.id,
    m.tenant_id,
    m.movement_type,
    m.quantity,
    m.unit_cost,
    m.created_at,
    p.sku as product_sku,
    p.name as product_name,
    w.code as warehouse_code,
    w.name as warehouse_name
FROM inv_stock_movements m
JOIN inv_products p ON m.product_id = p.id
JOIN inv_warehouses w ON m.warehouse_id = w.id
ORDER BY m.created_at DESC;

-- ============================================================================
-- Seed Demo Data
-- ============================================================================

-- Get demo tenant
DO $$
DECLARE
    demo_tenant_id UUID;
    main_warehouse_id UUID;
    secondary_warehouse_id UUID;
    electronics_cat_id UUID;
    office_cat_id UUID;
    product1_id UUID;
    product2_id UUID;
    product3_id UUID;
    product4_id UUID;
    product5_id UUID;
BEGIN
    -- Get the demo tenant
    SELECT id INTO demo_tenant_id FROM tenants WHERE slug = 'demo' LIMIT 1;

    IF demo_tenant_id IS NULL THEN
        RAISE NOTICE 'Demo tenant not found, skipping inventory seed data';
        RETURN;
    END IF;

    -- Create categories
    INSERT INTO inv_categories (id, tenant_id, name, code, description)
    VALUES
        (gen_random_uuid(), demo_tenant_id, 'Electronics', 'ELEC', 'Electronic devices and components'),
        (gen_random_uuid(), demo_tenant_id, 'Office Supplies', 'OFFICE', 'Office supplies and stationery'),
        (gen_random_uuid(), demo_tenant_id, 'Furniture', 'FURN', 'Office and warehouse furniture'),
        (gen_random_uuid(), demo_tenant_id, 'Packaging', 'PACK', 'Packaging materials')
    ON CONFLICT DO NOTHING;

    SELECT id INTO electronics_cat_id FROM inv_categories WHERE tenant_id = demo_tenant_id AND code = 'ELEC';
    SELECT id INTO office_cat_id FROM inv_categories WHERE tenant_id = demo_tenant_id AND code = 'OFFICE';

    -- Create warehouses
    INSERT INTO inv_warehouses (id, tenant_id, code, name, warehouse_type, city, state, country)
    VALUES
        (gen_random_uuid(), demo_tenant_id, 'MAIN', 'Main Distribution Center', 'DISTRIBUTION', 'Dallas', 'TX', 'USA'),
        (gen_random_uuid(), demo_tenant_id, 'EAST', 'East Coast Warehouse', 'DISTRIBUTION', 'Atlanta', 'GA', 'USA'),
        (gen_random_uuid(), demo_tenant_id, 'RETAIL-01', 'Downtown Retail Store', 'RETAIL', 'Dallas', 'TX', 'USA')
    ON CONFLICT DO NOTHING;

    SELECT id INTO main_warehouse_id FROM inv_warehouses WHERE tenant_id = demo_tenant_id AND code = 'MAIN';
    SELECT id INTO secondary_warehouse_id FROM inv_warehouses WHERE tenant_id = demo_tenant_id AND code = 'EAST';

    -- Create locations in main warehouse
    IF main_warehouse_id IS NOT NULL THEN
        INSERT INTO inv_locations (tenant_id, warehouse_id, code, name, location_type, zone, aisle, rack)
        VALUES
            (demo_tenant_id, main_warehouse_id, 'A-01-01', 'Aisle A, Rack 1, Shelf 1', 'RACK', 'A', '01', '01'),
            (demo_tenant_id, main_warehouse_id, 'A-01-02', 'Aisle A, Rack 1, Shelf 2', 'RACK', 'A', '01', '02'),
            (demo_tenant_id, main_warehouse_id, 'A-02-01', 'Aisle A, Rack 2, Shelf 1', 'RACK', 'A', '02', '01'),
            (demo_tenant_id, main_warehouse_id, 'B-01-01', 'Aisle B, Rack 1, Shelf 1', 'BULK', 'B', '01', '01'),
            (demo_tenant_id, main_warehouse_id, 'RECV-01', 'Receiving Dock 1', 'RECEIVING', 'DOCK', NULL, NULL),
            (demo_tenant_id, main_warehouse_id, 'SHIP-01', 'Shipping Dock 1', 'SHIPPING', 'DOCK', NULL, NULL)
        ON CONFLICT DO NOTHING;
    END IF;

    -- Create products
    INSERT INTO inv_products (id, tenant_id, sku, name, description, category_id, product_type, unit_of_measure, cost_price, sale_price, reorder_point, reorder_quantity)
    VALUES
        (gen_random_uuid(), demo_tenant_id, 'LAPTOP-PRO-15', 'Professional Laptop 15"', 'High-performance business laptop with 15" display', electronics_cat_id, 'STOCKABLE', 'EACH', 850.00, 1299.99, 10, 25),
        (gen_random_uuid(), demo_tenant_id, 'MONITOR-27', '27" LED Monitor', 'High-resolution LED monitor for professional use', electronics_cat_id, 'STOCKABLE', 'EACH', 220.00, 349.99, 15, 30),
        (gen_random_uuid(), demo_tenant_id, 'KEYBOARD-MECH', 'Mechanical Keyboard', 'Ergonomic mechanical keyboard with RGB', electronics_cat_id, 'STOCKABLE', 'EACH', 45.00, 89.99, 25, 50),
        (gen_random_uuid(), demo_tenant_id, 'PAPER-A4-500', 'A4 Copy Paper (500 sheets)', 'Standard white A4 copy paper, 500 sheet ream', office_cat_id, 'CONSUMABLE', 'BOX', 4.50, 8.99, 100, 200),
        (gen_random_uuid(), demo_tenant_id, 'PENS-BLUE-12', 'Blue Ballpoint Pens (12 pack)', 'Standard blue ballpoint pens, 12 per pack', office_cat_id, 'CONSUMABLE', 'BOX', 2.00, 5.99, 50, 100)
    ON CONFLICT DO NOTHING;

    SELECT id INTO product1_id FROM inv_products WHERE tenant_id = demo_tenant_id AND sku = 'LAPTOP-PRO-15';
    SELECT id INTO product2_id FROM inv_products WHERE tenant_id = demo_tenant_id AND sku = 'MONITOR-27';
    SELECT id INTO product3_id FROM inv_products WHERE tenant_id = demo_tenant_id AND sku = 'KEYBOARD-MECH';
    SELECT id INTO product4_id FROM inv_products WHERE tenant_id = demo_tenant_id AND sku = 'PAPER-A4-500';
    SELECT id INTO product5_id FROM inv_products WHERE tenant_id = demo_tenant_id AND sku = 'PENS-BLUE-12';

    -- Create initial stock levels
    IF main_warehouse_id IS NOT NULL AND product1_id IS NOT NULL THEN
        INSERT INTO inv_stock_levels (tenant_id, product_id, warehouse_id, quantity_on_hand, quantity_available, unit_cost, total_value)
        VALUES
            (demo_tenant_id, product1_id, main_warehouse_id, 45, 45, 850.00, 38250.00),
            (demo_tenant_id, product2_id, main_warehouse_id, 78, 78, 220.00, 17160.00),
            (demo_tenant_id, product3_id, main_warehouse_id, 156, 156, 45.00, 7020.00),
            (demo_tenant_id, product4_id, main_warehouse_id, 500, 500, 4.50, 2250.00),
            (demo_tenant_id, product5_id, main_warehouse_id, 200, 200, 2.00, 400.00)
        ON CONFLICT DO NOTHING;
    END IF;

    IF secondary_warehouse_id IS NOT NULL AND product1_id IS NOT NULL THEN
        INSERT INTO inv_stock_levels (tenant_id, product_id, warehouse_id, quantity_on_hand, quantity_available, unit_cost, total_value)
        VALUES
            (demo_tenant_id, product1_id, secondary_warehouse_id, 20, 20, 850.00, 17000.00),
            (demo_tenant_id, product2_id, secondary_warehouse_id, 35, 35, 220.00, 7700.00),
            (demo_tenant_id, product3_id, secondary_warehouse_id, 80, 80, 45.00, 3600.00)
        ON CONFLICT DO NOTHING;
    END IF;

    -- Create some stock movements
    IF main_warehouse_id IS NOT NULL AND product1_id IS NOT NULL THEN
        INSERT INTO inv_stock_movements (tenant_id, product_id, warehouse_id, movement_type, quantity, unit_cost, notes, created_at)
        VALUES
            (demo_tenant_id, product1_id, main_warehouse_id, 'RECEIPT', 50, 850.00, 'Initial stock receipt from supplier', NOW() - INTERVAL '30 days'),
            (demo_tenant_id, product1_id, main_warehouse_id, 'SHIPMENT', -5, NULL, 'Order #1001 - Customer shipment', NOW() - INTERVAL '15 days'),
            (demo_tenant_id, product2_id, main_warehouse_id, 'RECEIPT', 100, 220.00, 'Bulk order from manufacturer', NOW() - INTERVAL '25 days'),
            (demo_tenant_id, product2_id, main_warehouse_id, 'SHIPMENT', -22, NULL, 'Order #1002 - Customer shipment', NOW() - INTERVAL '10 days'),
            (demo_tenant_id, product3_id, main_warehouse_id, 'RECEIPT', 200, 45.00, 'Quarterly restock', NOW() - INTERVAL '20 days'),
            (demo_tenant_id, product3_id, main_warehouse_id, 'ADJUSTMENT', -44, NULL, 'Inventory count adjustment', NOW() - INTERVAL '5 days');
    END IF;

    RAISE NOTICE 'Inventory seed data created successfully';
END $$;
