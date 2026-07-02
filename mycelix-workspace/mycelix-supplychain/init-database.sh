#!/usr/bin/env bash
# Mycelix ERP Database Initialization Script
#
# This script sets up the PostgreSQL database for development

set -e  # Exit on error

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Database configuration (can be overridden with environment variables)
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"
DB_NAME="${DB_NAME:-mycelix_erp}"
DB_USER="${DB_USER:-postgres}"
DB_PASSWORD="${DB_PASSWORD:-postgres}"

export DATABASE_URL="postgresql://${DB_USER}:${DB_PASSWORD}@${DB_HOST}:${DB_PORT}/${DB_NAME}"

echo -e "${GREEN}🗄️  Mycelix ERP Database Setup${NC}"
echo "======================================"
echo ""

# Check if PostgreSQL is running
echo -e "${YELLOW}Checking PostgreSQL connection...${NC}"
if ! psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw template1; then
    echo "❌ Cannot connect to PostgreSQL at ${DB_HOST}:${DB_PORT}"
    echo "Please ensure PostgreSQL is running:"
    echo "  sudo systemctl start postgresql"
    echo "  OR"
    echo "  docker run -d -p 5432:5432 -e POSTGRES_PASSWORD=postgres postgres:15"
    exit 1
fi
echo -e "${GREEN}✅ PostgreSQL is running${NC}"
echo ""

# Create database if it doesn't exist
echo -e "${YELLOW}Creating database '${DB_NAME}' if not exists...${NC}"
if psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -lqt | cut -d \| -f 1 | grep -qw "$DB_NAME"; then
    echo -e "${GREEN}✅ Database '${DB_NAME}' already exists${NC}"
else
    createdb -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" "$DB_NAME"
    echo -e "${GREEN}✅ Database '${DB_NAME}' created${NC}"
fi
echo ""

# Run migrations using sqlx
echo -e "${YELLOW}Running database migrations...${NC}"
if command -v sqlx &> /dev/null; then
    sqlx migrate run
    echo -e "${GREEN}✅ Migrations applied successfully${NC}"
else
    echo -e "${YELLOW}⚠️  sqlx-cli not found. Run migrations manually:${NC}"
    echo "  nix develop"
    echo "  sqlx migrate run"
    echo ""
    echo -e "${YELLOW}Applying migrations directly with psql...${NC}"
    for migration in migrations/*.sql; do
        echo "  Applying: $(basename "$migration")"
        psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -f "$migration"
    done
    echo -e "${GREEN}✅ Migrations applied via psql${NC}"
fi
echo ""

# Verify tables were created
echo -e "${YELLOW}Verifying database schema...${NC}"
TABLE_COUNT=$(psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -t -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';")
echo -e "${GREEN}✅ Found ${TABLE_COUNT} tables in database${NC}"
echo ""

# Show table list
echo -e "${YELLOW}Tables created:${NC}"
psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "\dt"
echo ""

echo -e "${GREEN}🎉 Database setup complete!${NC}"
echo ""
echo "Connection details:"
echo "  DATABASE_URL=${DATABASE_URL}"
echo ""
echo "Next steps:"
echo "  1. Start the service: cargo run"
echo "  2. Test an endpoint: curl http://localhost:3000/v1/fin/accounts"
echo "  3. Create seed data: ./seed-demo-data.sh"
echo ""
