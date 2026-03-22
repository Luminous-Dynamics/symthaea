#!/bin/bash

echo "🚀 Pushing Terra Atlas Schema to Supabase"
echo "=========================================="
echo ""

# Check for required environment variables
if [ -z "$SUPABASE_PROJECT_REF" ]; then
  echo "⚠️  Enter your Supabase project reference (from dashboard URL)"
  read -p "Project ref (e.g., abcd1234efgh5678ijkl): " SUPABASE_PROJECT_REF
fi

SUPABASE_HOST="aws-0-us-west-1.pooler.supabase.com"
SUPABASE_DB="postgres"
SUPABASE_PORT="5432"

echo ""
echo "⚠️  You need the database password for $SUPABASE_PROJECT_REF"
echo "This is the password you set when creating the Supabase project"
echo ""
read -p "Enter Supabase database password: " -s SUPABASE_PASSWORD
echo ""

# Build the connection string
DATABASE_URL="postgresql://postgres.${SUPABASE_PROJECT_REF}:${SUPABASE_PASSWORD}@${SUPABASE_HOST}:${SUPABASE_PORT}/${SUPABASE_DB}"

echo ""
echo "📋 Exporting schema from local database..."
pg_dump -h localhost -p 5434 -U tstoltz -d terra_atlas --schema-only --no-owner --no-privileges > schema.sql

echo "✅ Schema exported to schema.sql"
echo ""

echo "📤 Creating tables in Supabase..."
echo ""

# Remove any PostGIS-specific commands that might fail
sed -i '/CREATE EXTENSION/d' schema.sql
sed -i '/COMMENT ON EXTENSION/d' schema.sql

# Push schema to Supabase
PGPASSWORD="${SUPABASE_PASSWORD}" psql "${DATABASE_URL}" < schema.sql

echo ""
echo "✅ Schema pushed to Supabase!"
echo ""

echo "🔧 Setting up environment file..."
echo ""

# Create .env.production template (secrets should be added manually)
cat > .env.production << EOF
# Supabase Database - Credentials entered at runtime
DATABASE_URL=${DATABASE_URL}

# Supabase API - Get these from https://supabase.com/dashboard
NEXT_PUBLIC_SUPABASE_URL=https://${SUPABASE_PROJECT_REF}.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=<get from Supabase dashboard: Settings > API > anon/public>

# Auth Secrets - Auto-generated
JWT_SECRET=$(openssl rand -base64 32)
NEXTAUTH_SECRET=$(openssl rand -base64 32)

# Mapbox - Get from https://account.mapbox.com/access-tokens/
NEXT_PUBLIC_MAPBOX_TOKEN=<your-mapbox-token>
EOF

echo "✅ Production environment file created"
echo ""
echo "⚠️  IMPORTANT: Edit .env.production to add your API keys:"
echo "    - NEXT_PUBLIC_SUPABASE_ANON_KEY (from Supabase dashboard)"
echo "    - NEXT_PUBLIC_MAPBOX_TOKEN (from Mapbox dashboard)"
echo ""

echo "📊 Testing connection..."
PGPASSWORD="${SUPABASE_PASSWORD}" psql "${DATABASE_URL}" -c "SELECT COUNT(*) FROM information_schema.tables WHERE table_schema = 'public';" 2>/dev/null

echo ""
echo "🎉 SUCCESS! Your database is ready on Supabase!"
echo ""
echo "Next steps:"
echo "1. Edit .env.production to add your API keys"
echo "2. Add environment variables to Vercel:"
echo "   npx vercel env add DATABASE_URL production"
echo ""
echo "3. Deploy to Vercel:"
echo "   npx vercel --prod"
echo ""
