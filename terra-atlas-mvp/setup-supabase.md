# 🚀 Setting Up Supabase for Terra Atlas Production Database

## Why Supabase?
- **Free Tier**: 500MB database, perfect for MVP
- **PostgreSQL**: Full PostgreSQL with PostGIS support
- **Built-in Auth**: Can replace our JWT system later if needed
- **Instant APIs**: Automatic REST and GraphQL APIs
- **Works with Drizzle**: Full compatibility

## Quick Setup Steps

### 1. Create Supabase Project

1. Go to https://supabase.com
2. Sign up/Login (use your GitHub account)
3. Click "New Project"
4. Settings:
   - Project name: `terra-atlas`
   - Database Password: (Generate a strong one and SAVE IT)
   - Region: Choose closest to you
   - Plan: Free tier

### 2. Get Connection String

Once created, go to:
- Settings → Database
- Connection string → Node.js
- Copy the connection string

It looks like:
```
postgresql://postgres.[project-ref]:[password]@aws-0-[region].pooler.supabase.com:5432/postgres
```

### 3. Update Vercel Environment

```bash
# Add the database URL to Vercel
npx vercel env add DATABASE_URL production

# Paste your Supabase connection string when prompted
```

### 4. Run Migrations on Supabase

```bash
# Update your local .env with Supabase URL
export DATABASE_URL="your-supabase-url"

# Push schema to Supabase
npm run db:push

# Or generate and run migrations
npm run db:generate
npm run db:migrate
```

### 5. Import Existing Data (Optional)

```bash
# Export from local PostgreSQL
pg_dump -h localhost -p 5434 -U tstoltz -d terra_atlas --data-only > data.sql

# Import to Supabase (use psql with Supabase connection string)
psql "your-supabase-connection-string" < data.sql
```

## Alternative: Use Existing Supabase Project

If you already have a Supabase project configured:
1. Run the schema migrations there
2. Update DATABASE_URL in Vercel to point to it
3. Configure all required environment variables in Vercel dashboard

## Vercel Environment Variables

Add these to Vercel (Settings → Environment Variables):

```bash
# Database
DATABASE_URL=postgresql://postgres.[project]:[password]@[host]:5432/postgres

# Supabase (if using their auth)
NEXT_PUBLIC_SUPABASE_URL=https://[project].supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=<your-anon-key>
SUPABASE_SERVICE_ROLE_KEY=<your-service-key>

# Your existing auth
JWT_SECRET=<generate with: openssl rand -base64 32>
NEXTAUTH_SECRET=<generate with: openssl rand -base64 32>

# Mapbox
NEXT_PUBLIC_MAPBOX_TOKEN=<your-mapbox-token>
```

## Quick Test

```bash
# Test connection from command line
psql "your-supabase-connection-string" -c "SELECT version();"

# Should return: PostgreSQL 15.x ...
```

## Migration Command for Supabase

```bash
# One command to rule them all
DATABASE_URL="your-supabase-url" npm run db:push
```

This will create all tables in Supabase with proper types!