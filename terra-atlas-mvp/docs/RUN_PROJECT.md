# 🚀 Quick Start Guide - Terra Atlas Unified

## Prerequisites
- Node.js 20+
- npm or yarn
- PostgreSQL or Supabase account
- Stripe account (for payments)

## Step 1: Install Dependencies
```bash
npm install
```

## Step 2: Setup Environment Variables
Create `.env.local` file:
```bash
# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://fyyszjyixenujgbjaqkd.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=[get from BWS: bws get supabase-prod-anon-key]
SUPABASE_SERVICE_ROLE_KEY=[get from BWS: bws get supabase-service-role-key]

# Stripe (test keys for now)
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=YOUR_STRIPE_PUBLISHABLE_KEY...
STRIPE_SECRET_KEY=YOUR_STRIPE_SECRET_KEY...

# App
NEXT_PUBLIC_APP_URL=http://localhost:3000
```

## Step 3: Run Development Server
```bash
npm run dev
```
Open http://localhost:3000

## Key Pages
- `/` - Main platform
- `/landing-pages/index.html` - Investment landing page
- `/api/stripe/create-payment-intent` - Payment API (Equal opportunity!)
- `/dashboard` - User dashboard (TO BUILD)
- `/portfolio` - Investment portfolio (TO BUILD)

## Import Real Data (Optional)
```bash
# Import 10,549 projects from SQLite to Supabase
node scripts/import-to-supabase.js

# Import USACE dam data
node scripts/import-usace-dams.js

# Import FERC queue data  
node scripts/import-ferc-queue.js
```

## Test Payment Integration
```bash
# Run Stripe test
node scripts/test-stripe-integration.js
```

## Production Deployment
1. Set production environment variables
2. Configure real Stripe keys
3. Deploy to Vercel: `vercel --prod`
4. Update DNS to point to Vercel

## Support
Part of Luminous Dynamics - Equal opportunity energy investment for all!
