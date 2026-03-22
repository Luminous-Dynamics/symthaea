# 🌍 Terra Atlas Unified - Complete Project Structure

## Project Overview
Terra Atlas is an energy investment democratization platform that allows anyone to invest as little as $10 in energy infrastructure projects, including Small Modular Reactors (SMRs).

## Data Assets
- **10,549 Real Energy Projects** in SQLite database
- **87,000 USACE Dam Sites** (80MB JSON) - Ready for hydro upgrades
- **5,290 FERC Queue Projects** (9.5MB JSON) - Active development pipeline
- **26 SMR Projects** - Next-gen nuclear opportunities
- **Equal Opportunity** - $10 minimum for ALL projects (no discrimination)

## Directory Structure

```
terra-atlas-unified/
│
├── app/                        # Next.js 15 App Router
│   ├── api/                    # API routes
│   │   ├── stripe/            # Payment processing (EQUAL OPPORTUNITY)
│   │   ├── projects/          # Project data endpoints
│   │   └── auth/              # Authentication endpoints
│   ├── dashboard/             # User dashboard (TO BUILD)
│   ├── portfolio/             # Investment portfolio (TO BUILD)
│   └── layout.tsx             # Root layout
│
├── components/                 # React components
│   ├── TerraGlobe/           # 3D visualization
│   ├── ProjectCard.tsx       # Project display
│   └── PaymentForm.tsx       # Stripe integration
│
├── data/                      # Data assets (VALUABLE!)
│   ├── terra-atlas-local.db  # 10,549 projects SQLite
│   ├── usace-dams-2024.json # 87,000 dam sites (80MB)
│   ├── ferc-queue-2024.json # 5,290 FERC projects (9.5MB)
│   └── smr-pipeline-projects.json # 26 SMR projects
│
├── landing-pages/             # Professional investment page
│   ├── index.html            # Clean landing page
│   ├── roi-calculator.html  # ROI calculator
│   └── data-room.html       # Investor data room
│
├── lib/                       # Utilities
│   ├── stripe.ts            # Stripe configuration
│   ├── supabase.ts          # Database client
│   └── drizzle/             # ORM configuration
│
├── scripts/                   # Import/setup scripts
│   ├── import-all-renewable-sites.js
│   ├── import-to-supabase.js
│   ├── setup-local-database.js
│   └── test-stripe-integration.js
│
├── public/                    # Static assets
│   └── images/              # Project images
│
└── docs/                      # Documentation
    ├── EQUAL_OPPORTUNITY_UPDATE.md
    ├── SYSTEM_ARCHITECTURE.md
    └── FEATURE_MAP.md
```

## Key Features Implemented
✅ Stripe Payment Integration (Equal opportunity - $10 minimum for all)
✅ 10,549 Real Projects Database
✅ Professional Investment Landing Page
✅ 3D Globe Visualization
✅ Authentication System (Header-based for now)

## Priority Development Tasks
1. 🚧 Import USACE dam data (87,000 sites)
2. 🚧 Add grid/transmission data
3. 🚧 Build Portfolio/Dashboard pages
4. 🚧 Setup Supabase production database
5. 🚧 Configure real Stripe API keys

## Technical Stack
- **Frontend**: Next.js 15.5.3, React 19.1.0, TypeScript
- **Database**: Supabase (PostgreSQL) + SQLite (local data)
- **Payments**: Stripe (Payment Intents API)
- **3D Viz**: Three.js + Three-Globe
- **Styling**: Tailwind CSS v4
- **Auth**: Header-based (temporary) → Supabase Auth (planned)

## Environment Variables Required
```env
# Supabase
NEXT_PUBLIC_SUPABASE_URL=https://fyyszjyixenujgbjaqkd.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=[from BWS: supabase-prod-anon-key]
SUPABASE_SERVICE_ROLE_KEY=[from BWS: supabase-service-role-key]

# Stripe
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=[your-stripe-pk]
STRIPE_SECRET_KEY=[your-stripe-sk]
STRIPE_WEBHOOK_SECRET=[your-webhook-secret]

# App
NEXT_PUBLIC_APP_URL=http://localhost:3000
NODE_ENV=development
```

## Database Schema Summary
- `projects` - Energy projects (solar, wind, hydro, nuclear, storage)
- `investments` - User investment records
- `users` - User accounts
- `portfolios` - User portfolios
- `transactions` - Payment records
- `project_updates` - Project news/updates

## Deployment Status
- **Local Development**: Ready ✅
- **Supabase Database**: Schema ready, needs execution
- **Stripe Integration**: Test mode working, needs production keys
- **Production Deploy**: Pending Vercel/hosting setup

## Value Proposition
- **Democratization**: Anyone can invest $10+ in clean energy
- **Scale**: 10,549 existing projects + 87,000 potential hydro sites
- **Equal Opportunity**: No wealth discrimination for nuclear investment
- **Community Ownership**: Path to local energy sovereignty

## Contact
Part of Luminous Dynamics - Consciousness-first technology serving all beings