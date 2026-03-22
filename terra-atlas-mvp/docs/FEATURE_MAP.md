# Terra Atlas Feature Map 🗺️
*Quick visual reference to prevent duplicate work*

## 🏗️ Current System State

```
Terra Atlas MVP
├── ✅ COMPLETED
│   ├── 🌍 Project Discovery
│   │   ├── Browse all projects (/projects)
│   │   ├── Filter by type (Solar/Wind/Hydro)
│   │   ├── Search functionality
│   │   └── Grid/List view toggle
│   │
│   ├── ⚛️ SMR Section
│   │   ├── Dedicated SMR page (/smr)
│   │   ├── $10 minimum - EQUAL OPPORTUNITY
│   │   ├── Advanced filter panel
│   │   └── Specialized project cards
│   │
│   ├── 💳 Payment Processing
│   │   ├── Stripe integration
│   │   ├── Payment intent creation
│   │   ├── Payment confirmation
│   │   ├── Webhook handling
│   │   └── Failed payment recovery
│   │
│   └── 🔍 Advanced Filtering
│       ├── Multi-select dropdowns
│       ├── Range sliders
│       ├── Location filter
│       └── Status filter
│
├── 🚧 IN PROGRESS
│   ├── 🔐 Authentication
│   │   └── Header-based (temporary)
│   │
│   └── 💾 Database
│       └── Schema created (needs execution)
│
└── 📋 TODO
    ├── 👤 User System
    │   ├── Registration/Login
    │   ├── Profile management
    │   ├── KYC verification
    │   └── Accredited investor status
    │
    ├── 📊 Dashboard & Portfolio
    │   ├── Investment overview
    │   ├── Performance tracking
    │   ├── Transaction history
    │   ├── Documents/Statements
    │   └── Tax reporting
    │
    ├── 🗺️ Maps & Visualization
    │   ├── Interactive project map
    │   ├── Geographic filtering
    │   ├── Cluster markers
    │   └── 3D globe view
    │
    ├── 📈 Data & Analytics
    │   ├── Import USACE dams (87K)
    │   ├── Real-time energy prices
    │   ├── ROI calculations
    │   └── Risk assessments
    │
    ├── 📱 Mobile & Responsive
    │   ├── Mobile app
    │   ├── PWA features
    │   └── Touch optimizations
    │
    └── 🔔 Communication
        ├── Email notifications
        ├── Investment updates
        ├── Newsletter system
        └── In-app messaging
```

## 📁 File Structure Map

```
/srv/luminous-dynamics/terra-atlas-mvp/
│
├── 📂 app/                      ← Next.js App Router
│   ├── page.tsx                 ✅ Landing page
│   ├── projects/page.tsx        ✅ Browse projects
│   ├── smr/page.tsx            ✅ SMR projects
│   ├── invest/
│   │   ├── [id]/page.tsx       ✅ Standard investment
│   │   └── smr/[id]/page.tsx   ✅ SMR investment
│   ├── dashboard/               📋 TODO
│   └── api/                    ← Backend endpoints
│       ├── projects/            ✅ Project data
│       ├── smr/                 ✅ SMR data
│       ├── stripe/              ✅ Payment processing
│       └── health/              ✅ Health check
│
├── 📂 components/               ← Reusable components
│   ├── ProjectCard.tsx         ✅ Project display
│   ├── SMRProjectCard.tsx      ✅ SMR display
│   ├── AdvancedFilterPanel.tsx ✅ Filtering UI
│   ├── PaymentForm.tsx         ✅ Stripe Elements
│   ├── Navigation.tsx          ✅ Site navigation
│   └── DashboardLayout.tsx     📋 TODO
│
├── 📂 lib/                      ← Utilities
│   ├── stripe.ts               ✅ Stripe config
│   ├── stripe-client.ts        ✅ Client Stripe
│   ├── supabase.ts             🚧 Database client
│   └── auth.ts                 📋 TODO
│
├── 📂 data/                     ← Static/mock data
│   ├── projects.db             ✅ SQLite database
│   ├── smr-projects.json       ✅ SMR data
│   └── usace-dams.json         📋 TODO (87K sites)
│
├── 📂 public/                   ← Static assets
│   ├── images/                 ✅ Project images
│   └── icons/                  ✅ UI icons
│
├── 📂 styles/                   ← CSS/Tailwind
│   └── globals.css             ✅ Global styles
│
├── 📂 supabase/                ← Database
│   └── setup_investments.sql   ✅ Schema (needs run)
│
└── 📂 docs/                    ← Documentation
    ├── STRIPE_INTEGRATION.md   ✅ Payment guide
    ├── SYSTEM_ARCHITECTURE.md  ✅ Full architecture
    └── API.md                  📋 TODO
```

## 🔄 Data Flow Diagram

```
User Journey:
    Browse Projects → Select Project → Enter Amount → Process Payment → View Portfolio
         ↓                ↓                ↓               ↓                ↓
    [Frontend]       [Frontend]      [Frontend]      [Stripe API]     [Dashboard]
         ↓                ↓                ↓               ↓                ↓
    [/api/projects] [/api/projects/id] [PaymentForm] [Payment Intent]  [/api/user]
         ↓                ↓                ↓               ↓                ↓
    [SQLite DB]     [SQLite DB]      [Stripe.js]    [Webhook]       [Supabase]
                                                          ↓
                                                   [Investment Record]
```

## 🎨 Component Hierarchy

```
<App>
  <Navigation />                    ✅ Global nav
  <main>
    <ProjectsPage>                  ✅ /projects
      <AdvancedFilterPanel />       ✅ Filters
      <ProjectGrid>                 ✅ Grid layout
        <ProjectCard />             ✅ Each project
      </ProjectGrid>
    </ProjectsPage>
    
    <SMRPage>                       ✅ /smr
      <AdvancedFilterPanel />       ✅ Filters
      <SMRGrid>                     ✅ Grid layout
        <SMRProjectCard />          ✅ Each SMR
      </SMRGrid>
    </SMRPage>
    
    <InvestPage>                    ✅ /invest/[id]
      <ProjectDetails />            ✅ Project info
      <InvestmentModal>             ✅ Amount input
        <PaymentForm />             ✅ Stripe Elements
      </InvestmentModal>
    </InvestPage>
    
    <DashboardPage>                 📋 TODO /dashboard
      <PortfolioOverview />         📋 TODO
      <InvestmentList />            📋 TODO
      <PerformanceChart />          📋 TODO
    </DashboardPage>
  </main>
</App>
```

## 🔑 Key Business Rules

### Investment Minimums - EQUAL OPPORTUNITY FOR ALL
- **ALL Projects (including SMR)**: $10 minimum
- **Nuclear is for everyone**: Same access regardless of wealth
- **Maximum**: $1,000,000 per investment

### Project Types
- Solar
- Wind
- Hydro
- Nuclear
- SMR (Small Modular Reactor)
- Battery (Energy Storage)

### Project Status
- Planning
- Development
- Construction
- Operational
- Decommissioned

### User Roles
- Visitor (no auth)
- Investor (registered)
- Accredited Investor (verified)
- Admin (platform management)

## 🚦 Status Legend

- ✅ **Complete** - Fully implemented and tested
- 🚧 **In Progress** - Partially implemented
- 📋 **TODO** - Not started
- ⚠️ **Needs Fix** - Implemented but has issues
- 🔄 **Refactor** - Works but needs improvement

## 📊 Database Tables Status

| Table | Created | Populated | In Use |
|-------|---------|-----------|---------|
| projects | ✅ | ✅ | ✅ |
| investments | ✅ | ❌ | 🚧 |
| users | ✅ | ❌ | ❌ |
| smr_projects | ✅ | ❌ | ❌ |
| failed_investment_records | ✅ | ❌ | 🚧 |

## 🎯 Quick Wins (Can Do Now)

1. **Execute database setup** - Run SQL in Supabase
2. **Add test Stripe keys** - Get from Stripe Dashboard
3. **Build Dashboard skeleton** - Basic layout/routing
4. **Import USACE data** - 87K projects ready
5. **Add authentication** - NextAuth or Supabase Auth

## ⚠️ Known Issues

1. **Authentication** - Using headers instead of proper auth
2. **Database** - Using SQLite for projects, need Supabase migration
3. **Stripe Keys** - Using placeholder keys
4. **Mobile View** - Not fully responsive yet
5. **TypeScript** - Some type errors to fix

## 📝 Notes for Future Development

### When Adding New Features:
1. Check this map first - don't rebuild existing features
2. Update the map after implementation
3. Follow existing patterns (see components/)
4. Add to SYSTEM_ARCHITECTURE.md for details

### Common Patterns:
- API routes return JSON with error handling
- Components use TypeScript interfaces
- Tailwind for styling (no separate CSS files)
- Async/await for all API calls
- Error boundaries for component failures

---

**Quick Reference Commands:**
```bash
npm run dev          # Start dev server (port 3002)
npm run build        # Build for production
npm run test         # Run tests (when added)

# Test endpoints
curl http://localhost:3002/api/health
curl http://localhost:3002/api/projects
curl http://localhost:3002/api/smr
```

**Last Updated:** January 29, 2025
**Next Review:** After next major feature