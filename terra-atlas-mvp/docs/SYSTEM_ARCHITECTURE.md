# Terra Atlas System Architecture & Feature Registry
*Master document to prevent rebuilding and maintain consistency*

## 🎯 Project Overview

**Terra Atlas** is a global energy investment platform democratizing access to renewable energy and SMR (Small Modular Reactor) projects. Users can invest as little as $10 in solar/wind/hydro projects or $100,000+ in SMR projects.

**Tech Stack:**
- Frontend: Next.js 15.5.3 with TypeScript, Tailwind CSS
- Backend: Next.js API Routes
- Database: Supabase (PostgreSQL)
- Payments: Stripe
- Maps: Mapbox/Cesium (future)
- Hosting: Vercel
- Authentication: NextAuth (planned) / Header-based (current)

## 📊 Database Schema

### Core Tables

#### 1. `projects` Table
```sql
CREATE TABLE projects (
  id SERIAL PRIMARY KEY,
  name VARCHAR(255) NOT NULL,
  type VARCHAR(50),              -- Solar, Wind, Hydro, Nuclear, SMR, Battery
  capacity_mw DECIMAL(10,2),
  location VARCHAR(255),
  latitude DECIMAL(10,8),
  longitude DECIMAL(11,8),
  state VARCHAR(2),
  country VARCHAR(3),
  status VARCHAR(50),             -- Planning, Development, Construction, Operational
  developer VARCHAR(255),
  total_raised DECIMAL(12,2) DEFAULT 0,
  target_amount DECIMAL(12,2),
  min_investment DECIMAL(10,2) DEFAULT 10,
  max_investment DECIMAL(10,2) DEFAULT 1000000,
  investors_count INTEGER DEFAULT 0,
  expected_roi DECIMAL(5,2),
  risk_level VARCHAR(20),         -- Low, Medium, High
  commissioning_date DATE,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### 2. `investments` Table
```sql
CREATE TABLE investments (
  id SERIAL PRIMARY KEY,
  user_id VARCHAR NOT NULL,
  user_email VARCHAR NOT NULL,
  project_id INTEGER REFERENCES projects(id),
  project_name VARCHAR,
  project_type VARCHAR,
  amount DECIMAL(10,2) NOT NULL,
  investment_term_months INTEGER DEFAULT 12,
  payment_method VARCHAR DEFAULT 'stripe',
  payment_status VARCHAR DEFAULT 'pending',   -- pending, completed, failed, refunded
  stripe_payment_intent_id VARCHAR UNIQUE,
  stripe_customer_id VARCHAR,
  stripe_status VARCHAR,
  failure_reason TEXT,
  refund_amount DECIMAL(10,2),
  refunded_at TIMESTAMPTZ,
  metadata JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW(),
  updated_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### 3. `users` Table (Future)
```sql
CREATE TABLE users (
  id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  email VARCHAR(255) UNIQUE NOT NULL,
  name VARCHAR(255),
  role VARCHAR(50) DEFAULT 'investor',  -- investor, accredited_investor, admin
  kyc_status VARCHAR(50),               -- pending, verified, rejected
  total_invested DECIMAL(12,2) DEFAULT 0,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

#### 4. `smr_projects` Table
```sql
CREATE TABLE smr_projects (
  id SERIAL PRIMARY KEY,
  project_id INTEGER REFERENCES projects(id),
  reactor_type VARCHAR(100),            -- NuScale, Rolls-Royce, TerraPower
  capacity_mwe INTEGER,
  modules_count INTEGER,
  nrc_status VARCHAR(100),              -- Pre-application, Under Review, Approved
  estimated_completion DATE,
  epc_contractor VARCHAR(255),
  offtake_agreements JSONB,
  detailed_specs JSONB
);
```

#### 5. `failed_investment_records` Table
```sql
CREATE TABLE failed_investment_records (
  id SERIAL PRIMARY KEY,
  payment_intent_id VARCHAR,
  investment_data JSONB,
  error_details JSONB,
  created_at TIMESTAMPTZ DEFAULT NOW()
);
```

## 🎨 Frontend Architecture

### Page Structure

```
app/
├── page.tsx                      # Landing page
├── projects/
│   └── page.tsx                  # Browse all projects
├── invest/
│   ├── [id]/
│   │   └── page.tsx             # Standard investment flow
│   └── smr/
│       └── [id]/
│           └── page.tsx         # SMR investment flow ($100K min)
├── smr/
│   └── page.tsx                  # SMR projects listing
├── dashboard/                    # User dashboard (TODO)
│   ├── page.tsx                 # Overview
│   └── portfolio/               # Investment portfolio
│       └── page.tsx
├── api/
│   ├── projects/
│   │   ├── route.ts            # GET all projects
│   │   └── [id]/
│   │       └── route.ts        # GET single project
│   ├── smr/
│   │   └── route.ts            # GET SMR projects
│   ├── stripe/
│   │   ├── create-payment-intent/
│   │   │   └── route.ts        # POST create payment
│   │   ├── confirm-payment/
│   │   │   └── route.ts        # POST confirm payment
│   │   └── webhook/
│   │       └── route.ts        # POST Stripe webhooks
│   └── health/
│       └── route.ts            # GET health check
```

### Key Components

#### 1. `ProjectCard` Component
- Location: `components/ProjectCard.tsx`
- Props: `project`, `onInvest`
- Features: Displays project info, progress bar, invest button

#### 2. `AdvancedFilterPanel` Component
- Location: `components/AdvancedFilterPanel.tsx`
- Features: Multi-select filters, range sliders, search
- Used in: Projects page, SMR page

#### 3. `PaymentForm` Component
- Location: `components/PaymentForm.tsx`
- Features: Stripe Elements integration, PCI compliance
- Props: `amount`, `projectId`, `projectName`, `projectType`, `onSuccess`, `onCancel`

#### 4. `InvestmentModal` Component
- Location: Inline in invest pages
- Features: Investment amount input, terms selection, payment flow

#### 5. `SMRProjectCard` Component
- Location: `components/SMRProjectCard.tsx`
- Features: Extended info for nuclear projects, $100K minimum

## 🔌 API Endpoints

### Public Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/api/health` | GET | Health check | No |
| `/api/projects` | GET | List all projects with filters | No |
| `/api/projects/[id]` | GET | Get single project | No |
| `/api/smr` | GET | List SMR projects | No |

### Protected Endpoints

| Endpoint | Method | Description | Auth Required |
|----------|--------|-------------|---------------|
| `/api/stripe/create-payment-intent` | POST | Create Stripe payment | Yes (headers) |
| `/api/stripe/confirm-payment` | POST | Confirm payment & record | Yes (headers) |
| `/api/stripe/webhook` | POST | Handle Stripe events | Webhook signature |

### Request/Response Examples

#### Create Payment Intent
```typescript
// Request
POST /api/stripe/create-payment-intent
Headers: {
  "x-user-id": "user-123",
  "x-user-email": "user@example.com"
}
Body: {
  "amount": 1000,
  "projectId": 1,
  "projectName": "Desert Solar Farm",
  "projectType": "Solar"
}

// Response
{
  "clientSecret": "pi_xxx_secret_xxx",
  "paymentIntentId": "pi_xxx",
  "amount": 1000,
  "customerId": "cus_xxx"
}
```

## 🎯 Feature Status

### ✅ Completed Features

1. **Project Browsing**
   - Grid/list view of all projects
   - Basic filtering by type
   - Individual project pages

2. **SMR Section**
   - Dedicated SMR listing page
   - $10 minimum - EQUAL ACCESS FOR ALL
   - Advanced filtering
   - Specialized project cards

3. **Payment Integration**
   - Stripe payment intents
   - Payment confirmation flow
   - Webhook handling
   - Failed payment recovery

4. **Search & Filter**
   - Advanced filter panel
   - Multi-select dropdowns
   - Range sliders for capacity/investment
   - Real-time filtering

### 🚧 In Progress

1. **Authentication**
   - Currently using header-based auth
   - Need NextAuth integration

2. **Database**
   - Schema created
   - Need to execute in Supabase

### 📋 TODO Features

1. **User Dashboard**
   - Portfolio overview
   - Investment history
   - Performance tracking
   - Documents/statements

2. **Project Details**
   - Rich project pages
   - Documents/reports
   - Updates/news
   - Q&A section

3. **Data Import**
   - USACE dam data (87,000 sites)
   - Solar/wind farm data
   - Real-time energy prices

4. **Maps Integration**
   - Interactive project map
   - Geographic filtering
   - Cluster visualization

## 🔐 Security Considerations

### Current Implementation
- Stripe webhook signature verification
- SQL injection prevention via parameterized queries
- CORS configuration
- Rate limiting (TODO)

### Needed Improvements
- Proper authentication (NextAuth)
- Row Level Security (RLS) in Supabase
- API rate limiting
- Input validation middleware
- CSRF protection

## 🚀 Deployment Configuration

### Environment Variables

```env
# Database
NEXT_PUBLIC_SUPABASE_URL=https://xxx.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=xxx
SUPABASE_SERVICE_ROLE_KEY=xxx
DATABASE_URL=postgresql://xxx

# Stripe
NEXT_PUBLIC_STRIPE_PUBLISHABLE_KEY=YOUR_STRIPE_PUBLISHABLE_KEY
STRIPE_SECRET_KEY=YOUR_STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET=YOUR_STRIPE_WEBHOOK_SECRET

# Maps (Future)
NEXT_PUBLIC_MAPBOX_TOKEN=xxx
NEXT_PUBLIC_CESIUM_TOKEN=xxx

# Energy APIs (Future)
EIA_API_KEY=xxx
NREL_API_KEY=xxx
```

### Vercel Configuration
- Framework: Next.js
- Build Command: `npm run build`
- Output Directory: `.next`
- Environment Variables: Set in Vercel dashboard

## 📈 Performance Optimizations

### Current
- Turbopack for faster dev builds
- API route caching
- Optimized images with Next.js Image

### Planned
- Database query optimization
- Redis caching layer
- CDN for static assets
- Lazy loading components

## 🧪 Testing

### Test Coverage
- Unit tests: TODO
- Integration tests: Basic Stripe tests created
- E2E tests: TODO

### Test Commands
```bash
# Run integration tests
node test-stripe-integration.js

# Test API endpoints
curl http://localhost:3002/api/health
```

## 📝 Development Workflow

### Branch Strategy
- `main`: Production
- `develop`: Staging
- Feature branches: `feature/dashboard`, `feature/auth`

### Commit Convention
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation
- `style:` Formatting
- `refactor:` Code restructuring
- `test:` Tests
- `chore:` Maintenance

## 🗺️ Roadmap

### Phase 1 (Current)
- ✅ Basic project browsing
- ✅ SMR section
- ✅ Payment integration
- 🚧 Authentication
- 📋 User dashboard

### Phase 2 (Next Month)
- 📋 Portfolio management
- 📋 Real project data import
- 📋 Map visualization
- 📋 Email notifications

### Phase 3 (Q2 2025)
- 📋 Mobile app
- 📋 Recurring investments
- 📋 Social features
- 📋 Advanced analytics

## 🔄 Version History

| Version | Date | Changes |
|---------|------|---------|
| 0.1.0 | Jan 2025 | Initial MVP with projects listing |
| 0.2.0 | Jan 2025 | Added SMR section |
| 0.3.0 | Jan 2025 | Stripe payment integration |
| 0.4.0 | TBD | Authentication system |
| 0.5.0 | TBD | User dashboard |

## 📚 Related Documentation

- [Payment Integration Status](./PAYMENT_INTEGRATION_STATUS.md)
- [Stripe Integration Guide](./docs/STRIPE_INTEGRATION.md)
- [Database Setup](./supabase/setup_investments.sql)
- [API Documentation](./docs/API.md) (TODO)

---

**Last Updated:** January 29, 2025
**Maintained By:** Development Team
**Review Frequency:** After each major feature