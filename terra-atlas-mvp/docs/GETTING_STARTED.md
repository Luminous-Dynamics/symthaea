# 🚀 Getting Started with Terra Atlas Development

Welcome to Terra Atlas! This guide will get you up and running in **15 minutes**.

---

## Prerequisites

Before you begin, make sure you have:
- **Node.js 20+** (`node --version`)
- **npm** or **yarn** (`npm --version`)
- **Git** (`git --version`)
- **Supabase account** (free tier works!) - [Sign up here](https://supabase.com)

---

## Quick Start (5 minutes)

### 1. Clone the Repository
```bash
git clone https://github.com/Luminous-Dynamics/terra-atlas.git
cd terra-atlas-mvp
```

### 2. Install Dependencies
```bash
npm install
```

This installs everything needed: Next.js, React, TypeScript, Supabase client, etc.

### 3. Set Up Environment Variables
```bash
cp .env.example .env.local
```

Now edit `.env.local` with your Supabase credentials:
```env
# Required for basic functionality
NEXT_PUBLIC_SUPABASE_URL=https://your-project.supabase.co
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key_here

# Required for database operations
SUPABASE_SERVICE_ROLE_KEY=your_service_role_key
DATABASE_URL=postgresql://postgres:[password]@db.your-project.supabase.co:5432/postgres

# Optional (for payments - use test keys)
STRIPE_SECRET_KEY=YOUR_STRIPE_SECRET_KEY
STRIPE_WEBHOOK_SECRET=YOUR_STRIPE_WEBHOOK_SECRET

# Optional (for maps)
NEXT_PUBLIC_MAPBOX_TOKEN=pk.your_mapbox_token
```

**Where to find Supabase credentials:**
1. Go to [app.supabase.com](https://app.supabase.com)
2. Select your project (or create a new one)
3. Go to Settings → API
4. Copy `URL` and `anon` key to `.env.local`

### 4. Run the Development Server
```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) in your browser. You should see the Terra Atlas homepage with the 3D globe!

---

## Database Setup (10 minutes)

### Option A: Use Existing Demo Data
The project includes a SQLite database with demo data at `data/terra-atlas-local.db`. This works out of the box for browsing.

### Option B: Set Up Supabase (Recommended for Development)

#### 1. Create Tables
```bash
# Run migrations in your Supabase project
# Go to: Supabase Dashboard → SQL Editor
# Copy and run the SQL from: supabase/migrations/*.sql
```

#### 2. Seed with Sample Data
```bash
# Option 1: Use the import scripts
npm run db:migrate

# Option 2: Manual SQL
# Run the MASTER_IMPORT_ALL_DATA.sql script in Supabase SQL Editor
```

#### 3. Verify Database Connection
```bash
# Test the connection
node test-supabase.js
```

If you see "✅ Supabase connection successful", you're all set!

---

## Development Workflow

### Daily Development Commands
```bash
# Start dev server (with hot reload)
npm run dev

# Clean build cache and restart
npm run dev:clean

# Nuclear option - clear everything
npm run dev:fresh
```

### Code Quality Checks
```bash
# Run all checks (lint + typecheck + test)
npm run check

# Individual checks
npm run lint       # ESLint
npm run typecheck  # TypeScript
npm run test       # Vitest
npm run test:watch # Vitest in watch mode
```

### Database Operations
```bash
# Generate Drizzle migrations
npm run db:generate

# Push schema changes
npm run db:push

# Open Drizzle Studio (DB GUI)
npm run db:studio

# Run migrations
npm run db:migrate
```

### Telemetry & Analytics
```bash
# View telemetry report
npm run telemetry:report

# Watch telemetry in real-time
tail -f data/telemetry-events.log
```

---

## Project Structure

```
terra-atlas-mvp/
├── app/                        # Next.js 15 App Router
│   ├── api/                   # API routes
│   │   ├── projects/          # Project data endpoints
│   │   ├── stats/             # Platform statistics
│   │   ├── stripe/            # Payment processing
│   │   └── telemetry/         # Performance tracking
│   ├── dashboard/             # User dashboard pages
│   ├── explore/               # Project exploration
│   └── page.tsx               # Homepage
│
├── components/                # React components
│   ├── TerraGlobeWithSites.tsx  # 3D globe visualization
│   ├── HeroSection.tsx          # Homepage hero
│   └── auth/                    # Authentication components
│
├── lib/                       # Utilities and helpers
│   ├── supabase.ts           # Supabase client
│   ├── drizzle/              # Database ORM
│   ├── utils.ts              # Shared utilities
│   └── schemas/              # Validation schemas
│
├── data/                      # Static data files
│   ├── terra-atlas-local.db  # SQLite demo database
│   └── demo-sites.json       # Sample project data
│
├── public/                    # Static assets
│   ├── data/                 # Public JSON data
│   └── textures/             # Globe textures
│
├── scripts/                   # Development scripts
│   ├── import-usace-dams.ts  # Import USACE dam data
│   └── analyze-telemetry.mjs # Telemetry analysis
│
└── docs/                      # Documentation
    ├── STATUS.md             # Current project status
    └── GETTING_STARTED.md    # This file!
```

---

## Common Tasks

### Adding a New Feature

1. **Create a branch**
```bash
git checkout -b feature/your-feature-name
```

2. **Make your changes**
```bash
# Edit files...
npm run dev  # Test locally
```

3. **Run quality checks**
```bash
npm run check  # Must pass!
```

4. **Commit and push**
```bash
git add .
git commit -m "feat: your feature description"
git push origin feature/your-feature-name
```

5. **Create a Pull Request** on GitHub

### Running Tests (When We Have Them 😅)
```bash
# Run all tests
npm test

# Run in watch mode
npm run test:watch

# Run with coverage
npm run test -- --coverage
```

**Note**: We're currently setting up the test infrastructure. See [STATUS.md](./STATUS.md) for progress.

### Fixing a Bug

1. Check if there's an issue on GitHub
2. Create a branch: `git checkout -b fix/issue-number-description`
3. Fix the bug
4. Add a test to prevent regression (once we have tests)
5. Run `npm run check`
6. Submit a PR

### Adding Data Sources

We track energy projects from multiple sources:

- **FERC Queue**: Renewable projects in interconnection queues
- **USACE Dams**: US Army Corps dam locations for retrofits
- **SMR Pipeline**: Small Modular Reactor projects

To add a new data source:

1. Create a script in `scripts/import-your-data-source.ts`
2. Define the schema in `lib/drizzle/schema.ts`
3. Run the import: `npx tsx scripts/import-your-data-source.ts`
4. Verify in Drizzle Studio: `npm run db:studio`

---

## Using Nix (Optional)

If you're using NixOS or have Nix installed:

```bash
# Enter development shell
nix develop

# Playwright environment (for screenshots/tests)
nix develop .#playwright

# Python environment (for data scripts)
nix develop .#python
```

Inside the Nix shell, use regular `npm` commands.

---

## Troubleshooting

### Port Already in Use
```bash
# Kill process on port 3000
lsof -ti:3000 | xargs kill -9

# Or use a different port
PORT=3001 npm run dev
```

### Database Connection Errors
```bash
# Check your .env.local has correct Supabase credentials
# Verify at: https://app.supabase.com → Project → Settings → API

# Test connection
node test-supabase.js
```

### Build Errors
```bash
# Clear Next.js cache
rm -rf .next

# Clear node_modules and reinstall
rm -rf node_modules
npm install

# Nuclear option
npm run dev:fresh
```

### Type Errors
```bash
# Run type check
npm run typecheck

# Common fix: restart TypeScript server in your editor
# VSCode: Cmd+Shift+P → "TypeScript: Restart TS Server"
```

### ESLint Errors
```bash
# Run linter
npm run lint

# Auto-fix many issues
npm run lint -- --fix
```

---

## Development Environment Setup

### Recommended VSCode Extensions
- **ESLint** - Linting
- **Prettier** - Code formatting
- **TypeScript** - Type checking
- **Tailwind CSS IntelliSense** - CSS utilities
- **Prisma** - Database schema editing
- **GitLens** - Git integration

### Editor Settings
Create `.vscode/settings.json`:
```json
{
  "editor.formatOnSave": true,
  "editor.defaultFormatter": "esbenp.prettier-vscode",
  "editor.codeActionsOnSave": {
    "source.fixAll.eslint": true
  },
  "typescript.tsdk": "node_modules/typescript/lib"
}
```

---

## Next Steps

1. **Read [STATUS.md](./STATUS.md)** - Understand current state and priorities
2. **Check [GitHub Issues](https://github.com/Luminous-Dynamics/terra-atlas/issues)** - Find good first issues
3. **Join Development** - Pick a task from the roadmap
4. **Run Tests** - Once we have them 😊
5. **Submit PRs** - Help us build the future of energy investment!

### Priority Areas for Contributors

🔴 **Critical** (Need help immediately):
- Writing tests (Vitest setup complete, need test files!)
- Fixing security issues (hard-coded secrets, etc.)
- Documentation improvements

🟡 **High Priority**:
- API improvements (pagination, rate limiting)
- Performance optimization (globe loading, caching)
- Data source expansion (more projects!)

🟢 **Good First Issues**:
- UI/UX improvements
- Documentation fixes
- Adding tooltips and help text
- Accessibility enhancements

---

## Getting Help

- **Documentation**: Check `docs/` folder first
- **GitHub Issues**: [github.com/Luminous-Dynamics/terra-atlas/issues](https://github.com/Luminous-Dynamics/terra-atlas/issues)
- **Email**: invest@luminousdynamics.org

---

## Code of Conduct

We're building public good infrastructure for energy democracy. Be:
- **Respectful** - Treat everyone with kindness
- **Collaborative** - Help each other learn
- **Honest** - About capabilities and limitations
- **Transparent** - Share knowledge and learnings

---

**Welcome to the team! Let's democratize energy investment together. 🌍💚**

*For the complete project vision, see [TERRA_ATLAS_UNIFIED_VISION.md](./TERRA_ATLAS_UNIFIED_VISION.md)*
