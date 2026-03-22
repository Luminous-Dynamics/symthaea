# 🌍 Terra Atlas MVP - The ONE Unified Version

## ⚠️ IMPORTANT: This is THE version we work on!

**Stop creating new demos!** This is the main Terra Atlas with the 3D globe that you liked. All improvements should be made HERE, not in new files.

> Democratizing Global Energy Investment - 4M+ Renewable Opportunities Worldwide

[![Live Platform](https://img.shields.io/badge/Live-atlas.luminousdynamics.io-green)](https://atlas.luminousdynamics.io)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Next.js](https://img.shields.io/badge/Next.js-15.5-black)](https://nextjs.org/)
[![Supabase](https://img.shields.io/badge/Supabase-Database-green)](https://supabase.com/)

## 🚀 The Platform

Terra Atlas is revolutionizing energy investment by connecting everyday investors with renewable energy opportunities globally. Starting with 87,000 untapped dams in the US ready for hydroelectric retrofitting, we're building the infrastructure for community-owned energy.

### Live at [atlas.luminousdynamics.io](https://atlas.luminousdynamics.io)

## 💡 Key Features

- **🌐 Interactive 3D Globe** - Visualize energy projects worldwide
- **📊 Real-Time Data** - FERC queue tracking (11,547 active projects)
- **💰 Investment Calculator** - Project ROI with transparency
- **🏛️ Regenerative Exit Model** - 7-year path to community ownership
- **🇨🇭 Swiss Foundation** - Immutable mission via Luminous Chimera structure
- **📈 Portfolio Management** - Track and manage your energy investments

## 🎯 The Opportunity

### Immediate Impact
- **$60B** opportunity in US dam retrofitting alone
- **87,000** existing dams ready for hydroelectric conversion
- **4M+** global renewable energy sites identified
- **11-14%** IRR for investors with tax optimization

### Revolutionary Model
- **Luminous Chimera**: Swiss foundation ensures mission permanence
- **Community Ownership**: Platform transitions to users over 7 years
- **Tax Efficiency**: 30-40% more capital to communities
- **Democratic Access**: Invest from $10 to $10M

## 🛠️ Tech Stack

- **Frontend**: Next.js 15.5, React 19, TypeScript, Tailwind CSS
- **Backend**: Supabase (PostgreSQL), Prisma ORM, Drizzle
- **Maps**: Mapbox GL for interactive visualizations
- **Deployment**: Vercel with edge functions
- **Data Sources**: FERC, USACE, SMR pipelines, GIS databases

## 🏗️ Development Setup

### Prerequisites
- Node.js 20+
- npm or yarn
- Supabase account (for database)

### Quick Start

```bash
# Clone the repository
git clone https://github.com/Luminous-Dynamics/terra-atlas.git
cd terra-atlas

# Install dependencies
npm install

# Set up environment variables
cp .env.example .env.local
# Edit .env.local with your Supabase credentials

# Run development server
npm run dev

# Open http://localhost:3000
```

### Environment Variables

```env
NEXT_PUBLIC_SUPABASE_URL=your_supabase_url
NEXT_PUBLIC_SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_service_key
DATABASE_URL=your_postgres_connection_string
```

## 📁 Project Structure

```
terra-atlas/
├── app/                    # Next.js app directory
│   ├── api/               # API routes
│   ├── dashboard/         # User dashboard
│   └── projects/          # Project pages
├── components/            # React components
│   ├── Globe3D.tsx       # Interactive 3D visualization
│   ├── ProjectCard.tsx   # Project display cards
│   └── ROICalculator.tsx # Investment calculator
├── lib/                   # Utilities and helpers
│   ├── supabase/         # Database client
│   └── utils/            # Helper functions
├── data/                  # Static data files
│   ├── ferc-queue.json  # FERC project data
│   └── dam-sites.json   # USACE dam database
└── public/               # Static assets
```

## 🗺️ Roadmap

### Phase 1: MVP (Complete ✅)
- Interactive globe with 250+ demo projects
- Basic investment calculator
- User authentication
- Project search and filtering

### Phase 2: Data Integration (In Progress 🚧)
- FERC queue real-time updates
- USACE dam database integration
- SMR project pipeline tracking
- Weather and grid data layers

### Phase 3: Investment Features (Q1 2025)
- Pledge system for projects
- Portfolio management
- Community forums
- Smart contract integration

### Phase 4: Community Ownership (Q2 2025)
- DAO governance structure
- Token economics design
- Voting mechanisms
- Revenue sharing system

## 🤝 Contributing

We welcome contributions! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Areas We Need Help
- Data integration from energy databases
- UI/UX improvements
- Smart contract development
- Community features
- Documentation
- Testing

## 📊 Current Stats

- **250+** demonstration projects live
- **11,547** FERC queue projects tracked
- **87,000** potential dam sites mapped
- **$60B** investment opportunity identified

## 🏛️ Governance

Terra Atlas operates under the Luminous Chimera model:
- Swiss Foundation holds golden share
- Mission legally immutable
- 7-year transition to community ownership
- Transparent governance via DAO

## 📄 License

MIT License - We're building public good infrastructure for energy democracy.

## 🔗 Links

- **Live Platform**: [atlas.luminousdynamics.io](https://atlas.luminousdynamics.io)
- **Investment Deck**: [terra.luminousdynamics.org](https://terra.luminousdynamics.org)
- **Documentation**: [docs.luminousdynamics.org](https://docs.luminousdynamics.org)
- **Organization**: [github.com/Luminous-Dynamics](https://github.com/Luminous-Dynamics)

## 💬 Contact

- **Email**: invest@luminousdynamics.org
- **Discord**: [Join our community](https://discord.gg/luminous)
- **Twitter**: [@LuminousDynamics](https://twitter.com/LuminousDynamics)

---

*Built with consciousness-first principles and love for our planet 🌍💚*

**Terra Atlas** - Where energy meets democracy, and investment serves community.