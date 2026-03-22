# 🚀 Terra Atlas Quick Launch Guide

## Option 1: Instant Demo (5 minutes) ⭐ RECOMMENDED

```bash
# Start the platform with sample data
cd /srv/luminous-dynamics/terra-atlas-mvp
npm run dev
```

Visit http://localhost:3002 to see:
- ✅ 3D Globe with 101 demo projects
- ✅ Authentication system
- ✅ Investment calculator
- ✅ Beautiful UI ready for investors

### Deploy to Production (10 minutes)
```bash
# Deploy to Vercel
vercel --prod

# Your site will be live at:
# https://terra-atlas.vercel.app
```

## Option 2: Full Data Integration (2-3 hours)

### Step 1: Load Environment Variables
```bash
source scripts/setup-env-from-bws.sh
```

### Step 2: Run Complete Data Migration
```bash
# This imports:
# - 79,193 energy projects from old demos
# - 87,000 USACE dams
# - 11,547 FERC queue projects
# - SMR pipeline projects
npm run migrate:data
```

### Step 3: Start All Services
```bash
./start-complete-platform.sh
```

## Option 3: Selective Feature Testing

### Test Python ML Backend Only
```bash
cd backend
python site_analysis_engine.py
# Visit http://localhost:8000/docs for API
```

### Test Real-time WebSocket
```bash
npm run dev
# Open multiple browser tabs
# Changes sync in real-time
```

### Test Payment Integration
```bash
# Set Stripe keys from BWS
export STRIPE_PUBLISHABLE_KEY=$(bws get stripe-publishable-key)
export STRIPE_SECRET_KEY=$(bws get stripe-secret-key)
npm run dev
# Test investment flow
```

## 🎯 Recommended Sequence

### Today (Day 1) - Demo & Validate
1. **Launch basic demo** (5 min)
2. **Deploy to Vercel** (10 min)  
3. **Share link with stakeholders**
4. **Gather feedback**

### Tomorrow (Day 2) - Data Integration
1. **Import 79,193 projects** (1 hour)
2. **Add USACE dam data** (30 min)
3. **Test site analysis** (30 min)
4. **Verify all integrations**

### Day 3 - Polish & Launch
1. **Customize branding**
2. **Add custom domain**
3. **Enable payments**
4. **Go live!**

## 🔧 Troubleshooting

### Database Connection Issues
```bash
# Test Supabase connection
curl https://fyyszjyixenujgbjaqkd.supabase.co/rest/v1/projects?select=count \
  -H "apikey: $(bws get supabase-prod-anon-key)" \
  -H "Authorization: Bearer $(bws get supabase-prod-anon-key)"
```

### Missing API Keys
```bash
# Check what's in BWS
bws list | grep -E "api|key|token"

# Load all at once
source scripts/setup-env-from-bws.sh
```

### Port Conflicts
```bash
# Check if ports are in use
lsof -i :3002  # Frontend
lsof -i :8000  # Python backend
lsof -i :6379  # Redis

# Kill if needed
kill $(lsof -t -i:3002)
```

## 📊 What You'll See

### With Basic Demo (Immediate)
- 101 energy projects on globe
- Investment calculator working
- User authentication ready
- Beautiful glass UI

### With Full Data (After Migration)
- 79,193 real energy projects
- 87,000 US dams for retrofit
- 11,547 FERC queue projects
- ML-powered site analysis
- Real-time updates
- Complete payment flow

## 🎯 Success Metrics

✅ **MVP Complete When:**
1. Globe shows real projects
2. Users can create accounts
3. Investment calculator works
4. Site analysis returns data
5. Deployed to public URL

✅ **Production Ready When:**
1. All 79,193 projects imported
2. Payment processing tested
3. Real-time sync working
4. Custom domain configured
5. SSL certificates active

## 💡 Pro Tips

1. **Start Simple** - Get demo running first
2. **Deploy Early** - Vercel deployment takes 5 min
3. **Import Data Later** - Can be done anytime
4. **Test Features Incrementally** - Don't try everything at once
5. **Use BWS** - All credentials are there

## 🚀 One-Line Quick Start

```bash
cd /srv/luminous-dynamics/terra-atlas-mvp && npm run dev
```

That's it! Platform running at http://localhost:3002 🎉