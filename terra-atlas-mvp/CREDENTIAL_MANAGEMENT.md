# 🔐 Terra Atlas Credential Management Guide

**Last Updated**: January 29, 2025
**Problem Solved**: "Why do we keep losing keys?"

---

## 🎯 TL;DR - Quick Fix

```bash
# When you feel like credentials are "lost":
./scripts/refresh-credentials.sh    # Refresh from BWS
npm run dev                          # Restart dev server
```

**That's it!** Your keys were never lost - Next.js just needed to see them.

---

## 📚 Understanding the Problem

### Why It Feels Like Keys Are Lost

**The Illusion**: Environment variables seem to disappear or stop working
**The Reality**: Keys are safe in BWS and `.env.local` - but Next.js caches them

### The Root Causes

1. **`.env.local` is gitignored** (correct security practice)
   - When you clone/pull the repo → `.env.local` doesn't exist
   - Solution: Copy from `.env.example` or run refresh script

2. **Next.js caches environment variables**
   - Changes to `.env.local` require dev server restart
   - Running `npm run dev` won't pick up new vars from running process
   - Solution: Stop server (Ctrl+C) and restart

3. **`NEXT_PUBLIC_*` vars are build-time**
   - These get baked into the JavaScript bundle
   - Changing them requires rebuild or dev server restart
   - Solution: Always restart after changing public env vars

4. **Component hot-reload doesn't reload env**
   - When you edit a component, it hot-reloads
   - But environment variables don't hot-reload with it
   - Solution: Full dev server restart

---

## ✅ The Correct Workflow

### Initial Setup (First Time)

```bash
cd /srv/luminous-dynamics/terra-atlas-mvp

# 1. Create .env.local from template
cp .env.example .env.local

# 2. Populate with real credentials from BWS
./scripts/refresh-credentials.sh

# 3. Start dev server
npm run dev
```

### Daily Development

```bash
# Just work normally
npm run dev

# If credentials seem "broken":
# 1. Stop server (Ctrl+C)
# 2. Refresh credentials
./scripts/refresh-credentials.sh
# 3. Restart server
npm run dev
```

### After Pulling Changes

```bash
git pull

# If .env.local was deleted/modified:
./scripts/refresh-credentials.sh
npm run dev
```

---

## 🔒 Our Credential Architecture

### Three-Layer Security

```
┌─────────────────────────────────────────┐
│  Layer 1: Bitwarden Secrets Manager     │  ← SOURCE OF TRUTH
│  (BWS)                                   │
│  - supabase-prod-url                    │
│  - supabase-prod-anon-key               │
└──────────────┬──────────────────────────┘
               │
               │ bws get
               ↓
┌─────────────────────────────────────────┐
│  Layer 2: .env.local (gitignored)       │  ← LOCAL DEVELOPMENT
│  NEXT_PUBLIC_SUPABASE_URL=...           │
│  NEXT_PUBLIC_SUPABASE_ANON_KEY=...      │
└──────────────┬──────────────────────────┘
               │
               │ npm run dev
               ↓
┌─────────────────────────────────────────┐
│  Layer 3: Next.js Runtime               │  ← YOUR APP
│  process.env.NEXT_PUBLIC_SUPABASE_URL   │
└─────────────────────────────────────────┘
```

### Why This Architecture?

- **BWS**: Single source of truth, never committed to git
- **.env.local**: Local development, gitignored for security
- **Runtime**: App reads from process.env at build/start time

---

## 🛠️ Available Scripts

### `./scripts/refresh-credentials.sh`

**What it does**:
1. Fetches latest credentials from BWS
2. Updates your `.env.local` file
3. Creates backup (`.env.local.backup`)

**When to use**:
- "Keys aren't working"
- After cloning the repo
- After pulling changes that affect env vars
- When you're unsure if keys are current

**Example**:
```bash
./scripts/refresh-credentials.sh
# ✅ Credentials refreshed!
# ⚠️  Restart Next.js dev server
```

---

## 🐛 Troubleshooting

### Problem: "Cannot read NEXT_PUBLIC_SUPABASE_URL"

**Symptoms**:
```javascript
// Component shows:
process.env.NEXT_PUBLIC_SUPABASE_URL // undefined
```

**Solution**:
```bash
# 1. Check if .env.local exists
cat .env.local

# 2. If it exists, refresh it
./scripts/refresh-credentials.sh

# 3. RESTART dev server (critical!)
# Stop current server (Ctrl+C)
npm run dev
```

### Problem: "Keys work locally but not in production"

**Symptoms**:
- Works on `localhost:3002`
- Fails on Vercel deployment

**Solution**:
```bash
# Check Vercel environment variables
vercel env ls

# Add missing variables
vercel env add NEXT_PUBLIC_SUPABASE_URL
vercel env add NEXT_PUBLIC_SUPABASE_ANON_KEY
vercel env add SUPABASE_SERVICE_ROLE_KEY

# Redeploy
vercel --prod
```

### Problem: "Old keys still being used after update"

**Symptoms**:
- Updated `.env.local`
- Dev server still uses old values

**Solution**:
```bash
# Clear Next.js cache
rm -rf .next/

# Restart dev server
npm run dev
```

---

## 📝 Best Practices

### ✅ DO

- **Keep `.env.local` gitignored** - Security first!
- **Use BWS as source of truth** - Never hardcode
- **Restart dev server** after env changes
- **Use `refresh-credentials.sh`** when in doubt
- **Prefix public vars** with `NEXT_PUBLIC_`
- **Document new env vars** in `.env.example`

### ❌ DON'T

- **Don't commit `.env.local`** - Ever!
- **Don't hardcode credentials** in components
- **Don't skip dev server restart** after env changes
- **Don't put secrets** in `NEXT_PUBLIC_*` vars
- **Don't share `.env.local`** file directly

---

## 🔍 Verifying Credentials

### Check BWS (Source of Truth)

```bash
# Verify credentials are in BWS
bws get supabase-prod-url
bws get supabase-prod-anon-key

# If these work, your credentials exist!
```

### Check .env.local (Development)

```bash
# View your local env file
cat .env.local | grep SUPABASE

# Should show:
# NEXT_PUBLIC_SUPABASE_URL=https://...
# NEXT_PUBLIC_SUPABASE_ANON_KEY=eyJ...
```

### Check Runtime (Next.js)

```javascript
// Add to any component temporarily
console.log('Supabase URL:', process.env.NEXT_PUBLIC_SUPABASE_URL)

// If undefined → restart dev server!
```

---

## 🎓 Key Concepts

### Build-time vs Runtime

**Build-time** (`NEXT_PUBLIC_*`):
- Embedded into JavaScript bundle
- Available in browser
- Requires rebuild to change

**Runtime** (no prefix):
- Only available on server
- Never sent to browser
- Can change without rebuild

### Hot Module Replacement (HMR)

**What HMR reloads**:
- ✅ Component code changes
- ✅ CSS/style changes
- ✅ Page content

**What HMR doesn't reload**:
- ❌ Environment variables
- ❌ Next.js configuration
- ❌ Package.json changes

---

## 📞 Quick Reference

| Scenario | Command | Why |
|----------|---------|-----|
| First setup | `cp .env.example .env.local && ./scripts/refresh-credentials.sh` | Create and populate env file |
| Keys not working | `./scripts/refresh-credentials.sh && npm run dev` | Refresh and restart |
| After git pull | `./scripts/refresh-credentials.sh` | Update from BWS |
| Production deploy | `vercel env add` | Set Vercel env vars |
| Clear cache | `rm -rf .next/ && npm run dev` | Fresh build |

---

## 🌟 Summary

**Your keys are not lost!** They're safely stored in:
1. ✅ Bitwarden Secrets Manager (BWS)
2. ✅ `.env.local` (gitignored)
3. ✅ Next.js runtime (after dev server starts)

**The "lost keys" feeling** comes from Next.js caching, not actual key loss.

**The solution** is simple: Restart your dev server! 🔄

---

## 🔗 Related Documentation

- **CLAUDE.md** - Credential management section
- **GLOBE_INTEGRATION_COMPLETE.md** - Globe component with data integration
- **.env.example** - Template for all environment variables
- **scripts/refresh-credentials.sh** - Automated credential refresh

---

*"Never hardcode. Always refresh. Restart when in doubt."* 🔐
