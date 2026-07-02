# 🚀 Vercel Deployment Guide - marketplace.mycelix.net

**Status**: Ready for deployment
**Target URL**: https://marketplace.mycelix.net
**Framework**: SvelteKit 2.0

---

## 🎯 Overview

This guide will help you deploy the Mycelix Marketplace frontend to Vercel with a custom domain (marketplace.mycelix.net). This provides a live demo of the marketplace even before the Holochain backend is fully integrated.

### What Will Be Deployed

- ✅ Complete frontend (10 pages)
- ✅ All UI components and interactions
- ✅ TypeScript-compiled JavaScript
- ✅ Static assets and styles
- ⚠️ Mock data (real Holochain integration in Phase 5)

---

## 📋 Prerequisites

### Required
- GitHub account (already have: Luminous-Dynamics)
- Vercel account (create at https://vercel.com)
- Domain access (mycelix.net DNS configuration)
- Repository pushed to GitHub ✅

### Optional
- Vercel CLI for local testing

---

## 🚀 Deployment Steps

### Step 1: Connect GitHub to Vercel (5 minutes)

1. **Go to Vercel**: https://vercel.com
2. **Sign up/Login** with GitHub
3. **Import Project**:
   - Click "Add New..." → "Project"
   - Select "Import Git Repository"
   - Authorize Vercel to access your GitHub account
   - Select `Luminous-Dynamics/Mycelix-Marketplace`

### Step 2: Configure Project Settings (3 minutes)

Vercel should auto-detect SvelteKit, but verify these settings:

**Framework Preset**: SvelteKit
**Root Directory**: `./` (leave default)
**Build Command**: `cd frontend && npm run build`
**Output Directory**: `frontend/build`
**Install Command**: `cd frontend && npm install`
**Development Command**: `cd frontend && npm run dev`

**Environment Variables**:
Add these if you have them (optional for demo):
```
VITE_HOLOCHAIN_URL=ws://localhost:8888
VITE_PINATA_JWT=your_jwt_here (optional)
VITE_PINATA_GATEWAY=gateway.pinata.cloud (optional)
```

### Step 3: Deploy (2 minutes)

1. Click **Deploy**
2. Wait for build to complete (~2-3 minutes)
3. Vercel will provide a URL: `https://mycelix-marketplace.vercel.app`

**Expected Build Output**:
```
✓ Building for production...
✓ Compiling TypeScript...
✓ Bundling JavaScript...
✓ Optimizing assets...
✓ Build completed successfully!
```

### Step 4: Configure Custom Domain (5 minutes)

#### 4.1 Add Domain in Vercel

1. Go to your project in Vercel
2. Click **Settings** → **Domains**
3. Click **Add Domain**
4. Enter: `marketplace.mycelix.net`
5. Click **Add**

Vercel will provide DNS configuration instructions.

#### 4.2 Configure DNS (Cloudflare)

You'll need to add a CNAME record in Cloudflare:

**Using Cloudflare API** (recommended):

```bash
# Get Cloudflare zone ID for mycelix.net
ZONE_ID="685364b37101f56f919dbd988f0f779a"  # mycelix.net

# Get API token from BWS
API_TOKEN=$(bws get cloudflare-api-token)

# Add CNAME record for marketplace subdomain
curl -X POST "https://api.cloudflare.com/client/v4/zones/${ZONE_ID}/dns_records" \
  -H "Authorization: Bearer ${API_TOKEN}" \
  -H "Content-Type: application/json" \
  --data '{
    "type": "CNAME",
    "name": "marketplace",
    "content": "cname.vercel-dns.com",
    "ttl": 1,
    "proxied": false
  }'
```

**Using Cloudflare Dashboard**:

1. Go to: https://dash.cloudflare.com
2. Select domain: `mycelix.net`
3. Go to **DNS** → **Records**
4. Click **Add record**
5. Configure:
   - **Type**: CNAME
   - **Name**: marketplace
   - **Target**: cname.vercel-dns.com
   - **Proxy status**: DNS only (gray cloud)
   - **TTL**: Auto
6. Click **Save**

#### 4.3 Verify Domain

1. Return to Vercel project settings
2. Wait 1-2 minutes for DNS propagation
3. Vercel will automatically verify the domain
4. SSL certificate will be provisioned automatically

**Your site will be live at**: https://marketplace.mycelix.net

---

## 🔧 Vercel Configuration File

We've created `vercel.json` with optimal settings:

```json
{
  "buildCommand": "cd frontend && npm run build",
  "outputDirectory": "frontend/build",
  "devCommand": "cd frontend && npm run dev",
  "installCommand": "cd frontend && npm install",
  "framework": "sveltekit",
  "regions": ["iad1"],
  "github": {
    "silent": true
  }
}
```

**Configuration Details**:
- **buildCommand**: Navigates to frontend directory and builds
- **outputDirectory**: SvelteKit build output location
- **framework**: Tells Vercel to use SvelteKit optimizations
- **regions**: US East (change if needed: sfo1, lhr1, etc.)
- **github.silent**: Reduces GitHub notification spam

---

## 🌐 Custom Domain Setup Summary

### DNS Configuration for mycelix.net

| Subdomain | Type | Target | Proxy | Purpose |
|-----------|------|--------|-------|---------|
| marketplace | CNAME | cname.vercel-dns.com | ❌ DNS only | Vercel app |

### Expected URLs

| URL | Purpose | Status |
|-----|---------|--------|
| https://marketplace.mycelix.net | Primary domain | ⏳ Pending DNS |
| https://mycelix-marketplace.vercel.app | Vercel default | ✅ After deploy |
| https://mycelix-marketplace-git-main-luminous-dynamics.vercel.app | Git branch preview | ✅ Auto |

---

## 🔐 Environment Variables (Optional)

For a fully functional demo, you can add these environment variables in Vercel:

### Production Environment

Go to **Settings** → **Environment Variables** in Vercel:

```bash
# Holochain (if running remotely)
VITE_HOLOCHAIN_URL=wss://conductor.mycelix.net

# IPFS/Pinata (for photo uploads)
VITE_PINATA_JWT=your_pinata_jwt_here
VITE_PINATA_GATEWAY=gateway.pinata.cloud

# Feature flags
VITE_DEMO_MODE=true
```

**Note**: For Phase 4 (current), the app works with mock data, so these are optional.

---

## 🚀 Deployment Modes

### Automatic Deployments

Vercel will automatically deploy:
- **Production**: Every push to `main` → https://marketplace.mycelix.net
- **Preview**: Every PR → Unique preview URL
- **Branch**: Every push to other branches → Branch-specific URL

### Manual Deployments

Using Vercel CLI:

```bash
# Install Vercel CLI
npm i -g vercel

# Login
vercel login

# Deploy to preview
vercel

# Deploy to production
vercel --prod
```

---

## 🎯 Post-Deployment Checklist

### Verify Deployment ✅

1. **Build Success**: Check Vercel dashboard for green checkmark
2. **Site Loads**: Visit https://mycelix-marketplace.vercel.app
3. **All Pages Work**: Test navigation to all 10 pages
4. **No Console Errors**: Open DevTools console
5. **TypeScript Compiled**: No type errors in build
6. **Assets Load**: Images, CSS, fonts all working

### Verify Custom Domain ✅

1. **DNS Propagated**: Run `dig marketplace.mycelix.net`
2. **Domain Loads**: Visit https://marketplace.mycelix.net
3. **SSL Active**: Check for green padlock
4. **Redirects Work**: HTTP → HTTPS automatic

### Performance Check ✅

1. **Lighthouse Score**: Run in Chrome DevTools
   - Target: 90+ performance
   - Target: 100 accessibility
   - Target: 100 best practices
   - Target: 100 SEO

2. **Load Time**: Should be <2 seconds
3. **Mobile Responsive**: Test on different devices

---

## 🔧 Troubleshooting

### Build Fails

**Error**: `Cannot find module '@sveltejs/kit'`
**Solution**: Check `frontend/package.json` includes all dependencies

**Error**: `TypeScript errors in build`
**Solution**: Run `npm run check` locally first

### Domain Not Working

**Error**: DNS not resolving
**Solution**:
- Wait 5-10 minutes for DNS propagation
- Verify CNAME record in Cloudflare
- Check proxy is disabled (DNS only)

**Error**: SSL certificate pending
**Solution**: Wait for automatic provisioning (up to 24 hours)

### Environment Variables

**Error**: Variables not loading
**Solution**:
- Variables must start with `VITE_` for SvelteKit
- Redeploy after adding variables
- Check variable names are exact

---

## 📊 Expected Performance

### Vercel Edge Network

- **Global CDN**: Content delivered from nearest edge location
- **HTTP/2 & HTTP/3**: Modern protocols for fast loading
- **Automatic compression**: Gzip and Brotli
- **Image optimization**: Next-gen formats (WebP, AVIF)

### Build Metrics

| Metric | Expected | Our Target |
|--------|----------|------------|
| **Build Time** | 2-3 minutes | ✅ |
| **Bundle Size** | ~500KB gzipped | ✅ |
| **First Load** | <2 seconds | ✅ |
| **Time to Interactive** | <3 seconds | ✅ |

---

## 🎨 Preview Deployments

Every pull request gets a unique preview URL:

**Example**:
- PR #5: `https://mycelix-marketplace-git-fix-cart-bug-luminous-dynamics.vercel.app`

**Benefits**:
- Test changes before merging
- Share with team/testers
- Automatic PR comments with preview link

---

## 🔄 Continuous Deployment Workflow

```
┌─────────────────┐
│  Push to GitHub │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Vercel Detects  │
│   New Commit    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Run npm ci     │
│  Run build      │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Deploy to Edge  │
│   Network       │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Update DNS      │
│ Provision SSL   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ ✅ Live at      │
│ marketplace.    │
│ mycelix.net     │
└─────────────────┘
```

---

## 📱 Mobile App Preparation

Vercel deployment also prepares for future PWA:

- Service worker ready
- Manifest.json configured
- Offline-first architecture
- App-like experience on mobile

---

## 🎊 Success Criteria

Your deployment is successful when:

- ✅ Build completes without errors
- ✅ Site loads at Vercel URL
- ✅ All 10 pages are accessible
- ✅ Custom domain resolves
- ✅ SSL certificate is active
- ✅ Lighthouse score >90
- ✅ Automatic deployments working
- ✅ Preview deployments on PRs

---

## 🚀 Next Steps After Deployment

### Phase 5: Backend Integration

Once deployed, you can:

1. **Test with real users**: Share the demo link
2. **Collect feedback**: Use GitHub Discussions
3. **Monitor performance**: Vercel Analytics
4. **Plan backend**: Holochain conductor integration
5. **Add real data**: Connect to DHT
6. **Enable features**: IPFS uploads, MRC voting

### Future Enhancements

- Analytics integration
- Error tracking (Sentry)
- Performance monitoring
- A/B testing
- Feature flags
- User authentication

---

## 📚 Resources

- **Vercel Docs**: https://vercel.com/docs
- **SvelteKit Deployment**: https://kit.svelte.dev/docs/adapter-vercel
- **Cloudflare DNS**: https://developers.cloudflare.com/dns
- **Custom Domains**: https://vercel.com/docs/custom-domains

---

## 🎯 Quick Commands Reference

```bash
# Get Cloudflare credentials
bws get cloudflare-api-token

# Check DNS
dig marketplace.mycelix.net

# Test local build
cd frontend && npm run build && npm run preview

# Deploy with Vercel CLI
vercel --prod

# Check deployment status
vercel ls

# View deployment logs
vercel logs <deployment-url>
```

---

**Repository**: https://github.com/Luminous-Dynamics/Mycelix-Marketplace
**Vercel Project**: (will be created during setup)
**Target URL**: https://marketplace.mycelix.net
**Status**: ✅ **READY TO DEPLOY**

---

*🍄 Let's make the marketplace live for the world to see!*

**Last Updated**: November 13, 2025
