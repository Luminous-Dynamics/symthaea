# 🔧 Cache Troubleshooting & Prevention Guide

## 🎯 The Problem

When developing with Next.js, changes sometimes don't appear even after hard refresh. This happens due to multiple cache layers.

---

## 📚 Root Causes Explained

### 1. **Next.js Build Cache (`.next/` folder)**
- **What**: Compiled pages, chunks, and optimized assets
- **Why it caches**: Speeds up rebuilds (incremental compilation)
- **When it fails**: Dynamic imports, style changes, global state changes

### 2. **Browser Cache**
- **Memory Cache**: RAM-based, cleared on tab close
- **Disk Cache**: Persists across sessions
- **Service Workers**: Can cache aggressively for PWAs
- **Prefetch Cache**: Next.js prefetches links, caches them

### 3. **React Fast Refresh Limitations**
- **Doesn't detect**:
  - Changes in parent components of dynamic imports
  - `<style jsx>` tag modifications sometimes
  - Context provider changes
  - Changes in `getStaticProps`/`getServerSideProps`

### 4. **Module Resolution Cache**
- **Node.js module cache**: `require.cache`
- **Webpack module cache**: Hot Module Replacement state
- **Import map cache**: Dynamic import() caching

---

## 🚀 Quick Solutions

### Method 1: Clean Restart (Recommended)
```bash
# Use the automated script
./scripts/dev-clean.sh

# Or manually:
rm -rf .next && npm run dev
```

### Method 2: Browser Hard Refresh
- **Chrome/Edge**: `Ctrl + Shift + R` or `Ctrl + F5`
- **Firefox**: `Ctrl + Shift + R`
- **Safari**: `Cmd + Shift + R`
- **Better**: Open DevTools → Network → Disable cache (while open)

### Method 3: Incognito/Private Window
Opens fresh browser context with no cache:
- **Chrome/Edge**: `Ctrl + Shift + N`
- **Firefox**: `Ctrl + Shift + P`

### Method 4: Clear Browser Data
1. `Ctrl + Shift + Delete`
2. Select "Cached images and files"
3. Time range: "Last hour"
4. Clear data

---

## 🔧 Automation Solutions

### 1. **Add to package.json**
```json
{
  "scripts": {
    "dev": "next dev",
    "dev:clean": "rm -rf .next && next dev",
    "dev:fresh": "rm -rf .next node_modules/.cache && npm run dev"
  }
}
```

Usage:
```bash
npm run dev:clean   # Quick clean
npm run dev:fresh   # Deep clean (slower)
```

### 2. **Next.js Config - Disable Caching in Dev**
Add to `next.config.js`:

```javascript
module.exports = {
  // ... existing config

  // Development-only settings
  ...(process.env.NODE_ENV === 'development' && {
    // Disable React Fast Refresh (for debugging cache issues)
    // reactStrictMode: false,

    // Custom headers to prevent caching
    async headers() {
      return [
        {
          source: '/:path*',
          headers: [
            {
              key: 'Cache-Control',
              value: 'no-store, no-cache, must-revalidate, proxy-revalidate, max-age=0',
            },
          ],
        },
      ]
    },
  }),
}
```

### 3. **Webpack Config - Aggressive Cache Busting**
Add to `next.config.js`:

```javascript
module.exports = {
  webpack: (config, { dev }) => {
    if (dev) {
      // Disable Webpack caching in development
      config.cache = false

      // Add timestamp to chunk names for cache busting
      config.output.filename = '[name].[contenthash].js'
      config.output.chunkFilename = '[name].[contenthash].js'
    }
    return config
  },
}
```

### 4. **Git Hook - Auto-clean on branch switch**
Create `.git/hooks/post-checkout`:

```bash
#!/usr/bin/env bash
# Auto-clean Next.js cache when switching branches

echo "🧹 Branch changed - cleaning Next.js cache..."
rm -rf .next
echo "✅ Cache cleared. Run 'npm run dev' to start fresh."
```

Make executable:
```bash
chmod +x .git/hooks/post-checkout
```

### 5. **VSCode Task - One-Click Clean Restart**
Add to `.vscode/tasks.json`:

```json
{
  "version": "2.0.0",
  "tasks": [
    {
      "label": "Dev: Clean Restart",
      "type": "shell",
      "command": "./scripts/dev-clean.sh",
      "problemMatcher": [],
      "presentation": {
        "reveal": "always",
        "panel": "new"
      }
    }
  ]
}
```

Usage: `Ctrl+Shift+P` → "Run Task" → "Dev: Clean Restart"

---

## 📊 When to Use Each Solution

| Scenario | Solution | Time | Effectiveness |
|----------|----------|------|---------------|
| Minor CSS changes not showing | Browser hard refresh | 1s | 70% |
| Component changes not updating | `rm -rf .next && npm run dev` | 10s | 95% |
| Persistent cache issues | Incognito window | 2s | 90% |
| Dynamic imports not updating | Full restart + browser clear | 15s | 99% |
| Production-like testing | `npm run build && npm start` | 60s | 100% |

---

## 🔍 Debugging Cache Issues

### Check if changes are in build:
```bash
# Search for your code in compiled output
grep -r "your-new-code" .next/

# Check when files were last modified
ls -lt .next/server/app/ | head
```

### Check browser cache:
1. Open DevTools (`F12`)
2. Network tab
3. Disable cache (checkbox)
4. Reload page
5. Look for `(from disk cache)` or `(from memory cache)` in Size column

### Check service workers:
1. DevTools → Application tab
2. Service Workers section
3. Unregister if present
4. Reload page

---

## 🎯 Best Practices

### During Active Development:
1. **Keep DevTools open** with "Disable cache" checked
2. **Use `dev:clean` script** when making major changes
3. **Test in incognito** before deploying
4. **Clear cache** when switching branches

### Before Deployment:
1. **Production build test**: `npm run build && npm start`
2. **Multiple browsers**: Test Chrome, Firefox, Safari
3. **Mobile testing**: Real devices (not just DevTools)
4. **Lighthouse audit**: Checks for caching issues

### For CI/CD:
```yaml
# .github/workflows/test.yml
- name: Clear Next.js cache
  run: rm -rf .next

- name: Build
  run: npm run build

- name: Test
  run: npm test
```

---

## 🚨 Emergency: "Nothing Works!"

If all else fails:

```bash
# Nuclear option - full clean slate
rm -rf .next
rm -rf node_modules
rm -rf node_modules/.cache
npm install
npm run dev

# In browser:
# 1. Clear all browsing data (Ctrl+Shift+Delete)
# 2. Close ALL browser windows
# 3. Reopen browser
# 4. Visit site in incognito
```

---

## 📈 Preventing Future Issues

### 1. **Document Cache-Sensitive Changes**
When making changes to:
- Dynamic imports
- Global CSS
- Context providers
- Configuration files

Add note in PR: "⚠️ Requires clean restart"

### 2. **Automate Clean Restarts**
Add to your workflow:
```bash
# Morning routine
git pull && npm run dev:clean

# Before testing feature
npm run dev:clean
```

### 3. **Monitor Build Times**
Slow builds = likely cache issues:
```bash
# Add to package.json
"dev:time": "time npm run dev:clean"
```

### 4. **Use Feature Flags**
Instead of changing code frequently:
```tsx
const ENABLE_NEW_DESIGN = process.env.NEXT_PUBLIC_NEW_DESIGN === 'true'

// Toggle via .env.local, no code changes needed
```

---

## 💡 Pro Tips

1. **Versioned Assets**: Add `?v=2` to critical assets in development
2. **Service Worker Bypass**: Use "Update on reload" in DevTools → Application
3. **Module Aliases**: Use `@/components` vs `../../components` - better HMR
4. **Smaller Components**: HMR works better with smaller, isolated components
5. **Avoid Global State**: Prefer props/context - better Fast Refresh support

---

## 📚 Further Reading

- [Next.js Caching](https://nextjs.org/docs/app/building-your-application/caching)
- [React Fast Refresh](https://nextjs.org/docs/architecture/fast-refresh)
- [Webpack Caching](https://webpack.js.org/configuration/cache/)
- [Browser Cache Control](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Cache-Control)

---

**Last Updated**: November 11, 2025
**Status**: ✅ Clean restart script automated
**Next**: Implement CI/CD cache management

---

💚 **Remember**: Cache issues are frustrating but fixable. When in doubt, clean restart!
