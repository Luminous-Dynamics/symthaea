# Shared Package Blueprint (`@luminous-dynamics/core`)

To let Terra Atlas and Mycelix Mail share UI + data utilities without duplicating files, publish a small package from this repo and consume it in downstream apps.

---

## 1. Proposed Folder Layout

Inside this repository, create a `packages/core` workspace:

```
packages/
  core/
    package.json
    tsconfig.json
    src/
      components/      // Re-export safe, design-system-grade components
      hooks/           // useSupabase, useRealtime, etc.
      lib/             // shared helpers (marker clustering, formatting)
      styles/          // optional base styles or Tailwind presets
```

Recommended exports:

### Components
- `Navigation`, `MobileNav`
- `PaymentForm`
- `AdvancedFilterPanel`
- `RealtimeCharts`
- `InvestmentCalculator` (and other self-contained widgets)
- `Globe`, `TerraGlobeWithSites` (abstracting data fetch from the core package)

### Hooks / Utilities
- `useSupabaseClient` wrapper (from `lib/supabase.ts`)
- `useStripeClient` (if applicable)
- `formatters` (currency, MW, ROI)
- Data helpers: `markerClustering`, `discovery-sdk`, `lib/utils`

Only move pieces that are design-system safe (no direct page-level side-effects). Page-specific logic should stay in each app.

---

## 2. Build & Publish Workflow

### package.json (packages/core)
```json
{
  "name": "@luminous-dynamics/core",
  "version": "0.1.0",
  "private": false,
  "main": "dist/index.js",
  "module": "dist/index.mjs",
  "types": "dist/index.d.ts",
  "scripts": {
    "build": "tsup src/index.ts --dts --format cjs,esm",
    "lint": "eslint src --ext .ts,.tsx",
    "test": "vitest run"
  },
  "peerDependencies": {
    "react": "^18.0.0",
    "react-dom": "^18.0.0"
  },
  "dependencies": {
    "lucide-react": "^0.x",
    "@supabase/supabase-js": "^2.x",
    // ...only packages required by exported modules
  }
}
```

### Build tool
- Use [tsup](https://tsup.egoist.dev/) (already added to root devDependencies) from within the Nix dev shell:
  ```bash
  nix develop            # brings in nodejs + npm via flake.nix
  npm run build --workspace @luminous-dynamics/core
  ```
- `tsconfig.json` inside the package references the root config; only `packages/core/src` is compiled.

### Publishing
- Option A: npm (public) – requires scoped package access.
- Option B: GitHub Packages – configure `.npmrc` and `NODE_AUTH_TOKEN` in CI.
- Tag releases via GitHub Actions: lint → test → build → `npm publish`.

---

## 3. Consuming the Package (Mycelix Mail)

1. Enter the flake dev shell, then add GitHub Packages auth (if private):
   ```bash
   nix develop
   ```
   ```
   // .npmrc (Mycelix Mail)
   @luminous-dynamics:registry=https://npm.pkg.github.com
   //npm.pkg.github.com/:_authToken=${NPM_TOKEN}
   ```
2. Install:
   ```
   npm install @luminous-dynamics/core
   ```
3. Import components/utilities:
   ```tsx
   import { PaymentForm, useSupabaseClient } from '@luminous-dynamics/core'
   ```
4. Pin to semantic versions (e.g., `^0.1.0`). Update Mycelix Mail only when ready.

---

## 4. Migration Steps

1. **Create workspace**
   - Add `"workspaces": ["packages/*"]` to the root `package.json`.
   - Move shared files into `packages/core/src`.
   - Keep page-specific code in `app/` (imports now come from `@luminous-dynamics/core`).
2. **Adjust imports**
   - Replace relative paths (`../../components/PaymentForm`) with package imports.
   - Ensure tree-shakeable entry points (`src/index.ts` re-exporting modules).
3. **CI/CD**
   - Add workflow `.github/workflows/publish-core.yml`.
   - Steps: install → test → build → npm publish (on tagged releases).
4. **Docs**
   - Update `docs/MYCELIX_MAIL_MIGRATION_PLAN.md` with install instructions.
   - Mention how to bump versions and changelog policy.

---

## 5. Guidelines

- Keep `@luminous-dynamics/core` lean: no environment-specific configs or page state.
- Avoid bundling large JSON/data seeds; expose helper functions to fetch them instead.
- Respect semver: breaking changes → major version.
- Provide Storybook or docs for each exported component to ease adoption.

With this package in place, Mycelix Mail can stay slim while pulling in proven Terra Atlas UI and helper code via standard npm workflows. Let me know when you want to scaffold the actual `packages/core` folder—I can help automate that next. 
