# Terra Atlas Comprehensive Improvement Plan

_Review date: 2025-11-16_

This plan distills the current state of the Terra Atlas platform and the moves required to turn it into a resilient, measurable product. Each theme below lists the risks we observed in the codebase, precise evidence, and the work required to close the gap. Use `Now → Next → Later` priorities to sequence execution; “Definition of Done” (DoD) bullets provide concrete acceptance criteria.

---

## Architecture Snapshot
- **Frontend (Next.js 15 / React 19)** – App Router with a still-heavy landing page (`app/page.tsx`) coordinating streamed sections, dynamic globe imports, and telemetry controls. Components live in `components/` with bespoke animation logic (`HeroSection.tsx`, `EnhancedStatCard.tsx`).
- **Data & APIs** – Most `app/api/*` routes proxy Supabase/Drizzle, but top-of-funnel endpoints (`app/api/stats/route.ts`, `app/api/projects/route.ts`) still query a bundled SQLite DB (`data/terra-atlas-local.db`), which drifts from Supabase and breaks across build targets.
- **Auth & Payments** – Supabase client in `lib/supabase.ts` powers the global `AuthContext`, while Stripe flows under `app/api/stripe/*` lack a unified webhook contract or request tracing.
- **Infrastructure** – Vercel hosts the Next app; Supabase/Postgres + Drizzle (`lib/drizzle/db.ts`) back persistence; telemetry writes NDJSON to `data/telemetry-events.log` and is only analyzed manually (`scripts/analyze-telemetry.mjs`).
- **Tooling** – `npm run check` promises lint/type/test gates but ESLint is skipped in `next.config.js`, Vitest config is missing, and there is no automated perf budget or request ID plumbing.

## System Review Highlights
- `/api/projects` now validates query params, rate-limits requests, and reads directly from Supabase via Drizzle (with SQLite as an error fallback); `/api/stats` follows the same pattern so top-of-funnel data stays in sync.
- `lib/drizzle/db.ts` exposes a developer-specific fallback DSN (`postgresql://tstoltz@localhost...`) which leaks secrets and silently swaps databases between environments.
- `components/TerraGlobeWithSites.tsx` fetches `/data/demo-sites.json` plus textures without cancellation, so memoized state is lost during navigation and FPS telemetry never reaches the API.
- Scroll-reveal logic was centralized, but lazy sections still render before their data is ready, causing content flashes until stats API responds.
- Status documentation is scattered across `PROJECT_STATUS.md`, `STRATEGIC_NEXT_STEPS.md`, `ULTRA_STRATEGIC_EXECUTION_PLAN.md`, etc., making onboarding slow and duplicative.

---

## 1. Developer Confidence & Tooling _(Now)_
- **Risks**
  - No Vitest suite or config even though `npm run test` exists; `docs/SYSTEM_ARCHITECTURE.md:345` calls out “Unit tests: TODO”.
  - Lint / type failures do not block the build locally because `next.config` still has `eslint: { ignoreDuringBuilds: true }` (see `next.config.js:6`).
  - Environment validation only covers public keys (`lib/env.ts`) but server routes continue to read `process.env` directly (e.g. `app/api/portfolio/route.ts:48`, `lib/drizzle/db.ts:10`).
- **Plan**
  1. Add `vitest.config.ts` + starter spec files for shared libs (`lib/utils`, `utils/performance`) to unblock CI.
  2. Remove `ignoreDuringBuilds`, keep `npm run check` wired to CI gates, and enforce a local `prepush` hook via Husky.
  3. Expand `lib/env.server.ts` to expose `requireServerEnv` helpers and migrate every server route / script to them.
- **DoD**
  - `npm run check` runs (and passes) locally in <60s with cached deps.
  - CI fails fast when lint/type/test fail; build no longer skips ESLint.
  - `.env.example` documents every required server secret and `node env:verify` script enforces it.

## 2. Data & API Integrity _(Now)_
- **Risks**
  - `/api/projects` and `/api/stats` hit Supabase first but still lack auth + cursor pagination; SQLite fallbacks remain for resilience and must be retired once Supabase parity is proven.
  - Projects search validates inputs and rate limits bursts, but responses remain unauthenticated and lack cursor pagination.
  - Portfolio API mixes Drizzle helpers with raw SQL snippets and has no pagination (`app/api/portfolio/route.ts:84-143`).
- **Plan**
  1. Stand up a read replica table in Supabase for high-traffic “atlas stats” queries; expose them through Drizzle models so both `/api/stats` and `/api/projects` run on the same source.
  2. Introduce shared Zod schemas under `lib/schemas/*` for all API payloads + query strings; reuse server-side and client-side to keep types in sync.
  3. Add pagination + rate limiting middleware for `/api/projects`; stream large responses via incremental rendering when possible.
- **DoD**
  - No route imports `better-sqlite3`; all DB access flows through Drizzle or Supabase SDK.
  - `/api/projects` rejects invalid parameters with 400s and documents them in `docs/API.md`.
  - Portfolio API returns stable `cursor`-based pages and surfaces `X-Request-Id` for traceability.

## 3. Security & Secrets _(Now)_
- **Risks**
  - JWT verification falls back to a hard-coded `'dev-secret'` (`app/api/portfolio/route.ts:48`).
  - Database fallback connection string checks in the repo (`lib/drizzle/db.ts:10-15`) leak local usernames/paths.
  - OAuth helpers expose `window.location.origin` directly (`lib/supabase.ts:167-209`) which breaks when called server-side.
- **Plan**
  1. Require `JWT_SECRET`, `SUPABASE_SERVICE_ROLE_KEY`, and `DATABASE_URL` through `requireServerEnv`, and add runtime checks during boot.
  2. Move OAuth redirect URLs into config derived from `NEXT_PUBLIC_SITE_URL` to avoid `window` usage server-side.
  3. Add a lightweight request firewall for API routes (rate limiting + origin allowlist) and document incident response in `CREDENTIAL_MANAGEMENT.md`.
- **DoD**
  - Secrets never default; missing vars crash boot with actionable errors.
  - repo no longer contains user-specific DSNs; `.env.example` is the single config surface.
  - API logs include structured auth failures and rate-limit decisions.

## 4. Observability & Telemetry _(Next)_
- **Risks**
  - Telemetry endpoint appends to a flat file indefinitely (`app/api/telemetry/route.ts:16-45`) with no rotation or aggregation.
  - `scripts/analyze-telemetry.mjs` only parses `globe_performance` events; no dashboards or alerting exist.
  - No tracing or request IDs across API handlers, making it hard to correlate Supabase queries with frontend issues.
- **Plan**
  1. Stream telemetry events into Supabase (or ClickHouse) through edge functions; add nightly retention job for the log file as a fallback.
  2. Build a minimal `/dashboard/performance` page that visualizes FPS/load trends using the stored telemetry.
  3. Add a `logger.withRequestId` helper + middleware so every API response carries `X-Request-Id` and all logs include it.
- **DoD**
  - Telemetry file never exceeds 10 MB; Supabase table holds 30 days of metrics with rollups.
  - Performance dashboard shows P50/P95 load metrics and alerts when FPS < 45 for 3 consecutive samples.
  - API logs/traces let us reconstruct any failed request within minutes.

## 5. Experience & Performance _(Next)_
- **Risks**
  - Hero now consumes live stats, but `/api/stats` still runs on SQLite and can lag behind Supabase updates.
  - Scroll-reveal hook is in place, yet sections that depend on remote data (projects, stats cards) still flash placeholders because data loading is not coordinated with the animation lifecycle.
  - `TerraGlobeWithSites` downloads `/public/data/demo-sites.json` on every mount without caching or aborting pending requests (`components/TerraGlobeWithSites.tsx:33-66`).
- **Plan**
  1. Feed real `/api/stats` data into `HeroSection` and other sections; fall back gracefully when the API is offline.
  2. Extract a reusable `useScrollReveal` hook that re-attaches observers when lazy components stream in; add unit tests.
  3. Memoize site data + textures for the globe, use `AbortController` for fetches, and expose loading metrics to the telemetry system.
  4. Add Lighthouse/Calibre budgets to CI (FCP, TTI, CLS) to keep regressions visible.
- **DoD**
  - Hero metrics always reflect latest stats and show skeletons while loading.
  - Scroll animations trigger for every lazily injected block.
  - Globe load telemetry contains accurate mount/init/markers timings in Supabase dashboards; FCP budget enforced in CI.

## 6. Documentation & Enablement _(Later)_
- **Risks**
  - Multiple status docs (e.g., `PROJECT_STATUS.md`, `STRATEGIC_NEXT_STEPS.md`, `IMMEDIATE_ACTION_PLAN.md`) diverge, confusing contributors.
  - No end-to-end “Run the stack locally” guide covering Supabase, Drizzle, Stripe, and telemetry.
- **Plan**
  1. Consolidate status documents into a single `docs/STATUS.md` with links to live trackers.
  2. Create a “Day 0” onboarding doc covering `npm run dev`, database seeding, Stripe webhooks, screenshot tooling, and telemetry analysis.
  3. Add architecture decision records (ADRs) for major choices (Globe rendering stack, Supabase vs. Postgres, etc.).
- **DoD**
  - New contributors follow one doc to stand up the stack and run `npm run check`.
  - ADRs capture trade-offs for any non-trivial platform decision.

---

### Immediate Execution Snapshot
1. **Projects API guardrails:** finish auth + cursor pagination, then swap SQLite for Supabase/Drizzle so the validated, rate-limited endpoint hits live data.
2. **Telemetry ingestion:** stream globe performance metrics into Supabase and build a basic dashboard to watch FPS/load trends.
3. **Secret management:** move JWT/database/Stripe secrets to `lib/env.server.ts`, remove plaintext fallbacks, and document rotation in `CREDENTIAL_MANAGEMENT.md`.

This creates a foundation for telemetry ingest + Supabase migrations in the next sprint.
