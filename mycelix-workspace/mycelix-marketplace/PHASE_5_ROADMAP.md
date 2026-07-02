# Phase 5: Testing, Enhancement & Production Readiness

**Status**: Ready to Begin
**Prerequisites**: Phase 4 Complete ✅ (10/10 pages integrated)
**Goal**: Transform working prototype into production-ready marketplace
**Estimated Duration**: 2-3 weeks
**Target Launch**: December 2025

---

## 🎯 Phase 5 Objectives

1. **Replace Mock Implementations** - Real IPFS uploads, auth flow
2. **Comprehensive Testing** - E2E tests, integration tests, unit tests
3. **Performance Optimization** - Bundle size, lazy loading, caching
4. **Production Infrastructure** - Deployment, monitoring, error tracking
5. **User Experience Polish** - Animations, loading states, accessibility

---

## 📋 Phase 5a: Critical Production Requirements (Week 1)

### Priority 1: Real IPFS Integration 🔴

**Current State**: Mock IPFS uploads with placeholder CIDs
**Target State**: Real IPFS uploads via Pinata or ipfs-http-client
**Estimated Effort**: 4-6 hours

#### Files to Update:
1. `frontend/src/routes/CreateListing.svelte` - Line ~86: `uploadPhotosToIPFS()`
2. `frontend/src/routes/FileDispute.svelte` - Line ~128: `uploadEvidenceToIPFS()`

#### Implementation Options:

**Option A: Pinata API (Recommended for MVP)**
```bash
# Install dependencies
npm install pinata-web3

# Environment variables needed
PINATA_JWT=your_jwt_token
PINATA_GATEWAY=your_gateway_url
```

**Implementation**:
```typescript
import { PinataSDK } from "pinata-web3";

const pinata = new PinataSDK({
  pinataJwt: import.meta.env.VITE_PINATA_JWT,
  pinataGateway: import.meta.env.VITE_PINATA_GATEWAY,
});

async function uploadPhotosToIPFS(files: File[]): Promise<string[]> {
  const uploadPromises = files.map(async (file) => {
    const upload = await pinata.upload.file(file);
    return upload.IpfsHash;
  });
  return await Promise.all(uploadPromises);
}
```

**Option B: ipfs-http-client (Self-hosted)**
```bash
# Install dependencies
npm install ipfs-http-client

# Requires running IPFS node
ipfs daemon
```

**Implementation**:
```typescript
import { create } from 'ipfs-http-client';

const ipfs = create({ url: 'http://localhost:5001' });

async function uploadPhotosToIPFS(files: File[]): Promise<string[]> {
  const uploadPromises = files.map(async (file) => {
    const { cid } = await ipfs.add(file);
    return cid.toString();
  });
  return await Promise.all(uploadPromises);
}
```

**Recommendation**: Start with Pinata (easier, no infrastructure), migrate to self-hosted later if needed.

#### Acceptance Criteria:
- ✅ Photos upload to real IPFS
- ✅ CIDs are valid and retrievable
- ✅ Upload progress shown to user
- ✅ Error handling for failed uploads
- ✅ File size validation (max 10MB per file)
- ✅ File type validation (images only)

---

### Priority 2: Authentication Flow 🔴

**Current State**: Auth store exists but not connected to Holochain
**Target State**: User authentication via Holochain agent pubkey
**Estimated Effort**: 3-4 hours

#### Files to Update:
1. `frontend/src/lib/stores/auth.ts` - Connect to Holochain agent
2. All route files - Check auth state before actions

#### Implementation:
```typescript
// frontend/src/lib/stores/auth.ts
import { writable } from 'svelte/store';
import { initHolochainClient } from '$lib/holochain/client';

async function initAuth() {
  try {
    const client = await initHolochainClient();
    const agentInfo = await client.appInfo();

    return {
      isAuthenticated: true,
      agentPubKey: agentInfo.agent_pub_key,
      username: '', // Fetch from user profile
      roles: [], // Fetch from user profile
    };
  } catch (e) {
    return {
      isAuthenticated: false,
      agentPubKey: null,
      username: null,
      roles: [],
    };
  }
}

export const authStore = writable(await initAuth());
```

#### Route Guards:
```typescript
// frontend/src/lib/guards/auth.ts
import { get } from 'svelte/store';
import { goto } from '$app/navigation';
import { authStore } from '$lib/stores/auth';

export function requireAuth() {
  const auth = get(authStore);
  if (!auth.isAuthenticated) {
    goto('/login');
    return false;
  }
  return true;
}

export function requireRole(role: string) {
  const auth = get(authStore);
  if (!auth.roles.includes(role)) {
    goto('/unauthorized');
    return false;
  }
  return true;
}
```

#### Protected Routes:
- `/dashboard` - Requires auth
- `/create-listing` - Requires auth
- `/transactions` - Requires auth
- `/mrc-arbitration` - Requires auth + arbitrator role

#### Acceptance Criteria:
- ✅ User authentication via Holochain agent
- ✅ Auth state persists across page refreshes
- ✅ Protected routes redirect to login
- ✅ Role-based access control (arbitrator)
- ✅ Logout functionality

---

### Priority 3: Trust Score Batch Fetching 🟡

**Current State**: Browse page uses placeholder trust score (85%)
**Target State**: Real trust scores fetched in batches
**Estimated Effort**: 2-3 hours

#### Files to Update:
1. `frontend/src/routes/Browse.svelte` - Line ~62: Batch fetch seller profiles

#### Implementation:
```typescript
// Batch fetch seller profiles
async function loadListingsWithTrustScores() {
  loading = true;
  try {
    const client = await initHolochainClient();

    // 1. Fetch all listings
    const listings = await getAllListings(client);

    // 2. Extract unique seller IDs
    const sellerIds = [...new Set(listings.map((l) => l.seller_agent_id))];

    // 3. Batch fetch seller profiles (parallel)
    const sellerProfiles = await Promise.all(
      sellerIds.map((id) => getUserProfile(client, id).catch(() => null))
    );

    // 4. Create seller lookup map
    const sellerMap = new Map();
    sellerProfiles.forEach((profile) => {
      if (profile) sellerMap.set(profile.agent_id, profile);
    });

    // 5. Merge trust scores into listings
    allListings = listings.map((listing) => ({
      ...listing,
      seller_trust_score: sellerMap.get(listing.seller_agent_id)?.trust_score || 0,
    }));
  } catch (e: any) {
    notifications.error('Load Failed', e.message);
  } finally {
    loading = false;
  }
}
```

#### Optimization Strategies:
1. **Caching**: Cache seller profiles for 5 minutes
2. **Pagination**: Only fetch profiles for visible listings
3. **Background Loading**: Show placeholder, load in background

#### Acceptance Criteria:
- ✅ Real trust scores displayed
- ✅ Batch fetching (not one-by-one)
- ✅ Caching to avoid repeated fetches
- ✅ Graceful fallback to default if fetch fails
- ✅ Performance < 1 second for 100 listings

---

### Priority 4: Error Boundaries 🟡

**Current State**: Try/catch in individual functions
**Target State**: Svelte error boundaries for graceful failures
**Estimated Effort**: 2 hours

#### Implementation:
```typescript
// frontend/src/lib/components/ErrorBoundary.svelte
<script lang="ts">
  import { onMount } from 'svelte';

  export let fallback: string = 'Something went wrong';

  let error: Error | null = null;

  onMount(() => {
    window.addEventListener('error', handleError);
    window.addEventListener('unhandledrejection', handleRejection);

    return () => {
      window.removeEventListener('error', handleError);
      window.removeEventListener('unhandledrejection', handleRejection);
    };
  });

  function handleError(event: ErrorEvent) {
    error = event.error;
  }

  function handleRejection(event: PromiseRejectionEvent) {
    error = new Error(event.reason);
  }

  function reset() {
    error = null;
    location.reload();
  }
</script>

{#if error}
  <div class="error-boundary">
    <h2>Oops! Something went wrong</h2>
    <p>{fallback}</p>
    <button on:click={reset}>Reload Page</button>
  </div>
{:else}
  <slot />
{/if}
```

#### Wrap Routes:
```typescript
// frontend/src/routes/+layout.svelte
<script lang="ts">
  import ErrorBoundary from '$lib/components/ErrorBoundary.svelte';
</script>

<ErrorBoundary>
  <slot />
</ErrorBoundary>
```

#### Acceptance Criteria:
- ✅ Errors caught at route level
- ✅ User-friendly error display
- ✅ Reload/retry functionality
- ✅ Error logging (console/Sentry)

---

## 📋 Phase 5b: Testing Infrastructure (Week 2)

### Priority 5: Integration Testing 🔴

**Current State**: No automated tests
**Target State**: E2E tests for all critical user flows
**Estimated Effort**: 8-12 hours

#### Testing Framework: Playwright (Recommended)

```bash
# Install Playwright
npm install -D @playwright/test
npx playwright install

# Create test directory
mkdir -p tests/integration
```

#### Test Cases:

**1. Browse → Purchase Flow** (`tests/integration/purchase.spec.ts`)
```typescript
import { test, expect } from '@playwright/test';

test('complete purchase flow', async ({ page }) => {
  // 1. Browse marketplace
  await page.goto('/browse');
  await expect(page.locator('h1')).toContainText('Browse');

  // 2. Click first listing
  await page.locator('.listing-card').first().click();
  await expect(page).toHaveURL(/\/listing\/.+/);

  // 3. Add to cart
  await page.locator('button:has-text("Add to Cart")').click();
  await expect(page.locator('.notification')).toContainText('Added to Cart');

  // 4. Go to cart
  await page.goto('/cart');
  await expect(page.locator('.cart-item')).toHaveCount(1);

  // 5. Proceed to checkout
  await page.locator('button:has-text("Proceed to Checkout")').click();
  await expect(page).toHaveURL('/checkout');

  // 6. Fill shipping info
  await page.fill('#name', 'Test User');
  await page.fill('#address', '123 Test St');
  await page.fill('#city', 'Test City');
  await page.fill('#state', 'TS');
  await page.fill('#postal_code', '12345');

  // 7. Complete checkout
  await page.locator('button:has-text("Complete Order")').click();
  await expect(page.locator('.notification')).toContainText('Order Placed');
});
```

**2. Seller Flow** (`tests/integration/seller.spec.ts`)
```typescript
test('create listing flow', async ({ page }) => {
  // 1. Navigate to create listing
  await page.goto('/create-listing');

  // 2. Fill form
  await page.fill('#title-input', 'Test Product');
  await page.fill('#description-input', 'This is a test product description');
  await page.fill('#price-input', '99.99');
  await page.selectOption('#category-input', 'Electronics');

  // 3. Upload photo (mock)
  const fileInput = page.locator('#photos-input');
  await fileInput.setInputFiles('tests/fixtures/test-image.jpg');

  // 4. Submit
  await page.locator('button[type="submit"]').click();

  // 5. Verify redirect to listing detail
  await expect(page).toHaveURL(/\/listing\/.+/);
  await expect(page.locator('h1')).toContainText('Test Product');
});
```

**3. Review Flow** (`tests/integration/review.spec.ts`)
```typescript
test('submit review flow', async ({ page }) => {
  // Requires completed transaction
  await page.goto('/submit-review?transaction=test&listing=test&title=Test Product&seller=Test Seller');

  // 1. Select rating
  await page.locator('.star').nth(4).click(); // 5 stars

  // 2. Write comment
  await page.fill('#comment-input', 'Great product! Highly recommend.');

  // 3. Submit
  await page.locator('button[type="submit"]').click();

  // 4. Verify success
  await expect(page.locator('.notification')).toContainText('Review Submitted');
});
```

**4. Dispute Flow** (`tests/integration/dispute.spec.ts`)
```typescript
test('file dispute flow', async ({ page }) => {
  await page.goto('/file-dispute?transaction=test&title=Test Product&seller=Test Seller');

  // 1. Select reason
  await page.selectOption('#reason-input', 'not_as_described');

  // 2. Write description
  await page.fill('#description-input', 'Product does not match description provided.');

  // 3. Upload evidence
  const fileInput = page.locator('#evidence-input');
  await fileInput.setInputFiles('tests/fixtures/evidence.jpg');

  // 4. Submit
  await page.locator('button[type="submit"]').click();

  // 5. Verify success
  await expect(page.locator('.notification')).toContainText('Dispute Filed');
});
```

#### Run Tests:
```bash
# Run all tests
npx playwright test

# Run specific test
npx playwright test tests/integration/purchase.spec.ts

# Run with UI
npx playwright test --ui

# Generate report
npx playwright show-report
```

#### Acceptance Criteria:
- ✅ All critical flows tested
- ✅ Tests run in CI/CD pipeline
- ✅ > 80% code coverage
- ✅ Tests pass consistently

---

### Priority 6: Unit Testing 🟡

**Testing Framework**: Vitest + Testing Library

```bash
npm install -D vitest @testing-library/svelte @testing-library/jest-dom
```

#### Test Cases:

**1. Cart Store** (`tests/unit/stores/cart.test.ts`)
```typescript
import { describe, it, expect, beforeEach } from 'vitest';
import { get } from 'svelte/store';
import { cartItems, subtotal, total } from '$lib/stores/cart';

describe('Cart Store', () => {
  beforeEach(() => {
    cartItems.clear();
  });

  it('adds item to cart', () => {
    cartItems.addItem({
      listing_hash: '123',
      title: 'Test Product',
      price: 99.99,
      quantity: 1,
      photo_cid: 'QmTest',
      seller_agent_id: 'seller123',
      seller_name: 'Test Seller',
    });

    const items = get(cartItems);
    expect(items).toHaveLength(1);
    expect(items[0].title).toBe('Test Product');
  });

  it('calculates subtotal correctly', () => {
    cartItems.addItem({
      listing_hash: '123',
      title: 'Product 1',
      price: 10.00,
      quantity: 2,
      photo_cid: 'QmTest',
      seller_agent_id: 'seller123',
      seller_name: 'Test Seller',
    });

    cartItems.addItem({
      listing_hash: '456',
      title: 'Product 2',
      price: 25.00,
      quantity: 1,
      photo_cid: 'QmTest',
      seller_agent_id: 'seller123',
      seller_name: 'Test Seller',
    });

    expect(get(subtotal)).toBe(45.00); // 10*2 + 25*1
  });

  it('removes item from cart', () => {
    cartItems.addItem({
      listing_hash: '123',
      title: 'Test Product',
      price: 99.99,
      quantity: 1,
      photo_cid: 'QmTest',
      seller_agent_id: 'seller123',
      seller_name: 'Test Seller',
    });

    cartItems.removeItem('123');

    expect(get(cartItems)).toHaveLength(0);
  });
});
```

**2. Holochain Client** (`tests/unit/holochain/client.test.ts`)
```typescript
import { describe, it, expect, vi } from 'vitest';
import { initHolochainClient } from '$lib/holochain/client';

describe('Holochain Client', () => {
  it('initializes client successfully', async () => {
    // Mock WebSocket
    vi.mock('@holochain/client', () => ({
      AppWebsocket: {
        connect: vi.fn().mockResolvedValue({
          appInfo: vi.fn().mockResolvedValue({ agent_pub_key: 'test' }),
        }),
      },
    }));

    const client = await initHolochainClient();
    expect(client).toBeDefined();
  });

  it('retries on connection failure', async () => {
    // Test retry logic
  });
});
```

#### Run Tests:
```bash
npm run test
npm run test:coverage
```

---

## 📋 Phase 5c: Performance Optimization (Week 2-3)

### Priority 7: Bundle Analysis 🟡

```bash
# Install analyzer
npm install -D rollup-plugin-visualizer

# Add to vite.config.ts
import { visualizer } from 'rollup-plugin-visualizer';

export default defineConfig({
  plugins: [
    svelte(),
    visualizer({
      open: true,
      filename: 'bundle-analysis.html',
    }),
  ],
});

# Build and analyze
npm run build
# Opens bundle-analysis.html in browser
```

#### Optimization Targets:
- Total bundle size < 200KB gzipped
- Initial load < 1 second
- Time to Interactive < 3 seconds

---

### Priority 8: Code Splitting 🟡

**Current State**: All routes loaded upfront
**Target State**: Lazy loading with dynamic imports

```typescript
// frontend/src/routes/+layout.svelte
<script lang="ts">
  import { page } from '$app/stores';
  import { onMount } from 'svelte';

  // Preload next likely route
  onMount(() => {
    if ($page.url.pathname === '/browse') {
      // Preload listing detail
      import('./listing/[listing_hash]/+page.svelte');
    }
  });
</script>
```

#### Route Chunking:
- Core bundle: Dashboard, Browse, ListingDetail
- Seller bundle: CreateListing
- Transaction bundle: Checkout, Cart, Transactions
- Review bundle: SubmitReview, FileDispute
- Admin bundle: MRCArbitration

---

### Priority 9: Image Optimization 🟡

```typescript
// frontend/src/lib/components/OptimizedImage.svelte
<script lang="ts">
  export let src: string;
  export let alt: string;
  export let width: number = 300;
  export let height: number = 300;

  // Use IPFS gateway with resizing
  const optimizedSrc = `https://ipfs.io/ipfs/${src}?w=${width}&h=${height}&fit=cover`;
</script>

<img
  {src}={optimizedSrc}
  {alt}
  {width}
  {height}
  loading="lazy"
  decoding="async"
/>
```

---

## 📋 Phase 5d: User Experience Polish (Week 3)

### Priority 10: Loading Skeletons 🟢

Replace all `<p>Loading...</p>` with skeleton screens:

```typescript
// frontend/src/lib/components/ListingSkeleton.svelte
<div class="listing-skeleton">
  <div class="skeleton skeleton-image"></div>
  <div class="skeleton skeleton-title"></div>
  <div class="skeleton skeleton-price"></div>
  <div class="skeleton skeleton-seller"></div>
</div>

<style>
  .skeleton {
    background: linear-gradient(90deg, #f0f0f0 25%, #e0e0e0 50%, #f0f0f0 75%);
    background-size: 200% 100%;
    animation: loading 1.5s ease-in-out infinite;
  }

  @keyframes loading {
    0% { background-position: 200% 0; }
    100% { background-position: -200% 0; }
  }

  .skeleton-image {
    width: 100%;
    height: 200px;
    border-radius: 8px;
  }

  .skeleton-title {
    width: 80%;
    height: 24px;
    border-radius: 4px;
    margin-top: 1rem;
  }

  .skeleton-price {
    width: 40%;
    height: 20px;
    border-radius: 4px;
    margin-top: 0.5rem;
  }

  .skeleton-seller {
    width: 60%;
    height: 16px;
    border-radius: 4px;
    margin-top: 0.5rem;
  }
</style>
```

---

### Priority 11: Animations & Transitions 🟢

```typescript
// frontend/src/lib/transitions.ts
import { fade, slide, scale } from 'svelte/transition';
import { quintOut } from 'svelte/easing';

export const fadeIn = {
  duration: 300,
  easing: quintOut,
};

export const slideIn = {
  duration: 300,
  easing: quintOut,
  axis: 'y',
};

export const scaleIn = {
  duration: 200,
  start: 0.95,
  opacity: 0,
};
```

**Apply to components**:
```svelte
<div in:fade={fadeIn} out:fade={fadeIn}>
  <!-- Content -->
</div>
```

---

### Priority 12: Accessibility Audit 🟢

**WCAG 2.1 AA Compliance Checklist**:

- [ ] All images have alt text
- [ ] All forms have labels
- [ ] All interactive elements keyboard accessible
- [ ] Color contrast ratios ≥ 4.5:1
- [ ] Focus indicators visible
- [ ] Skip navigation links
- [ ] ARIA labels for complex components
- [ ] Screen reader testing
- [ ] Keyboard-only navigation testing

**Tools**:
- axe DevTools (browser extension)
- Lighthouse accessibility audit
- NVDA/JAWS screen reader testing

---

## 📋 Phase 5e: Production Infrastructure (Week 3)

### Priority 13: Deployment Configuration 🔴

#### Vercel Deployment (Recommended)

```bash
# Install Vercel CLI
npm install -g vercel

# Configure
vercel init

# Deploy
vercel --prod
```

**vercel.json**:
```json
{
  "buildCommand": "npm run build",
  "outputDirectory": "build",
  "framework": "sveltekit",
  "routes": [
    {
      "src": "/(.*)",
      "dest": "/"
    }
  ],
  "env": {
    "VITE_HOLOCHAIN_URL": "@holochain-url",
    "VITE_PINATA_JWT": "@pinata-jwt"
  }
}
```

---

### Priority 14: Environment Configuration 🔴

```bash
# .env.example (commit this)
VITE_HOLOCHAIN_URL=ws://localhost:8888
VITE_PINATA_JWT=your_jwt_here
VITE_PINATA_GATEWAY=https://gateway.pinata.cloud

# .env.local (gitignore this)
# Actual secrets go here

# .env.production
VITE_HOLOCHAIN_URL=wss://holochain.mycelix.net
VITE_PINATA_JWT=production_jwt
```

---

### Priority 15: Monitoring & Error Tracking 🟡

**Sentry Integration**:

```bash
npm install @sentry/sveltekit
```

```typescript
// src/hooks.client.ts
import * as Sentry from '@sentry/sveltekit';

Sentry.init({
  dsn: import.meta.env.VITE_SENTRY_DSN,
  environment: import.meta.env.MODE,
  tracesSampleRate: 1.0,
});
```

**Analytics**:
- Plausible (privacy-friendly)
- PostHog (self-hostable)

---

## 📊 Success Metrics

### Technical Metrics
- [ ] Lighthouse Score > 90 (all categories)
- [ ] Bundle size < 200KB gzipped
- [ ] First Contentful Paint < 1.5s
- [ ] Time to Interactive < 3s
- [ ] 0 console errors in production

### User Experience Metrics
- [ ] All user flows work end-to-end
- [ ] No broken links
- [ ] All forms validate correctly
- [ ] Error messages are clear and helpful
- [ ] Mobile experience is smooth

### Code Quality Metrics
- [ ] TypeScript strict mode: 0 errors
- [ ] ESLint: 0 errors
- [ ] Test coverage > 80%
- [ ] All critical paths tested
- [ ] Documentation complete

---

## 🗓️ Timeline

### Week 1: Critical Requirements
- Day 1-2: Real IPFS integration
- Day 3: Authentication flow
- Day 4: Trust score fetching
- Day 5: Error boundaries

### Week 2: Testing
- Day 1-2: E2E test setup
- Day 3-4: Write integration tests
- Day 5: Unit tests for stores/utilities

### Week 3: Polish & Deploy
- Day 1-2: Performance optimization
- Day 3: Accessibility audit
- Day 4: Loading skeletons & animations
- Day 5: Production deployment

---

## 🚀 Launch Checklist

### Pre-Launch
- [ ] All Priority 1-2 tasks complete
- [ ] All tests passing
- [ ] Performance benchmarks met
- [ ] Security audit complete
- [ ] Legal review (terms, privacy policy)

### Launch Day
- [ ] Deploy to production
- [ ] Monitor error rates
- [ ] Test all critical flows
- [ ] Prepare rollback plan

### Post-Launch
- [ ] Monitor analytics
- [ ] Collect user feedback
- [ ] Fix critical bugs within 24h
- [ ] Plan Phase 6 enhancements

---

**Status**: Phase 5 Ready to Begin
**Next Action**: Start with Priority 1 (Real IPFS Integration)

🌊 From completion to excellence - the journey continues!
