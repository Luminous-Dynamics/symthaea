# Performance

*Speed in Service of Truth*

---

> "Performance is not about being fast—it's about not being slow when it matters."

---

## Introduction

Epistemic Markets must be fast enough that performance never interferes with truth-seeking. This document covers our performance philosophy, architecture, optimization strategies, and monitoring.

---

## Part I: Performance Philosophy

### Core Principles

**1. User-Perceived Performance**
- What matters is how fast it *feels*, not how fast it *is*
- Perceived latency < actual latency
- Progressive loading over blocking

**2. Right-Size Performance**
- Not everything needs to be instant
- Match performance to user expectations
- Optimize hot paths, not everything

**3. Predictable > Fast**
- Consistent 100ms beats variable 50-500ms
- Users adapt to consistent latency
- Spikes destroy trust

**4. Degrade Gracefully**
- Partial results > no results
- Slow is better than broken
- Always show something

### Performance Budget

| Operation | Budget | Priority |
|-----------|--------|----------|
| Page load (initial) | < 2s | Critical |
| Page load (cached) | < 500ms | Critical |
| Market list load | < 300ms | High |
| Single market load | < 150ms | High |
| Submit prediction | < 500ms | High |
| Search markets | < 500ms | Medium |
| Load predictions | < 400ms | Medium |
| Start resolution | < 1s | Medium |
| Historical data | < 2s | Low |

---

## Part II: Architecture for Performance

### Data Flow Optimization

```
Optimal Path:
┌──────────┐    ┌──────────┐    ┌──────────┐
│  Cache   │ →  │   SDK    │ →  │    UI    │
│  (hit)   │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘
     ~5ms           ~10ms           ~15ms
     Total: ~30ms

Standard Path:
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│Conductor │ →  │   Zome   │ →  │   SDK    │ →  │    UI    │
│          │    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘
    ~20ms          ~50ms           ~10ms           ~15ms
    Total: ~95ms

Cold Path (with network):
┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐
│  Network │ →  │Conductor │ →  │   Zome   │ →  │   SDK    │ →  │    UI    │
│  (DHT)   │    │          │    │          │    │          │    │          │
└──────────┘    └──────────┘    └──────────┘    └──────────┘    └──────────┘
   ~100-500ms      ~20ms          ~50ms           ~10ms           ~15ms
   Total: ~195-545ms
```

### Caching Strategy

```typescript
// Multi-level caching
interface CacheHierarchy {
  // Level 1: In-memory (SDK)
  memory: {
    ttl: 30_000;  // 30 seconds
    maxSize: 1000;
    strategy: "LRU";
  };

  // Level 2: IndexedDB (Browser)
  indexedDB: {
    ttl: 300_000;  // 5 minutes
    maxSize: 50_000_000;  // 50MB
    strategy: "LRU";
  };

  // Level 3: Conductor cache
  conductor: {
    ttl: 600_000;  // 10 minutes
    strategy: "DHT-aware";
  };
}

class CacheManager {
  private memory = new LRUCache<string, CachedItem>(1000);
  private db: IDBDatabase;

  async get<T>(key: string): Promise<T | null> {
    // Try memory first
    const memCached = this.memory.get(key);
    if (memCached && !this.isExpired(memCached)) {
      return memCached.value as T;
    }

    // Try IndexedDB
    const dbCached = await this.getFromDB(key);
    if (dbCached && !this.isExpired(dbCached)) {
      // Promote to memory
      this.memory.set(key, dbCached);
      return dbCached.value as T;
    }

    return null;
  }

  async set<T>(key: string, value: T, ttl?: number): Promise<void> {
    const item = { value, expiresAt: Date.now() + (ttl ?? 30_000) };
    this.memory.set(key, item);
    await this.setInDB(key, item);
  }

  invalidate(pattern: string | RegExp): void {
    // Invalidate matching keys across all levels
  }
}
```

### Lazy Loading

```typescript
// Lazy load heavy components
const MarketChart = lazy(() => import("./MarketChart"));
const PredictionHistory = lazy(() => import("./PredictionHistory"));

function MarketDetail({ marketId }: { marketId: string }) {
  const { market, loading } = useMarket(marketId);

  if (loading) return <MarketSkeleton />;

  return (
    <div>
      <MarketHeader market={market} />

      <Suspense fallback={<ChartSkeleton />}>
        <MarketChart marketId={marketId} />
      </Suspense>

      <Suspense fallback={<HistorySkeleton />}>
        <PredictionHistory marketId={marketId} />
      </Suspense>
    </div>
  );
}
```

### Pagination and Virtualization

```typescript
// Virtual list for long market lists
import { useVirtualizer } from "@tanstack/react-virtual";

function MarketList({ markets }: { markets: Market[] }) {
  const parentRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: markets.length,
    getScrollElement: () => parentRef.current,
    estimateSize: () => 120,  // Estimated row height
    overscan: 5,
  });

  return (
    <div ref={parentRef} style={{ height: "600px", overflow: "auto" }}>
      <div style={{ height: `${virtualizer.getTotalSize()}px`, position: "relative" }}>
        {virtualizer.getVirtualItems().map((virtualItem) => (
          <MarketCard
            key={virtualItem.key}
            market={markets[virtualItem.index]}
            style={{
              position: "absolute",
              top: virtualItem.start,
              left: 0,
              right: 0,
            }}
          />
        ))}
      </div>
    </div>
  );
}

// Paginated API calls
async function listMarkets(params: ListParams): Promise<PaginatedResult<Market>> {
  const { cursor, limit = 20 } = params;

  // Efficient cursor-based pagination
  const markets = await client.markets.list({
    cursor,
    limit: limit + 1,  // Fetch one extra to detect hasMore
  });

  return {
    items: markets.slice(0, limit),
    nextCursor: markets.length > limit ? markets[limit - 1].hash : null,
    hasMore: markets.length > limit,
  };
}
```

---

## Part III: Zome Performance

### Efficient Entry Retrieval

```rust
// GOOD: Batch retrieval
#[hdk_extern]
pub fn get_markets_batch(hashes: Vec<EntryHash>) -> ExternResult<Vec<Option<Market>>> {
    hashes
        .into_iter()
        .map(|hash| {
            match get(hash, GetOptions::default())? {
                Some(record) => {
                    let market: Market = record.entry().to_app_option()?.ok_or(
                        wasm_error!(WasmErrorInner::Guest("Not a market".into()))
                    )?;
                    Ok(Some(market))
                }
                None => Ok(None),
            }
        })
        .collect()
}

// BAD: Individual calls from client
// for hash in hashes {
//     const market = await client.markets.getMarket(hash);  // N round trips!
// }
```

### Link Optimization

```rust
// Use link types for efficient filtering
#[hdk_link_types]
pub enum LinkTypes {
    MarketToCreator,
    MarketToPredictions,
    MarketToTag,
    TagToMarkets,
    CreatorToMarkets,
}

// Efficient tag-based query
#[hdk_extern]
pub fn get_markets_by_tag(tag: String) -> ExternResult<Vec<MarketWithHash>> {
    // Get tag anchor
    let tag_anchor = anchor("tag".into(), tag)?;

    // Get links (fast)
    let links = get_links(tag_anchor, Some(LinkTypes::TagToMarkets), None)?;

    // Batch fetch markets
    let hashes: Vec<EntryHash> = links
        .into_iter()
        .filter_map(|l| l.target.into_entry_hash())
        .collect();

    get_markets_batch(hashes)
}

// Avoid expensive patterns
// BAD: Full table scan with filter
// let all_markets = list_all_markets()?;
// let filtered: Vec<_> = all_markets.iter().filter(|m| m.tags.contains(&tag)).collect();
```

### Computation Optimization

```rust
// Cache expensive calculations
use std::collections::HashMap;
use std::sync::Mutex;
use once_cell::sync::Lazy;

static MATL_CACHE: Lazy<Mutex<HashMap<AgentPubKey, (MatlScore, Timestamp)>>> =
    Lazy::new(|| Mutex::new(HashMap::new()));

const MATL_CACHE_TTL: i64 = 60_000_000;  // 1 minute in microseconds

pub fn get_cached_matl(agent: AgentPubKey) -> ExternResult<MatlScore> {
    let now = sys_time()?;

    // Check cache
    {
        let cache = MATL_CACHE.lock().unwrap();
        if let Some((score, cached_at)) = cache.get(&agent) {
            if now.as_micros() - cached_at.as_micros() < MATL_CACHE_TTL {
                return Ok(score.clone());
            }
        }
    }

    // Fetch fresh
    let score = fetch_matl_score(agent.clone())?;

    // Update cache
    {
        let mut cache = MATL_CACHE.lock().unwrap();
        cache.insert(agent, (score.clone(), now));
    }

    Ok(score)
}
```

---

## Part IV: SDK Performance

### Request Batching

```typescript
// Automatic request batching
class BatchedZomeCaller {
  private pendingCalls: Map<string, BatchedCall[]> = new Map();
  private batchTimeout: number = 10;  // ms

  async call<T>(zome: string, fn: string, payload: unknown): Promise<T> {
    const key = `${zome}:${fn}`;

    return new Promise((resolve, reject) => {
      const call: BatchedCall = { payload, resolve, reject };

      if (!this.pendingCalls.has(key)) {
        this.pendingCalls.set(key, [call]);

        // Schedule batch execution
        setTimeout(() => this.executeBatch(key), this.batchTimeout);
      } else {
        this.pendingCalls.get(key)!.push(call);
      }
    });
  }

  private async executeBatch(key: string): Promise<void> {
    const calls = this.pendingCalls.get(key);
    if (!calls) return;
    this.pendingCalls.delete(key);

    const [zome, fn] = key.split(":");
    const batchFn = `${fn}_batch`;

    try {
      // Try batch call first
      const results = await this.rawCall(zome, batchFn, calls.map(c => c.payload));
      results.forEach((result, i) => calls[i].resolve(result));
    } catch (e) {
      // Fallback to individual calls
      for (const call of calls) {
        try {
          const result = await this.rawCall(zome, fn, call.payload);
          call.resolve(result);
        } catch (e) {
          call.reject(e);
        }
      }
    }
  }
}
```

### Connection Pooling

```typescript
// Connection management
class ConnectionPool {
  private connections: AppWebsocket[] = [];
  private maxConnections = 3;
  private currentIndex = 0;

  async getConnection(): Promise<AppWebsocket> {
    // Round-robin for load distribution
    const conn = this.connections[this.currentIndex];
    this.currentIndex = (this.currentIndex + 1) % this.connections.length;

    if (conn.isOpen) {
      return conn;
    }

    // Reconnect if needed
    await this.reconnect(this.currentIndex);
    return this.connections[this.currentIndex];
  }

  async initialize(urls: string[]): Promise<void> {
    this.connections = await Promise.all(
      urls.slice(0, this.maxConnections).map(url =>
        AppWebsocket.connect(url)
      )
    );
  }
}
```

### Optimistic Updates

```typescript
// Optimistic UI updates
function usePrediction(marketId: string) {
  const [predictions, setPredictions] = useState<Prediction[]>([]);
  const client = useClient();

  const submitPrediction = useCallback(async (input: PredictionInput) => {
    // Create optimistic prediction
    const optimisticPrediction: Prediction = {
      ...input,
      hash: `temp-${Date.now()}`,
      createdAt: Date.now(),
      status: "pending",
    };

    // Update UI immediately
    setPredictions(prev => [...prev, optimisticPrediction]);

    try {
      // Actually submit
      const hash = await client.predictions.submit(input);

      // Replace optimistic with real
      setPredictions(prev =>
        prev.map(p =>
          p.hash === optimisticPrediction.hash
            ? { ...p, hash, status: "confirmed" }
            : p
        )
      );
    } catch (error) {
      // Rollback on failure
      setPredictions(prev =>
        prev.filter(p => p.hash !== optimisticPrediction.hash)
      );
      throw error;
    }
  }, [client]);

  return { predictions, submitPrediction };
}
```

---

## Part V: Frontend Performance

### Bundle Optimization

```typescript
// vite.config.ts
export default defineConfig({
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          // Separate vendor chunks
          "vendor-react": ["react", "react-dom"],
          "vendor-holochain": ["@holochain/client"],
          "vendor-charts": ["recharts"],
        },
      },
    },
    // Target modern browsers
    target: "es2020",
    // Minification
    minify: "terser",
    terserOptions: {
      compress: {
        drop_console: true,
      },
    },
  },
});
```

### Asset Optimization

```typescript
// Image optimization
import { Image } from "next/image";

function MarketImage({ src, alt }: { src: string; alt: string }) {
  return (
    <Image
      src={src}
      alt={alt}
      width={400}
      height={300}
      loading="lazy"
      placeholder="blur"
      blurDataURL={BLUR_PLACEHOLDER}
    />
  );
}

// Font optimization
import { Inter } from "next/font/google";

const inter = Inter({
  subsets: ["latin"],
  display: "swap",  // Prevent FOIT
  preload: true,
});
```

### Render Optimization

```typescript
// Memoization
const MarketCard = memo(function MarketCard({ market }: { market: Market }) {
  return (
    <div className="market-card">
      <h3>{market.question}</h3>
      <div>{market.outcomes.join(" | ")}</div>
    </div>
  );
});

// Avoid re-renders
function MarketList() {
  const { markets } = useMarkets();

  // Stable callback reference
  const handleSelect = useCallback((id: string) => {
    navigate(`/market/${id}`);
  }, [navigate]);

  // Memoized derived data
  const sortedMarkets = useMemo(
    () => markets.sort((a, b) => b.createdAt - a.createdAt),
    [markets]
  );

  return (
    <div>
      {sortedMarkets.map(market => (
        <MarketCard
          key={market.hash}
          market={market}
          onSelect={handleSelect}
        />
      ))}
    </div>
  );
}
```

### Progressive Loading

```typescript
// Progressive enhancement
function MarketDetail({ marketId }: { marketId: string }) {
  // Critical data first
  const { market } = useMarket(marketId);

  // Secondary data deferred
  const { predictions } = usePredictions(marketId, { defer: true });
  const { history } = useHistory(marketId, { defer: true });

  return (
    <>
      {/* Always render critical content */}
      <MarketHeader market={market} />
      <MarketActions market={market} />

      {/* Progressive enhancement */}
      <Suspense fallback={<PredictionsSkeleton />}>
        <PredictionList predictions={predictions} />
      </Suspense>

      <Suspense fallback={<HistorySkeleton />}>
        <MarketHistory history={history} />
      </Suspense>
    </>
  );
}
```

---

## Part VI: Network Performance

### DHT Optimization

```rust
// Minimize DHT hops
impl Market {
    // Store commonly-accessed data together
    pub fn with_summary(&self) -> MarketWithSummary {
        MarketWithSummary {
            market: self.clone(),
            prediction_count: self.cached_prediction_count,
            latest_price: self.cached_latest_price,
            volume: self.cached_volume,
        }
    }
}

// Update cached summaries on changes
fn update_market_summary(market_hash: EntryHash) -> ExternResult<()> {
    let predictions = get_predictions_for_market(market_hash.clone())?;

    let summary = MarketSummary {
        prediction_count: predictions.len(),
        latest_price: calculate_price(&predictions),
        volume: calculate_volume(&predictions),
        updated_at: sys_time()?,
    };

    // Store summary as link tag for fast retrieval
    create_link(
        market_hash,
        market_hash,
        LinkTypes::MarketSummary,
        serialize(&summary)?,
    )?;

    Ok(())
}
```

### Request Compression

```typescript
// Compress large payloads
import { compress, decompress } from "lz-string";

class CompressedClient {
  async sendLarge<T>(zome: string, fn: string, payload: unknown): Promise<T> {
    const serialized = JSON.stringify(payload);

    // Only compress if beneficial
    if (serialized.length > 1000) {
      const compressed = compress(serialized);
      return this.rawCall(zome, `${fn}_compressed`, compressed);
    }

    return this.rawCall(zome, fn, payload);
  }
}
```

### Signal Optimization

```rust
// Efficient signal emission
fn emit_market_signal(market: &Market, event: SignalEvent) -> ExternResult<()> {
    // Minimal payload for signals
    let signal = MarketSignal {
        hash: hash_entry(market)?,
        event,
        // Don't include full market - let clients fetch if needed
        summary: market.summary(),
    };

    emit_signal(&signal)?;
    Ok(())
}

// Client subscribes to relevant signals only
client.onSignal((signal) => {
    if (signal.type === "market" && watchedMarkets.has(signal.hash)) {
        // Refresh only the affected market
        refreshMarket(signal.hash);
    }
});
```

---

## Part VII: Monitoring and Profiling

### Performance Metrics

```typescript
// Collect performance metrics
class PerformanceMonitor {
  private metrics: Map<string, number[]> = new Map();

  measure<T>(name: string, fn: () => T): T {
    const start = performance.now();
    const result = fn();
    const duration = performance.now() - start;

    this.record(name, duration);
    return result;
  }

  async measureAsync<T>(name: string, fn: () => Promise<T>): Promise<T> {
    const start = performance.now();
    const result = await fn();
    const duration = performance.now() - start;

    this.record(name, duration);
    return result;
  }

  private record(name: string, duration: number): void {
    if (!this.metrics.has(name)) {
      this.metrics.set(name, []);
    }
    this.metrics.get(name)!.push(duration);

    // Alert on slow operations
    if (duration > this.getThreshold(name)) {
      console.warn(`Slow operation: ${name} took ${duration}ms`);
    }
  }

  getStats(name: string): PerformanceStats {
    const values = this.metrics.get(name) || [];
    return {
      count: values.length,
      min: Math.min(...values),
      max: Math.max(...values),
      avg: values.reduce((a, b) => a + b, 0) / values.length,
      p50: this.percentile(values, 50),
      p95: this.percentile(values, 95),
      p99: this.percentile(values, 99),
    };
  }
}

// Usage
const marketId = await perf.measureAsync("createMarket", () =>
  client.markets.createMarket(input)
);
```

### Real User Monitoring

```typescript
// RUM integration
class RealUserMonitoring {
  reportNavigation(route: string, loadTime: number): void {
    this.send({
      type: "navigation",
      route,
      loadTime,
      timestamp: Date.now(),
      userAgent: navigator.userAgent,
    });
  }

  reportInteraction(action: string, responseTime: number): void {
    this.send({
      type: "interaction",
      action,
      responseTime,
      timestamp: Date.now(),
    });
  }

  reportError(error: Error, context: string): void {
    this.send({
      type: "error",
      error: error.message,
      stack: error.stack,
      context,
      timestamp: Date.now(),
    });
  }

  private send(data: unknown): void {
    // Batch and send to analytics endpoint
    navigator.sendBeacon("/api/rum", JSON.stringify(data));
  }
}
```

### Profiling Tools

```bash
# Rust profiling
cargo flamegraph --bin epistemic-markets

# Chrome DevTools
# 1. Open DevTools
# 2. Performance tab
# 3. Record interaction
# 4. Analyze flame chart

# Lighthouse audit
npx lighthouse https://app.epistemic-markets.com --view
```

---

## Part VIII: Performance Testing

### Load Testing

```typescript
// k6 load test
import http from "k6/http";
import { check, sleep } from "k6";

export const options = {
  stages: [
    { duration: "1m", target: 50 },   // Ramp up
    { duration: "5m", target: 50 },   // Steady state
    { duration: "1m", target: 100 },  // Peak
    { duration: "1m", target: 0 },    // Ramp down
  ],
  thresholds: {
    http_req_duration: ["p(95)<500"],  // 95% under 500ms
    http_req_failed: ["rate<0.01"],    // <1% failure rate
  },
};

export default function () {
  // List markets
  const listRes = http.get("http://localhost:3000/api/markets");
  check(listRes, {
    "list status is 200": (r) => r.status === 200,
    "list time < 300ms": (r) => r.timings.duration < 300,
  });

  sleep(1);

  // Create prediction
  const predRes = http.post("http://localhost:3000/api/predictions", {
    marketId: "test-market",
    outcome: "Yes",
    confidence: 0.7,
  });
  check(predRes, {
    "predict status is 200": (r) => r.status === 200,
    "predict time < 500ms": (r) => r.timings.duration < 500,
  });

  sleep(1);
}
```

### Benchmark Tests

```typescript
// Vitest benchmarks
import { bench, describe } from "vitest";

describe("Market Operations", () => {
  bench("create market", async () => {
    await client.markets.createMarket(testMarketInput);
  });

  bench("list markets", async () => {
    await client.markets.listOpenMarkets();
  });

  bench("submit prediction", async () => {
    await client.predictions.submitPrediction(testPredictionInput);
  });

  bench("search markets", async () => {
    await client.markets.searchByTag("crypto");
  });
});
```

---

## Part IX: Performance Optimization Checklist

### Before Launch

- [ ] Bundle size < 200KB (gzipped)
- [ ] Time to First Byte < 200ms
- [ ] First Contentful Paint < 1.5s
- [ ] Largest Contentful Paint < 2.5s
- [ ] Time to Interactive < 3s
- [ ] All API calls < 500ms (p95)
- [ ] No memory leaks
- [ ] Lighthouse score > 90

### Ongoing

- [ ] Monitor RUM metrics
- [ ] Weekly performance reviews
- [ ] Automated performance tests in CI
- [ ] Performance budgets enforced
- [ ] Regular profiling sessions
- [ ] Cache hit rates > 80%
- [ ] Error rates < 1%

---

## Conclusion

Performance is a feature, not an afterthought. Every millisecond of latency is friction between the user and truth-seeking. We optimize relentlessly so that performance never becomes an obstacle.

Fast systems feel responsive. Responsive systems feel alive. Alive systems invite participation.

---

> "The best performance optimization is the one that makes the next optimization unnecessary."

---

*Speed serves truth. We serve both.*

---

## Quick Reference

### Common Performance Patterns

| Pattern | Use When | Example |
|---------|----------|---------|
| Caching | Repeated reads | Market details |
| Batching | Multiple similar requests | Loading market list |
| Virtualization | Long lists | Market listings |
| Lazy loading | Heavy components | Charts, history |
| Optimistic updates | User mutations | Submitting predictions |
| Pagination | Large datasets | Historical data |
| Compression | Large payloads | Bulk exports |

### Performance Commands

```bash
# Analyze bundle
npm run analyze

# Run benchmarks
npm run bench

# Load test
k6 run load-test.js

# Profile production
npm run profile

# Lighthouse audit
npm run lighthouse
```
