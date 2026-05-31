# Performance Guide

Optimizing Knowledge hApp for production workloads.

## Performance Targets

### Response Time Goals

| Operation | Target | Acceptable | Degraded |
|-----------|--------|------------|----------|
| Get claim | <50ms | <100ms | >200ms |
| Search (simple) | <200ms | <500ms | >1s |
| Search (complex) | <500ms | <1s | >2s |
| Create claim | <100ms | <200ms | >500ms |
| Fact-check | <1s | <2s | >5s |
| Credibility score | <200ms | <500ms | >1s |
| Belief propagation | <2s | <5s | >10s |

### Throughput Goals

| Metric | Target | Per Node |
|--------|--------|----------|
| Claims/second | 100 | 10 |
| Queries/second | 500 | 50 |
| Fact-checks/second | 50 | 5 |
| Relationships/second | 200 | 20 |

## Architecture for Performance

### DHT Optimization

```
┌─────────────────────────────────────────────────────────────┐
│                    KNOWLEDGE DHT                             │
│                                                              │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐   │
│  │  Claims  │  │  Graph   │  │  Scores  │  │  Markets │   │
│  │  Shard   │  │  Shard   │  │  Cache   │  │  Links   │   │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘   │
│        │            │             │             │           │
│        └────────────┼─────────────┼─────────────┘           │
│                     │             │                          │
│              ┌──────▼─────────────▼──────┐                  │
│              │     Local Agent Cache     │                  │
│              │   (Hot data, 100ms TTL)   │                  │
│              └───────────────────────────┘                  │
└─────────────────────────────────────────────────────────────┘
```

### Caching Strategy

#### L1: In-Memory Cache (Hot)
- Recent claims and scores
- TTL: 60 seconds
- Size: 10,000 entries
- Hit rate target: >80%

#### L2: Agent-Local Store (Warm)
- Frequently accessed data
- TTL: 5 minutes
- Size: 100,000 entries
- Hit rate target: >95%

#### L3: DHT (Cold)
- All data
- Persistent
- Global availability

### Cache Configuration

```typescript
// SDK cache configuration
const knowledge = new KnowledgeClient(client, {
  cache: {
    enabled: true,
    l1: {
      maxSize: 10000,
      ttlSeconds: 60,
    },
    l2: {
      maxSize: 100000,
      ttlSeconds: 300,
    },
    preload: ['hot-domains'], // Preload frequently accessed
  },
});
```

## Query Optimization

### Efficient Queries

```typescript
// ❌ Inefficient: Fetches all, filters client-side
const allClaims = await knowledge.claims.listClaimsByDomain('climate');
const filtered = allClaims.filter(c => c.classification.empirical > 0.7);

// ✅ Efficient: Server-side filtering
const filtered = await knowledge.query.queryByEpistemic(0.7, 1.0, null, null, null, null, 50);
```

### Batch Operations

```typescript
// ❌ Inefficient: Sequential requests
for (const id of claimIds) {
  const cred = await knowledge.inference.calculateEnhancedCredibility(id, 'Claim');
  results.push(cred);
}

// ✅ Efficient: Batch request
const results = await knowledge.inference.batchCredibilityAssessment(claimIds);
```

### Pagination

```typescript
// ✅ Always paginate large result sets
async function* getAllClaims(domain: string) {
  let offset = 0;
  const limit = 100;

  while (true) {
    const result = await knowledge.query.advancedQuery({
      filters: [{ field: 'domain', operator: { Equals: domain } }],
      limit,
      offset,
    });

    yield* result.items;

    if (!result.hasMore) break;
    offset += limit;
  }
}
```

## Write Optimization

### Batch Writes

```typescript
// ❌ Inefficient: One at a time
for (const claim of claims) {
  await knowledge.claims.createClaim(claim);
}

// ✅ Efficient: Batch create
await knowledge.claims.batchCreateClaims(claims);
```

### Async Processing

```typescript
// For non-critical updates, use fire-and-forget
knowledge.inference.calculateEnhancedCredibility(claimId, 'Claim')
  .catch(err => console.error('Background calculation failed:', err));

// Critical operations still await
const hash = await knowledge.claims.createClaim(input);
```

### Deferred Propagation

```typescript
// Defer cascade updates for bulk operations
const options = { deferPropagation: true };

for (const rel of relationships) {
  await knowledge.graph.createRelationship(rel, options);
}

// Trigger single propagation at end
await knowledge.graph.propagateBelief(rootClaimId);
```

## Network Optimization

### Connection Pooling

```typescript
// Reuse connections
const client = await AppWebsocket.connect('ws://localhost:8888');
const knowledge = new KnowledgeClient(client);

// Use single instance throughout application
export { knowledge };
```

### Request Deduplication

```typescript
// SDK automatically deduplicates concurrent identical requests
const [result1, result2] = await Promise.all([
  knowledge.claims.getClaim(claimId),
  knowledge.claims.getClaim(claimId), // Same request, deduped
]);
// Only one network call made
```

### Prefetching

```typescript
// Prefetch related data
async function getClaimWithContext(claimId: string) {
  const [claim, relationships, credibility] = await Promise.all([
    knowledge.claims.getClaim(claimId),
    knowledge.graph.getRelationships(claimId),
    knowledge.inference.calculateEnhancedCredibility(claimId, 'Claim'),
  ]);

  return { claim, relationships, credibility };
}
```

## Belief Propagation Optimization

### Limiting Propagation Depth

```typescript
// Limit cascade depth for performance
await knowledge.claims.cascadeUpdate(claimId, {
  maxDepth: 3,        // Don't go beyond 3 hops
  maxNodes: 1000,     // Stop after 1000 nodes
  convergenceThreshold: 0.01, // Stop when changes < 1%
});
```

### Incremental Updates

```typescript
// Only propagate changed beliefs
await knowledge.graph.propagateBelief(claimId, {
  incremental: true,  // Only update changed nodes
  changedOnly: true,  // Skip unchanged paths
});
```

## Monitoring Performance

### Built-in Metrics

```typescript
// Enable metrics collection
const knowledge = new KnowledgeClient(client, {
  metrics: {
    enabled: true,
    sampleRate: 0.1, // 10% sampling
  },
});

// Get metrics
const metrics = knowledge.getMetrics();
console.log(`Average query time: ${metrics.queryTimeMs}ms`);
console.log(`Cache hit rate: ${metrics.cacheHitRate}%`);
```

### Custom Instrumentation

```typescript
// Wrap operations with timing
async function timedOperation<T>(name: string, fn: () => Promise<T>): Promise<T> {
  const start = performance.now();
  try {
    return await fn();
  } finally {
    const duration = performance.now() - start;
    console.log(`${name}: ${duration.toFixed(2)}ms`);
    metrics.record(name, duration);
  }
}

const claim = await timedOperation('getClaim', () =>
  knowledge.claims.getClaim(claimId)
);
```

### Health Checks

```typescript
// Regular health check
async function healthCheck(): Promise<HealthStatus> {
  const start = Date.now();

  try {
    // Test basic operations
    await knowledge.query.search('test', { limit: 1 });
    const latency = Date.now() - start;

    return {
      healthy: latency < 500,
      latencyMs: latency,
      cacheHitRate: knowledge.getMetrics().cacheHitRate,
    };
  } catch (error) {
    return { healthy: false, error: error.message };
  }
}
```

## Performance Anti-Patterns

### Avoid These

| Anti-Pattern | Problem | Solution |
|--------------|---------|----------|
| N+1 queries | Sequential fetches | Batch operations |
| No pagination | Memory exhaustion | Always paginate |
| Sync everything | Blocking UI | Async where possible |
| No caching | Repeated fetches | Enable caching |
| Deep propagation | Exponential growth | Limit depth |
| Large payloads | Network bottleneck | Compress, summarize |

## Scaling Considerations

### Horizontal Scaling

```
┌─────────────────────────────────────────────────────────────┐
│                      LOAD BALANCER                          │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│    ┌──────────┐  ┌──────────┐  ┌──────────┐               │
│    │  Node 1  │  │  Node 2  │  │  Node 3  │  ...          │
│    │  Agent   │  │  Agent   │  │  Agent   │               │
│    └──────────┘  └──────────┘  └──────────┘               │
│         │             │             │                       │
│         └─────────────┼─────────────┘                       │
│                       │                                     │
│                 ┌─────▼─────┐                               │
│                 │    DHT    │                               │
│                 └───────────┘                               │
└─────────────────────────────────────────────────────────────┘
```

### Read Replicas

For read-heavy workloads:
- Deploy read-only agents
- Sync from primary agents
- Route queries to replicas
- Writes go to primary

## Troubleshooting Performance

### Slow Queries

1. **Check cache hit rate** - Should be >80%
2. **Review query filters** - Use server-side filtering
3. **Check DHT health** - Network connectivity
4. **Profile operation** - Find bottleneck

### High Latency

1. **Check network** - Peer connectivity
2. **Check agent load** - CPU/memory usage
3. **Check DHT size** - Shard distribution
4. **Enable compression** - Reduce payload size

### Memory Issues

1. **Check cache size** - May need reduction
2. **Review pagination** - Limit result sets
3. **Check for leaks** - Long-running connections
4. **Restart agent** - Clear accumulated state

---

## Related Documentation

- [Security](./SECURITY.md) - Security considerations
- [Metrics](./METRICS.md) - Monitoring and metrics

---

*Fast knowledge is accessible knowledge.*
