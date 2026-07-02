# Error Codes Reference

Complete reference for error codes in the Knowledge hApp.

## Error Format

All errors follow a consistent format:

```rust
pub struct KnowledgeError {
    pub code: ErrorCode,
    pub message: String,
    pub details: Option<HashMap<String, String>>,
}
```

In the SDK:

```typescript
interface KnowledgeError {
  code: string;
  message: string;
  details?: Record<string, string>;
}
```

## Error Code Categories

| Prefix | Category | Description |
|--------|----------|-------------|
| `E1xx` | Validation | Input validation failures |
| `E2xx` | Authorization | Permission/access errors |
| `E3xx` | Not Found | Resource doesn't exist |
| `E4xx` | Conflict | State conflicts |
| `E5xx` | External | External service errors |
| `E6xx` | Rate Limit | Rate limiting errors |
| `E9xx` | Internal | Internal system errors |

---

## Validation Errors (E1xx)

### E100: INVALID_INPUT
General input validation failure.

```typescript
// Example
{
  code: "E100",
  message: "Invalid input provided",
  details: { field: "content", reason: "cannot be empty" }
}
```

---

### E101: INVALID_CLASSIFICATION
Epistemic classification values out of range.

```typescript
// Triggers when E, N, or M is outside [0.0, 1.0]
{
  code: "E101",
  message: "Classification values must be between 0.0 and 1.0",
  details: {
    empirical: "1.5",  // Invalid
    normative: "0.5",
    mythic: "0.3"
  }
}
```

**Resolution**: Ensure all E-N-M values are in range [0.0, 1.0].

---

### E102: EMPTY_CONTENT
Claim content is empty or whitespace-only.

```typescript
{
  code: "E102",
  message: "Claim content cannot be empty",
  details: null
}
```

**Resolution**: Provide non-empty content string.

---

### E103: CONTENT_TOO_LONG
Claim content exceeds maximum length.

```typescript
{
  code: "E103",
  message: "Claim content exceeds maximum length",
  details: {
    maxLength: "10000",
    actualLength: "15234"
  }
}
```

**Resolution**: Reduce content to under 10,000 characters.

---

### E104: INVALID_DOMAIN
Domain is not in the allowed domains list.

```typescript
{
  code: "E104",
  message: "Invalid domain specified",
  details: {
    provided: "invalid-domain",
    allowed: "climate,energy,finance,governance,health,science,technology"
  }
}
```

**Resolution**: Use a domain from the allowed list or request new domain addition.

---

### E105: TOO_MANY_TOPICS
Claim has too many topic tags.

```typescript
{
  code: "E105",
  message: "Too many topics specified",
  details: {
    maxTopics: "20",
    actualTopics: "25"
  }
}
```

**Resolution**: Reduce topic tags to 20 or fewer.

---

### E106: TOO_MANY_EVIDENCE
Claim has too many evidence items.

```typescript
{
  code: "E106",
  message: "Too many evidence items",
  details: {
    maxEvidence: "50",
    actualEvidence: "62"
  }
}
```

**Resolution**: Reduce evidence items to 50 or fewer.

---

### E107: INVALID_WEIGHT
Relationship or dependency weight out of range.

```typescript
{
  code: "E107",
  message: "Weight must be between 0.0 and 1.0",
  details: { weight: "-0.5" }
}
```

**Resolution**: Use weight in range [0.0, 1.0].

---

### E108: SELF_REFERENCE
Attempting to create self-referential relationship.

```typescript
{
  code: "E108",
  message: "Cannot create self-referential relationship",
  details: { claimId: "uhCkk..." }
}
```

**Resolution**: Use different source and target claims.

---

### E109: INVALID_VERDICT
Invalid fact-check verdict value.

```typescript
{
  code: "E109",
  message: "Invalid verdict value",
  details: {
    provided: "maybe",
    allowed: "True,MostlyTrue,Mixed,MostlyFalse,False,Unverifiable,InsufficientEvidence"
  }
}
```

---

## Authorization Errors (E2xx)

### E200: NOT_AUTHORIZED
General authorization failure.

```typescript
{
  code: "E200",
  message: "Not authorized to perform this action",
  details: null
}
```

---

### E201: NOT_AUTHOR
Attempting to modify a claim you didn't author.

```typescript
{
  code: "E201",
  message: "Only the original author can modify this claim",
  details: {
    author: "uhCAk...",
    caller: "uhCAk..."
  }
}
```

**Resolution**: Only the original author can update/delete claims.

---

### E202: HAPP_NOT_REGISTERED
External hApp attempting access without registration.

```typescript
{
  code: "E202",
  message: "hApp not registered for bridge access",
  details: { happId: "my-happ" }
}
```

**Resolution**: Register your hApp via the bridge zome first.

---

### E203: INSUFFICIENT_PERMISSION
hApp lacks required permission level.

```typescript
{
  code: "E203",
  message: "Insufficient permission level for this operation",
  details: {
    required: "Standard",
    actual: "Limited"
  }
}
```

**Resolution**: Request higher permission level for your hApp.

---

### E204: DOMAIN_ACCESS_DENIED
hApp attempting to write to unauthorized domain.

```typescript
{
  code: "E204",
  message: "Not authorized to write to this domain",
  details: {
    domain: "finance",
    allowedDomains: "energy,climate"
  }
}
```

**Resolution**: Only write to domains your hApp is authorized for.

---

## Not Found Errors (E3xx)

### E300: CLAIM_NOT_FOUND
Referenced claim doesn't exist.

```typescript
{
  code: "E300",
  message: "Claim not found",
  details: { claimId: "uhCkk..." }
}
```

**Resolution**: Verify the claim ID is correct.

---

### E301: RELATIONSHIP_NOT_FOUND
Referenced relationship doesn't exist.

```typescript
{
  code: "E301",
  message: "Relationship not found",
  details: { relationshipId: "uhCkk..." }
}
```

---

### E302: AUTHOR_NOT_FOUND
Referenced author doesn't exist.

```typescript
{
  code: "E302",
  message: "Author not found",
  details: { did: "did:key:..." }
}
```

---

### E303: MARKET_NOT_FOUND
Referenced market doesn't exist.

```typescript
{
  code: "E303",
  message: "Market not found",
  details: { marketId: "market-123" }
}
```

---

### E304: NO_RELEVANT_CLAIMS
Fact-check found no relevant claims.

```typescript
{
  code: "E304",
  message: "No relevant claims found for fact-check",
  details: {
    statement: "The sky is purple",
    searchTerms: "sky,purple,color"
  }
}
```

**Resolution**: The knowledge graph has no information on this topic.

---

### E305: FACT_CHECK_REQUEST_NOT_FOUND
Fact-check request doesn't exist.

```typescript
{
  code: "E305",
  message: "Fact-check request not found",
  details: { requestId: "fc-123" }
}
```

---

## Conflict Errors (E4xx)

### E400: DUPLICATE_CLAIM
Attempting to create a duplicate claim.

```typescript
{
  code: "E400",
  message: "A similar claim already exists",
  details: {
    existingClaimId: "uhCkk...",
    similarity: "0.95"
  }
}
```

**Resolution**: Reference the existing claim instead.

---

### E401: RELATIONSHIP_EXISTS
Relationship between claims already exists.

```typescript
{
  code: "E401",
  message: "Relationship already exists between these claims",
  details: {
    sourceId: "uhCkk...",
    targetId: "uhCkk...",
    existingType: "Supports"
  }
}
```

**Resolution**: Update the existing relationship instead.

---

### E402: CIRCULAR_DEPENDENCY
Creating this dependency would create a cycle.

```typescript
{
  code: "E402",
  message: "Cannot create circular dependency",
  details: {
    cycle: "A -> B -> C -> A"
  }
}
```

**Resolution**: Restructure dependencies to avoid cycles.

---

### E403: MARKET_ALREADY_EXISTS
Verification market already exists for this claim.

```typescript
{
  code: "E403",
  message: "Active verification market already exists",
  details: {
    claimId: "uhCkk...",
    existingMarketId: "market-123"
  }
}
```

**Resolution**: Wait for existing market to resolve.

---

### E404: CLAIM_STATUS_CONFLICT
Claim status doesn't allow requested operation.

```typescript
{
  code: "E404",
  message: "Operation not allowed in current claim status",
  details: {
    currentStatus: "Retracted",
    operation: "spawn_verification_market"
  }
}
```

**Resolution**: Only active claims can have markets.

---

### E405: MARKET_CLOSED
Market has already closed.

```typescript
{
  code: "E405",
  message: "Market has closed and cannot accept new operations",
  details: {
    marketId: "market-123",
    closedAt: "1704067200000"
  }
}
```

---

## External Service Errors (E5xx)

### E500: MARKETS_UNAVAILABLE
Epistemic Markets service unavailable.

```typescript
{
  code: "E500",
  message: "Epistemic Markets service unavailable",
  details: {
    lastAttempt: "1704067200000",
    retryAfter: "60"
  }
}
```

**Resolution**: Retry after the specified interval.

---

### E501: MARKETS_CREATION_FAILED
Failed to create verification market.

```typescript
{
  code: "E501",
  message: "Failed to create verification market",
  details: {
    reason: "Insufficient subsidy",
    minimumSubsidy: "100"
  }
}
```

---

### E502: BRIDGE_COMMUNICATION_FAILED
Cross-hApp bridge communication failed.

```typescript
{
  code: "E502",
  message: "Bridge communication failed",
  details: {
    targetHapp: "media",
    error: "Connection timeout"
  }
}
```

---

### E503: IDENTITY_VERIFICATION_FAILED
Failed to verify caller identity.

```typescript
{
  code: "E503",
  message: "Identity verification failed",
  details: null
}
```

---

## Rate Limit Errors (E6xx)

### E600: RATE_LIMITED
Too many requests.

```typescript
{
  code: "E600",
  message: "Rate limit exceeded",
  details: {
    limit: "100",
    period: "60",
    retryAfter: "45"
  }
}
```

**Resolution**: Wait before retrying. Consider caching results.

---

### E601: CLAIM_CREATION_LIMIT
Too many claims created in time period.

```typescript
{
  code: "E601",
  message: "Claim creation rate limit exceeded",
  details: {
    limit: "10",
    period: "3600",
    remaining: "0"
  }
}
```

**Resolution**: Wait before creating more claims.

---

### E602: FACT_CHECK_LIMIT
Too many fact-check requests.

```typescript
{
  code: "E602",
  message: "Fact-check rate limit exceeded",
  details: {
    limit: "50",
    period: "3600"
  }
}
```

---

## Internal Errors (E9xx)

### E900: INTERNAL_ERROR
Unexpected internal error.

```typescript
{
  code: "E900",
  message: "Internal error occurred",
  details: {
    errorId: "err-abc123"  // For support
  }
}
```

**Resolution**: Report to developers with the errorId.

---

### E901: DHT_ERROR
DHT operation failed.

```typescript
{
  code: "E901",
  message: "DHT operation failed",
  details: {
    operation: "get",
    hash: "uhCkk..."
  }
}
```

---

### E902: SERIALIZATION_ERROR
Data serialization/deserialization failed.

```typescript
{
  code: "E902",
  message: "Serialization error",
  details: {
    type: "Claim",
    error: "Invalid UTF-8"
  }
}
```

---

### E903: PROPAGATION_FAILED
Belief propagation failed to converge.

```typescript
{
  code: "E903",
  message: "Belief propagation failed to converge",
  details: {
    iterations: "1000",
    maxChange: "0.05"
  }
}
```

---

### E904: CASCADE_FAILED
Cascade update failed.

```typescript
{
  code: "E904",
  message: "Cascade update failed",
  details: {
    claimsAttempted: "50",
    claimsSucceeded: "45",
    failures: ["uhCkk...1", "uhCkk...2"]
  }
}
```

---

## SDK Error Handling

### TypeScript

```typescript
import { KnowledgeClient, KnowledgeError } from '@mycelix/knowledge-sdk';

try {
  await knowledge.claims.createClaim(input);
} catch (error) {
  if (error.code === 'E101') {
    // Invalid classification
    console.error('Fix your E-N-M values');
  } else if (error.code === 'E600') {
    // Rate limited
    const retryAfter = parseInt(error.details?.retryAfter || '60');
    await sleep(retryAfter * 1000);
    // Retry...
  } else if (error.code.startsWith('E9')) {
    // Internal error - report to developers
    reportError(error);
  }
}
```

### Error Recovery Patterns

```typescript
// Retry with exponential backoff
async function withRetry<T>(
  fn: () => Promise<T>,
  maxRetries: number = 3
): Promise<T> {
  for (let i = 0; i < maxRetries; i++) {
    try {
      return await fn();
    } catch (error) {
      if (error.code === 'E600' || error.code?.startsWith('E5')) {
        // Retryable error
        const delay = Math.pow(2, i) * 1000;
        await sleep(delay);
        continue;
      }
      throw error;  // Non-retryable
    }
  }
  throw new Error('Max retries exceeded');
}

// Use it
const claim = await withRetry(() => knowledge.claims.createClaim(input));
```

---

## Reporting Errors

When reporting errors, include:
1. Error code and message
2. The `errorId` if present (E9xx errors)
3. Input that caused the error
4. Timestamp and context

Report issues at: https://github.com/mycelix/knowledge/issues

---

*Complete error reference for Knowledge hApp.*
