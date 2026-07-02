# ADR 0002: Consistent API Response Envelope

## Status
Accepted

## Context
Clients consuming our API need predictable response structures for:
- Successful data retrieval
- Error handling
- Pagination metadata
- Performance metrics

Different response formats make client implementation complex and error-prone.

## Decision
All API responses will follow a consistent envelope structure:

```typescript
interface ApiResponse<T> {
  success: boolean;
  data?: T;
  error?: {
    code: string;
    message: string;
    details?: unknown;
  };
  meta?: {
    pagination?: {
      total: number;
      limit: number;
      offset: number;
      hasMore: boolean;
      nextCursor?: string;
    };
    timing?: {
      startedAt: string;
      duration: number;
    };
    version?: string;
    requestId?: string;
  };
}
```

### Error Codes
We define standard error codes for client handling:
- `BAD_REQUEST` - Invalid input
- `UNAUTHORIZED` - Authentication required
- `FORBIDDEN` - Permission denied
- `NOT_FOUND` - Resource not found
- `VALIDATION_ERROR` - Field validation failed
- `RATE_LIMITED` - Too many requests
- `INTERNAL_ERROR` - Server error

### Implementation
- `ResponseBuilder` class for fluent response construction
- Helper functions: `success()`, `paginated()`, `error()`, `errors.*`
- Automatic metadata injection (timing, requestId, version)

## Consequences

### Positive
- Clients always know where to find data, errors, and metadata
- Error handling is consistent across all endpoints
- Easy to add new metadata without breaking changes
- Response builder makes code cleaner

### Negative
- Slightly larger response payloads
- All endpoints must use the response utilities
- Breaking change for existing clients (if any)

### Migration
For existing endpoints, wrap responses in the envelope structure during v2 API migration.
