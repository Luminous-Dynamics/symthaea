# ADR 0004: Audit Logging for Security and Compliance

## Status
Accepted

## Context
As a platform handling payments and user data, we need:
- Security monitoring and incident investigation
- Compliance with data protection requirements
- Understanding of user behavior patterns
- Debugging of production issues

## Decision
Implement a comprehensive **audit logging system** with:

### Event Types
```typescript
enum AuditEventType {
  // Authentication
  AUTH_LOGIN, AUTH_LOGOUT, AUTH_FAILED, AUTH_TOKEN_REFRESH,

  // Resources
  RESOURCE_READ, RESOURCE_CREATE, RESOURCE_UPDATE, RESOURCE_DELETE,

  // Business events
  SONG_REGISTERED, PAYMENT_PROCESSED, PAYMENT_FAILED, CLAIM_CREATED,

  // Security
  RATE_LIMITED, INVALID_INPUT, UNAUTHORIZED_ACCESS, SUSPICIOUS_ACTIVITY
}
```

### Event Structure
```typescript
interface AuditEvent {
  timestamp: string;
  eventType: AuditEventType;
  severity: 'info' | 'warning' | 'error' | 'critical';
  requestId?: string;
  userId?: string;
  walletAddress?: string;
  ip: string;
  userAgent: string;
  method: string;
  path: string;
  statusCode?: number;
  resourceType?: string;
  resourceId?: string;
  details?: Record<string, unknown>;
  duration?: number;
}
```

### Implementation
1. **Middleware-based** for automatic request logging
2. **Helper functions** for explicit business event logging
3. **Pluggable backends** (console, JSON file, external service)
4. **Non-blocking** - audit failures don't break requests

### What We Log
| Event | When | Severity |
|-------|------|----------|
| Auth attempts | All | info/warning |
| Payment operations | All | info/error |
| Song registration | All | info |
| Rate limiting | Triggered | warning |
| 5xx errors | All | error |
| Admin actions | All | info |

### What We Don't Log
- Request/response bodies (privacy)
- Health check endpoints (noise)
- Successful GET requests to public endpoints (volume)

## Consequences

### Positive
- Full audit trail for security incidents
- Compliance with data protection requirements
- Useful for debugging and analytics
- Pattern detection for fraud/abuse

### Negative
- Storage requirements for logs
- Performance overhead (minimal due to async)
- Privacy considerations for stored data

### Data Retention
- Security events: 1 year
- Business events: 2 years
- General access logs: 30 days

### Future Enhancements
- Real-time alerting on suspicious patterns
- Integration with SIEM systems
- Anonymization for analytics use
