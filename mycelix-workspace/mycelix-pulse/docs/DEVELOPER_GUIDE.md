# Mycelix Mail Developer Guide

This guide covers everything you need to get started developing on Mycelix Mail.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Development Setup](#development-setup)
3. [Project Structure](#project-structure)
4. [Backend Development](#backend-development)
5. [Frontend Development](#frontend-development)
6. [Database](#database)
7. [Testing](#testing)
8. [Deployment](#deployment)

---

## Architecture Overview

Mycelix Mail is a privacy-focused email system with these core components:

```
┌─────────────────────────────────────────────────────────────────┐
│                        Client Layer                              │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│   Web App    │  Mobile App  │  Desktop App │   Browser Ext     │
│  (React/TS)  │(React Native)│  (Electron)  │    (Chrome)       │
└──────┬───────┴──────┬───────┴──────┬───────┴─────────┬─────────┘
       │              │              │                  │
       └──────────────┴──────────────┴──────────────────┘
                              │
                    ┌─────────▼─────────┐
                    │   API Gateway     │
                    │  (Rate Limiting)  │
                    └─────────┬─────────┘
                              │
┌─────────────────────────────▼─────────────────────────────────┐
│                      Backend Services                          │
├───────────────┬───────────────┬───────────────┬──────────────┤
│  Email API    │  Trust API    │  Search API   │  Auth API    │
│   (Rust)      │   (Rust)      │  (Rust+Meil)  │  (Rust)      │
└───────┬───────┴───────┬───────┴───────┬───────┴──────┬───────┘
        │               │               │              │
        └───────────────┴───────────────┴──────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐   ┌─────────────────┐   ┌─────────────────┐
│  PostgreSQL   │   │   Holochain     │   │     Redis       │
│  (Primary)    │   │  (Trust DHT)    │   │   (Cache/Queue) │
└───────────────┘   └─────────────────┘   └─────────────────┘
```

### Key Technologies

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Backend | Rust + Axum | High-performance API server |
| Frontend | React + TypeScript | Modern web application |
| Mobile | React Native + Expo | Cross-platform mobile |
| Desktop | Electron | Native desktop experience |
| Database | PostgreSQL | Primary data store |
| Cache | Redis | Caching and job queue |
| Search | Meilisearch | Full-text email search |
| Trust | Holochain | Decentralized trust network |
| Encryption | CRYSTALS-Kyber/Dilithium | Post-quantum cryptography |

---

## Development Setup

### Prerequisites

- Rust 1.75+ (`rustup install stable`)
- Node.js 20+ (recommend using `fnm` or `nvm`)
- PostgreSQL 15+
- Redis 7+
- Docker & Docker Compose (optional but recommended)

### Quick Start with Docker

```bash
# Clone the repository
git clone https://github.com/luminous-dynamics/mycelix-mail.git
cd mycelix-mail

# Start all services
docker-compose up -d

# Run database migrations
cargo sqlx migrate run

# Start the backend
cargo run --release

# In another terminal, start the frontend
cd ui/frontend
npm install
npm run dev
```

### Manual Setup

1. **Database**
   ```bash
   # Create database
   createdb mycelix_mail

   # Set environment variable
   export DATABASE_URL=postgres://localhost/mycelix_mail

   # Run migrations
   cargo sqlx migrate run
   ```

2. **Redis**
   ```bash
   # Start Redis
   redis-server
   ```

3. **Backend**
   ```bash
   cd backend/api
   cp .env.example .env
   # Edit .env with your configuration
   cargo run
   ```

4. **Frontend**
   ```bash
   cd ui/frontend
   npm install
   npm run dev
   ```

### Environment Variables

Create a `.env` file in the backend directory:

```env
# Database
DATABASE_URL=postgres://user:pass@localhost/mycelix_mail

# Redis
REDIS_URL=redis://localhost:6379

# JWT
JWT_SECRET=your-secret-key-here
JWT_EXPIRATION=3600

# SMTP
SMTP_HOST=localhost
SMTP_PORT=587
SMTP_USER=
SMTP_PASS=

# Holochain
HOLOCHAIN_URL=ws://localhost:8888

# Telemetry (optional)
OTLP_ENDPOINT=http://localhost:4317
SENTRY_DSN=
```

---

## Project Structure

```
mycelix-mail/
├── backend/
│   ├── api/                 # Main API server
│   │   ├── src/
│   │   │   ├── auth/        # Authentication
│   │   │   ├── email/       # Email operations
│   │   │   ├── trust/       # Trust network
│   │   │   ├── search/      # Search functionality
│   │   │   ├── middleware/  # Request middleware
│   │   │   ├── webhooks/    # Webhook system
│   │   │   ├── plugins/     # Plugin architecture
│   │   │   ├── gdpr/        # GDPR compliance
│   │   │   └── main.rs
│   │   ├── tests/
│   │   └── Cargo.toml
│   └── shared/              # Shared libraries
│
├── ui/
│   └── frontend/            # React web app
│       ├── src/
│       │   ├── components/
│       │   ├── pages/
│       │   ├── hooks/
│       │   ├── lib/
│       │   └── App.tsx
│       └── package.json
│
├── mobile/                  # React Native app
├── desktop/                 # Electron app
├── holochain/               # Trust network DNA
├── docs/                    # Documentation
│   ├── api/                 # OpenAPI specs
│   └── adr/                 # Architecture decisions
├── tests/
│   ├── e2e/                 # Playwright tests
│   └── load/                # k6 load tests
└── docker-compose.yml
```

---

## Backend Development

### Creating a New Endpoint

1. Define the handler in the appropriate module:

```rust
// src/email/handlers.rs
pub async fn get_email(
    State(state): State<AppState>,
    Path(id): Path<Uuid>,
    auth: AuthUser,
) -> Result<Json<Email>, ApiError> {
    let email = state.email_service
        .get_email(id, auth.user_id)
        .await?
        .ok_or(ApiError::NotFound)?;

    Ok(Json(email))
}
```

2. Add the route:

```rust
// src/email/routes.rs
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/emails/:id", get(handlers::get_email))
        .route("/emails", post(handlers::send_email))
}
```

3. Register in main router:

```rust
// src/main.rs
let app = Router::new()
    .nest("/api/v1/emails", email::routes())
    .with_state(state);
```

### Database Operations

We use sqlx for type-safe database queries:

```rust
// Query with automatic type checking
let emails = sqlx::query_as::<_, Email>(
    r#"
    SELECT * FROM emails
    WHERE user_id = $1 AND folder_id = $2
    ORDER BY received_at DESC
    LIMIT $3
    "#
)
.bind(user_id)
.bind(folder_id)
.bind(limit)
.fetch_all(&pool)
.await?;
```

### Adding Migrations

```bash
# Create a new migration
cargo sqlx migrate add create_new_table

# Run migrations
cargo sqlx migrate run

# Generate query metadata (for offline mode)
cargo sqlx prepare
```

### Error Handling

Use the `ApiError` type for consistent error responses:

```rust
#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("Not found")]
    NotFound,

    #[error("Unauthorized")]
    Unauthorized,

    #[error("Validation error: {0}")]
    Validation(String),

    #[error("Internal error")]
    Internal(#[from] anyhow::Error),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let (status, message) = match self {
            Self::NotFound => (StatusCode::NOT_FOUND, "Not found"),
            Self::Unauthorized => (StatusCode::UNAUTHORIZED, "Unauthorized"),
            Self::Validation(msg) => (StatusCode::BAD_REQUEST, msg.as_str()),
            Self::Internal(_) => (StatusCode::INTERNAL_SERVER_ERROR, "Internal error"),
        };

        (status, Json(json!({ "error": message }))).into_response()
    }
}
```

---

## Frontend Development

### Component Structure

```typescript
// src/components/EmailList/EmailList.tsx
import { useEmails } from '@/hooks/useEmails';
import { EmailListItem } from './EmailListItem';

interface EmailListProps {
  folderId: string;
}

export function EmailList({ folderId }: EmailListProps) {
  const { data, isLoading, error } = useEmails(folderId);

  if (isLoading) return <LoadingSpinner />;
  if (error) return <ErrorMessage error={error} />;

  return (
    <div className="email-list">
      {data?.emails.map((email) => (
        <EmailListItem key={email.id} email={email} />
      ))}
    </div>
  );
}
```

### State Management

We use React Query for server state:

```typescript
// src/hooks/useEmails.ts
import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { emailApi } from '@/lib/api';

export function useEmails(folderId: string) {
  return useQuery({
    queryKey: ['emails', folderId],
    queryFn: () => emailApi.list(folderId),
  });
}

export function useSendEmail() {
  const queryClient = useQueryClient();

  return useMutation({
    mutationFn: emailApi.send,
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['emails'] });
    },
  });
}
```

### Styling

We use Tailwind CSS with a custom design system:

```typescript
// tailwind.config.js
module.exports = {
  theme: {
    extend: {
      colors: {
        primary: {
          50: '#f0f9ff',
          // ...
          900: '#0c4a6e',
        },
        trust: {
          high: '#22c55e',
          medium: '#eab308',
          low: '#ef4444',
        },
      },
    },
  },
};
```

---

## Database

### Schema Overview

Core tables:
- `users` - User accounts
- `emails` - Email messages
- `folders` - Email folders (inbox, sent, etc.)
- `contacts` - Contact information
- `attachments` - Email attachments
- `trust_scores` - Calculated trust scores
- `attestations` - Trust attestations

### Query Optimization

Use the built-in performance analyzer:

```rust
// Analyze slow queries
let analyzer = QueryAnalyzer::new(pool);
let slow_queries = analyzer.get_slow_queries(100.0).await?;
let suggestions = analyzer.suggest_indexes().await?;
```

Recommended indexes are defined in `src/performance/mod.rs`.

---

## Testing

### Unit Tests

```bash
# Run all tests
cargo test

# Run specific test
cargo test test_email_send

# Run with logging
RUST_LOG=debug cargo test -- --nocapture
```

### Integration Tests

```bash
# Run integration tests (requires Docker)
cargo test --test integration
```

### E2E Tests

```bash
cd ui/frontend

# Run Playwright tests
npx playwright test

# Run with UI
npx playwright test --ui

# Run specific test
npx playwright test inbox.spec.ts
```

### Load Tests

```bash
# Run k6 load tests
k6 run tests/load/k6-load-test.js

# With custom options
k6 run --vus 50 --duration 5m tests/load/k6-load-test.js
```

---

## Deployment

### Docker Deployment

```bash
# Build images
docker-compose -f docker-compose.prod.yml build

# Deploy
docker-compose -f docker-compose.prod.yml up -d
```

### Kubernetes

Helm charts are available in `deploy/helm/`:

```bash
helm install mycelix-mail ./deploy/helm/mycelix-mail \
  --set database.host=postgres.default.svc \
  --set redis.host=redis.default.svc
```

### Environment Configuration

Production settings:
- Use managed PostgreSQL (RDS, Cloud SQL)
- Use managed Redis (ElastiCache, Memorystore)
- Enable HTTPS with proper certificates
- Configure rate limiting appropriately
- Set up monitoring with Prometheus/Grafana

---

## Additional Resources

- [API Documentation](./api/openapi.yaml)
- [Architecture Decision Records](./adr/)
- [Contributing Guide](../CONTRIBUTING.md)
- [Security Policy](../SECURITY.md)

## Getting Help

- GitHub Issues: Report bugs and feature requests
- Discussions: Ask questions and share ideas
- Discord: Real-time community chat

---

Happy coding!
