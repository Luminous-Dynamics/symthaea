# 🔐 Mycelix ERP - Authentication & Multi-Tenancy Design

**Enterprise-Grade Security + Multi-Tenant Architecture**

---

## 🎯 Design Goals

1. **Enterprise Security**: OAuth 2.0 + OIDC + MFA
2. **Multi-Tenancy**: Complete data isolation between companies
3. **Role-Based Access Control (RBAC)**: Fine-grained permissions
4. **API Keys**: For programmatic access
5. **Audit Logging**: Every action logged with cryptographic proof
6. **SSO Support**: SAML 2.0 + OAuth for enterprise
7. **Zero-Trust Architecture**: Verify every request

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    CLIENT REQUEST                        │
│          (Browser/API with JWT or API Key)              │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               AUTHENTICATION LAYER                       │
│  ┌─────────────┐  ┌──────────────┐  ┌───────────────┐  │
│  │   JWT       │  │   API Keys   │  │   OAuth 2.0   │  │
│  │ Validator   │  │  Validator   │  │   Provider    │  │
│  └─────────────┘  └──────────────┘  └───────────────┘  │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│               TENANT RESOLUTION                          │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Extract tenant_id from:                        │    │
│  │  - JWT claims                                   │    │
│  │  - API key metadata                             │    │
│  │  - Subdomain (acme.mycelix.net)                 │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│           AUTHORIZATION & RBAC                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Check permissions:                             │    │
│  │  - User roles (admin, accountant, viewer)       │    │
│  │  - Resource ownership                           │    │
│  │  - Field-level permissions                      │    │
│  └─────────────────────────────────────────────────┘    │
└──────────────────────┬──────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────┐
│              DATA ACCESS LAYER                           │
│  ┌─────────────────────────────────────────────────┐    │
│  │  All queries filtered by tenant_id:             │    │
│  │  SELECT * FROM invoices WHERE tenant_id = $1    │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 📊 Database Schema

### **Core Auth Tables**

```sql
-- Tenants (Companies)
CREATE TABLE tenants (
    id UUID PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    subdomain VARCHAR(100) UNIQUE NOT NULL,  -- acme.mycelix.net
    plan VARCHAR(50) NOT NULL,  -- free, pro, enterprise
    status VARCHAR(50) NOT NULL DEFAULT 'active',  -- active, suspended, cancelled
    settings JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Users
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) NOT NULL UNIQUE,
    email_verified BOOLEAN NOT NULL DEFAULT false,
    password_hash VARCHAR(255),  -- NULL if OAuth-only
    full_name VARCHAR(255) NOT NULL,
    avatar_url VARCHAR(500),
    phone_number VARCHAR(50),
    phone_verified BOOLEAN NOT NULL DEFAULT false,
    mfa_enabled BOOLEAN NOT NULL DEFAULT false,
    mfa_secret VARCHAR(100),  -- TOTP secret
    last_login_at TIMESTAMPTZ,
    last_login_ip INET,
    status VARCHAR(50) NOT NULL DEFAULT 'active',  -- active, suspended, deleted
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- User-Tenant Relationship (Many-to-Many)
CREATE TABLE user_tenants (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    role VARCHAR(50) NOT NULL,  -- owner, admin, accountant, viewer
    permissions JSONB NOT NULL DEFAULT '{}',  -- Custom permissions
    is_primary BOOLEAN NOT NULL DEFAULT false,  -- Primary tenant for user
    joined_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_accessed_at TIMESTAMPTZ,
    UNIQUE(user_id, tenant_id)
);

-- Sessions
CREATE TABLE sessions (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    tenant_id UUID REFERENCES tenants(id) ON DELETE CASCADE,
    token_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 of JWT
    expires_at TIMESTAMPTZ NOT NULL,
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- API Keys (for programmatic access)
CREATE TABLE api_keys (
    id UUID PRIMARY KEY,
    tenant_id UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    name VARCHAR(255) NOT NULL,
    key_prefix VARCHAR(20) NOT NULL UNIQUE,  -- First 8 chars for display
    key_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256 of full key
    scopes JSONB NOT NULL DEFAULT '[]',  -- ["invoices:read", "payments:write"]
    last_used_at TIMESTAMPTZ,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    revoked_at TIMESTAMPTZ
);

-- OAuth Connections (for SSO)
CREATE TABLE oauth_connections (
    id UUID PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    provider VARCHAR(50) NOT NULL,  -- google, microsoft, okta
    provider_user_id VARCHAR(255) NOT NULL,
    access_token_encrypted TEXT,
    refresh_token_encrypted TEXT,
    expires_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE(provider, provider_user_id)
);

-- Audit Log
CREATE TABLE audit_log (
    id UUID PRIMARY KEY,
    tenant_id UUID REFERENCES tenants(id) ON DELETE SET NULL,
    user_id UUID REFERENCES users(id) ON DELETE SET NULL,
    action VARCHAR(100) NOT NULL,  -- "invoice.created", "payment.deleted"
    resource_type VARCHAR(50) NOT NULL,  -- "invoice", "payment"
    resource_id UUID,
    details JSONB NOT NULL DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Indexes for performance
CREATE INDEX idx_user_tenants_user ON user_tenants(user_id);
CREATE INDEX idx_user_tenants_tenant ON user_tenants(tenant_id);
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_token ON sessions(token_hash);
CREATE INDEX idx_api_keys_tenant ON api_keys(tenant_id);
CREATE INDEX idx_audit_log_tenant ON audit_log(tenant_id);
CREATE INDEX idx_audit_log_user ON audit_log(user_id);
CREATE INDEX idx_audit_log_created ON audit_log(created_at DESC);
```

---

## 🔑 JWT Structure

```json
{
  "sub": "550e8400-e29b-41d4-a716-446655440000",  // user_id
  "email": "john@acme.com",
  "tenant_id": "650e8400-e29b-41d4-a716-446655440001",
  "tenant_subdomain": "acme",
  "role": "admin",
  "permissions": {
    "invoices": ["read", "write", "delete"],
    "payments": ["read", "write"],
    "reports": ["read"]
  },
  "iat": 1735564800,  // Issued at
  "exp": 1735651200,  // Expires (24 hours)
  "iss": "mycelix-auth",
  "aud": "mycelix-api"
}
```

**Signing**: RS256 (RSA 2048-bit private key)

---

## 🔐 Authentication Flows

### **1. Email/Password Login**

```rust
// POST /auth/login
#[derive(Deserialize)]
struct LoginRequest {
    email: String,
    password: String,
    tenant_subdomain: Option<String>,  // If multi-tenant at subdomain level
}

async fn login(
    State(auth_service): State<AuthService>,
    Json(req): Json<LoginRequest>,
) -> Result<Json<LoginResponse>, AuthError> {
    // 1. Find user by email
    let user = auth_service.find_user_by_email(&req.email).await?
        .ok_or(AuthError::InvalidCredentials)?;

    // 2. Verify password (using argon2)
    auth_service.verify_password(&user.password_hash, &req.password)?;

    // 3. Check if MFA enabled
    if user.mfa_enabled {
        return Ok(Json(LoginResponse::MfaRequired {
            user_id: user.id,
            methods: vec!["totp", "sms"],
        }));
    }

    // 4. Get tenant (either from subdomain or user's primary)
    let tenant = if let Some(subdomain) = req.tenant_subdomain {
        auth_service.get_tenant_by_subdomain(&subdomain).await?
    } else {
        auth_service.get_user_primary_tenant(user.id).await?
    };

    // 5. Verify user has access to this tenant
    let user_tenant = auth_service
        .get_user_tenant(user.id, tenant.id).await?
        .ok_or(AuthError::TenantAccessDenied)?;

    // 6. Generate JWT
    let jwt = auth_service.generate_jwt(&user, &tenant, &user_tenant)?;

    // 7. Create session
    auth_service.create_session(&user, &tenant, &jwt).await?;

    // 8. Update last login
    auth_service.update_last_login(&user).await?;

    Ok(Json(LoginResponse::Success {
        access_token: jwt,
        token_type: "Bearer".to_string(),
        expires_in: 86400,  // 24 hours
        user: UserInfo::from(user),
        tenant: TenantInfo::from(tenant),
    }))
}
```

### **2. MFA Verification**

```rust
// POST /auth/mfa/verify
#[derive(Deserialize)]
struct MfaVerifyRequest {
    user_id: Uuid,
    code: String,  // 6-digit TOTP code
}

async fn verify_mfa(
    State(auth_service): State<AuthService>,
    Json(req): Json<MfaVerifyRequest>,
) -> Result<Json<LoginResponse>, AuthError> {
    // 1. Get user
    let user = auth_service.get_user(req.user_id).await?;

    // 2. Verify TOTP code
    auth_service.verify_totp(&user.mfa_secret, &req.code)?;

    // 3. Continue with login flow...
    // (same as steps 4-8 from login)
}
```

### **3. OAuth 2.0 Login (Google, Microsoft)**

```rust
// GET /auth/oauth/google
async fn oauth_google_login(
    State(auth_service): State<AuthService>,
    Query(params): Query<OAuthParams>,
) -> Redirect {
    // 1. Generate OAuth state (CSRF protection)
    let state = auth_service.generate_oauth_state().await;

    // 2. Build Google OAuth URL
    let oauth_url = format!(
        "https://accounts.google.com/o/oauth2/v2/auth?\
         client_id={}&\
         redirect_uri={}&\
         response_type=code&\
         scope=openid email profile&\
         state={}",
        GOOGLE_CLIENT_ID,
        REDIRECT_URI,
        state
    );

    Redirect::to(&oauth_url)
}

// GET /auth/oauth/callback
async fn oauth_callback(
    State(auth_service): State<AuthService>,
    Query(params): Query<OAuthCallbackParams>,
) -> Result<Redirect, AuthError> {
    // 1. Verify state (CSRF protection)
    auth_service.verify_oauth_state(&params.state).await?;

    // 2. Exchange code for access token
    let token_response = auth_service
        .exchange_oauth_code(&params.code).await?;

    // 3. Get user info from Google
    let google_user = auth_service
        .get_google_user_info(&token_response.access_token).await?;

    // 4. Find or create user
    let user = auth_service
        .find_or_create_user_from_oauth(&google_user).await?;

    // 5. Create or update OAuth connection
    auth_service.save_oauth_connection(&user, &google_user, &token_response).await?;

    // 6. Generate JWT and redirect
    // ...
}
```

### **4. API Key Authentication**

```rust
// Middleware for API key auth
async fn api_key_middleware(
    mut req: Request,
    next: Next,
) -> Result<Response, StatusCode> {
    // 1. Extract API key from header
    let api_key = req.headers()
        .get("X-API-Key")
        .and_then(|v| v.to_str().ok())
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // 2. Hash the key
    let key_hash = sha256(api_key);

    // 3. Lookup in database
    let api_key_record = auth_service
        .get_api_key_by_hash(&key_hash).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::UNAUTHORIZED)?;

    // 4. Check if revoked or expired
    if api_key_record.revoked_at.is_some() {
        return Err(StatusCode::UNAUTHORIZED);
    }
    if let Some(expires) = api_key_record.expires_at {
        if expires < Utc::now() {
            return Err(StatusCode::UNAUTHORIZED);
        }
    }

    // 5. Check scopes
    let required_scope = get_required_scope(&req.uri().path());
    if !api_key_record.scopes.contains(&required_scope) {
        return Err(StatusCode::FORBIDDEN);
    }

    // 6. Add tenant context to request
    req.extensions_mut().insert(TenantContext {
        tenant_id: api_key_record.tenant_id,
        user_id: api_key_record.user_id,
        scopes: api_key_record.scopes.clone(),
    });

    // 7. Update last_used_at
    auth_service.update_api_key_last_used(api_key_record.id).await.ok();

    Ok(next.run(req).await)
}
```

---

## 🏢 Multi-Tenancy Implementation

### **Strategy: Shared Database with tenant_id Column**

**Why?**
- ✅ Cost-effective (one database)
- ✅ Easy backups
- ✅ Simple maintenance
- ✅ Good for 0-1000 tenants

**How?**
Every table gets a `tenant_id` column:

```sql
-- Add tenant_id to all tables
ALTER TABLE gl_accounts ADD COLUMN tenant_id UUID NOT NULL REFERENCES tenants(id);
ALTER TABLE invoices ADD COLUMN tenant_id UUID NOT NULL REFERENCES tenants(id);
ALTER TABLE payments ADD COLUMN tenant_id UUID NOT NULL REFERENCES tenants(id);
ALTER TABLE journal_entries ADD COLUMN tenant_id UUID NOT NULL REFERENCES tenants(id);

-- Add composite indexes for performance
CREATE INDEX idx_gl_accounts_tenant ON gl_accounts(tenant_id, id);
CREATE INDEX idx_invoices_tenant ON invoices(tenant_id, id);
CREATE INDEX idx_payments_tenant ON payments(tenant_id, id);

-- Row-Level Security (RLS) for extra safety
ALTER TABLE invoices ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation_policy ON invoices
    USING (tenant_id = current_setting('app.tenant_id')::uuid);
```

### **Automatic Tenant Filtering**

```rust
// Middleware injects tenant_id into all queries
pub struct TenantContext {
    pub tenant_id: Uuid,
    pub user_id: Option<Uuid>,
}

// All database queries automatically filtered
async fn list_invoices(
    State(db): State<PgPool>,
    Extension(ctx): Extension<TenantContext>,
) -> Result<Json<Vec<Invoice>>, Error> {
    let invoices = sqlx::query_as::<_, Invoice>(
        "SELECT * FROM invoices WHERE tenant_id = $1 ORDER BY created_at DESC"
    )
    .bind(ctx.tenant_id)  // ✅ Automatic tenant isolation
    .fetch_all(&db)
    .await?;

    Ok(Json(invoices))
}

// Use PostgreSQL RLS for defense-in-depth
async fn set_tenant_context(pool: &PgPool, tenant_id: Uuid) -> Result<()> {
    sqlx::query("SET app.tenant_id = $1")
        .bind(tenant_id)
        .execute(pool)
        .await?;
    Ok(())
}
```

---

## 🎭 Role-Based Access Control (RBAC)

### **Built-in Roles**

```rust
#[derive(Debug, Serialize, Deserialize)]
enum Role {
    // Tenant roles
    Owner,       // Full access, can delete tenant
    Admin,       // Full access except tenant deletion
    Accountant,  // FIN module full access, SCM read-only
    Manager,     // All modules read/write, no admin
    Viewer,      // Read-only across all modules

    // Module-specific roles
    ScmManager,  // SCM module only
    FinManager,  // FIN module only
    HrManager,   // HR module only (future)
}

impl Role {
    fn permissions(&self) -> Vec<Permission> {
        match self {
            Role::Owner => vec![Permission::All],
            Role::Admin => vec![
                Permission::Read(Resource::All),
                Permission::Write(Resource::All),
                Permission::Delete(Resource::All),
            ],
            Role::Accountant => vec![
                Permission::Read(Resource::All),
                Permission::Write(Resource::Finance),
                Permission::Delete(Resource::Finance),
            ],
            Role::Manager => vec![
                Permission::Read(Resource::All),
                Permission::Write(Resource::All),
            ],
            Role::Viewer => vec![
                Permission::Read(Resource::All),
            ],
            // ...
        }
    }
}
```

### **Permission Checking**

```rust
// Middleware for permission checks
async fn require_permission(
    req: Request,
    next: Next,
    required_permission: Permission,
) -> Result<Response, StatusCode> {
    let ctx = req.extensions()
        .get::<TenantContext>()
        .ok_or(StatusCode::UNAUTHORIZED)?;

    let user_tenant = get_user_tenant(ctx.user_id, ctx.tenant_id).await
        .map_err(|_| StatusCode::INTERNAL_SERVER_ERROR)?
        .ok_or(StatusCode::FORBIDDEN)?;

    let role: Role = user_tenant.role.parse().unwrap();

    if !role.permissions().contains(&required_permission) {
        return Err(StatusCode::FORBIDDEN);
    }

    Ok(next.run(req).await)
}

// Usage in routes
Router::new()
    .route("/v1/fin/invoices/:id", delete(delete_invoice))
    .layer(middleware::from_fn(|req, next| {
        require_permission(
            req,
            next,
            Permission::Delete(Resource::Finance)
        )
    }));
```

---

## 📝 Audit Logging

### **Automatic Audit Trail**

```rust
// Middleware logs all mutations
async fn audit_logger_middleware(
    req: Request,
    next: Next,
) -> Response {
    let method = req.method().clone();
    let path = req.uri().path().to_string();
    let ctx = req.extensions().get::<TenantContext>().cloned();

    let response = next.run(req).await;

    // Only log mutations (POST, PUT, PATCH, DELETE)
    if matches!(method, Method::POST | Method::PUT | Method::PATCH | Method::DELETE) {
        if let Some(ctx) = ctx {
            let (resource_type, action) = extract_resource_and_action(&path, &method);

            tokio::spawn(async move {
                create_audit_log(AuditLog {
                    tenant_id: Some(ctx.tenant_id),
                    user_id: ctx.user_id,
                    action,
                    resource_type,
                    details: json!({}),
                    ip_address: None,  // Extract from request
                    user_agent: None,  // Extract from request
                }).await.ok();
            });
        }
    }

    response
}

// Query audit logs
async fn get_audit_logs(
    State(db): State<PgPool>,
    Extension(ctx): Extension<TenantContext>,
    Query(params): Query<AuditLogQuery>,
) -> Result<Json<Vec<AuditLog>>, Error> {
    let logs = sqlx::query_as::<_, AuditLog>(
        r#"
        SELECT * FROM audit_log
        WHERE tenant_id = $1
        AND ($2::UUID IS NULL OR user_id = $2)
        AND ($3::TIMESTAMPTZ IS NULL OR created_at >= $3)
        AND ($4::TIMESTAMPTZ IS NULL OR created_at <= $4)
        ORDER BY created_at DESC
        LIMIT 100
        "#
    )
    .bind(ctx.tenant_id)
    .bind(params.user_id)
    .bind(params.start_date)
    .bind(params.end_date)
    .fetch_all(&db)
    .await?;

    Ok(Json(logs))
}
```

---

## 🔒 Security Best Practices

### **1. Password Security**

```rust
use argon2::{Argon2, PasswordHash, PasswordHasher, PasswordVerifier};
use argon2::password_hash::{rand_core::OsRng, SaltString};

// Hash password on registration
async fn hash_password(password: &str) -> Result<String, Error> {
    let salt = SaltString::generate(&mut OsRng);
    let argon2 = Argon2::default();
    let password_hash = argon2
        .hash_password(password.as_bytes(), &salt)?
        .to_string();
    Ok(password_hash)
}

// Verify password on login
async fn verify_password(hash: &str, password: &str) -> Result<bool, Error> {
    let parsed_hash = PasswordHash::new(hash)?;
    Ok(Argon2::default()
        .verify_password(password.as_bytes(), &parsed_hash)
        .is_ok())
}
```

### **2. JWT Security**

```rust
use jsonwebtoken::{encode, decode, Header, Algorithm, Validation};

// Generate JWT with RS256
fn generate_jwt(user: &User, tenant: &Tenant, user_tenant: &UserTenant) -> Result<String> {
    let claims = Claims {
        sub: user.id.to_string(),
        email: user.email.clone(),
        tenant_id: tenant.id.to_string(),
        tenant_subdomain: tenant.subdomain.clone(),
        role: user_tenant.role.clone(),
        permissions: user_tenant.permissions.clone(),
        exp: (Utc::now() + Duration::hours(24)).timestamp() as usize,
        iat: Utc::now().timestamp() as usize,
        iss: "mycelix-auth".to_string(),
        aud: "mycelix-api".to_string(),
    };

    let header = Header::new(Algorithm::RS256);
    let token = encode(&header, &claims, &PRIVATE_KEY)?;

    Ok(token)
}

// Validate JWT
fn validate_jwt(token: &str) -> Result<Claims> {
    let validation = Validation::new(Algorithm::RS256);
    let token_data = decode::<Claims>(token, &PUBLIC_KEY, &validation)?;
    Ok(token_data.claims)
}
```

### **3. API Key Security**

```rust
use rand::Rng;

// Generate API key
fn generate_api_key() -> String {
    let mut rng = rand::thread_rng();
    let key: String = (0..32)
        .map(|_| {
            let idx = rng.gen_range(0..62);
            "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                .chars().nth(idx).unwrap()
        })
        .collect();

    format!("mk_live_{}", key)  // Prefix for easy identification
}

// Store only hash
async fn create_api_key(
    tenant_id: Uuid,
    name: String,
    scopes: Vec<String>,
) -> Result<ApiKeyResponse> {
    let key = generate_api_key();
    let key_hash = sha256(&key);
    let key_prefix = &key[..12];  // First 12 chars for display

    sqlx::query(
        "INSERT INTO api_keys (id, tenant_id, name, key_prefix, key_hash, scopes)
         VALUES ($1, $2, $3, $4, $5, $6)"
    )
    .bind(Uuid::new_v4())
    .bind(tenant_id)
    .bind(&name)
    .bind(key_prefix)
    .bind(&key_hash)
    .bind(json!(scopes))
    .execute(&pool)
    .await?;

    // Return full key ONLY once
    Ok(ApiKeyResponse {
        key,  // User must save this!
        key_prefix: key_prefix.to_string(),
        scopes,
        message: "⚠️ Save this key now! It will not be shown again.".to_string(),
    })
}
```

---

## 🚀 Implementation Roadmap

### **Week 1-2: Core Auth**
- [ ] Database schema
- [ ] User registration/login
- [ ] JWT generation/validation
- [ ] Session management
- [ ] Password reset

### **Week 3-4: Multi-Tenancy**
- [ ] Tenant creation
- [ ] Subdomain routing
- [ ] Tenant-scoped queries
- [ ] User-tenant relationships
- [ ] Role-based access control

### **Week 5-6: Advanced Features**
- [ ] MFA (TOTP)
- [ ] API keys
- [ ] OAuth 2.0 (Google, Microsoft)
- [ ] SAML 2.0 (enterprise SSO)
- [ ] Audit logging

### **Week 7-8: Security & Compliance**
- [ ] Penetration testing
- [ ] SOC 2 readiness
- [ ] GDPR compliance
- [ ] Rate limiting
- [ ] IP whitelisting

---

## 📊 Success Metrics

- **Authentication Success Rate**: >99.9%
- **Session Duration**: 24 hours (configurable)
- **MFA Adoption**: >70% of enterprise users
- **API Key Usage**: >30% of requests
- **Audit Log Completeness**: 100% of mutations

---

**Status**: Design Complete, Ready for Implementation
**Priority**: Critical (blocks multi-tenant SaaS)
**Estimated Effort**: 8 weeks (1 engineer full-time)
**Dependencies**: PostgreSQL, Redis (for sessions)

🔐 **Security is not optional. Let's build this right!**
