# Phase 11: Operational Excellence & Deployment Readiness

**Date**: 2025-11-16
**Previous Phase**: Phase 10 (Production Hardening & Performance)
**Focus**: Operations, deployment, monitoring, production best practices
**Estimated Duration**: 3-4 hours
**Status**: 🚧 In Progress

---

## Overview

Phase 11 is the **final phase** to achieve full production readiness. This phase focuses on operational excellence - the critical infrastructure needed for deploying, monitoring, and maintaining the platform in production environments.

After Phase 11, the platform will be **deployment-ready** with:
- Complete operational tooling
- Zero-downtime deployment support
- Production monitoring and alerting
- Comprehensive deployment documentation
- 12-factor app compliance

---

## Success Criteria

### Must Have (Priority 1-3)
- ✅ Rate limiting on all endpoints
- ✅ Enhanced health checks (readiness + liveness)
- ✅ Graceful shutdown handling
- ✅ 12-factor configuration management
- ✅ Production deployment guide

### Should Have (Priority 4-5)
- ✅ Environment-based configuration
- ✅ Startup validation
- ✅ Docker configuration
- ✅ Kubernetes manifests
- ✅ Monitoring best practices

### Nice to Have (Priority 6)
- 📋 Grafana dashboard templates
- 📋 Alert rule examples
- 📋 Load testing guide
- 📋 Backup/restore procedures

---

## Priority 1: Rate Limiting (40 min)

### Objective
Implement production-grade rate limiting to protect API from abuse.

### Dependencies

**Cargo.toml**:
```toml
[dependencies]
governor = "0.6"
```

### Implementation

#### Rate Limiting Module

**File**: `rust/service/src/middleware/rate_limit.rs`

```rust
//! Rate limiting middleware using token bucket algorithm
//!
//! Provides configurable per-endpoint rate limiting with burst support

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use governor::{
    clock::DefaultClock,
    state::{InMemoryState, NotKeyed},
    Quota, RateLimiter,
};
use serde_json::json;
use std::num::NonZeroU32;
use std::sync::Arc;

/// Rate limiter configuration
#[derive(Debug, Clone)]
pub struct RateLimitConfig {
    /// Requests per second allowed
    pub requests_per_second: NonZeroU32,
    /// Burst size (allow temporary spikes)
    pub burst_size: NonZeroU32,
}

impl Default for RateLimitConfig {
    fn default() -> Self {
        Self {
            // 100 requests per second
            requests_per_second: NonZeroU32::new(100).unwrap(),
            // Allow bursts of 20 requests
            burst_size: NonZeroU32::new(20).unwrap(),
        }
    }
}

impl RateLimitConfig {
    /// Create from environment variables
    pub fn from_env() -> Self {
        let rps = std::env::var("RATE_LIMIT_RPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .and_then(NonZeroU32::new)
            .unwrap_or_else(|| NonZeroU32::new(100).unwrap());

        let burst = std::env::var("RATE_LIMIT_BURST")
            .ok()
            .and_then(|s| s.parse().ok())
            .and_then(NonZeroU32::new)
            .unwrap_or_else(|| NonZeroU32::new(20).unwrap());

        Self {
            requests_per_second: rps,
            burst_size: burst,
        }
    }
}

/// Global rate limiter state
pub type GlobalRateLimiter = Arc<RateLimiter<NotKeyed, InMemoryState, DefaultClock>>;

/// Create a rate limiter with the given configuration
pub fn create_rate_limiter(config: RateLimitConfig) -> GlobalRateLimiter {
    let quota = Quota::per_second(config.requests_per_second)
        .allow_burst(config.burst_size);

    Arc::new(RateLimiter::direct(quota))
}

/// Rate limiting middleware
///
/// Returns 429 Too Many Requests when limit exceeded
pub async fn rate_limit_middleware(
    State(limiter): State<GlobalRateLimiter>,
    req: Request,
    next: Next,
) -> Response {
    match limiter.check() {
        Ok(_) => {
            // Request allowed, proceed
            next.run(req).await
        }
        Err(_) => {
            // Rate limit exceeded
            tracing::warn!("Rate limit exceeded");

            (
                StatusCode::TOO_MANY_REQUESTS,
                Json(json!({
                    "error": "Rate limit exceeded",
                    "message": "Too many requests. Please try again later.",
                    "retry_after": 1
                }))
            ).into_response()
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_rate_limiter_creation() {
        let config = RateLimitConfig::default();
        let limiter = create_rate_limiter(config);

        // Should allow first request
        assert!(limiter.check().is_ok());
    }

    #[test]
    fn test_rate_limiter_burst() {
        let config = RateLimitConfig {
            requests_per_second: NonZeroU32::new(10).unwrap(),
            burst_size: NonZeroU32::new(5).unwrap(),
        };
        let limiter = create_rate_limiter(config);

        // Should allow burst
        for _ in 0..5 {
            assert!(limiter.check().is_ok());
        }
    }

    #[test]
    fn test_config_from_env() {
        std::env::set_var("RATE_LIMIT_RPS", "50");
        std::env::set_var("RATE_LIMIT_BURST", "10");

        let config = RateLimitConfig::from_env();
        assert_eq!(config.requests_per_second.get(), 50);
        assert_eq!(config.burst_size.get(), 10);

        std::env::remove_var("RATE_LIMIT_RPS");
        std::env::remove_var("RATE_LIMIT_BURST");
    }
}
```

#### Integration

**File**: `rust/service/src/middleware/mod.rs`

```rust
pub mod rate_limit;
pub mod security;
pub mod tracing;

pub use self::rate_limit::{create_rate_limiter, rate_limit_middleware, RateLimitConfig};
pub use self::security::security_headers;
pub use self::tracing::trace_request;
```

**File**: `rust/service/src/main.rs`

```rust
// Create rate limiter from environment
let rate_limit_config = provenance_service::middleware::RateLimitConfig::from_env();
let rate_limiter = provenance_service::middleware::create_rate_limiter(rate_limit_config);

info!(
    "Rate limiting: {} req/s (burst: {})",
    rate_limit_config.requests_per_second,
    rate_limit_config.burst_size
);

// Add to router
let app = Router::new()
    .route(/* ... */)
    .layer(middleware::from_fn_with_state(
        rate_limiter.clone(),
        provenance_service::middleware::rate_limit_middleware
    ))
    .layer(/* other middleware */)
    .with_state(state);
```

### Configuration

**Environment Variables**:
```bash
# Default: 100 req/s, burst 20
export RATE_LIMIT_RPS=100
export RATE_LIMIT_BURST=20

# Production (higher limits)
export RATE_LIMIT_RPS=1000
export RATE_LIMIT_BURST=100
```

### Files to Create/Modify
- `rust/service/src/middleware/rate_limit.rs` (NEW)
- `rust/service/src/middleware/mod.rs` (MODIFY)
- `rust/service/src/main.rs` (MODIFY)
- `rust/service/Cargo.toml` (MODIFY - add governor)

### Success Criteria
- ✅ Rate limiting active on all endpoints
- ✅ Returns 429 with retry_after header
- ✅ Configurable via environment
- ✅ Tests validate burst behavior

### Time Estimate
**40 minutes**

---

## Priority 2: Enhanced Health Checks (30 min)

### Objective
Implement comprehensive health checks for Kubernetes readiness and liveness probes.

### Implementation

#### Health Check Module

**File**: `rust/service/src/health.rs`

```rust
//! Health check endpoints for Kubernetes and monitoring
//!
//! Provides separate readiness and liveness probes

use axum::{
    extract::State,
    http::StatusCode,
    response::IntoResponse,
    Json,
};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::AppState;

/// Health check response
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthResponse {
    /// Service status
    pub status: String,
    /// Service version
    pub version: String,
    /// Database status
    pub database: DatabaseStatus,
    /// Uptime in seconds
    pub uptime_seconds: u64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct DatabaseStatus {
    pub connected: bool,
    pub migrations_applied: bool,
}

/// Liveness probe
///
/// Returns 200 if service is alive (can handle requests)
/// Used by Kubernetes to detect if pod needs restart
pub async fn liveness() -> impl IntoResponse {
    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "alive",
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))
    )
}

/// Readiness probe
///
/// Returns 200 if service is ready to accept traffic
/// Checks database connectivity and critical dependencies
/// Used by Kubernetes to control traffic routing
pub async fn readiness(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    // Check database connectivity
    let db_status = if let Some(ref db) = state.db {
        match db.health_check().await {
            Ok(_) => DatabaseStatus {
                connected: true,
                migrations_applied: true,
            },
            Err(e) => {
                tracing::error!("Database health check failed: {}", e);
                return (
                    StatusCode::SERVICE_UNAVAILABLE,
                    Json(serde_json::json!({
                        "status": "not_ready",
                        "reason": "database_unavailable",
                        "timestamp": chrono::Utc::now().to_rfc3339()
                    }))
                );
            }
        }
    } else {
        DatabaseStatus {
            connected: false,
            migrations_applied: false,
        }
    };

    (
        StatusCode::OK,
        Json(serde_json::json!({
            "status": "ready",
            "database": db_status,
            "timestamp": chrono::Utc::now().to_rfc3339()
        }))
    )
}

/// Detailed health check
///
/// Returns comprehensive service health information
/// Suitable for monitoring dashboards
pub async fn health(
    State(state): State<Arc<AppState>>,
) -> impl IntoResponse {
    let start_time = std::time::SystemTime::now();
    let uptime = start_time
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_secs();

    let db_status = if let Some(ref db) = state.db {
        match db.health_check().await {
            Ok(_) => DatabaseStatus {
                connected: true,
                migrations_applied: true,
            },
            Err(_) => DatabaseStatus {
                connected: false,
                migrations_applied: false,
            },
        }
    } else {
        DatabaseStatus {
            connected: false,
            migrations_applied: false,
        }
    };

    let response = HealthResponse {
        status: if db_status.connected { "healthy" } else { "degraded" }.to_string(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        database: db_status,
        uptime_seconds: uptime,
    };

    (StatusCode::OK, Json(response))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_liveness_always_succeeds() {
        let response = liveness().await.into_response();
        assert_eq!(response.status(), StatusCode::OK);
    }
}
```

#### Routes

**File**: `rust/service/src/main.rs`

```rust
let app = Router::new()
    .route("/health", get(provenance_service::health::health))
    .route("/health/live", get(provenance_service::health::liveness))
    .route("/health/ready", get(provenance_service::health::readiness))
    // ... other routes
```

### Kubernetes Integration

**Example Pod Spec**:
```yaml
livenessProbe:
  httpGet:
    path: /health/live
    port: 8080
  initialDelaySeconds: 10
  periodSeconds: 30

readinessProbe:
  httpGet:
    path: /health/ready
    port: 8080
  initialDelaySeconds: 5
  periodSeconds: 10
```

### Files to Create/Modify
- `rust/service/src/health.rs` (NEW)
- `rust/service/src/lib.rs` (MODIFY - export health)
- `rust/service/src/main.rs` (MODIFY - add routes)

### Success Criteria
- ✅ Liveness probe always returns 200
- ✅ Readiness checks database connectivity
- ✅ Health endpoint returns detailed status
- ✅ Kubernetes-compatible format

### Time Estimate
**30 minutes**

---

## Priority 3: Graceful Shutdown (20 min)

### Objective
Implement graceful shutdown to support zero-downtime deployments.

### Implementation

**File**: `rust/service/src/main.rs`

```rust
use tokio::signal;

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install signal handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {
            tracing::info!("Received Ctrl+C, initiating graceful shutdown");
        },
        _ = terminate => {
            tracing::info!("Received SIGTERM, initiating graceful shutdown");
        },
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // ... initialization

    let addr = "0.0.0.0:8080";
    info!("Starting server on {}", addr);

    let listener = tokio::net::TcpListener::bind(addr).await?;

    // Server with graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await?;

    info!("Server shutdown complete");

    Ok(())
}
```

### Behavior

1. **SIGTERM received** (Kubernetes pod termination)
   - Log shutdown initiation
   - Stop accepting new connections
   - Finish processing in-flight requests (30s timeout)
   - Close database connections
   - Exit cleanly

2. **Zero-downtime deployment**:
   - New pods start and become ready
   - Old pods receive SIGTERM
   - Traffic shifts to new pods
   - Old pods finish existing requests
   - Old pods terminate

### Files to Modify
- `rust/service/src/main.rs` (MODIFY)

### Success Criteria
- ✅ Responds to SIGTERM gracefully
- ✅ Finishes in-flight requests
- ✅ Logs shutdown events
- ✅ Compatible with Kubernetes

### Time Estimate
**20 minutes**

---

## Priority 4: Configuration Management (40 min)

### Objective
Implement 12-factor app configuration with environment variables and validation.

### Implementation

#### Configuration Module

**File**: `rust/service/src/config.rs`

```rust
//! Configuration management
//!
//! Implements 12-factor app configuration with environment variables

use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Server configuration
    pub server: ServerConfig,
    /// Database configuration
    pub database: DatabaseConfig,
    /// Rate limiting configuration
    pub rate_limit: RateLimitConfig,
    /// CORS configuration
    pub cors: CorsConfig,
    /// Logging configuration
    pub logging: LoggingConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServerConfig {
    pub host: String,
    pub port: u16,
    pub request_body_limit_mb: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatabaseConfig {
    pub url: String,
    pub max_connections: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RateLimitConfig {
    pub requests_per_second: u32,
    pub burst_size: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorsConfig {
    pub allowed_origins: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggingConfig {
    pub level: String,
    pub format: String,  // "json" or "text"
}

impl Config {
    /// Load configuration from environment variables
    pub fn from_env() -> Result<Self> {
        Ok(Self {
            server: ServerConfig {
                host: std::env::var("HOST").unwrap_or_else(|_| "0.0.0.0".to_string()),
                port: std::env::var("PORT")
                    .ok()
                    .and_then(|p| p.parse().ok())
                    .unwrap_or(8080),
                request_body_limit_mb: std::env::var("REQUEST_BODY_LIMIT_MB")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(2),
            },
            database: DatabaseConfig {
                url: std::env::var("DATABASE_URL")
                    .unwrap_or_else(|_| "sqlite://data/claims.db".to_string()),
                max_connections: std::env::var("DATABASE_MAX_CONNECTIONS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(10),
            },
            rate_limit: RateLimitConfig {
                requests_per_second: std::env::var("RATE_LIMIT_RPS")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(100),
                burst_size: std::env::var("RATE_LIMIT_BURST")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(20),
            },
            cors: CorsConfig {
                allowed_origins: std::env::var("ALLOWED_ORIGINS")
                    .unwrap_or_else(|_| "http://localhost:3000,http://localhost:8080".to_string())
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .collect(),
            },
            logging: LoggingConfig {
                level: std::env::var("RUST_LOG").unwrap_or_else(|_| "info".to_string()),
                format: std::env::var("LOG_FORMAT").unwrap_or_else(|_| "text".to_string()),
            },
        })
    }

    /// Validate configuration
    pub fn validate(&self) -> Result<()> {
        if self.server.port == 0 {
            anyhow::bail!("Server port cannot be 0");
        }

        if self.database.max_connections == 0 {
            anyhow::bail!("Database max_connections must be > 0");
        }

        if self.rate_limit.requests_per_second == 0 {
            anyhow::bail!("Rate limit requests_per_second must be > 0");
        }

        if self.cors.allowed_origins.is_empty() {
            tracing::warn!("No CORS origins configured, will use permissive CORS");
        }

        Ok(())
    }

    /// Get server socket address
    pub fn server_addr(&self) -> SocketAddr {
        format!("{}:{}", self.server.host, self.server.port)
            .parse()
            .expect("Invalid server address")
    }

    /// Print configuration summary (safe for logs)
    pub fn log_summary(&self) {
        tracing::info!("Configuration loaded:");
        tracing::info!("  Server: {}:{}", self.server.host, self.server.port);
        tracing::info!("  Database: {}", mask_db_url(&self.database.url));
        tracing::info!("  Rate Limit: {} req/s (burst: {})",
            self.rate_limit.requests_per_second,
            self.rate_limit.burst_size
        );
        tracing::info!("  CORS Origins: {} configured", self.cors.allowed_origins.len());
        tracing::info!("  Log Level: {}", self.logging.level);
        tracing::info!("  Log Format: {}", self.logging.format);
    }
}

/// Mask sensitive parts of database URL for logging
fn mask_db_url(url: &str) -> String {
    if let Some(idx) = url.find("://") {
        if let Some(at_idx) = url[idx+3..].find('@') {
            let prefix = &url[..idx+3];
            let suffix = &url[idx+3+at_idx..];
            return format!("{}***{}", prefix, suffix);
        }
    }
    url.to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = Config::from_env().unwrap();
        assert_eq!(config.server.port, 8080);
        config.validate().unwrap();
    }

    #[test]
    fn test_mask_db_url() {
        assert_eq!(
            mask_db_url("postgres://user:pass@localhost/db"),
            "postgres://***@localhost/db"
        );
        assert_eq!(
            mask_db_url("sqlite://data/claims.db"),
            "sqlite://data/claims.db"
        );
    }
}
```

### Environment Variables Reference

Create `.env.example`:
```bash
# Server Configuration
HOST=0.0.0.0
PORT=8080
REQUEST_BODY_LIMIT_MB=2

# Database Configuration
DATABASE_URL=sqlite://data/claims.db
DATABASE_MAX_CONNECTIONS=10

# Rate Limiting
RATE_LIMIT_RPS=100
RATE_LIMIT_BURST=20

# CORS Configuration
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:8080

# Logging
RUST_LOG=info,provenance_service=debug
LOG_FORMAT=text  # or "json" for production
```

### Files to Create/Modify
- `rust/service/src/config.rs` (NEW)
- `rust/service/src/lib.rs` (MODIFY - export config)
- `rust/service/src/main.rs` (MODIFY - use Config)
- `.env.example` (NEW)

### Success Criteria
- ✅ All config from environment
- ✅ Validation on startup
- ✅ Safe logging (no secrets)
- ✅ 12-factor compliant

### Time Estimate
**40 minutes**

---

## Priority 5: Production Deployment Guide (45 min)

### Objective
Create comprehensive deployment documentation for production.

### Deliverables

#### Deployment Guide

**File**: `docs/DEPLOYMENT.md`

Contents:
1. Prerequisites
2. Environment Configuration
3. Docker Deployment
4. Kubernetes Deployment
5. Monitoring Setup
6. Backup and Recovery
7. Troubleshooting

#### Docker Configuration

**File**: `Dockerfile`

Multi-stage build for minimal image size.

**File**: `docker-compose.yml`

Complete stack for local testing.

#### Kubernetes Manifests

**File**: `k8s/deployment.yaml`
**File**: `k8s/service.yaml`
**File**: `k8s/configmap.yaml`
**File**: `k8s/ingress.yaml`

### Time Estimate
**45 minutes**

---

## Implementation Order

### Phase 1: Core Features (1.5 hours)
1. Rate limiting (40 min)
2. Enhanced health checks (30 min)
3. Graceful shutdown (20 min)

### Phase 2: Configuration (40 min)
4. Configuration management (40 min)

### Phase 3: Documentation (1 hour)
5. Production deployment guide (45 min)
6. Phase 11 summary (15 min)

### Phase 4: Finalization (30 min)
7. Testing all features
8. Commit and push

**Total Time**: ~3.5 hours

---

## Expected Outcomes

### Operational Excellence
- **Rate Limiting**: 100-1000 req/s (configurable)
- **Health Checks**: Kubernetes-ready probes
- **Graceful Shutdown**: Zero-downtime deployments
- **Configuration**: 12-factor app compliant
- **Documentation**: Complete deployment guide

### Production Readiness Score: 6/6

After Phase 11:
- [x] Testing (Phase 9)
- [x] Observability (Phase 9)
- [x] Security (Phase 10)
- [x] Performance (Phase 10)
- [x] Operations (Phase 11) ✅
- [x] Deployment (Phase 11) ✅

---

## Success Metrics

### Reliability
- ✅ Zero-downtime deployments
- ✅ Graceful shutdown (30s timeout)
- ✅ Health check response < 100ms

### Scalability
- ✅ Configurable rate limits
- ✅ Horizontal pod autoscaling ready
- ✅ Database connection pooling

### Maintainability
- ✅ Environment-based configuration
- ✅ Comprehensive deployment docs
- ✅ Clear error messages

---

## Phase 11 Completion Checklist

- [ ] Rate limiting implemented and tested
- [ ] Health checks (liveness + readiness)
- [ ] Graceful shutdown handling
- [ ] Configuration management module
- [ ] .env.example created
- [ ] Dockerfile created
- [ ] docker-compose.yml created
- [ ] Kubernetes manifests created
- [ ] Deployment guide written
- [ ] All tests passing
- [ ] Phase 11 summary documented
- [ ] All changes committed and pushed

---

**Phase 11 Status**: 🚧 **IN PROGRESS**
**Target Completion**: 2025-11-16
**Production Deployment**: ✅ **READY AFTER COMPLETION**
