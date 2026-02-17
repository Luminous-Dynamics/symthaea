# Multi-stage Dockerfile for Mycelix Music API
# Optimized for production with minimal image size and security best practices

# =============================================================================
# Stage 1: Base image with dependencies
# =============================================================================
FROM node:20-alpine AS base

# Install build dependencies for native modules
RUN apk add --no-cache python3 make g++ libc6-compat

WORKDIR /app

# =============================================================================
# Stage 2: Install dependencies
# =============================================================================
FROM base AS deps

# Copy package files for dependency installation
COPY package.json package-lock.json* ./
COPY apps/api/package.json ./apps/api/
COPY packages/sdk/package.json ./packages/sdk/

# Install all dependencies (including dev for build)
RUN npm ci --workspace=@mycelix/api --workspace=@mycelix/sdk

# =============================================================================
# Stage 3: Build the application
# =============================================================================
FROM base AS builder

WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/apps/api/node_modules ./apps/api/node_modules
COPY --from=deps /app/packages/sdk/node_modules ./packages/sdk/node_modules

# Copy source code
COPY . .

# Build SDK first (API depends on it)
RUN npm run build --workspace=@mycelix/sdk

# Build API
RUN npm run build --workspace=@mycelix/api

# Prune dev dependencies for production
RUN npm prune --production --workspace=@mycelix/api --workspace=@mycelix/sdk

# =============================================================================
# Stage 4: Production runtime
# =============================================================================
FROM node:20-alpine AS runner

WORKDIR /app

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 api

# Install runtime dependencies only
RUN apk add --no-cache tini curl

# Copy built application and production dependencies
COPY --from=builder --chown=api:nodejs /app/node_modules ./node_modules
COPY --from=builder --chown=api:nodejs /app/apps/api/node_modules ./apps/api/node_modules
COPY --from=builder --chown=api:nodejs /app/apps/api/dist ./apps/api/dist
COPY --from=builder --chown=api:nodejs /app/apps/api/package.json ./apps/api/
COPY --from=builder --chown=api:nodejs /app/packages/sdk/dist ./packages/sdk/dist
COPY --from=builder --chown=api:nodejs /app/packages/sdk/package.json ./packages/sdk/

# Set environment variables
ENV NODE_ENV=production
ENV PORT=3100

# Expose API port
EXPOSE 3100

# Switch to non-root user
USER api

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:3100/health || exit 1

# Use tini as init system for proper signal handling
ENTRYPOINT ["/sbin/tini", "--"]

# Start the API server
CMD ["node", "apps/api/dist/index.js"]
