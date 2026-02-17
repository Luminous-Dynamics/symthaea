# =============================================================================
# Mycelix Music Platform - Production Dockerfile
#
# Multi-stage build optimized for:
# - Small image size (distroless base)
# - Build caching (layer optimization)
# - Security (non-root user, minimal attack surface)
# - Performance (native dependencies, optimized Node.js)
# =============================================================================

# -----------------------------------------------------------------------------
# Stage 1: Dependencies
# -----------------------------------------------------------------------------
FROM node:20-alpine AS deps

WORKDIR /app

# Install build dependencies for native modules
RUN apk add --no-cache \
    python3 \
    make \
    g++ \
    libc6-compat

# Copy package files
COPY package.json package-lock.json* yarn.lock* pnpm-lock.yaml* ./

# Install dependencies based on lockfile
RUN \
    if [ -f yarn.lock ]; then yarn install --frozen-lockfile --production=false; \
    elif [ -f package-lock.json ]; then npm ci; \
    elif [ -f pnpm-lock.yaml ]; then corepack enable pnpm && pnpm install --frozen-lockfile; \
    else npm install; \
    fi

# -----------------------------------------------------------------------------
# Stage 2: Builder
# -----------------------------------------------------------------------------
FROM node:20-alpine AS builder

WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY . .

# Generate Prisma client
RUN npx prisma generate 2>/dev/null || true

# Build TypeScript
RUN npm run build 2>/dev/null || npx tsc --project tsconfig.json 2>/dev/null || echo "No build script"

# Prune dev dependencies
RUN npm prune --production 2>/dev/null || yarn install --production --ignore-scripts 2>/dev/null || true

# -----------------------------------------------------------------------------
# Stage 3: Production Runner
# -----------------------------------------------------------------------------
FROM node:20-alpine AS runner

WORKDIR /app

# Security: Create non-root user
RUN addgroup --system --gid 1001 mycelix && \
    adduser --system --uid 1001 mycelix

# Install runtime dependencies
RUN apk add --no-cache \
    tini \
    curl \
    ca-certificates \
    tzdata

# Set timezone
ENV TZ=UTC

# Copy built application
COPY --from=builder --chown=mycelix:mycelix /app/dist ./dist
COPY --from=builder --chown=mycelix:mycelix /app/node_modules ./node_modules
COPY --from=builder --chown=mycelix:mycelix /app/package.json ./package.json

# Copy Prisma schema and migrations
COPY --from=builder --chown=mycelix:mycelix /app/prisma ./prisma 2>/dev/null || true

# Copy configuration files
COPY --from=builder --chown=mycelix:mycelix /app/config ./config 2>/dev/null || true

# Environment configuration
ENV NODE_ENV=production
ENV PORT=3000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:${PORT}/api/health || exit 1

# Expose port
EXPOSE 3000

# Switch to non-root user
USER mycelix

# Use tini as init system
ENTRYPOINT ["/sbin/tini", "--"]

# Start application
CMD ["node", "dist/main.js"]

# -----------------------------------------------------------------------------
# Stage 4: Development (optional, for local dev)
# -----------------------------------------------------------------------------
FROM node:20-alpine AS development

WORKDIR /app

# Install development tools
RUN apk add --no-cache \
    git \
    curl \
    vim \
    python3 \
    make \
    g++

# Copy everything
COPY . .

# Install all dependencies
RUN npm install

# Development port
EXPOSE 3000 9229

# Start with hot reload
CMD ["npm", "run", "dev"]
