# Multi-stage Dockerfile for Mycelix Music Web App (Next.js)
# Optimized for production with standalone output mode

# =============================================================================
# Stage 1: Base image with dependencies
# =============================================================================
FROM node:20-alpine AS base

# Install build dependencies
RUN apk add --no-cache libc6-compat

WORKDIR /app

# =============================================================================
# Stage 2: Install dependencies
# =============================================================================
FROM base AS deps

# Copy package files for dependency installation
COPY package.json package-lock.json* ./
COPY apps/web/package.json ./apps/web/
COPY packages/sdk/package.json ./packages/sdk/

# Install all dependencies
RUN npm ci --workspace=@mycelix/web --workspace=@mycelix/sdk

# =============================================================================
# Stage 3: Build the application
# =============================================================================
FROM base AS builder

WORKDIR /app

# Copy dependencies from deps stage
COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/apps/web/node_modules ./apps/web/node_modules
COPY --from=deps /app/packages/sdk/node_modules ./packages/sdk/node_modules

# Copy source code
COPY . .

# Build SDK first (web depends on it)
RUN npm run build --workspace=@mycelix/sdk

# Build Next.js app with standalone output
# Environment variables needed at build time
ARG NEXT_PUBLIC_PRIVY_APP_ID
ARG NEXT_PUBLIC_ROUTER_ADDRESS
ARG NEXT_PUBLIC_FLOW_TOKEN_ADDRESS
ARG NEXT_PUBLIC_CHAIN_ID=100
ARG NEXT_PUBLIC_RPC_URL=https://rpc.gnosischain.com
ARG NEXT_PUBLIC_API_URL

ENV NEXT_PUBLIC_PRIVY_APP_ID=$NEXT_PUBLIC_PRIVY_APP_ID
ENV NEXT_PUBLIC_ROUTER_ADDRESS=$NEXT_PUBLIC_ROUTER_ADDRESS
ENV NEXT_PUBLIC_FLOW_TOKEN_ADDRESS=$NEXT_PUBLIC_FLOW_TOKEN_ADDRESS
ENV NEXT_PUBLIC_CHAIN_ID=$NEXT_PUBLIC_CHAIN_ID
ENV NEXT_PUBLIC_RPC_URL=$NEXT_PUBLIC_RPC_URL
ENV NEXT_PUBLIC_API_URL=$NEXT_PUBLIC_API_URL
ENV NODE_ENV=production

# Build Next.js
RUN npm run build --workspace=@mycelix/web

# =============================================================================
# Stage 4: Production runtime
# =============================================================================
FROM node:20-alpine AS runner

WORKDIR /app

# Create non-root user for security
RUN addgroup --system --gid 1001 nodejs && \
    adduser --system --uid 1001 nextjs

# Install runtime dependencies
RUN apk add --no-cache tini curl

# Set environment
ENV NODE_ENV=production
ENV NEXT_TELEMETRY_DISABLED=1
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"

# Copy public assets and static files
COPY --from=builder /app/apps/web/public ./apps/web/public

# Copy standalone output
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/apps/web/.next/static ./apps/web/.next/static

# Expose port
EXPOSE 3000

# Switch to non-root user
USER nextjs

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:3000/api/health || exit 1

# Use tini as init system
ENTRYPOINT ["/sbin/tini", "--"]

# Start Next.js server
CMD ["node", "apps/web/server.js"]
