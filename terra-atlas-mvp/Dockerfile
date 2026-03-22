# Terra Atlas — Multi-stage Next.js production build
#
# Build:  docker build -t terra-atlas .
# Run:    docker run -p 3002:3000 terra-atlas

# ── Stage 1: Dependencies ─────────────────────────────────────────
FROM node:20-alpine AS deps
WORKDIR /app
COPY package.json package-lock.json* ./
RUN npm ci --omit=dev

# ── Stage 2: Build ────────────────────────────────────────────────
FROM node:20-alpine AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npm run build

# ── Stage 3: Production ──────────────────────────────────────────
FROM node:20-alpine AS runner
WORKDIR /app

RUN addgroup --system --gid 1001 atlas && \
    adduser --system --uid 1001 atlas

COPY --from=builder /app/public ./public
COPY --from=builder --chown=atlas:atlas /app/.next/standalone/terra-atlas-mvp ./
COPY --from=builder --chown=atlas:atlas /app/.next/static ./.next/static

# Static data files (USACE, FERC, SMR)
COPY --from=builder /app/data ./data

USER atlas

EXPOSE 3000
ENV PORT=3000
ENV HOSTNAME="0.0.0.0"
ENV NODE_ENV=production

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3000/api/health || exit 1

CMD ["node", "server.js"]
