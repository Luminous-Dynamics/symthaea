# Streaming Service Dockerfile
# Includes ffmpeg for audio transcoding

# Stage 1: Builder
FROM node:20-alpine AS builder
WORKDIR /app

# Copy package files
COPY package.json package-lock.json* ./
COPY services/streaming/package.json ./services/streaming/

# Install dependencies
RUN npm ci

# Copy source code
COPY services/streaming ./services/streaming

# Build
RUN npm run build --workspace=services/streaming

# Stage 2: Runner
FROM node:20-alpine AS runner
WORKDIR /app

ENV NODE_ENV=production

# Install ffmpeg
RUN apk add --no-cache ffmpeg

# Create non-root user
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 streaming

# Create temp directory for transcoding
RUN mkdir -p /tmp/mycelix-transcode && chown streaming:nodejs /tmp/mycelix-transcode

# Copy production dependencies and built app
COPY --from=builder /app/node_modules ./node_modules
COPY --from=builder /app/services/streaming/node_modules ./services/streaming/node_modules
COPY --from=builder /app/services/streaming/dist ./services/streaming/dist
COPY --from=builder /app/services/streaming/package.json ./services/streaming/

USER streaming

EXPOSE 3003
ENV PORT=3003
ENV TRANSCODE_TEMP_DIR=/tmp/mycelix-transcode

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD wget --no-verbose --tries=1 --spider http://localhost:3003/health || exit 1

CMD ["node", "services/streaming/dist/index.js"]
