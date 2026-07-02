# Realtime WebSocket Service Dockerfile
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY services/realtime/package*.json ./services/realtime/

# Install dependencies
RUN npm ci --workspace=services/realtime

# Copy source
COPY services/realtime ./services/realtime

# Build
WORKDIR /app/services/realtime
RUN npm run build

# Production image
FROM node:20-alpine

WORKDIR /app

# Copy built files
COPY --from=builder /app/services/realtime/dist ./dist
COPY --from=builder /app/services/realtime/package*.json ./

# Install production dependencies only
RUN npm ci --omit=dev

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

USER nodejs

EXPOSE 3005

ENV NODE_ENV=production

CMD ["node", "dist/index.js"]
