# Search Service Dockerfile
FROM node:20-alpine AS builder

WORKDIR /app

# Copy package files
COPY package*.json ./
COPY services/search/package*.json ./services/search/

# Install dependencies
RUN npm ci --workspace=services/search

# Copy source
COPY services/search ./services/search

# Build
WORKDIR /app/services/search
RUN npm run build

# Production image
FROM node:20-alpine

WORKDIR /app

# Copy built files
COPY --from=builder /app/services/search/dist ./dist
COPY --from=builder /app/services/search/package*.json ./

# Install production dependencies only
RUN npm ci --omit=dev

# Create non-root user
RUN addgroup -g 1001 -S nodejs && \
    adduser -S nodejs -u 1001

USER nodejs

EXPOSE 3004

ENV NODE_ENV=production

CMD ["node", "dist/index.js"]
