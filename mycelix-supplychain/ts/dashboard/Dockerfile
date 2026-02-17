# Multi-stage build for Next.js dashboard

FROM node:20-alpine as deps

WORKDIR /app

# Install SDK first
COPY ../sdk/package.json ../sdk/package-lock.json ../sdk/
RUN cd ../sdk && npm ci

# Install dashboard deps
COPY package.json package-lock.json ./
RUN npm ci

# Builder stage
FROM node:20-alpine as builder

WORKDIR /app

COPY --from=deps /app/node_modules ./node_modules
COPY --from=deps /app/../sdk ../sdk

COPY . .

RUN npm run build

# Runner stage
FROM node:20-alpine as runner

WORKDIR /app

ENV NODE_ENV production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static

USER nextjs

EXPOSE 3000

ENV PORT 3000
ENV HOSTNAME "0.0.0.0"

CMD ["node", "server.js"]
