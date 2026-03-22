const path = require('path')

/** @type {import('next').NextConfig} */
const nextConfig = {
  /* config options here */
  eslint: {
    // TODO: Fix ~20 ESLint errors and remove this flag
    // Current errors include missing dependencies, unused vars, etc.
    // Priority: HIGH - tracked in docs/STATUS.md
    ignoreDuringBuilds: true,
  },
  typescript: {
    // TODO: Fix ~70 TypeScript errors and remove this flag
    // Current errors include:
    // - better-sqlite3 type definitions (will be removed with SQLite cleanup)
    // - Next.js 15 params changes (async params)
    // - Missing type declarations for some modules
    // - Various type mismatches in components
    // Priority: HIGH - tracked in docs/STATUS.md
    ignoreBuildErrors: true,
  },
  // Disable image optimization for now
  images: {
    unoptimized: true,
  },
  // Ensure pages directory is included
  pageExtensions: ['tsx', 'ts', 'jsx', 'js'],
  // Skip static generation for problematic routes
  skipMiddlewareUrlNormalize: true,
  skipTrailingSlashRedirect: true,
  output: 'standalone',
  outputFileTracingRoot: path.resolve(__dirname, '..'),
  // Generate 404 at runtime instead of build time
  generateBuildId: async () => {
    return 'build-' + Date.now()
  },
};

module.exports = nextConfig;
