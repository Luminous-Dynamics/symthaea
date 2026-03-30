// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  swcMinify: true,
  transpilePackages: ['@mycelix/sdk'],
  // Use standalone output for deployment
  output: 'standalone',
  // Disable static optimization to avoid SSR issues with wallet providers
  experimental: {
    // This helps with some SSR issues
    serverComponentsExternalPackages: ['@privy-io/react-auth'],
  },
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  env: {
    NEXT_PUBLIC_ROUTER_ADDRESS: process.env.NEXT_PUBLIC_ROUTER_ADDRESS,
    NEXT_PUBLIC_FLOW_TOKEN_ADDRESS: process.env.NEXT_PUBLIC_FLOW_TOKEN_ADDRESS,
    NEXT_PUBLIC_CHAIN_ID: process.env.NEXT_PUBLIC_CHAIN_ID || '31337',
    NEXT_PUBLIC_RPC_URL: process.env.NEXT_PUBLIC_RPC_URL || 'http://localhost:8545',
  },
  images: {
    domains: ['w3s.link', 'ipfs.io'],
  },
};

module.exports = nextConfig;
