// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useState, useEffect } from 'react';
import type { AppProps } from 'next/app';
import { PrivyProvider } from '@privy-io/react-auth';
import { ToastProvider } from '../src/components/ToastProvider';
import '../styles/globals.css';

// Chain configurations
const gnosisChain = {
  id: 100,
  name: 'Gnosis Chain',
  nativeCurrency: { name: 'xDAI', symbol: 'xDAI', decimals: 18 },
  rpcUrls: { default: { http: ['https://rpc.gnosischain.com'] } },
};

const localhostChain = {
  id: 31337,
  name: 'Localhost',
  nativeCurrency: { name: 'Ether', symbol: 'ETH', decimals: 18 },
  rpcUrls: { default: { http: ['http://localhost:8545'] } },
};

export default function App({ Component, pageProps }: AppProps) {
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Show loading state during SSR to avoid hydration mismatches
  if (!mounted) {
    return (
      <ToastProvider>
        <div className="min-h-screen bg-gradient-to-br from-gray-900 via-purple-900 to-gray-900">
          <div className="flex items-center justify-center min-h-screen">
            <div className="w-16 h-16 border-4 border-purple-500 border-t-transparent rounded-full animate-spin" />
          </div>
        </div>
      </ToastProvider>
    );
  }

  return (
    <PrivyProvider
      appId={process.env.NEXT_PUBLIC_PRIVY_APP_ID || 'clzxv89ey0089l80fyukufq9a'}
      config={{
        appearance: {
          theme: 'dark',
          accentColor: '#8B5CF6',
          logo: 'https://music.mycelix.net/logo.png',
          walletList: ['metamask', 'coinbase_wallet', 'wallet_connect', 'rainbow', 'phantom'],
        },
        loginMethods: ['wallet', 'email', 'sms', 'google', 'discord'],
        embeddedWallets: {
          createOnLogin: 'users-without-wallets',
          requireUserPasswordOnCreate: false,
        },
        supportedChains: [gnosisChain, localhostChain],
        defaultChain: process.env.NEXT_PUBLIC_CHAIN_ID === '31337' ? localhostChain : gnosisChain,
      }}
    >
      <ToastProvider>
        <Component {...pageProps} />
      </ToastProvider>
    </PrivyProvider>
  );
}
