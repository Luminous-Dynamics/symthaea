// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Metadata, Viewport } from 'next';
import { Inter } from 'next/font/google';
import './globals.css';
import { Providers } from '@/components/providers';
import { Player } from '@/components/player';
import { Sidebar } from '@/components/sidebar';
import { Toaster } from '@/components/ui/toaster';

const inter = Inter({ subsets: ['latin'], variable: '--font-inter' });

export const metadata: Metadata = {
  title: {
    default: 'Mycelix Music',
    template: '%s | Mycelix',
  },
  description: 'Discover, stream, and connect through music on the decentralized music platform.',
  keywords: ['music', 'streaming', 'web3', 'nft', 'artists', 'discovery'],
  authors: [{ name: 'Mycelix' }],
  creator: 'Mycelix',
  openGraph: {
    type: 'website',
    locale: 'en_US',
    url: 'https://mycelix.io',
    siteName: 'Mycelix Music',
    title: 'Mycelix Music',
    description: 'Discover, stream, and connect through music.',
    images: [
      {
        url: '/og-image.png',
        width: 1200,
        height: 630,
        alt: 'Mycelix Music',
      },
    ],
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Mycelix Music',
    description: 'Discover, stream, and connect through music.',
    images: ['/og-image.png'],
  },
  robots: {
    index: true,
    follow: true,
  },
  manifest: '/manifest.json',
};

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: '#ffffff' },
    { media: '(prefers-color-scheme: dark)', color: '#0a0a0a' },
  ],
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <body className={`${inter.variable} font-sans antialiased`}>
        <Providers>
          <div className="flex h-screen overflow-hidden bg-background">
            {/* Sidebar Navigation */}
            <Sidebar />

            {/* Main Content */}
            <main className="flex-1 overflow-y-auto pb-24">
              {children}
            </main>
          </div>

          {/* Fixed Player */}
          <Player />

          {/* Notifications */}
          <Toaster />
        </Providers>
      </body>
    </html>
  );
}
