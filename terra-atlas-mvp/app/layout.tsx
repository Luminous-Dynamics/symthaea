import type { Metadata, Viewport } from 'next'
import { Inter } from 'next/font/google'
import { Providers } from './providers'
import './globals.css'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'Terra Atlas - Global Energy Investment Platform',
  description: 'Democratizing energy investment with 166,000+ projects worldwide. Invest from $10 in solar, wind, hydro, nuclear, and storage projects.',
  keywords: 'energy investment, renewable energy, solar investment, wind power, sustainable investing, climate finance',
  authors: [{ name: 'Terra Atlas Team' }],
  manifest: '/manifest.json',
  appleWebApp: {
    capable: true,
    statusBarStyle: 'black-translucent',
    title: 'Terra Atlas',
  },
  openGraph: {
    title: 'Terra Atlas - Global Energy Investment Platform',
    description: 'Invest in the energy revolution. 166,000+ projects, starting at $10.',
    url: 'https://terra-atlas.earth',
    siteName: 'Terra Atlas',
    type: 'website',
  },
  twitter: {
    card: 'summary_large_image',
    title: 'Terra Atlas',
    description: 'Global Energy Investment Platform - 166,000+ Projects',
  },
}

export const viewport: Viewport = {
  width: 'device-width',
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: '#030712',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <head>
        <link rel="manifest" href="/manifest.json" />
        <link rel="apple-touch-icon" href="/icon-192.png" />
        <meta name="apple-mobile-web-app-capable" content="yes" />
        <meta name="apple-mobile-web-app-status-bar-style" content="black-translucent" />
      </head>
      <body className={inter.className}>
        <Providers>
          {children}
        </Providers>
      </body>
    </html>
  )
}