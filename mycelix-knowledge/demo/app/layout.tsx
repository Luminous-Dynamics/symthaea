// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import './globals.css';
import type { Metadata } from 'next';

export const metadata: Metadata = {
  title: 'Mycelix Knowledge - Demo',
  description: 'Interactive demo of the Mycelix Knowledge decentralized knowledge graph',
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
