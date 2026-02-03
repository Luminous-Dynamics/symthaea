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
