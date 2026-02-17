export const metadata = {
  title: 'Mycelix Supply Chain Dashboard',
  description: 'Verifiable supply chain provenance and lineage tracking',
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
