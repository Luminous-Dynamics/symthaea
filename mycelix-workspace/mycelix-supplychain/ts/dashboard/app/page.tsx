// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Dashboard Home - Supply Chain Provenance Viewer
 */

export default function Home() {
  return (
    <div style={{ padding: '2rem', fontFamily: 'system-ui, sans-serif' }}>
      <h1>Mycelix Supply Chain Dashboard</h1>
      <p>Verifiable supply chain provenance and lineage tracking</p>

      <div style={{ marginTop: '2rem' }}>
        <h2>Quick Links</h2>
        <ul>
          <li><a href="/claims">View Claims</a></li>
          <li><a href="/verify">Verify Credential</a></li>
          <li><a href="/lineage">Explore Lineage</a></li>
        </ul>
      </div>

      <div style={{ marginTop: '2rem', padding: '1rem', background: '#f5f5f5', borderRadius: '8px' }}>
        <h3>Status</h3>
        <p>Service: <strong>Connected</strong></p>
        <p>Version: <strong>0.1.0</strong></p>
      </div>

      <div style={{ marginTop: '2rem' }}>
        <h2>About</h2>
        <p>
          This dashboard provides visibility into supply chain events, verifiable credentials,
          and lineage tracking on the Mycelix network.
        </p>
        <ul>
          <li><strong>Claims:</strong> View all supply chain claims and their provenance</li>
          <li><strong>Verify:</strong> Validate verifiable credentials and signatures</li>
          <li><strong>Lineage:</strong> Explore batch lineage and transformations</li>
        </ul>
      </div>
    </div>
  );
}
