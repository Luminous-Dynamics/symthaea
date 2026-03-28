// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { lazy, Suspense } from 'react';
import { BrowserRouter, Routes, Route, Link, NavLink } from 'react-router-dom';
import './App.css';
import './styles/transitions.css';
import './styles/toast.css';

// Lazy-loaded pages for code splitting
const CoursesPage = lazy(() => import('./pages/CoursesPage').then(m => ({ default: m.CoursesPage })));
const FLRoundsPage = lazy(() => import('./pages/FLRoundsPage').then(m => ({ default: m.FLRoundsPage })));
const CredentialsPage = lazy(() => import('./pages/CredentialsPage').then(m => ({ default: m.CredentialsPage })));
const CurriculumExplorer = lazy(() => import('./pages/CurriculumExplorer'));

// Components
import { ErrorBoundary } from './components/ErrorBoundary';
import { LoadingPage } from './components/LoadingSkeleton';
import { ToastProvider } from './contexts/ToastContext';
import { HolochainProvider, useHolochain } from './contexts/HolochainContext';

function HomePage() {
  return (
    <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '40px 24px' }}>
      <div style={{ textAlign: 'center', marginBottom: '60px' }}>
        <h1 style={{ margin: 0, fontSize: '48px', fontWeight: '700', color: '#111827' }}>
          Welcome to EduNet
        </h1>
        <p
          style={{
            margin: '16px 0 0 0',
            fontSize: '20px',
            color: '#6b7280',
            maxWidth: '600px',
            marginLeft: 'auto',
            marginRight: 'auto',
          }}
        >
          Privacy-preserving decentralized education powered by Holochain and Federated Learning
        </p>
      </div>

      <div
        style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))',
          gap: '24px',
          marginBottom: '60px',
        }}
      >
        <FeatureCard
          icon="📚"
          title="Discover Courses"
          description="Browse courses from expert instructors and start learning"
          link="/courses"
        />
        <FeatureCard
          icon="🤝"
          title="FL Rounds"
          description="Participate in privacy-preserving collaborative model training"
          link="/rounds"
        />
        <FeatureCard
          icon="🎓"
          title="My Credentials"
          description="View and share your verifiable achievements"
          link="/credentials"
        />
        <FeatureCard
          icon="🗺️"
          title="Curriculum Explorer"
          description="Interactive K-through-PhD knowledge graph with 1,718 nodes and career pathways"
          link="/curriculum"
        />
      </div>

      <div style={{ marginTop: '80px', padding: '40px', backgroundColor: '#f9fafb', borderRadius: '12px' }}>
        <h2 style={{ margin: '0 0 24px 0', fontSize: '24px', fontWeight: '600', textAlign: 'center' }}>
          Key Features
        </h2>
        <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(250px, 1fr))', gap: '24px' }}>
          <div>
            <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '8px' }}>
              🔒 Privacy-First
            </h3>
            <p style={{ fontSize: '14px', color: '#6b7280', lineHeight: '1.6' }}>
              Federated learning with gradient clipping and optional differential privacy
            </p>
          </div>
          <div>
            <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '8px' }}>
              🌐 Decentralized
            </h3>
            <p style={{ fontSize: '14px', color: '#6b7280', lineHeight: '1.6' }}>
              Built on Holochain for peer-to-peer, agent-centric learning
            </p>
          </div>
          <div>
            <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '8px' }}>
              ✅ Verifiable
            </h3>
            <p style={{ fontSize: '14px', color: '#6b7280', lineHeight: '1.6' }}>
              W3C Verifiable Credentials tied to model provenance
            </p>
          </div>
          <div>
            <h3 style={{ fontSize: '16px', fontWeight: '600', marginBottom: '8px' }}>
              🗳️ Community-Governed
            </h3>
            <p style={{ fontSize: '14px', color: '#6b7280', lineHeight: '1.6' }}>
              DAO governance for curricula and quality standards
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}

function FeatureCard({
  icon,
  title,
  description,
  link,
}: {
  icon: string;
  title: string;
  description: string;
  link: string;
}) {
  return (
    <Link
      to={link}
      aria-label={`${title}: ${description}`}
      style={{
        padding: '32px',
        backgroundColor: '#ffffff',
        border: '2px solid #e5e7eb',
        borderRadius: '12px',
        textDecoration: 'none',
        color: 'inherit',
        transition: 'all 0.2s ease',
        display: 'block',
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.borderColor = '#3b82f6';
        e.currentTarget.style.transform = 'translateY(-4px)';
        e.currentTarget.style.boxShadow = '0 8px 24px rgba(0, 0, 0, 0.1)';
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.borderColor = '#e5e7eb';
        e.currentTarget.style.transform = 'translateY(0)';
        e.currentTarget.style.boxShadow = 'none';
      }}
    >
      <div aria-hidden="true" style={{ fontSize: '48px', marginBottom: '16px' }}>{icon}</div>
      <h3 style={{ margin: '0 0 8px 0', fontSize: '20px', fontWeight: '600' }}>{title}</h3>
      <p style={{ margin: 0, fontSize: '14px', color: '#6b7280', lineHeight: '1.6' }}>
        {description}
      </p>
    </Link>
  );
}

function NavBar() {
  const { status, mode, reconnect } = useHolochain();
  const connected = status === 'connected';
  const isReal = mode === 'real';

  const navLinkStyle = ({ isActive }: { isActive: boolean }) => ({
    padding: '8px 16px',
    borderRadius: '6px',
    textDecoration: 'none',
    color: isActive ? '#3b82f6' : '#4b5563',
    backgroundColor: isActive ? '#eff6ff' : 'transparent',
    fontWeight: isActive ? '600' : '500',
    fontSize: '14px',
    transition: 'all 0.2s ease',
  });

  return (
    <nav
      role="navigation"
      aria-label="Main navigation"
      style={{
        backgroundColor: '#ffffff',
        borderBottom: '1px solid #e5e7eb',
        padding: '16px 24px',
      }}
    >
      <div
        style={{
          maxWidth: '1200px',
          margin: '0 auto',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
        }}
      >
        {/* Logo */}
        <Link
          to="/"
          aria-label="EduNet home"
          style={{
            fontSize: '20px',
            fontWeight: '700',
            color: '#111827',
            textDecoration: 'none',
          }}
        >
          EduNet
        </Link>

        {/* Navigation Links */}
        <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
          <NavLink to="/courses" style={navLinkStyle}>
            Courses
          </NavLink>
          <NavLink to="/rounds" style={navLinkStyle}>
            FL Rounds
          </NavLink>
          <NavLink to="/credentials" style={navLinkStyle}>
            Credentials
          </NavLink>
          <NavLink to="/curriculum" style={navLinkStyle}>
            Curriculum
          </NavLink>

          {/* Connection Status */}
          <div
            role="status"
            aria-live="polite"
            aria-label={`Holochain connection status: ${connected ? 'Connected' : status}`}
            style={{
              marginLeft: '16px',
              padding: '6px 12px',
              borderRadius: '6px',
              fontSize: '12px',
              fontWeight: '500',
              backgroundColor: connected ? '#ecfdf5' : status === 'connecting' ? '#fef3c7' : '#fef2f2',
              color: connected ? '#059669' : status === 'connecting' ? '#d97706' : '#dc2626',
              display: 'flex',
              alignItems: 'center',
              gap: '6px',
              cursor: status === 'error' ? 'pointer' : 'default',
            }}
            onClick={status === 'error' ? () => reconnect() : undefined}
            title={status === 'error' ? 'Click to retry connection' : undefined}
          >
            <span
              aria-hidden="true"
              style={{
                width: '6px',
                height: '6px',
                borderRadius: '50%',
                backgroundColor: connected ? '#10b981' : status === 'connecting' ? '#f59e0b' : '#ef4444',
              }}
            />
            {connected ? (isReal ? 'Connected (Real)' : 'Connected (Mock)') : status === 'connecting' ? 'Connecting...' : 'Disconnected'}
          </div>
        </div>
      </div>
    </nav>
  );
}

function App() {
  return (
    <ErrorBoundary>
      <HolochainProvider>
        <ToastProvider>
          <BrowserRouter>
            <div style={{ minHeight: '100vh', backgroundColor: '#ffffff' }}>
            {/* Skip Navigation Link */}
            <a
              href="#main-content"
              style={{
                position: 'absolute',
                left: '-9999px',
                zIndex: 999,
                padding: '8px 16px',
                backgroundColor: '#3b82f6',
                color: '#ffffff',
                textDecoration: 'none',
                borderRadius: '4px',
                fontWeight: '600',
              }}
              onFocus={(e) => {
                e.currentTarget.style.left = '10px';
                e.currentTarget.style.top = '10px';
              }}
              onBlur={(e) => {
                e.currentTarget.style.left = '-9999px';
              }}
            >
              Skip to main content
            </a>

            <NavBar />

            <main id="main-content" role="main">
              <Suspense fallback={<LoadingPage message="Loading..." />}>
                <Routes>
                  <Route path="/" element={<HomePage />} />
                  <Route path="/courses" element={<CoursesPage />} />
                  <Route path="/rounds" element={<FLRoundsPage />} />
                  <Route path="/credentials" element={<CredentialsPage />} />
                  <Route path="/curriculum" element={<CurriculumExplorer />} />
                </Routes>
              </Suspense>
            </main>

            <footer
              role="contentinfo"
              style={{
                marginTop: '80px',
                padding: '24px',
                borderTop: '1px solid #e5e7eb',
                textAlign: 'center',
                color: '#6b7280',
                fontSize: '14px',
              }}
            >
              <p style={{ margin: '0 0 8px 0' }}>
                Built with ❤️ by{' '}
                <a
                  href="https://github.com/Luminous-Dynamics"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="Luminous Dynamics GitHub organization"
                  style={{ color: '#3b82f6', textDecoration: 'none' }}
                >
                  Luminous Dynamics
                </a>
              </p>
              <p style={{ margin: 0 }}>
                <a
                  href="https://github.com/Luminous-Dynamics/mycelix-edunet"
                  target="_blank"
                  rel="noopener noreferrer"
                  aria-label="View mycelix-edunet repository on GitHub"
                  style={{ color: '#3b82f6', textDecoration: 'none' }}
                >
                  View on GitHub
                </a>
              </p>
              </footer>
            </div>
          </BrowserRouter>
        </ToastProvider>
      </HolochainProvider>
    </ErrorBoundary>
  );
}

export default App;
