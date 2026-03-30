// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { useEffect, Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { useAuthStore } from './store/authStore';
import { useThemeStore } from './store/themeStore';
import { useWebSocket } from './hooks/useWebSocket';
import ErrorBoundary from './components/ErrorBoundary';
import ToastContainer from './components/ToastContainer';
import { I18nProvider } from './lib/i18n';
import { AccessibilityProvider, SkipLink, accessibilityStyles } from './lib/a11y/AccessibilityProvider';
import LoginPage from './pages/LoginPage';
import RegisterPage from './pages/RegisterPage';
import DashboardPage from './pages/DashboardPage';
import SettingsPage from './pages/SettingsPage';

// Lazy-loaded pages for code splitting
const InboxPage = lazy(() => import('./pages/InboxPage'));
const ContactsPage = lazy(() => import('./pages/ContactsPage'));
const TrustNetworkPage = lazy(() => import('./pages/TrustNetworkPage'));
const TrustDemoPage = lazy(() => import('./pages/TrustDemoPage'));

// Query client for data fetching
const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30 * 1000,
      gcTime: 5 * 60 * 1000,
      retry: 2,
      refetchOnWindowFocus: true,
    },
  },
});

function App() {
  const { isAuthenticated, isLoading, loadUser } = useAuthStore();
  const { theme, setTheme } = useThemeStore();
  const { isConnected } = useWebSocket();

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  // Initialize theme on mount
  useEffect(() => {
    setTheme(theme);
  }, []);

  if (isLoading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading...</p>
        </div>
      </div>
    );
  }

  return (
    <>
      <style>{accessibilityStyles}</style>
      <QueryClientProvider client={queryClient}>
        <I18nProvider>
          <AccessibilityProvider>
            <ErrorBoundary>
              <ToastContainer />
              <BrowserRouter>
                <SkipLink targetId="main-content">Skip to content</SkipLink>
                <main id="main-content" tabIndex={-1}>
                  <Suspense fallback={
                    <div className="min-h-screen flex items-center justify-center">
                      <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
                    </div>
                  }>
                    <Routes>
                      <Route
                        path="/login"
                        element={!isAuthenticated ? <LoginPage /> : <Navigate to="/" replace />}
                      />
                      <Route
                        path="/register"
                        element={!isAuthenticated ? <RegisterPage /> : <Navigate to="/" replace />}
                      />
                      <Route
                        path="/"
                        element={isAuthenticated ? <DashboardPage /> : <Navigate to="/login" replace />}
                      />
                      <Route
                        path="/inbox"
                        element={isAuthenticated ? <InboxPage /> : <Navigate to="/login" replace />}
                      />
                      <Route
                        path="/contacts"
                        element={isAuthenticated ? <ContactsPage /> : <Navigate to="/login" replace />}
                      />
                      <Route
                        path="/trust"
                        element={isAuthenticated ? <TrustNetworkPage /> : <Navigate to="/login" replace />}
                      />
                      <Route
                        path="/settings"
                        element={isAuthenticated ? <SettingsPage /> : <Navigate to="/login" replace />}
                      />
                      {/* Public demo page for Holochain trust demo */}
                      <Route
                        path="/trust-demo"
                        element={<TrustDemoPage />}
                      />
                    </Routes>
                  </Suspense>
                </main>
              </BrowserRouter>
            </ErrorBoundary>
          </AccessibilityProvider>
        </I18nProvider>
      </QueryClientProvider>
    </>
  );
}

export default App;
