// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Epistemic App Integration
 *
 * Main application shell integrating all epistemic features:
 * - Routing and navigation
 * - Global state providers
 * - Error boundaries
 * - Keyboard shortcuts
 * - Notifications
 */

import { useState, useEffect, Suspense, lazy } from 'react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

// UI Components
import { ErrorBoundary, NetworkErrorBoundary } from '../ui/ErrorBoundary';
import { DashboardSkeleton, EmailListSkeleton, SettingsSkeleton } from '../ui/Skeleton';

// Epistemic Components
import KeyboardShortcuts, { useKeyboardShortcuts } from '../help/KeyboardShortcuts';
import TrustNotifications, { NotificationBell, useTrustNotifications } from '../notifications/TrustNotifications';
import { useEpistemicSettings } from '../settings/EpistemicSettings';
import { TrustConnectionStatus } from '@/hooks/useTrustWebSocket';

// Lazy load heavy components
const TrustDashboard = lazy(() => import('../trust/TrustDashboard'));
const EpistemicSettings = lazy(() => import('../settings/EpistemicSettings'));
const EpistemicOnboarding = lazy(() => import('../onboarding/EpistemicOnboarding'));
const ExternalIdentityProviders = lazy(() => import('../identity/ExternalProviders'));

// ============================================
// Query Client
// ============================================

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 5 * 60 * 1000, // 5 minutes
      retry: 2,
      refetchOnWindowFocus: false,
    },
    mutations: {
      retry: 1,
    },
  },
});

// ============================================
// Navigation Types
// ============================================

type AppView = 'inbox' | 'dashboard' | 'settings' | 'identity' | 'onboarding';

interface NavItem {
  id: AppView;
  label: string;
  icon: string;
  badge?: number;
}

// ============================================
// Navigation Component
// ============================================

interface NavigationProps {
  currentView: AppView;
  onNavigate: (view: AppView) => void;
  unreadCount?: number;
  notificationCount?: number;
}

function Navigation({ currentView, onNavigate, unreadCount, notificationCount }: NavigationProps) {
  const navItems: NavItem[] = [
    { id: 'inbox', label: 'Inbox', icon: '📥', badge: unreadCount },
    { id: 'dashboard', label: 'Trust Network', icon: '🔗' },
    { id: 'identity', label: 'Identity', icon: '🪪' },
    { id: 'settings', label: 'Settings', icon: '⚙️' },
  ];

  return (
    <nav className="flex items-center gap-1 p-1 bg-gray-100 dark:bg-gray-800 rounded-lg">
      {navItems.map((item) => (
        <button
          key={item.id}
          onClick={() => onNavigate(item.id)}
          className={`
            relative flex items-center gap-2 px-4 py-2 rounded-md text-sm font-medium transition-colors
            ${currentView === item.id
              ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm'
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
            }
          `}
        >
          <span>{item.icon}</span>
          <span className="hidden sm:inline">{item.label}</span>
          {item.badge && item.badge > 0 && (
            <span className="absolute -top-1 -right-1 min-w-[18px] h-[18px] flex items-center justify-center text-xs font-bold text-white bg-blue-600 rounded-full">
              {item.badge > 99 ? '99+' : item.badge}
            </span>
          )}
        </button>
      ))}
    </nav>
  );
}

// ============================================
// Header Component
// ============================================

interface HeaderProps {
  userDid: string;
  onShowShortcuts: () => void;
  isConnected: boolean;
  isConnecting: boolean;
  connectionError: Error | null;
  onReconnect: () => void;
}

function Header({
  userDid,
  onShowShortcuts,
  isConnected,
  isConnecting,
  connectionError,
  onReconnect,
}: HeaderProps) {
  return (
    <header className="flex items-center justify-between px-4 py-3 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-800">
      <div className="flex items-center gap-4">
        <div className="flex items-center gap-2">
          <span className="text-2xl">🍄</span>
          <span className="font-bold text-xl text-gray-900 dark:text-gray-100">
            Mycelix
          </span>
        </div>

        <TrustConnectionStatus
          isConnected={isConnected}
          isConnecting={isConnecting}
          error={connectionError}
          onReconnect={onReconnect}
        />
      </div>

      <div className="flex items-center gap-3">
        {/* Keyboard shortcuts button */}
        <button
          onClick={onShowShortcuts}
          className="p-2 text-gray-500 hover:text-gray-700 dark:hover:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 rounded-lg transition-colors"
          title="Keyboard shortcuts (?)"
        >
          <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8.228 9c.549-1.165 2.03-2 3.772-2 2.21 0 4 1.343 4 3 0 1.4-1.278 2.575-3.006 2.907-.542.104-.994.54-.994 1.093m0 3h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
        </button>

        {/* Notifications */}
        <NotificationBell />

        {/* User DID */}
        <div className="hidden md:flex items-center gap-2 px-3 py-1.5 bg-gray-100 dark:bg-gray-800 rounded-lg">
          <div className="w-2 h-2 rounded-full bg-emerald-500" />
          <span className="text-sm font-mono text-gray-600 dark:text-gray-400">
            {userDid.slice(-12)}
          </span>
        </div>
      </div>
    </header>
  );
}

// ============================================
// Main App Shell
// ============================================

interface EpistemicAppShellProps {
  userDid: string;
  children?: React.ReactNode;
}

export function EpistemicAppShell({ userDid, children }: EpistemicAppShellProps) {
  const [currentView, setCurrentView] = useState<AppView>('inbox');
  const [showOnboarding, setShowOnboarding] = useState(false);

  // Hooks
  const shortcutsModal = useKeyboardShortcuts();
  const settings = useEpistemicSettings();
  const notifications = useTrustNotifications(userDid);

  // Check if onboarding needed (first time user)
  useEffect(() => {
    const hasCompletedOnboarding = localStorage.getItem('mycelix_onboarding_complete');
    if (!hasCompletedOnboarding) {
      setShowOnboarding(true);
    }
  }, []);

  const handleOnboardingComplete = () => {
    localStorage.setItem('mycelix_onboarding_complete', 'true');
    setShowOnboarding(false);
  };

  // Mock connection state - replace with actual WebSocket hook
  const [isConnected, setIsConnected] = useState(true);
  const [isConnecting, setIsConnecting] = useState(false);
  const [connectionError, setConnectionError] = useState<Error | null>(null);

  const handleReconnect = () => {
    setIsConnecting(true);
    setTimeout(() => {
      setIsConnecting(false);
      setIsConnected(true);
      setConnectionError(null);
    }, 1000);
  };

  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        <div className="min-h-screen bg-gray-50 dark:bg-gray-950">
          {/* Header */}
          <Header
            userDid={userDid}
            onShowShortcuts={shortcutsModal.open}
            isConnected={isConnected}
            isConnecting={isConnecting}
            connectionError={connectionError}
            onReconnect={handleReconnect}
          />

          {/* Main content */}
          <main className="max-w-7xl mx-auto px-4 py-6">
            {/* Navigation */}
            <div className="mb-6 flex items-center justify-between">
              <Navigation
                currentView={currentView}
                onNavigate={setCurrentView}
                unreadCount={5}
              />
            </div>

            {/* View content */}
            <NetworkErrorBoundary>
              {currentView === 'inbox' && (
                <div className="bg-white dark:bg-gray-900 rounded-xl shadow-sm border border-gray-200 dark:border-gray-800">
                  {children || (
                    <div className="p-8 text-center text-gray-500">
                      Email list would go here
                    </div>
                  )}
                </div>
              )}

              {currentView === 'dashboard' && (
                <Suspense fallback={<DashboardSkeleton />}>
                  <TrustDashboard userDid={userDid} />
                </Suspense>
              )}

              {currentView === 'identity' && (
                <Suspense fallback={<SettingsSkeleton />}>
                  <div className="bg-white dark:bg-gray-900 rounded-xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
                    <ExternalIdentityProviders />
                  </div>
                </Suspense>
              )}

              {currentView === 'settings' && (
                <Suspense fallback={<SettingsSkeleton />}>
                  <div className="bg-white dark:bg-gray-900 rounded-xl shadow-sm border border-gray-200 dark:border-gray-800 p-6">
                    <EpistemicSettings />
                  </div>
                </Suspense>
              )}
            </NetworkErrorBoundary>
          </main>

          {/* Notifications panel */}
          <TrustNotifications />

          {/* Keyboard shortcuts modal */}
          <KeyboardShortcuts
            isOpen={shortcutsModal.isOpen}
            onClose={shortcutsModal.close}
          />

          {/* Onboarding modal */}
          {showOnboarding && (
            <Suspense fallback={null}>
              <EpistemicOnboarding onComplete={handleOnboardingComplete} />
            </Suspense>
          )}
        </div>
      </ErrorBoundary>
    </QueryClientProvider>
  );
}

// ============================================
// Standalone Providers Wrapper
// ============================================

interface EpistemicProvidersProps {
  children: React.ReactNode;
}

export function EpistemicProviders({ children }: EpistemicProvidersProps) {
  return (
    <QueryClientProvider client={queryClient}>
      <ErrorBoundary>
        {children}
      </ErrorBoundary>
    </QueryClientProvider>
  );
}

// ============================================
// Export App Entry Point
// ============================================

interface EpistemicAppProps {
  userDid?: string;
}

export default function EpistemicApp({ userDid = 'did:mycelix:self' }: EpistemicAppProps) {
  return <EpistemicAppShell userDid={userDid} />;
}
