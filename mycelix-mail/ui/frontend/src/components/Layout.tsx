// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import { Link } from 'react-router-dom';
import { useQuery } from '@tanstack/react-query';
import { useAuthStore } from '@/store/authStore';
import ThemeToggle from '@/components/ThemeToggle';
import { api } from '@/services/api';

interface LayoutProps {
  children: React.ReactNode;
}

export default function Layout({ children }: LayoutProps) {
  const { user, logout } = useAuthStore();
  const { data: trustHealth, error: trustHealthError } = useQuery({
    queryKey: ['trust-health-header'],
    queryFn: () => api.getTrustHealth(),
    retry: 0,
  });

  const trustStatus = (() => {
    if (trustHealthError) return { label: 'Trust: Unreachable', classes: 'bg-rose-100 text-rose-700 dark:bg-rose-900/30 dark:text-rose-200' };
    if (trustHealth?.providerConfigured) return { label: 'Trust: Live', classes: 'bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-200' };
    return { label: 'Trust: Not configured', classes: 'bg-gray-100 text-gray-700 dark:bg-gray-800 dark:text-gray-200' };
  })();

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900 transition-colors">
      <header className="bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700">
        <div className="px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center">
              <Link to="/" className="text-2xl font-bold text-primary-600 dark:text-primary-400">
                Mycelix Mail
              </Link>
            </div>

            <div className="flex items-center space-x-4">
              <span
                className={`text-xs font-semibold px-2 py-1 rounded-full border ${trustStatus.classes} border-transparent`}
                role="status"
                aria-live="polite"
              >
                {trustStatus.label}
              </span>
              <span className="text-sm text-gray-700 dark:text-gray-300">
                {user?.firstName || user?.email}
              </span>
              <ThemeToggle />
              <Link
                to="/settings"
                className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100"
              >
                Settings
              </Link>
              <button
                onClick={logout}
                className="text-sm text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100"
              >
                Logout
              </button>
            </div>
          </div>
        </div>
      </header>

      <main>{children}</main>
    </div>
  );
}
