// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Preview } from '@storybook/react';
import React from 'react';
import '../src/index.css';

// Mock I18n context for Storybook
const mockI18n = {
  locale: 'en' as const,
  setLocale: () => {},
  t: (key: string, params?: Record<string, string | number>) => {
    const translations: Record<string, string> = {
      'email.star': 'Star',
      'email.unstar': 'Unstar',
      'email.markRead': 'Mark as read',
      'email.markUnread': 'Mark as unread',
      'email.archive': 'Archive',
      'email.delete': 'Delete',
      'email.noSubject': '(No subject)',
      'email.attachments': 'attachments',
      'trust.trusted': 'Trusted',
      'trust.known': 'Known',
      'trust.caution': 'Caution',
      'trust.unknown': 'Unknown',
    };
    let result = translations[key] || key;
    if (params) {
      Object.entries(params).forEach(([k, v]) => {
        result = result.replace(`{${k}}`, String(v));
      });
    }
    return result;
  },
  formatDate: (date: Date | string) => {
    const d = typeof date === 'string' ? new Date(date) : date;
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  },
  formatNumber: (num: number) => num.toLocaleString('en-US'),
  formatRelativeTime: (date: Date | string) => {
    const d = typeof date === 'string' ? new Date(date) : date;
    const now = new Date();
    const diffMs = now.getTime() - d.getTime();
    const diffMins = Math.floor(diffMs / 60000);
    const diffHours = Math.floor(diffMins / 60);
    const diffDays = Math.floor(diffHours / 24);

    if (diffMins < 1) return 'just now';
    if (diffMins < 60) return `${diffMins}m ago`;
    if (diffHours < 24) return `${diffHours}h ago`;
    if (diffDays < 7) return `${diffDays}d ago`;
    return d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' });
  },
  isRTL: false,
  dir: 'ltr' as const,
};

// Mock Accessibility context for Storybook
const mockAccessibility = {
  reduceMotion: false,
  highContrast: false,
  fontSize: 'medium' as const,
  focusVisible: true,
  announcePageChanges: true,
  keyboardShortcutsEnabled: true,
  updateSetting: () => {},
  announce: (message: string) => console.log('[A11y Announce]:', message),
  focusTrap: () => undefined,
};

// Create mock provider contexts
const I18nContext = React.createContext(mockI18n);
const AccessibilityContext = React.createContext(mockAccessibility);

// Mock provider wrapper
const MockProviders: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  return React.createElement(
    I18nContext.Provider,
    { value: mockI18n },
    React.createElement(
      AccessibilityContext.Provider,
      { value: mockAccessibility },
      React.createElement('div', { className: 'storybook-root p-4' }, children)
    )
  );
};

// Export mock hooks for use in stories
(window as any).__STORYBOOK_MOCKS__ = {
  useI18n: () => React.useContext(I18nContext),
  useAccessibility: () => React.useContext(AccessibilityContext),
  useTrustScore: () => ({ data: { score: 0.75 }, isLoading: false, error: null }),
};

const preview: Preview = {
  parameters: {
    actions: { argTypesRegex: '^on[A-Z].*' },
    controls: {
      matchers: {
        color: /(background|color)$/i,
        date: /Date$/i,
      },
    },
    backgrounds: {
      default: 'light',
      values: [
        { name: 'light', value: '#f9fafb' },
        { name: 'dark', value: '#111827' },
        { name: 'white', value: '#ffffff' },
        { name: 'mycelix-light', value: '#f0fdf4' },
        { name: 'mycelix-dark', value: '#052e16' },
      ],
    },
    layout: 'centered',
    viewport: {
      viewports: {
        mobile: { name: 'Mobile', styles: { width: '375px', height: '667px' } },
        tablet: { name: 'Tablet', styles: { width: '768px', height: '1024px' } },
        desktop: { name: 'Desktop', styles: { width: '1440px', height: '900px' } },
      },
    },
    docs: {
      toc: true,
      source: { language: 'tsx' },
    },
  },
  globalTypes: {
    theme: {
      name: 'Theme',
      description: 'Global theme for components',
      defaultValue: 'light',
      toolbar: {
        icon: 'circlehollow',
        items: [
          { value: 'light', title: 'Light', icon: 'sun' },
          { value: 'dark', title: 'Dark', icon: 'moon' },
        ],
        showName: true,
      },
    },
    trustLevel: {
      name: 'Trust Level',
      description: 'Simulated trust score for testing',
      defaultValue: 'trusted',
      toolbar: {
        icon: 'shield',
        items: [
          { value: 'trusted', title: 'Trusted (>=80%)' },
          { value: 'known', title: 'Known (50-79%)' },
          { value: 'caution', title: 'Caution (20-49%)' },
          { value: 'unknown', title: 'Unknown (<20%)' },
        ],
        showName: true,
      },
    },
  },
  decorators: [
    (Story, context) => {
      const theme = context.globals.theme;
      document.documentElement.classList.toggle('dark', theme === 'dark');

      // Provide mock trust score based on global
      const trustLevels: Record<string, number> = {
        trusted: 0.92,
        known: 0.65,
        caution: 0.35,
        unknown: 0.1,
      };
      const mockTrustScore = trustLevels[context.globals.trustLevel] || 0.5;

      // Inject mock trust score into window for components that need it
      (window as any).__STORYBOOK_MOCK_TRUST_SCORE__ = mockTrustScore;

      return React.createElement(MockProviders, null, Story());
    },
  ],
};

export default preview;
