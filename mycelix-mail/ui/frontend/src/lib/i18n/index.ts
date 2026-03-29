// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Internationalization (i18n) for Mycelix Mail
 *
 * Provides translation support with React context and hooks
 */

import React, { createContext, useContext, useState, useCallback, useEffect } from 'react';

// Types
export type Locale = 'en' | 'es' | 'fr' | 'de' | 'ja' | 'zh' | 'ar' | 'he';

export interface TranslationStrings {
  [key: string]: string | TranslationStrings;
}

export interface I18nContextValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: string, params?: Record<string, string | number>) => string;
  formatDate: (date: Date | string, style?: 'short' | 'medium' | 'long') => string;
  formatNumber: (num: number, options?: Intl.NumberFormatOptions) => string;
  formatRelativeTime: (date: Date | string) => string;
  isRTL: boolean;
  dir: 'ltr' | 'rtl';
}

// RTL languages
const RTL_LANGUAGES: Locale[] = ['ar', 'he'];

// Default English translations
const defaultTranslations: TranslationStrings = {
  common: {
    loading: 'Loading...',
    error: 'An error occurred',
    retry: 'Retry',
    cancel: 'Cancel',
    save: 'Save',
    delete: 'Delete',
    edit: 'Edit',
    close: 'Close',
    search: 'Search',
    noResults: 'No results found',
  },
  auth: {
    login: 'Sign In',
    logout: 'Sign Out',
    loginWith: 'Sign in with {provider}',
    loggingIn: 'Signing in...',
    sessionExpired: 'Your session has expired. Please sign in again.',
  },
  email: {
    inbox: 'Inbox',
    sent: 'Sent',
    drafts: 'Drafts',
    trash: 'Trash',
    spam: 'Spam',
    starred: 'Starred',
    archived: 'Archived',
    compose: 'Compose',
    reply: 'Reply',
    replyAll: 'Reply All',
    forward: 'Forward',
    markRead: 'Mark as read',
    markUnread: 'Mark as unread',
    star: 'Star',
    unstar: 'Unstar',
    archive: 'Archive',
    unarchive: 'Unarchive',
    delete: 'Delete',
    moveToTrash: 'Move to Trash',
    subject: 'Subject',
    to: 'To',
    cc: 'Cc',
    bcc: 'Bcc',
    from: 'From',
    date: 'Date',
    attachments: 'Attachments',
    noEmails: 'No emails to display',
    send: 'Send',
    saveDraft: 'Save Draft',
    discard: 'Discard',
    sending: 'Sending...',
    sent: 'Email sent successfully',
    unreadCount: '{count} unread',
  },
  trust: {
    score: 'Trust Score',
    trusted: 'Trusted',
    known: 'Known',
    caution: 'Caution',
    unknown: 'Unknown',
    attestation: 'Trust Attestation',
    attestations: 'Trust Attestations',
    createAttestation: 'Create Attestation',
    revokeAttestation: 'Revoke Attestation',
    trustPath: 'Trust Path',
    noPath: 'No trust path found',
    pathLength: '{length} degrees of separation',
  },
  contacts: {
    contacts: 'Contacts',
    addContact: 'Add Contact',
    editContact: 'Edit Contact',
    deleteContact: 'Delete Contact',
    block: 'Block',
    unblock: 'Unblock',
    favorite: 'Add to Favorites',
    unfavorite: 'Remove from Favorites',
    noContacts: 'No contacts yet',
    searchContacts: 'Search contacts...',
  },
  settings: {
    settings: 'Settings',
    profile: 'Profile',
    security: 'Security',
    notifications: 'Notifications',
    appearance: 'Appearance',
    language: 'Language',
    theme: 'Theme',
    themeLight: 'Light',
    themeDark: 'Dark',
    themeSystem: 'System',
    accessibility: 'Accessibility',
    reduceMotion: 'Reduce motion',
    highContrast: 'High contrast',
    fontSize: 'Font size',
    fontSizeSmall: 'Small',
    fontSizeMedium: 'Medium',
    fontSizeLarge: 'Large',
  },
  accessibility: {
    skipToContent: 'Skip to content',
    openMenu: 'Open menu',
    closeMenu: 'Close menu',
    loading: 'Loading content',
    required: 'Required field',
    selected: 'Selected',
    expanded: 'Expanded',
    collapsed: 'Collapsed',
  },
  errors: {
    networkError: 'Network error. Please check your connection.',
    serverError: 'Server error. Please try again later.',
    notFound: 'Not found',
    unauthorized: 'You are not authorized to perform this action.',
    validationError: 'Please check your input and try again.',
  },
};

// Translation storage
const translations: Record<Locale, TranslationStrings> = {
  en: defaultTranslations,
  es: {}, // Spanish translations would go here
  fr: {}, // French translations
  de: {}, // German translations
  ja: {}, // Japanese translations
  zh: {}, // Chinese translations
  ar: {}, // Arabic translations
  he: {}, // Hebrew translations
};

// Context
const I18nContext = createContext<I18nContextValue | null>(null);

// Helper to get nested translation
function getNestedValue(obj: TranslationStrings, path: string): string | undefined {
  const parts = path.split('.');
  let current: TranslationStrings | string = obj;

  for (const part of parts) {
    if (typeof current === 'object' && part in current) {
      current = current[part];
    } else {
      return undefined;
    }
  }

  return typeof current === 'string' ? current : undefined;
}

// Interpolate parameters
function interpolate(template: string, params: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (_, key) => {
    return params[key]?.toString() ?? `{${key}}`;
  });
}

// Provider component
export function I18nProvider({ children }: { children: React.ReactNode }) {
  const [locale, setLocaleState] = useState<Locale>(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('mycelix_locale') as Locale;
      if (stored && translations[stored]) return stored;

      // Detect browser language
      const browserLang = navigator.language.split('-')[0] as Locale;
      if (translations[browserLang]) return browserLang;
    }
    return 'en';
  });

  const isRTL = RTL_LANGUAGES.includes(locale);
  const dir = isRTL ? 'rtl' : 'ltr';

  // Update document direction
  useEffect(() => {
    document.documentElement.dir = dir;
    document.documentElement.lang = locale;
  }, [locale, dir]);

  const setLocale = useCallback((newLocale: Locale) => {
    setLocaleState(newLocale);
    localStorage.setItem('mycelix_locale', newLocale);
  }, []);

  const t = useCallback(
    (key: string, params?: Record<string, string | number>): string => {
      // Try current locale first
      let value = getNestedValue(translations[locale], key);

      // Fall back to English
      if (!value && locale !== 'en') {
        value = getNestedValue(translations.en, key);
      }

      // Return key if not found
      if (!value) return key;

      // Interpolate parameters
      if (params) {
        return interpolate(value, params);
      }

      return value;
    },
    [locale]
  );

  const formatDate = useCallback(
    (date: Date | string, style: 'short' | 'medium' | 'long' = 'medium'): string => {
      const d = typeof date === 'string' ? new Date(date) : date;

      const options: Intl.DateTimeFormatOptions = {
        short: { month: 'numeric', day: 'numeric' },
        medium: { month: 'short', day: 'numeric', year: 'numeric' },
        long: { weekday: 'long', month: 'long', day: 'numeric', year: 'numeric' },
      }[style];

      return new Intl.DateTimeFormat(locale, options).format(d);
    },
    [locale]
  );

  const formatNumber = useCallback(
    (num: number, options?: Intl.NumberFormatOptions): string => {
      return new Intl.NumberFormat(locale, options).format(num);
    },
    [locale]
  );

  const formatRelativeTime = useCallback(
    (date: Date | string): string => {
      const d = typeof date === 'string' ? new Date(date) : date;
      const now = new Date();
      const diffMs = now.getTime() - d.getTime();
      const diffSecs = Math.floor(diffMs / 1000);
      const diffMins = Math.floor(diffSecs / 60);
      const diffHours = Math.floor(diffMins / 60);
      const diffDays = Math.floor(diffHours / 24);

      const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });

      if (diffSecs < 60) return rtf.format(-diffSecs, 'seconds');
      if (diffMins < 60) return rtf.format(-diffMins, 'minutes');
      if (diffHours < 24) return rtf.format(-diffHours, 'hours');
      if (diffDays < 7) return rtf.format(-diffDays, 'days');

      return formatDate(d, 'short');
    },
    [locale, formatDate]
  );

  const value: I18nContextValue = {
    locale,
    setLocale,
    t,
    formatDate,
    formatNumber,
    formatRelativeTime,
    isRTL,
    dir,
  };

  return React.createElement(I18nContext.Provider, { value }, children);
}

// Hook
export function useI18n(): I18nContextValue {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return context;
}

// Shorthand hook for translations
export function useTranslation() {
  const { t, locale } = useI18n();
  return { t, locale };
}

// Available locales for settings
export const AVAILABLE_LOCALES: { code: Locale; name: string; nativeName: string }[] = [
  { code: 'en', name: 'English', nativeName: 'English' },
  { code: 'es', name: 'Spanish', nativeName: 'Espanol' },
  { code: 'fr', name: 'French', nativeName: 'Francais' },
  { code: 'de', name: 'German', nativeName: 'Deutsch' },
  { code: 'ja', name: 'Japanese', nativeName: '日本語' },
  { code: 'zh', name: 'Chinese', nativeName: '中文' },
  { code: 'ar', name: 'Arabic', nativeName: 'العربية' },
  { code: 'he', name: 'Hebrew', nativeName: 'עברית' },
];
