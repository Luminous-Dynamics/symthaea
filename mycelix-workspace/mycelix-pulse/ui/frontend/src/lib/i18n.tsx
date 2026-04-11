// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Internationalization (i18n) Framework
 *
 * Multi-language support for Mycelix Mail:
 * - Translation loading and caching
 * - Locale detection
 * - Pluralization support
 * - Date/number formatting
 * - React hooks and components
 */

import { createContext, useContext, useState, useCallback, useEffect, ReactNode } from 'react';

// ============================================
// Types
// ============================================

export type Locale = 'en' | 'es' | 'fr' | 'de' | 'ja' | 'zh' | 'ko' | 'pt' | 'ru' | 'ar';

export interface TranslationData {
  [key: string]: string | TranslationData;
}

export interface I18nConfig {
  defaultLocale: Locale;
  supportedLocales: Locale[];
  fallbackLocale: Locale;
}

export interface I18nContextValue {
  locale: Locale;
  setLocale: (locale: Locale) => void;
  t: (key: string, params?: Record<string, string | number>) => string;
  formatDate: (date: Date | string, options?: Intl.DateTimeFormatOptions) => string;
  formatNumber: (num: number, options?: Intl.NumberFormatOptions) => string;
  formatRelativeTime: (date: Date | string) => string;
  isLoading: boolean;
}

// ============================================
// Translations
// ============================================

const translations: Record<Locale, TranslationData> = {
  en: {
    common: {
      loading: 'Loading...',
      error: 'An error occurred',
      retry: 'Retry',
      save: 'Save',
      cancel: 'Cancel',
      delete: 'Delete',
      edit: 'Edit',
      close: 'Close',
      search: 'Search',
      filter: 'Filter',
      sort: 'Sort',
      settings: 'Settings',
      help: 'Help',
    },
    nav: {
      inbox: 'Inbox',
      sent: 'Sent',
      drafts: 'Drafts',
      starred: 'Starred',
      trash: 'Trash',
      trustNetwork: 'Trust Network',
      identity: 'Identity',
      settings: 'Settings',
    },
    email: {
      compose: 'Compose',
      reply: 'Reply',
      replyAll: 'Reply All',
      forward: 'Forward',
      archive: 'Archive',
      markRead: 'Mark as Read',
      markUnread: 'Mark as Unread',
      star: 'Star',
      unstar: 'Unstar',
      noEmails: 'No emails',
      emailCount: '{count} email(s)',
    },
    trust: {
      trustScore: 'Trust Score',
      trustLevel: 'Trust Level',
      highTrust: 'High Trust',
      mediumTrust: 'Medium Trust',
      lowTrust: 'Low Trust',
      unknownTrust: 'Unknown',
      tier: 'Tier {tier}',
      tier0: 'Unverifiable',
      tier1: 'Email Verified',
      tier2: 'Identity Verified',
      tier3: 'In Trust Network',
      tier4: 'Fully Attested',
      connections: 'Connections',
      directTrust: 'Direct Trust',
      transitiveTrust: 'Transitive Trust',
      attestation: 'Attestation',
      attestations: 'Attestations',
      createAttestation: 'Create Attestation',
      revokeAttestation: 'Revoke Attestation',
      pathLength: 'Path Length',
    },
    ai: {
      aiInsights: 'AI Insights',
      intent: 'Intent',
      priority: 'Priority',
      summary: 'Summary',
      actionItems: 'Action Items',
      replySuggestions: 'Reply Suggestions',
      analyzing: 'Analyzing...',
    },
    settings: {
      general: 'General',
      trustSettings: 'Trust Settings',
      aiSettings: 'AI Settings',
      notificationSettings: 'Notifications',
      privacySettings: 'Privacy',
      displaySettings: 'Display',
      enabled: 'Enabled',
      disabled: 'Disabled',
      language: 'Language',
      theme: 'Theme',
      darkMode: 'Dark Mode',
      lightMode: 'Light Mode',
      systemDefault: 'System Default',
    },
    onboarding: {
      welcome: 'Welcome to Epistemic Mail',
      welcomeDesc: 'A new paradigm for trusted communication',
      epistemicTiers: 'Understanding Epistemic Tiers',
      verifyIdentity: 'Verify Your Identity',
      buildNetwork: 'Build Your Trust Network',
      aiFeatures: 'AI-Powered Features',
      ready: "You're All Set!",
      next: 'Next',
      back: 'Back',
      skip: 'Skip',
      finish: 'Get Started',
    },
    notifications: {
      newAttestation: 'New Attestation',
      trustScoreChanged: 'Trust Score Changed',
      introductionRequest: 'Introduction Request',
      credentialExpiring: 'Credential Expiring',
      noNotifications: 'No notifications',
      markAllRead: 'Mark all as read',
    },
    errors: {
      networkError: 'Network error. Please check your connection.',
      authError: 'Authentication failed. Please log in again.',
      notFound: 'Not found',
      serverError: 'Server error. Please try again later.',
      validationError: 'Please check your input',
    },
  },

  es: {
    common: {
      loading: 'Cargando...',
      error: 'Ocurrió un error',
      retry: 'Reintentar',
      save: 'Guardar',
      cancel: 'Cancelar',
      delete: 'Eliminar',
      edit: 'Editar',
      close: 'Cerrar',
      search: 'Buscar',
      filter: 'Filtrar',
      sort: 'Ordenar',
      settings: 'Configuración',
      help: 'Ayuda',
    },
    nav: {
      inbox: 'Bandeja de entrada',
      sent: 'Enviados',
      drafts: 'Borradores',
      starred: 'Destacados',
      trash: 'Papelera',
      trustNetwork: 'Red de Confianza',
      identity: 'Identidad',
      settings: 'Configuración',
    },
    trust: {
      trustScore: 'Puntuación de Confianza',
      highTrust: 'Alta Confianza',
      mediumTrust: 'Confianza Media',
      lowTrust: 'Baja Confianza',
      attestation: 'Atestación',
      createAttestation: 'Crear Atestación',
    },
  },

  fr: {
    common: {
      loading: 'Chargement...',
      error: 'Une erreur est survenue',
      retry: 'Réessayer',
      save: 'Enregistrer',
      cancel: 'Annuler',
      delete: 'Supprimer',
      edit: 'Modifier',
      close: 'Fermer',
      search: 'Rechercher',
      settings: 'Paramètres',
    },
    nav: {
      inbox: 'Boîte de réception',
      sent: 'Envoyés',
      drafts: 'Brouillons',
      trustNetwork: 'Réseau de Confiance',
      identity: 'Identité',
      settings: 'Paramètres',
    },
    trust: {
      trustScore: 'Score de Confiance',
      highTrust: 'Haute Confiance',
      mediumTrust: 'Confiance Moyenne',
      lowTrust: 'Faible Confiance',
    },
  },

  de: {
    common: {
      loading: 'Laden...',
      error: 'Ein Fehler ist aufgetreten',
      retry: 'Wiederholen',
      save: 'Speichern',
      cancel: 'Abbrechen',
      delete: 'Löschen',
      settings: 'Einstellungen',
    },
    nav: {
      inbox: 'Posteingang',
      sent: 'Gesendet',
      trustNetwork: 'Vertrauensnetzwerk',
      settings: 'Einstellungen',
    },
    trust: {
      trustScore: 'Vertrauenswert',
      highTrust: 'Hohes Vertrauen',
      lowTrust: 'Niedriges Vertrauen',
    },
  },

  ja: {
    common: {
      loading: '読み込み中...',
      error: 'エラーが発生しました',
      save: '保存',
      cancel: 'キャンセル',
      settings: '設定',
    },
    nav: {
      inbox: '受信トレイ',
      sent: '送信済み',
      trustNetwork: '信頼ネットワーク',
    },
    trust: {
      trustScore: '信頼スコア',
    },
  },

  zh: {
    common: {
      loading: '加载中...',
      save: '保存',
      cancel: '取消',
      settings: '设置',
    },
    nav: {
      inbox: '收件箱',
      trustNetwork: '信任网络',
    },
  },

  ko: {
    common: {
      loading: '로딩 중...',
      save: '저장',
      settings: '설정',
    },
    nav: {
      inbox: '받은편지함',
    },
  },

  pt: {
    common: {
      loading: 'Carregando...',
      save: 'Salvar',
      settings: 'Configurações',
    },
    nav: {
      inbox: 'Caixa de entrada',
      trustNetwork: 'Rede de Confiança',
    },
  },

  ru: {
    common: {
      loading: 'Загрузка...',
      save: 'Сохранить',
      settings: 'Настройки',
    },
    nav: {
      inbox: 'Входящие',
      trustNetwork: 'Сеть доверия',
    },
  },

  ar: {
    common: {
      loading: 'جاري التحميل...',
      save: 'حفظ',
      settings: 'الإعدادات',
    },
    nav: {
      inbox: 'صندوق الوارد',
    },
  },
};

// ============================================
// Utility Functions
// ============================================

function getNestedValue(obj: TranslationData, path: string): string | undefined {
  const keys = path.split('.');
  let current: TranslationData | string = obj;

  for (const key of keys) {
    if (typeof current === 'string') return undefined;
    current = current[key];
    if (current === undefined) return undefined;
  }

  return typeof current === 'string' ? current : undefined;
}

function interpolate(template: string, params: Record<string, string | number>): string {
  return template.replace(/\{(\w+)\}/g, (_, key) => String(params[key] ?? `{${key}}`));
}

function detectLocale(): Locale {
  // Check localStorage first
  const stored = localStorage.getItem('mycelix_locale');
  if (stored && translations[stored as Locale]) {
    return stored as Locale;
  }

  // Check browser language
  const browserLang = navigator.language.split('-')[0];
  if (translations[browserLang as Locale]) {
    return browserLang as Locale;
  }

  return 'en';
}

// ============================================
// React Context
// ============================================

const I18nContext = createContext<I18nContextValue | null>(null);

export function I18nProvider({ children, initialLocale }: { children: ReactNode; initialLocale?: Locale }) {
  const [locale, setLocaleState] = useState<Locale>(initialLocale || detectLocale());
  const [isLoading, setIsLoading] = useState(false);

  const setLocale = useCallback((newLocale: Locale) => {
    setLocaleState(newLocale);
    localStorage.setItem('mycelix_locale', newLocale);
    document.documentElement.lang = newLocale;
    document.documentElement.dir = newLocale === 'ar' ? 'rtl' : 'ltr';
  }, []);

  // Set initial document attributes
  useEffect(() => {
    document.documentElement.lang = locale;
    document.documentElement.dir = locale === 'ar' ? 'rtl' : 'ltr';
  }, [locale]);

  const t = useCallback(
    (key: string, params?: Record<string, string | number>): string => {
      // Try current locale
      let value = getNestedValue(translations[locale], key);

      // Fallback to English
      if (!value && locale !== 'en') {
        value = getNestedValue(translations.en, key);
      }

      // Return key if not found
      if (!value) return key;

      // Interpolate params
      return params ? interpolate(value, params) : value;
    },
    [locale]
  );

  const formatDate = useCallback(
    (date: Date | string, options?: Intl.DateTimeFormatOptions): string => {
      const d = typeof date === 'string' ? new Date(date) : date;
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
      const diffSec = Math.floor(diffMs / 1000);
      const diffMin = Math.floor(diffSec / 60);
      const diffHour = Math.floor(diffMin / 60);
      const diffDay = Math.floor(diffHour / 24);

      const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });

      if (diffDay > 0) return rtf.format(-diffDay, 'day');
      if (diffHour > 0) return rtf.format(-diffHour, 'hour');
      if (diffMin > 0) return rtf.format(-diffMin, 'minute');
      return rtf.format(-diffSec, 'second');
    },
    [locale]
  );

  const value: I18nContextValue = {
    locale,
    setLocale,
    t,
    formatDate,
    formatNumber,
    formatRelativeTime,
    isLoading,
  };

  return <I18nContext.Provider value={value}>{children}</I18nContext.Provider>;
}

// ============================================
// Hooks
// ============================================

export function useI18n(): I18nContextValue {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return context;
}

export function useTranslation(namespace?: string) {
  const { t, ...rest } = useI18n();

  const tNamespaced = useCallback(
    (key: string, params?: Record<string, string | number>) => {
      const fullKey = namespace ? `${namespace}.${key}` : key;
      return t(fullKey, params);
    },
    [t, namespace]
  );

  return { t: tNamespaced, ...rest };
}

// ============================================
// Components
// ============================================

interface TransProps {
  i18nKey: string;
  params?: Record<string, string | number>;
  as?: keyof JSX.IntrinsicElements;
  className?: string;
}

export function Trans({ i18nKey, params, as: Component = 'span', className }: TransProps) {
  const { t } = useI18n();
  return <Component className={className}>{t(i18nKey, params)}</Component>;
}

export function LocaleSelector({ className = '' }: { className?: string }) {
  const { locale, setLocale } = useI18n();

  const localeNames: Record<Locale, string> = {
    en: 'English',
    es: 'Español',
    fr: 'Français',
    de: 'Deutsch',
    ja: '日本語',
    zh: '中文',
    ko: '한국어',
    pt: 'Português',
    ru: 'Русский',
    ar: 'العربية',
  };

  return (
    <select
      value={locale}
      onChange={(e) => setLocale(e.target.value as Locale)}
      className={`px-3 py-2 bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-lg ${className}`}
    >
      {Object.entries(localeNames).map(([code, name]) => (
        <option key={code} value={code}>
          {name}
        </option>
      ))}
    </select>
  );
}

// ============================================
// Export helpers
// ============================================

export const supportedLocales: Locale[] = ['en', 'es', 'fr', 'de', 'ja', 'zh', 'ko', 'pt', 'ru', 'ar'];

export function getLocaleDirection(locale: Locale): 'ltr' | 'rtl' {
  return locale === 'ar' ? 'rtl' : 'ltr';
}

export default {
  I18nProvider,
  useI18n,
  useTranslation,
  Trans,
  LocaleSelector,
  supportedLocales,
};
