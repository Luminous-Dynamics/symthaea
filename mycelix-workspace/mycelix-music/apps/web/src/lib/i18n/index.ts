// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Internationalization (i18n) System
 *
 * Complete i18n solution with:
 * - 20+ language support
 * - RTL layout support
 * - Pluralization rules
 * - Date/time/number formatting
 * - Dynamic loading of translations
 * - React context and hooks
 */

import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';

// ==================== Types ====================

export type LocaleCode =
  | 'en' | 'en-GB' | 'es' | 'es-MX' | 'fr' | 'fr-CA' | 'de' | 'it' | 'pt' | 'pt-BR'
  | 'nl' | 'pl' | 'ru' | 'uk' | 'ja' | 'ko' | 'zh' | 'zh-TW' | 'ar' | 'he'
  | 'hi' | 'th' | 'vi' | 'id' | 'tr' | 'sv' | 'no' | 'da' | 'fi';

export interface LocaleConfig {
  code: LocaleCode;
  name: string;
  nativeName: string;
  direction: 'ltr' | 'rtl';
  dateFormat: string;
  timeFormat: string;
  numberFormat: Intl.NumberFormatOptions;
  currencyCode: string;
  pluralRules: Intl.PluralRules;
}

export type TranslationValue = string | { [key: string]: TranslationValue };
export type Translations = Record<string, TranslationValue>;

export interface I18nContextValue {
  locale: LocaleCode;
  direction: 'ltr' | 'rtl';
  setLocale: (locale: LocaleCode) => Promise<void>;
  t: (key: string, params?: Record<string, string | number>) => string;
  tc: (key: string, count: number, params?: Record<string, string | number>) => string;
  formatNumber: (value: number, options?: Intl.NumberFormatOptions) => string;
  formatCurrency: (value: number, currency?: string) => string;
  formatDate: (date: Date | number, options?: Intl.DateTimeFormatOptions) => string;
  formatTime: (date: Date | number, options?: Intl.DateTimeFormatOptions) => string;
  formatRelativeTime: (date: Date | number) => string;
  formatDuration: (seconds: number) => string;
  availableLocales: LocaleCode[];
  isLoading: boolean;
}

// ==================== Locale Configurations ====================

export const localeConfigs: Record<LocaleCode, LocaleConfig> = {
  'en': {
    code: 'en',
    name: 'English',
    nativeName: 'English',
    direction: 'ltr',
    dateFormat: 'MM/DD/YYYY',
    timeFormat: 'h:mm A',
    numberFormat: { notation: 'standard' },
    currencyCode: 'USD',
    pluralRules: new Intl.PluralRules('en'),
  },
  'en-GB': {
    code: 'en-GB',
    name: 'English (UK)',
    nativeName: 'English (UK)',
    direction: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'GBP',
    pluralRules: new Intl.PluralRules('en-GB'),
  },
  'es': {
    code: 'es',
    name: 'Spanish',
    nativeName: 'Espanol',
    direction: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'EUR',
    pluralRules: new Intl.PluralRules('es'),
  },
  'es-MX': {
    code: 'es-MX',
    name: 'Spanish (Mexico)',
    nativeName: 'Espanol (Mexico)',
    direction: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'h:mm A',
    numberFormat: { notation: 'standard' },
    currencyCode: 'MXN',
    pluralRules: new Intl.PluralRules('es-MX'),
  },
  'fr': {
    code: 'fr',
    name: 'French',
    nativeName: 'Francais',
    direction: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'EUR',
    pluralRules: new Intl.PluralRules('fr'),
  },
  'fr-CA': {
    code: 'fr-CA',
    name: 'French (Canada)',
    nativeName: 'Francais (Canada)',
    direction: 'ltr',
    dateFormat: 'YYYY-MM-DD',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'CAD',
    pluralRules: new Intl.PluralRules('fr-CA'),
  },
  'de': {
    code: 'de',
    name: 'German',
    nativeName: 'Deutsch',
    direction: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'EUR',
    pluralRules: new Intl.PluralRules('de'),
  },
  'it': {
    code: 'it',
    name: 'Italian',
    nativeName: 'Italiano',
    direction: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'EUR',
    pluralRules: new Intl.PluralRules('it'),
  },
  'pt': {
    code: 'pt',
    name: 'Portuguese',
    nativeName: 'Portugues',
    direction: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'EUR',
    pluralRules: new Intl.PluralRules('pt'),
  },
  'pt-BR': {
    code: 'pt-BR',
    name: 'Portuguese (Brazil)',
    nativeName: 'Portugues (Brasil)',
    direction: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'BRL',
    pluralRules: new Intl.PluralRules('pt-BR'),
  },
  'nl': {
    code: 'nl',
    name: 'Dutch',
    nativeName: 'Nederlands',
    direction: 'ltr',
    dateFormat: 'DD-MM-YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'EUR',
    pluralRules: new Intl.PluralRules('nl'),
  },
  'pl': {
    code: 'pl',
    name: 'Polish',
    nativeName: 'Polski',
    direction: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'PLN',
    pluralRules: new Intl.PluralRules('pl'),
  },
  'ru': {
    code: 'ru',
    name: 'Russian',
    nativeName: 'Russkij',
    direction: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'RUB',
    pluralRules: new Intl.PluralRules('ru'),
  },
  'uk': {
    code: 'uk',
    name: 'Ukrainian',
    nativeName: 'Ukrayins\'ka',
    direction: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'UAH',
    pluralRules: new Intl.PluralRules('uk'),
  },
  'ja': {
    code: 'ja',
    name: 'Japanese',
    nativeName: 'Nihongo',
    direction: 'ltr',
    dateFormat: 'YYYY/MM/DD',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'JPY',
    pluralRules: new Intl.PluralRules('ja'),
  },
  'ko': {
    code: 'ko',
    name: 'Korean',
    nativeName: 'Hangugeo',
    direction: 'ltr',
    dateFormat: 'YYYY. MM. DD.',
    timeFormat: 'a h:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'KRW',
    pluralRules: new Intl.PluralRules('ko'),
  },
  'zh': {
    code: 'zh',
    name: 'Chinese (Simplified)',
    nativeName: 'Zhongwen',
    direction: 'ltr',
    dateFormat: 'YYYY/MM/DD',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'CNY',
    pluralRules: new Intl.PluralRules('zh'),
  },
  'zh-TW': {
    code: 'zh-TW',
    name: 'Chinese (Traditional)',
    nativeName: 'Zhongwen (Taiwan)',
    direction: 'ltr',
    dateFormat: 'YYYY/MM/DD',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'TWD',
    pluralRules: new Intl.PluralRules('zh-TW'),
  },
  'ar': {
    code: 'ar',
    name: 'Arabic',
    nativeName: 'Al-Arabiyya',
    direction: 'rtl',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'SAR',
    pluralRules: new Intl.PluralRules('ar'),
  },
  'he': {
    code: 'he',
    name: 'Hebrew',
    nativeName: 'Ivrit',
    direction: 'rtl',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'ILS',
    pluralRules: new Intl.PluralRules('he'),
  },
  'hi': {
    code: 'hi',
    name: 'Hindi',
    nativeName: 'Hindi',
    direction: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'h:mm A',
    numberFormat: { notation: 'standard' },
    currencyCode: 'INR',
    pluralRules: new Intl.PluralRules('hi'),
  },
  'th': {
    code: 'th',
    name: 'Thai',
    nativeName: 'Phasa Thai',
    direction: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'THB',
    pluralRules: new Intl.PluralRules('th'),
  },
  'vi': {
    code: 'vi',
    name: 'Vietnamese',
    nativeName: 'Tieng Viet',
    direction: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'VND',
    pluralRules: new Intl.PluralRules('vi'),
  },
  'id': {
    code: 'id',
    name: 'Indonesian',
    nativeName: 'Bahasa Indonesia',
    direction: 'ltr',
    dateFormat: 'DD/MM/YYYY',
    timeFormat: 'HH.mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'IDR',
    pluralRules: new Intl.PluralRules('id'),
  },
  'tr': {
    code: 'tr',
    name: 'Turkish',
    nativeName: 'Turkce',
    direction: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'TRY',
    pluralRules: new Intl.PluralRules('tr'),
  },
  'sv': {
    code: 'sv',
    name: 'Swedish',
    nativeName: 'Svenska',
    direction: 'ltr',
    dateFormat: 'YYYY-MM-DD',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'SEK',
    pluralRules: new Intl.PluralRules('sv'),
  },
  'no': {
    code: 'no',
    name: 'Norwegian',
    nativeName: 'Norsk',
    direction: 'ltr',
    dateFormat: 'DD.MM.YYYY',
    timeFormat: 'HH:mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'NOK',
    pluralRules: new Intl.PluralRules('no'),
  },
  'da': {
    code: 'da',
    name: 'Danish',
    nativeName: 'Dansk',
    direction: 'ltr',
    dateFormat: 'DD-MM-YYYY',
    timeFormat: 'HH.mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'DKK',
    pluralRules: new Intl.PluralRules('da'),
  },
  'fi': {
    code: 'fi',
    name: 'Finnish',
    nativeName: 'Suomi',
    direction: 'ltr',
    dateFormat: 'D.M.YYYY',
    timeFormat: 'H.mm',
    numberFormat: { notation: 'standard' },
    currencyCode: 'EUR',
    pluralRules: new Intl.PluralRules('fi'),
  },
};

// ==================== Translation Cache ====================

const translationCache = new Map<LocaleCode, Translations>();

async function loadTranslations(locale: LocaleCode): Promise<Translations> {
  // Check cache
  if (translationCache.has(locale)) {
    return translationCache.get(locale)!;
  }

  try {
    // Dynamic import of translation file
    const translations = await import(`./locales/${locale}.json`);
    translationCache.set(locale, translations.default);
    return translations.default;
  } catch (error) {
    console.error(`Failed to load translations for ${locale}:`, error);

    // Fallback to English
    if (locale !== 'en') {
      return loadTranslations('en');
    }

    return {};
  }
}

// ==================== Translation Functions ====================

function getNestedValue(obj: Translations, path: string): string | undefined {
  const keys = path.split('.');
  let current: TranslationValue = obj;

  for (const key of keys) {
    if (current && typeof current === 'object' && key in current) {
      current = current[key];
    } else {
      return undefined;
    }
  }

  return typeof current === 'string' ? current : undefined;
}

function interpolate(template: string, params: Record<string, string | number>): string {
  return template.replace(/\{\{(\w+)\}\}/g, (_, key) => {
    return params[key]?.toString() ?? `{{${key}}}`;
  });
}

function getPluralKey(count: number, pluralRules: Intl.PluralRules): string {
  return pluralRules.select(count);
}

// ==================== Context ====================

const I18nContext = createContext<I18nContextValue | null>(null);

// ==================== Provider ====================

interface I18nProviderProps {
  children: ReactNode;
  defaultLocale?: LocaleCode;
  onLocaleChange?: (locale: LocaleCode) => void;
}

export function I18nProvider({
  children,
  defaultLocale = 'en',
  onLocaleChange,
}: I18nProviderProps) {
  const [locale, setLocaleState] = useState<LocaleCode>(() => {
    if (typeof window !== 'undefined') {
      const stored = localStorage.getItem('mycelix-locale') as LocaleCode;
      if (stored && localeConfigs[stored]) {
        return stored;
      }

      // Try to detect from browser
      const browserLang = navigator.language.split('-')[0] as LocaleCode;
      if (localeConfigs[browserLang]) {
        return browserLang;
      }
    }
    return defaultLocale;
  });

  const [translations, setTranslations] = useState<Translations>({});
  const [isLoading, setIsLoading] = useState(true);

  const config = localeConfigs[locale];

  // Load translations when locale changes
  useEffect(() => {
    setIsLoading(true);
    loadTranslations(locale).then((trans) => {
      setTranslations(trans);
      setIsLoading(false);
    });
  }, [locale]);

  // Update document direction and lang
  useEffect(() => {
    if (typeof document !== 'undefined') {
      document.documentElement.lang = locale;
      document.documentElement.dir = config.direction;
    }
  }, [locale, config.direction]);

  // Set locale
  const setLocale = useCallback(async (newLocale: LocaleCode) => {
    if (!localeConfigs[newLocale]) {
      console.error(`Unknown locale: ${newLocale}`);
      return;
    }

    setLocaleState(newLocale);
    localStorage.setItem('mycelix-locale', newLocale);
    onLocaleChange?.(newLocale);
  }, [onLocaleChange]);

  // Translate function
  const t = useCallback((key: string, params?: Record<string, string | number>): string => {
    const value = getNestedValue(translations, key);

    if (!value) {
      console.warn(`Missing translation: ${key}`);
      return key;
    }

    return params ? interpolate(value, params) : value;
  }, [translations]);

  // Translate with count (pluralization)
  const tc = useCallback((
    key: string,
    count: number,
    params?: Record<string, string | number>
  ): string => {
    const pluralKey = getPluralKey(count, config.pluralRules);
    const fullKey = `${key}.${pluralKey}`;
    const value = getNestedValue(translations, fullKey) || getNestedValue(translations, `${key}.other`);

    if (!value) {
      console.warn(`Missing pluralized translation: ${fullKey}`);
      return key;
    }

    return interpolate(value, { count, ...params });
  }, [translations, config.pluralRules]);

  // Format number
  const formatNumber = useCallback((
    value: number,
    options?: Intl.NumberFormatOptions
  ): string => {
    return new Intl.NumberFormat(locale, {
      ...config.numberFormat,
      ...options,
    }).format(value);
  }, [locale, config.numberFormat]);

  // Format currency
  const formatCurrency = useCallback((
    value: number,
    currency?: string
  ): string => {
    return new Intl.NumberFormat(locale, {
      style: 'currency',
      currency: currency || config.currencyCode,
    }).format(value);
  }, [locale, config.currencyCode]);

  // Format date
  const formatDate = useCallback((
    date: Date | number,
    options?: Intl.DateTimeFormatOptions
  ): string => {
    const d = typeof date === 'number' ? new Date(date) : date;
    return new Intl.DateTimeFormat(locale, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
      ...options,
    }).format(d);
  }, [locale]);

  // Format time
  const formatTime = useCallback((
    date: Date | number,
    options?: Intl.DateTimeFormatOptions
  ): string => {
    const d = typeof date === 'number' ? new Date(date) : date;
    return new Intl.DateTimeFormat(locale, {
      hour: 'numeric',
      minute: 'numeric',
      ...options,
    }).format(d);
  }, [locale]);

  // Format relative time
  const formatRelativeTime = useCallback((date: Date | number): string => {
    const d = typeof date === 'number' ? new Date(date) : date;
    const now = new Date();
    const diffMs = d.getTime() - now.getTime();
    const diffSec = Math.round(diffMs / 1000);
    const diffMin = Math.round(diffSec / 60);
    const diffHour = Math.round(diffMin / 60);
    const diffDay = Math.round(diffHour / 24);

    const rtf = new Intl.RelativeTimeFormat(locale, { numeric: 'auto' });

    if (Math.abs(diffSec) < 60) {
      return rtf.format(diffSec, 'second');
    } else if (Math.abs(diffMin) < 60) {
      return rtf.format(diffMin, 'minute');
    } else if (Math.abs(diffHour) < 24) {
      return rtf.format(diffHour, 'hour');
    } else {
      return rtf.format(diffDay, 'day');
    }
  }, [locale]);

  // Format duration (for audio)
  const formatDuration = useCallback((seconds: number): string => {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);

    if (mins >= 60) {
      const hours = Math.floor(mins / 60);
      const remainingMins = mins % 60;
      return `${hours}:${remainingMins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    return `${mins}:${secs.toString().padStart(2, '0')}`;
  }, []);

  const value: I18nContextValue = {
    locale,
    direction: config.direction,
    setLocale,
    t,
    tc,
    formatNumber,
    formatCurrency,
    formatDate,
    formatTime,
    formatRelativeTime,
    formatDuration,
    availableLocales: Object.keys(localeConfigs) as LocaleCode[],
    isLoading,
  };

  return React.createElement(I18nContext.Provider, { value }, children);
}

// ==================== Hooks ====================

export function useI18n(): I18nContextValue {
  const context = useContext(I18nContext);
  if (!context) {
    throw new Error('useI18n must be used within an I18nProvider');
  }
  return context;
}

export function useTranslation() {
  const { t, tc, isLoading } = useI18n();
  return { t, tc, isLoading };
}

export function useLocale() {
  const { locale, setLocale, availableLocales, direction } = useI18n();
  return { locale, setLocale, availableLocales, direction };
}

export function useFormatters() {
  const { formatNumber, formatCurrency, formatDate, formatTime, formatRelativeTime, formatDuration } = useI18n();
  return { formatNumber, formatCurrency, formatDate, formatTime, formatRelativeTime, formatDuration };
}

// ==================== Components ====================

interface TransProps {
  id: string;
  params?: Record<string, string | number>;
  count?: number;
  as?: keyof JSX.IntrinsicElements;
  className?: string;
}

export function Trans({ id, params, count, as: Component = 'span', className }: TransProps) {
  const { t, tc } = useI18n();
  const text = count !== undefined ? tc(id, count, params) : t(id, params);

  return React.createElement(Component, { className }, text);
}

interface LocaleSwitcherProps {
  className?: string;
  showNativeName?: boolean;
}

export function LocaleSwitcher({ className, showNativeName = false }: LocaleSwitcherProps) {
  const { locale, setLocale, availableLocales } = useI18n();

  return React.createElement(
    'select',
    {
      value: locale,
      onChange: (e: React.ChangeEvent<HTMLSelectElement>) => setLocale(e.target.value as LocaleCode),
      className,
      'aria-label': 'Select language',
    },
    availableLocales.map(code => {
      const config = localeConfigs[code];
      return React.createElement(
        'option',
        { key: code, value: code },
        showNativeName ? config.nativeName : config.name
      );
    })
  );
}

// ==================== Utility Functions ====================

export function detectLocale(): LocaleCode {
  if (typeof navigator === 'undefined') return 'en';

  const browserLangs = navigator.languages || [navigator.language];

  for (const lang of browserLangs) {
    const exact = lang as LocaleCode;
    if (localeConfigs[exact]) return exact;

    const base = lang.split('-')[0] as LocaleCode;
    if (localeConfigs[base]) return base;
  }

  return 'en';
}

export function isRTL(locale: LocaleCode): boolean {
  return localeConfigs[locale]?.direction === 'rtl';
}

export function getLocaleConfig(locale: LocaleCode): LocaleConfig {
  return localeConfigs[locale] || localeConfigs['en'];
}

export default { I18nProvider, useI18n, useTranslation, useLocale, useFormatters, Trans, LocaleSwitcher };
