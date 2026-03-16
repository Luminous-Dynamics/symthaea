/**
 * i18n scaffolding — lightweight string lookup for future multi-language support.
 *
 * Current: English only. To add a language:
 * 1. Add a new object in `translations` (e.g., `af` for Afrikaans)
 * 2. Set `currentLocale` via `setLocale()`
 * 3. All `t('key')` calls resolve to the new language
 *
 * Planned for v0.2: Afrikaans, Zulu, Sotho
 */

import { writable, derived, type Readable } from 'svelte/store';

export type Locale = 'en' | 'af' | 'zu' | 'st';

// ---------------------------------------------------------------------------
// String tables
// ---------------------------------------------------------------------------

const translations: Record<Locale, Record<string, string>> = {
  en: {
    // Navigation
    'nav.dashboard': 'Dashboard',
    'nav.resilience': 'Resilience',
    'nav.tend': 'TEND',
    'nav.food': 'Food',
    'nav.mutual_aid': 'Mutual Aid',
    'nav.emergency': 'Emergency',
    'nav.value_anchor': 'Value Anchor',
    'nav.water': 'Water',
    'nav.household': 'Household',
    'nav.knowledge': 'Knowledge',
    'nav.care_circles': 'Care Circles',
    'nav.shelter': 'Shelter',
    'nav.supplies': 'Supplies',
    'nav.operator': 'Operator',
    'nav.governance': 'Governance',
    'nav.network': 'Network',
    'nav.analytics': 'Analytics',
    'nav.attribution': 'Attribution',

    // Common actions
    'action.save': 'Save',
    'action.cancel': 'Cancel',
    'action.submit': 'Submit',
    'action.delete': 'Delete',
    'action.export': 'Export',
    'action.refresh': 'Refresh',
    'action.search': 'Search',
    'action.filter': 'Filter',
    'action.back': 'Back',
    'action.next': 'Next',
    'action.join': 'Join',
    'action.sync': 'Sync',

    // Status
    'status.loading': 'Loading...',
    'status.offline': 'Offline',
    'status.connected': 'Connected',
    'status.disconnected': 'Disconnected',
    'status.demo_mode': 'Demo Mode',
    'status.syncing': 'Syncing...',

    // Domains
    'tend.balance': 'TEND Balance',
    'tend.record_exchange': 'Record Exchange',
    'tend.marketplace': 'Marketplace',
    'food.plots': 'Food Plots',
    'food.harvests': 'Harvests',
    'emergency.channels': 'Channels',
    'emergency.send_message': 'Send Message',
    'water.systems': 'Water Systems',
    'water.alerts': 'Water Alerts',
    'supplies.inventory': 'Inventory',
    'supplies.low_stock': 'Low Stock',

    // Welcome / Onboarding
    'welcome.title': 'Welcome to Mycelix Resilience Kit',
    'welcome.subtitle': 'Community-powered tools for mutual aid, food security, and emergency coordination',
    'welcome.step1': 'What is this?',
    'welcome.step2': 'Your Community',
    'welcome.step3': 'Get Started',
    'welcome.dont_show': "Don't show this again",
    'welcome.enter': 'Enter the Kit',

    // Errors
    'error.load_failed': 'Failed to load data',
    'error.save_failed': 'Failed to save',
    'error.connection_failed': 'Connection failed',
  },

  // Placeholder stubs — fill in for v0.2
  af: {},
  zu: {},
  st: {},
};

// ---------------------------------------------------------------------------
// Store
// ---------------------------------------------------------------------------

export const locale = writable<Locale>('en');

export function setLocale(l: Locale): void {
  locale.set(l);
}

/**
 * Translate a key. Returns the translation for the current locale,
 * falling back to English, then to the key itself.
 */
export const t: Readable<(key: string, params?: Record<string, string>) => string> = derived(
  locale,
  ($locale) => (key: string, params?: Record<string, string>): string => {
    let str = translations[$locale]?.[key] ?? translations.en[key] ?? key;
    if (params) {
      for (const [k, v] of Object.entries(params)) {
        str = str.replace(`{${k}}`, v);
      }
    }
    return str;
  },
);

/**
 * Synchronous translate for non-reactive contexts (e.g., aria-labels in helpers).
 * Uses the current locale at call time.
 */
export function tSync(key: string, params?: Record<string, string>): string {
  let currentLocale: Locale = 'en';
  locale.subscribe((l) => { currentLocale = l; })();
  let str = translations[currentLocale]?.[key] ?? translations.en[key] ?? key;
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      str = str.replace(`{${k}}`, v);
    }
  }
  return str;
}
