# Internationalization

*Truth Has No Language*

---

> "The pursuit of truth is a human endeavor, not a linguistic one. Our system must speak every tongue."

---

## Introduction

Epistemic Markets is designed for global collective intelligence. Truth-seeking should not be limited by language, culture, or geography. This document describes our approach to making the system accessible worldwide.

---

## Part I: Philosophy

### Core Principles

**1. Language Equality**
- No language is privileged over another
- All languages are first-class citizens
- Translation is investment, not afterthought

**2. Cultural Sensitivity**
- Concepts must translate culturally, not just linguistically
- Different cultures have different epistemic traditions
- Honor diverse ways of knowing

**3. Inclusive by Default**
- Language selection should be automatic
- Fallbacks should be graceful
- No dead ends for any language

**4. Community-Driven**
- Translations done with community
- Native speakers validate
- Continuous improvement

### What We Internationalize

| Category | Examples |
|----------|----------|
| **UI Text** | Buttons, labels, messages, tooltips |
| **Content** | Help articles, tutorials, documentation |
| **System Messages** | Errors, confirmations, notifications |
| **User Content** | Prediction reasoning, wisdom seeds (user choice) |
| **Legal** | Terms, privacy policy, disclaimers |
| **Formats** | Dates, numbers, currencies, units |

---

## Part II: Technical Architecture

### Localization Stack

```
┌─────────────────────────────────────────────────────────────┐
│                      UI Layer                               │
│  React/Svelte components use translation hooks              │
├─────────────────────────────────────────────────────────────┤
│                  Translation Layer                          │
│  i18n library (e.g., i18next, formatjs)                    │
├─────────────────────────────────────────────────────────────┤
│                   Message Catalog                           │
│  JSON/YAML files per language per domain                   │
├─────────────────────────────────────────────────────────────┤
│                  Translation Memory                         │
│  Shared translations, glossary, style guides              │
├─────────────────────────────────────────────────────────────┤
│                Community Translation                        │
│  Crowdsourced, validated, version-controlled               │
└─────────────────────────────────────────────────────────────┘
```

### Message Format

We use ICU MessageFormat for complex translations:

```typescript
// Simple string
"greeting": "Welcome to Epistemic Markets"

// With variable
"welcome_user": "Welcome, {name}"

// Pluralization
"prediction_count": "{count, plural,
  =0 {No predictions yet}
  one {# prediction}
  other {# predictions}
}"

// Gender (where relevant)
"user_action": "{gender, select,
  female {She made a prediction}
  male {He made a prediction}
  other {They made a prediction}
}"

// Number formatting
"stake_amount": "Staked {amount, number, ::currency/USD}"

// Date formatting
"resolution_date": "Resolves {date, date, long}"

// Nested
"market_summary": "{creator} created a market about {topic} with {count, plural, one {# prediction} other {# predictions}}"
```

### Language Detection

```typescript
interface LanguageDetection {
  // Priority order for language detection
  priority: [
    "explicit_setting",     // User explicitly set language
    "url_parameter",        // ?lang=es
    "cookie",               // Previously selected
    "browser_preference",   // Accept-Language header
    "geolocation",          // IP-based (with consent)
    "default"               // System default (English)
  ];

  // Detection function
  detectLanguage(context: RequestContext): Language;

  // Validation
  isSupported(lang: Language): boolean;
  findFallback(lang: Language): Language;
}
```

### File Organization

```
locales/
├── en/                     # English (source)
│   ├── common.json        # Shared strings
│   ├── markets.json       # Market-related
│   ├── predictions.json   # Prediction-related
│   ├── resolution.json    # Resolution-related
│   ├── governance.json    # Governance-related
│   ├── errors.json        # Error messages
│   └── help.json          # Help content
├── es/                     # Spanish
│   ├── common.json
│   └── ...
├── zh/                     # Chinese (Simplified)
├── ar/                     # Arabic
├── hi/                     # Hindi
├── pt/                     # Portuguese
└── ...

# Metadata
├── languages.json          # Supported languages config
├── glossary.json           # Term consistency
└── style-guide.md          # Translation guidelines
```

### Language Configuration

```typescript
interface LanguageConfig {
  code: string;           // ISO 639-1 (e.g., "en", "es")
  name: string;           // Native name (e.g., "Español")
  nameEnglish: string;    // English name (e.g., "Spanish")
  direction: "ltr" | "rtl";

  // Formatting
  dateFormat: string;
  timeFormat: string;
  numberFormat: NumberFormatOptions;
  currencyDisplay: "symbol" | "code" | "name";

  // Completeness
  completeness: number;   // Percentage translated
  status: "full" | "partial" | "machine" | "community";

  // Fallback chain
  fallback: string[];     // e.g., ["en-GB", "en"]
}

const languages: LanguageConfig[] = [
  {
    code: "en",
    name: "English",
    nameEnglish: "English",
    direction: "ltr",
    dateFormat: "MM/DD/YYYY",
    completeness: 100,
    status: "full",
    fallback: []
  },
  {
    code: "es",
    name: "Español",
    nameEnglish: "Spanish",
    direction: "ltr",
    dateFormat: "DD/MM/YYYY",
    completeness: 95,
    status: "community",
    fallback: ["en"]
  },
  {
    code: "ar",
    name: "العربية",
    nameEnglish: "Arabic",
    direction: "rtl",
    dateFormat: "DD/MM/YYYY",
    completeness: 80,
    status: "community",
    fallback: ["en"]
  },
  // ...
];
```

---

## Part III: Right-to-Left (RTL) Support

### Design Principles for RTL

**1. Mirroring**
- Layout flips horizontally
- Reading direction reverses
- Icons with direction meaning flip

**2. Bi-directional Text**
- Handle mixed LTR/RTL content
- Numbers remain LTR within RTL text
- URLs and code remain LTR

**3. CSS Implementation**

```css
/* Base styles using logical properties */
.container {
  padding-inline-start: 1rem;  /* Not padding-left */
  margin-inline-end: 1rem;     /* Not margin-right */
  text-align: start;           /* Not text-align: left */
}

/* RTL-specific adjustments */
[dir="rtl"] {
  /* Flip icons that have directional meaning */
  .icon-arrow { transform: scaleX(-1); }

  /* Adjust for RTL-aware layouts */
  .sidebar { order: 1; }
}

/* Bidirectional text handling */
.ltr-always {
  direction: ltr;
  unicode-bidi: embed;
}
```

**4. Component Considerations**

```typescript
// RTL-aware component example
interface ProgressBarProps {
  value: number;
  direction?: "ltr" | "rtl" | "auto";
}

function ProgressBar({ value, direction = "auto" }: ProgressBarProps) {
  const dir = direction === "auto" ? document.dir : direction;

  return (
    <div
      class="progress-bar"
      dir={dir}
      style={{
        "--progress": `${value}%`,
        "--origin": dir === "rtl" ? "right" : "left"
      }}
    >
      <div class="progress-fill" />
    </div>
  );
}
```

---

## Part IV: Cultural Adaptation

### Beyond Translation

True internationalization goes beyond words:

**1. Conceptual Adaptation**
- "Staking reputation" may need different framing in collectivist cultures
- "Being wrong" has different connotations across cultures
- "Wisdom" means different things in different traditions

**2. Visual Considerations**
- Colors have cultural meanings (red = danger vs. prosperity)
- Imagery should be culturally appropriate
- Icons may need adaptation

**3. Interaction Patterns**
- Formality levels vary by culture
- Relationship to authority differs
- Individual vs. collective framing

### Cultural Localization Matrix

| Dimension | Western | East Asian | Middle Eastern | South Asian |
|-----------|---------|------------|----------------|-------------|
| **Stakes Framing** | Personal risk | Group harmony | Honor/duty | Dharmic balance |
| **Being Wrong** | Learning opportunity | Face consideration | Humility virtue | Karma adjustment |
| **Authority** | Challenge expected | Respect elders | Scholar reverence | Guru deference |
| **Collectivism** | Individual predictions | Group consensus | Tribal wisdom | Community dharma |

### Adaptation Examples

**"Stake your reputation"**

| Language | Adaptation |
|----------|------------|
| English | "Stake your reputation" |
| Japanese | "信用を賭ける" (Invest your credibility) |
| Arabic | "ضع سمعتك على المحك" (Put your reputation to the test) |
| Spanish | "Apuesta tu reputación" (Bet your reputation) |

**"You were wrong, and that's okay"**

| Language | Cultural Adaptation |
|----------|---------------------|
| English | "Being wrong is how we learn" |
| Japanese | "間違いを通じて、私たちは共に成長します" (Through mistakes, we grow together) |
| Arabic | "الخطأ بداية الحكمة" (Error is the beginning of wisdom) |
| Hindi | "गलतियाँ ज्ञान की सीढ़ियाँ हैं" (Mistakes are steps to knowledge) |

---

## Part V: Translation Workflow

### Translation Process

```
1. String Extraction
   └── Extract new/changed strings from code

2. Context Addition
   └── Developer adds context, screenshots, notes

3. Translation
   ├── Machine translation (initial draft)
   ├── Community translation (improvement)
   └── Native speaker review

4. Validation
   ├── Linguistic review
   ├── Cultural review
   └── Technical validation

5. Integration
   ├── Merge translations
   ├── Run tests
   └── Deploy

6. Feedback
   └── User reports, quality monitoring
```

### Context for Translators

Every translatable string should include:

```typescript
interface TranslationContext {
  key: string;
  source: string;

  // Context for translator
  context: {
    description: string;     // What this string is for
    maxLength?: number;      // Space constraints
    screenshot?: string;     // Where it appears
    placeholders: {          // What variables mean
      [key: string]: string;
    };
  };

  // Previous translations for reference
  previousVersions?: Translation[];

  // Related strings
  relatedKeys?: string[];
}

// Example
{
  key: "prediction.confidence.label",
  source: "How confident are you?",
  context: {
    description: "Label above the confidence slider when making a prediction",
    maxLength: 30,
    screenshot: "prediction-form-confidence.png",
    placeholders: {}
  },
  relatedKeys: [
    "prediction.confidence.help",
    "prediction.confidence.low",
    "prediction.confidence.high"
  ]
}
```

### Translation Memory

Maintain consistency with translation memory:

```typescript
interface TranslationMemory {
  // Exact matches
  getExactMatch(source: string, lang: string): Translation | null;

  // Fuzzy matches for suggestions
  getFuzzyMatches(source: string, lang: string, threshold: number): FuzzyMatch[];

  // Terminology consistency
  getTerminology(source: string): Term[];

  // Add new translations
  addTranslation(translation: Translation): void;

  // Quality metrics
  getQualityScore(translation: Translation): QualityScore;
}
```

### Quality Assurance

```typescript
interface TranslationQA {
  // Automated checks
  checks: {
    placeholderIntegrity: Check;    // All placeholders present
    lengthLimit: Check;             // Within length constraints
    noEmptyTranslations: Check;     // No empty strings
    consistentTerminology: Check;   // Uses approved terms
    noHtmlCorruption: Check;        // HTML tags intact
    validICUFormat: Check;          // ICU syntax correct
  };

  // Human review
  reviewProcess: {
    linguisticReview: boolean;
    culturalReview: boolean;
    technicalReview: boolean;
    nativeSpeakerValidation: boolean;
  };

  // Continuous monitoring
  monitoring: {
    userReports: boolean;
    completenessTracking: boolean;
    qualityMetrics: boolean;
  };
}
```

---

## Part VI: Content Translation

### User-Generated Content

How we handle translation of user content:

**Prediction Reasoning**
- Original language preserved
- Optional machine translation available
- Community translation for popular content
- Language tag displayed

**Wisdom Seeds**
- Encouraged to be written universally
- Translation welcomed
- Multi-language versions supported

**Market Questions**
- Original language is authoritative
- Translations provided
- Disputes resolved in original language

### Machine Translation Integration

```typescript
interface MachineTranslation {
  // Translation service
  translate(text: string, sourceLang: string, targetLang: string): Promise<Translation>;

  // Quality indicators
  getConfidence(translation: Translation): number;

  // Caching
  getCached(text: string, sourceLang: string, targetLang: string): Translation | null;

  // Rate limiting
  canTranslate(): boolean;
}

interface Translation {
  original: string;
  translated: string;
  sourceLang: string;
  targetLang: string;
  method: "machine" | "human" | "community";
  confidence: number;
  timestamp: number;
}
```

### Display Strategy

```
Original Content (en): "Bitcoin will reach $100k because of institutional adoption"

┌─────────────────────────────────────────────────────────────┐
│ Prediction by @user_name [en]                              │
│                                                             │
│ "Bitcoin will reach $100k because of institutional        │
│  adoption and limited supply dynamics."                    │
│                                                             │
│ [View in: 日本語 | Español | العربية] [Translate ▼]      │
└─────────────────────────────────────────────────────────────┘

When "日本語" selected:
┌─────────────────────────────────────────────────────────────┐
│ @user_name による予測 [en → ja 🤖]                        │
│                                                             │
│ 「機関投資家の参入と供給の限定性から、ビットコインは      │
│  10万ドルに達するだろう。」                               │
│                                                             │
│ 🤖 Machine translated • [View original] [Improve]          │
└─────────────────────────────────────────────────────────────┘
```

---

## Part VII: Date, Time, and Number Formatting

### Date Formatting

```typescript
// Use Intl APIs for localized formatting
function formatDate(date: Date, locale: string, options?: Intl.DateTimeFormatOptions): string {
  return new Intl.DateTimeFormat(locale, options).format(date);
}

// Examples
formatDate(new Date("2025-12-31"), "en-US", { dateStyle: "long" });
// → "December 31, 2025"

formatDate(new Date("2025-12-31"), "de-DE", { dateStyle: "long" });
// → "31. Dezember 2025"

formatDate(new Date("2025-12-31"), "ja-JP", { dateStyle: "long" });
// → "2025年12月31日"
```

### Time Formatting

```typescript
// Market closes display
function formatMarketClose(closeTime: Date, userLocale: string, userTimezone: string): string {
  const formatted = new Intl.DateTimeFormat(userLocale, {
    dateStyle: "medium",
    timeStyle: "short",
    timeZone: userTimezone
  }).format(closeTime);

  return `${formatted} (${userTimezone})`;
}

// Relative time
function formatRelativeTime(date: Date, locale: string): string {
  const rtf = new Intl.RelativeTimeFormat(locale, { numeric: "auto" });
  const diff = (date.getTime() - Date.now()) / 1000;

  if (Math.abs(diff) < 60) return rtf.format(Math.round(diff), "second");
  if (Math.abs(diff) < 3600) return rtf.format(Math.round(diff / 60), "minute");
  if (Math.abs(diff) < 86400) return rtf.format(Math.round(diff / 3600), "hour");
  return rtf.format(Math.round(diff / 86400), "day");
}

// "in 3 days" / "vor 3 Tagen" / "3日後"
```

### Number Formatting

```typescript
// Currency
function formatCurrency(amount: number, currency: string, locale: string): string {
  return new Intl.NumberFormat(locale, {
    style: "currency",
    currency: currency
  }).format(amount);
}

// Percentage
function formatPercentage(value: number, locale: string): string {
  return new Intl.NumberFormat(locale, {
    style: "percent",
    minimumFractionDigits: 0,
    maximumFractionDigits: 1
  }).format(value);
}

// Large numbers
function formatNumber(value: number, locale: string): string {
  return new Intl.NumberFormat(locale).format(value);
}

// Examples for 1234567.89:
// en-US: "1,234,567.89"
// de-DE: "1.234.567,89"
// hi-IN: "12,34,567.89"
```

---

## Part VIII: Language Prioritization

### Launch Languages

**Tier 1 (Full support at launch)**
- English (en)
- Spanish (es)
- Chinese Simplified (zh-CN)
- Arabic (ar)

**Tier 2 (Following launch)**
- Hindi (hi)
- Portuguese (pt)
- French (fr)
- German (de)
- Japanese (ja)
- Russian (ru)

**Tier 3 (Community-driven)**
- All other languages via community contribution

### Prioritization Criteria

| Criterion | Weight | Notes |
|-----------|--------|-------|
| Speaker population | 25% | Global reach |
| Internet users | 25% | Digital access |
| Crypto/prediction market activity | 20% | Target audience |
| Community interest | 15% | Request volume |
| Translation resources | 15% | Feasibility |

---

## Part IX: Testing and Validation

### Internationalization Testing

```typescript
interface I18nTestSuite {
  // String tests
  stringTests: {
    allKeysTranslated: Test;          // No missing translations
    noHardcodedStrings: Test;         // All strings externalized
    placeholderConsistency: Test;     // Placeholders match
  };

  // Formatting tests
  formattingTests: {
    dateFormattingWorks: Test;
    numberFormattingWorks: Test;
    currencyFormattingWorks: Test;
    pluralizationWorks: Test;
  };

  // Layout tests
  layoutTests: {
    textExpansionHandled: Test;       // German is ~30% longer
    rtlLayoutCorrect: Test;           // Right-to-left works
    fontRenderingCorrect: Test;       // All characters display
  };

  // Functional tests
  functionalTests: {
    languageSwitchingWorks: Test;
    fallbacksWork: Test;
    contentMixingWorks: Test;         // LTR/RTL mixed content
  };
}
```

### Pseudo-Localization

For development testing:

```typescript
// Pseudo-locale transforms source text to test:
// - Character rendering: Replaces ASCII with accented versions
// - Text expansion: Adds padding to simulate longer translations
// - RTL simulation: Wraps text with RTL markers

function pseudoLocalize(text: string): string {
  const charMap: Record<string, string> = {
    'a': 'à', 'e': 'è', 'i': 'ì', 'o': 'ò', 'u': 'ù',
    'A': 'À', 'E': 'È', 'I': 'Ì', 'O': 'Ò', 'U': 'Ù'
  };

  let result = text
    .split('')
    .map(c => charMap[c] || c)
    .join('');

  // Add expansion padding (30%)
  const padding = '~'.repeat(Math.ceil(text.length * 0.3));

  return `[${result}${padding}]`;
}

// "Welcome" → "[Wèlcòmè~~~]"
```

---

## Part X: Community Translation Program

### How to Contribute

**1. Join the Program**
- Sign up as community translator
- Take language proficiency assessment
- Complete translation guidelines training

**2. Start Translating**
- Access translation platform
- Choose strings to translate
- Submit for review

**3. Review Process**
- Peer review by other translators
- Native speaker validation
- Technical verification

**4. Recognition**
- Translator credits
- Contribution badges
- MATL reputation boost

### Translation Platform

```typescript
interface TranslationPlatform {
  // For translators
  translator: {
    getAssignedStrings(): TranslationTask[];
    submitTranslation(key: string, translation: string): void;
    addComment(key: string, comment: string): void;
    reportIssue(key: string, issue: Issue): void;
  };

  // For reviewers
  reviewer: {
    getPendingReviews(): Translation[];
    approve(translationId: string): void;
    reject(translationId: string, reason: string): void;
    suggest(translationId: string, alternative: string): void;
  };

  // For admins
  admin: {
    getStats(): TranslationStats;
    assignTasks(translatorId: string, tasks: TranslationTask[]): void;
    manageGlossary(): void;
    export(lang: string): TranslationBundle;
  };
}
```

### Translator Guidelines

**Do**:
- Maintain meaning over literal translation
- Use consistent terminology from glossary
- Preserve all placeholders
- Consider cultural context
- Ask questions when unclear

**Don't**:
- Use machine translation without review
- Invent terminology without discussion
- Change meaning for "better" phrasing
- Skip context review
- Translate technical terms inconsistently

---

## Conclusion

Internationalization is not a feature—it's a commitment to global truth-seeking. Every language we support is an invitation to millions more minds to join our collective intelligence.

We build for all of humanity, not just the English-speaking world. Truth has no language, and neither should our pursuit of it.

---

> "When we translate, we don't just change words—we build bridges between minds across the world."

---

*Truth transcends language. We follow.*

---

## Appendix: Quick Reference

### Adding a New Translatable String

```typescript
// In component:
import { useTranslation } from 'react-i18next';

function MyComponent() {
  const { t } = useTranslation('markets');

  return (
    <button>{t('create_market.submit')}</button>
  );
}

// In locales/en/markets.json:
{
  "create_market": {
    "submit": "Create Market"
  }
}
```

### Formatting Reference

```typescript
// Date
t('date', { date: new Date(), formatParams: { date: { dateStyle: 'long' } } })

// Number
t('count', { count: 1234 })

// Currency
t('amount', { amount: 99.99, formatParams: { amount: { style: 'currency', currency: 'USD' } } })

// Plural
t('items', { count: 5 })  // Uses ICU plural rules
```

### Language Codes

| Language | Code | Direction |
|----------|------|-----------|
| English | en | LTR |
| Spanish | es | LTR |
| Chinese (Simplified) | zh-CN | LTR |
| Arabic | ar | RTL |
| Hindi | hi | LTR |
| Portuguese | pt | LTR |
| French | fr | LTR |
| German | de | LTR |
| Japanese | ja | LTR |
| Russian | ru | LTR |
| Korean | ko | LTR |
| Hebrew | he | RTL |
| Persian | fa | RTL |
| Urdu | ur | RTL |
