# Accessibility Guide

Making Knowledge hApp accessible to all users.

## Accessibility Commitment

Knowledge is committed to being accessible to everyone, regardless of:
- Visual ability
- Motor ability
- Cognitive differences
- Technical expertise
- Language
- Network conditions

## Standards Compliance

### WCAG 2.1 Level AA

All Knowledge interfaces target WCAG 2.1 Level AA compliance:

| Principle | Implementation |
|-----------|---------------|
| **Perceivable** | Alt text, captions, color contrast |
| **Operable** | Keyboard navigation, skip links |
| **Understandable** | Clear language, consistent UI |
| **Robust** | Screen reader support, valid markup |

## Visual Accessibility

### Color and Contrast

```css
/* Minimum contrast ratios */
:root {
  /* Text on background: 4.5:1 minimum */
  --text-primary: #1a1a1a;
  --bg-primary: #ffffff;

  /* Large text: 3:1 minimum */
  --heading-color: #2d2d2d;

  /* Interactive elements: 3:1 minimum */
  --link-color: #0066cc;
  --button-bg: #0052a3;
}
```

### Color Independence

Never rely on color alone to convey information:

```typescript
// ❌ Bad: Only color indicates status
<div className={claim.verified ? 'green' : 'red'}>
  {claim.content}
</div>

// ✅ Good: Icon + color + text
<div className={claim.verified ? 'verified' : 'unverified'}>
  {claim.verified ? '✓' : '⚠'}
  <span className="status-text">
    {claim.verified ? 'Verified' : 'Unverified'}
  </span>
  {claim.content}
</div>
```

### Text Sizing

- Base font size: 16px minimum
- Support 200% zoom without horizontal scroll
- Use relative units (rem, em) not fixed pixels
- Line height: 1.5 minimum for body text

### Focus Indicators

```css
/* Visible focus indicators */
:focus {
  outline: 3px solid #0066cc;
  outline-offset: 2px;
}

/* Never hide focus */
:focus:not(:focus-visible) {
  outline: 3px solid #0066cc; /* Keep visible */
}
```

## Keyboard Accessibility

### Full Keyboard Support

All functionality available via keyboard:

| Action | Key |
|--------|-----|
| Navigate forward | Tab |
| Navigate backward | Shift+Tab |
| Activate button/link | Enter |
| Toggle checkbox | Space |
| Close modal | Escape |
| Navigate menu | Arrow keys |

### Skip Links

```html
<!-- First element in body -->
<a href="#main-content" class="skip-link">
  Skip to main content
</a>

<nav>...</nav>

<main id="main-content">...</main>
```

### Focus Management

```typescript
// Focus management for dynamic content
function showClaimDetails(claimId: string) {
  const details = loadClaimDetails(claimId);
  renderDetails(details);

  // Move focus to new content
  const heading = document.querySelector('#claim-details h2');
  heading?.focus();
}

// Trap focus in modals
function trapFocus(modal: HTMLElement) {
  const focusable = modal.querySelectorAll(
    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
  );
  const first = focusable[0] as HTMLElement;
  const last = focusable[focusable.length - 1] as HTMLElement;

  modal.addEventListener('keydown', (e) => {
    if (e.key === 'Tab') {
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault();
        first.focus();
      }
    }
  });
}
```

## Screen Reader Support

### Semantic HTML

```html
<!-- Use proper semantic elements -->
<header>
  <nav aria-label="Main navigation">...</nav>
</header>

<main>
  <article>
    <h1>Claim Title</h1>
    <section aria-labelledby="evidence-heading">
      <h2 id="evidence-heading">Evidence</h2>
      ...
    </section>
  </article>
</main>

<footer>...</footer>
```

### ARIA Labels

```html
<!-- Descriptive labels -->
<button aria-label="Fact-check this claim">
  <svg>...</svg> <!-- Icon only -->
</button>

<!-- Live regions for updates -->
<div role="status" aria-live="polite">
  Credibility score: 0.85
</div>

<!-- Expanded state -->
<button aria-expanded="false" aria-controls="details-panel">
  Show Details
</button>
<div id="details-panel" hidden>...</div>
```

### Announcements

```typescript
// Announce dynamic changes
function announceToScreenReader(message: string, priority: 'polite' | 'assertive' = 'polite') {
  const announcer = document.getElementById('sr-announcer');
  if (announcer) {
    announcer.setAttribute('aria-live', priority);
    announcer.textContent = message;

    // Clear after announcement
    setTimeout(() => {
      announcer.textContent = '';
    }, 1000);
  }
}

// Usage
await knowledge.claims.createClaim(input);
announceToScreenReader('Claim created successfully');
```

## Cognitive Accessibility

### Clear Language

- Use plain language (avoid jargon)
- Short sentences (20 words or less)
- One idea per paragraph
- Define technical terms

```typescript
// ❌ Complex
"The claim's epistemic position within the E-N-M manifold indicates
a high degree of empirical verifiability."

// ✅ Clear
"This claim can be verified with evidence. Verifiability score: 0.8 out of 1.0"
```

### Consistent Navigation

- Same navigation on every page
- Consistent button/link styling
- Predictable behavior
- Breadcrumbs for orientation

### Error Prevention

```typescript
// Confirm destructive actions
async function deleteClaim(claimId: string) {
  const confirmed = await showConfirmDialog({
    title: 'Delete Claim?',
    message: 'This will permanently delete the claim. This cannot be undone.',
    confirmText: 'Delete',
    cancelText: 'Keep Claim',
  });

  if (confirmed) {
    await knowledge.claims.deleteClaim(claimId);
    announceToScreenReader('Claim deleted');
  }
}
```

### Clear Instructions

```html
<form>
  <label for="claim-content">
    Claim Content
    <span class="required">(required)</span>
  </label>
  <textarea
    id="claim-content"
    aria-describedby="content-help content-error"
    required
  ></textarea>
  <p id="content-help" class="help-text">
    Enter a factual statement. Be specific and cite sources if possible.
  </p>
  <p id="content-error" class="error-text" role="alert" hidden>
    Please enter the claim content.
  </p>
</form>
```

## Motor Accessibility

### Large Touch Targets

```css
/* Minimum 44x44px touch targets */
button, a, input[type="checkbox"], input[type="radio"] {
  min-width: 44px;
  min-height: 44px;
}

/* Adequate spacing between targets */
.button-group button {
  margin: 8px;
}
```

### Timeout Extensions

```typescript
// Allow users to extend timeouts
function startTimedOperation(duration: number, onTimeout: () => void) {
  let timeRemaining = duration;

  const timer = setInterval(() => {
    timeRemaining -= 1000;

    if (timeRemaining === 30000) {
      // Warn with 30 seconds left
      showTimeoutWarning({
        timeRemaining: 30,
        onExtend: () => {
          timeRemaining += 60000; // Add 1 minute
        },
        onContinue: () => {}, // Do nothing, let it expire
      });
    }

    if (timeRemaining <= 0) {
      clearInterval(timer);
      onTimeout();
    }
  }, 1000);
}
```

### Alternative Input

Support alternative input methods:
- Voice control compatibility
- Switch access support
- Eye tracking compatibility
- Reduced motion support

```css
/* Respect motion preferences */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

## Language and Internationalization

### Multiple Languages

```typescript
// Language-aware content
const translations = {
  en: {
    verified: 'Verified',
    credibility: 'Credibility Score',
  },
  es: {
    verified: 'Verificado',
    credibility: 'Puntuación de Credibilidad',
  },
};

function t(key: string): string {
  const lang = navigator.language.split('-')[0];
  return translations[lang]?.[key] || translations.en[key];
}
```

### Language Attributes

```html
<html lang="en">
  <body>
    <!-- Content in different language -->
    <blockquote lang="fr">
      "La vérité est un miroir tombé de la main de Dieu et qui s'est brisé."
    </blockquote>
  </body>
</html>
```

## Low Bandwidth / Offline Support

### Progressive Enhancement

```typescript
// Core functionality works without JavaScript
// Enhanced features added progressively

// Check for feature support
if ('serviceWorker' in navigator) {
  // Enable offline support
  navigator.serviceWorker.register('/sw.js');
}

// Graceful degradation
function getCredibility(claimId: string) {
  if (navigator.onLine) {
    return knowledge.inference.calculateEnhancedCredibility(claimId, 'Claim');
  } else {
    return getCachedCredibility(claimId); // Offline fallback
  }
}
```

### Data Efficiency

```typescript
// Configurable data usage
const knowledge = new KnowledgeClient(client, {
  dataEfficiency: {
    compressResponses: true,
    thumbnailsOnly: isMobile,
    lazyLoadRelationships: true,
  },
});
```

## Testing Accessibility

### Automated Testing

```typescript
// Jest + axe-core
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

test('claim card is accessible', async () => {
  const { container } = render(<ClaimCard claim={mockClaim} />);
  const results = await axe(container);
  expect(results).toHaveNoViolations();
});
```

### Manual Testing Checklist

- [ ] Navigate entire interface with keyboard only
- [ ] Test with screen reader (NVDA, VoiceOver)
- [ ] Test with 200% zoom
- [ ] Test with high contrast mode
- [ ] Test with reduced motion
- [ ] Test with slow network
- [ ] Test with JavaScript disabled

## Reporting Accessibility Issues

If you encounter accessibility barriers:

1. **Describe the issue** - What were you trying to do?
2. **Your setup** - Browser, assistive technology, OS
3. **What happened** - Expected vs actual behavior
4. **Screenshots/recordings** if possible

Report at: https://github.com/mycelix/knowledge/issues/new?template=accessibility.md

---

## Related Documentation

- [Security](./SECURITY.md) - Security considerations
- [Performance](./PERFORMANCE.md) - Performance optimization

---

*Knowledge for all.*
