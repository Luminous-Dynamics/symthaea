# Accessibility

*Truth-Seeking for Every Mind*

---

> "The power of the Web is in its universality. Access by everyone regardless of disability is an essential aspect."
> — Tim Berners-Lee

> "If collective intelligence excludes any mind, it isn't truly collective."
> — Us

---

## Introduction

Epistemic Markets is built on the principle that collective intelligence requires diverse perspectives. Accessibility isn't an afterthought or a compliance checkbox—it's fundamental to our mission. If we exclude minds, we exclude truths.

This document outlines our accessibility commitments, standards, and implementation across all interfaces and interactions.

---

## Part I: Accessibility Principles

### Principle 1: Universal Design First

We design for the full range of human diversity from the start, not as an accommodation added later.

**In Practice**:
- Every feature is designed with accessibility in mind from conception
- Accessibility review is part of every design and code review
- We test with diverse users, not just diverse simulators

### Principle 2: Multiple Modalities

Every piece of information and every action should be achievable through multiple modalities.

**In Practice**:
- Visual information has text alternatives
- Audio information has text alternatives
- Mouse actions have keyboard alternatives
- Touch actions have alternatives for limited mobility

### Principle 3: Progressive Enhancement

Core functionality works in the simplest environments; enhanced experiences layer on top.

**In Practice**:
- Basic predictions work without JavaScript
- Core features work on slow connections
- Essential information available without images loading
- Graceful degradation when features unavailable

### Principle 4: User Control

Users control their experience—speed, contrast, animation, complexity.

**In Practice**:
- Respect system preferences (reduced motion, high contrast)
- Provide in-app controls for key settings
- Remember preferences across sessions
- Allow customization without penalty

### Principle 5: Cognitive Accessibility

Accessibility includes cognitive and neurological diversity, not just sensory and motor.

**In Practice**:
- Clear, simple language
- Consistent navigation
- Predictable interactions
- Appropriate time limits (or none)
- Error prevention and recovery

---

## Part II: Standards Compliance

### WCAG 2.2 Conformance

We target **WCAG 2.2 Level AA** conformance, with Level AAA where feasible.

#### Level A (Minimum)

| Guideline | Implementation |
|-----------|----------------|
| 1.1.1 Non-text Content | All images have alt text, charts have descriptions |
| 1.2.1 Audio-only/Video-only | Transcripts for all audio, descriptions for video |
| 1.3.1 Info and Relationships | Semantic HTML, ARIA landmarks |
| 1.3.2 Meaningful Sequence | DOM order matches visual order |
| 1.3.3 Sensory Characteristics | Instructions don't rely solely on shape/color/sound |
| 1.4.1 Use of Color | Color never sole conveyor of information |
| 2.1.1 Keyboard | All functionality keyboard accessible |
| 2.1.2 No Keyboard Trap | Focus can always be moved |
| 2.4.1 Bypass Blocks | Skip links, landmarks |
| 2.4.2 Page Titled | Descriptive, unique titles |
| 2.4.3 Focus Order | Logical focus sequence |
| 2.4.4 Link Purpose | Links describe destination |
| 3.1.1 Language of Page | HTML lang attribute set |
| 3.2.1 On Focus | No unexpected changes on focus |
| 3.2.2 On Input | No unexpected changes on input |
| 3.3.1 Error Identification | Errors clearly identified |
| 3.3.2 Labels or Instructions | Form inputs labeled |
| 4.1.1 Parsing | Valid HTML |
| 4.1.2 Name, Role, Value | Custom controls have accessible names |

#### Level AA (Target)

| Guideline | Implementation |
|-----------|----------------|
| 1.2.4 Captions (Live) | Live events captioned |
| 1.2.5 Audio Description | Complex visuals described |
| 1.3.4 Orientation | Works in any orientation |
| 1.3.5 Identify Input Purpose | Autocomplete attributes |
| 1.4.3 Contrast (Minimum) | 4.5:1 for text, 3:1 for large |
| 1.4.4 Resize Text | 200% zoom without loss |
| 1.4.5 Images of Text | Real text, not images |
| 1.4.10 Reflow | No horizontal scroll at 320px |
| 1.4.11 Non-text Contrast | 3:1 for UI components |
| 1.4.12 Text Spacing | Customizable spacing |
| 1.4.13 Content on Hover/Focus | Dismissible, hoverable, persistent |
| 2.4.5 Multiple Ways | Multiple navigation methods |
| 2.4.6 Headings and Labels | Descriptive headings |
| 2.4.7 Focus Visible | Clear focus indicators |
| 3.1.2 Language of Parts | Language changes marked |
| 3.2.3 Consistent Navigation | Navigation consistent |
| 3.2.4 Consistent Identification | Components identified consistently |
| 3.3.3 Error Suggestion | Helpful error messages |
| 3.3.4 Error Prevention | Confirmation for important actions |
| 4.1.3 Status Messages | Status communicated to AT |

#### Level AAA (Where Feasible)

| Guideline | Status |
|-----------|--------|
| 1.2.6 Sign Language | Planned for key content |
| 1.4.6 Contrast (Enhanced) | 7:1 available via high contrast mode |
| 2.2.3 No Timing | No time limits on predictions |
| 2.4.9 Link Purpose (Link Only) | Implemented |
| 3.1.5 Reading Level | Simplified language mode available |
| 3.2.5 Change on Request | User-initiated changes only |

---

## Part III: Implementation by Component

### Navigation

```typescript
// Landmark structure
<header role="banner">
  <nav role="navigation" aria-label="Main">
    <!-- Skip link -->
    <a href="#main-content" class="skip-link">
      Skip to main content
    </a>
    <!-- Navigation items -->
  </nav>
</header>

<main id="main-content" role="main" tabindex="-1">
  <!-- Main content -->
</main>

<aside role="complementary" aria-label="Sidebar">
  <!-- Sidebar content -->
</aside>

<footer role="contentinfo">
  <!-- Footer -->
</footer>
```

**Keyboard Navigation**:
- `Tab`: Move forward through focusable elements
- `Shift+Tab`: Move backward
- `Enter/Space`: Activate buttons and links
- `Arrow keys`: Navigate within components (menus, tabs)
- `Escape`: Close modals, cancel operations
- `/`: Focus search (when implemented)
- `?`: Open keyboard shortcuts help

### Forms and Inputs

```typescript
// Accessible form pattern
<form>
  <div class="form-group">
    <label for="prediction-confidence" id="confidence-label">
      Confidence Level
      <span class="required" aria-hidden="true">*</span>
    </label>

    <input
      type="range"
      id="prediction-confidence"
      name="confidence"
      min="1"
      max="99"
      value="50"
      aria-labelledby="confidence-label"
      aria-describedby="confidence-help confidence-value"
      aria-required="true"
    />

    <output id="confidence-value" for="prediction-confidence">
      50%
    </output>

    <p id="confidence-help" class="help-text">
      How certain are you? 50% means you're unsure,
      90% means you're very confident.
    </p>
  </div>

  <!-- Error handling -->
  <div role="alert" aria-live="polite" id="form-errors">
    <!-- Errors announced here -->
  </div>
</form>
```

**Form Accessibility Features**:
- All inputs have visible labels
- Required fields clearly marked
- Error messages associated with fields
- Helpful descriptions for complex inputs
- No time limits for form completion
- Data preserved on errors

### Data Visualization

```typescript
// Accessible chart pattern
<figure role="img" aria-labelledby="chart-title" aria-describedby="chart-desc">
  <figcaption id="chart-title">
    Market Price History: "Will X happen?"
  </figcaption>

  <p id="chart-desc" class="sr-only">
    Line chart showing price movement from January to March 2024.
    Price started at 45%, peaked at 72% in mid-February, and
    currently sits at 68%. Overall trend is upward.
  </p>

  <!-- Chart canvas -->
  <canvas id="price-chart" aria-hidden="true"></canvas>

  <!-- Data table alternative -->
  <details>
    <summary>View data as table</summary>
    <table>
      <caption>Price history data</caption>
      <thead>
        <tr>
          <th scope="col">Date</th>
          <th scope="col">Price</th>
        </tr>
      </thead>
      <tbody>
        <tr><td>Jan 1</td><td>45%</td></tr>
        <!-- ... -->
      </tbody>
    </table>
  </details>
</figure>
```

**Visualization Accessibility**:
- Text descriptions for all charts
- Data tables as alternatives
- Pattern fills in addition to colors
- High contrast mode support
- Screen reader announcements for updates

### Predictions Interface

```typescript
// Accessible prediction card
<article
  class="prediction-card"
  aria-labelledby="pred-title-123"
  aria-describedby="pred-summary-123"
>
  <header>
    <h3 id="pred-title-123">
      <a href="/markets/123">
        Will renewable energy exceed 50% of US grid by 2030?
      </a>
    </h3>

    <div class="market-status" role="status">
      <span class="sr-only">Market status:</span>
      Open until March 15, 2024
    </div>
  </header>

  <div id="pred-summary-123">
    <p class="current-prediction">
      Current aggregate prediction:
      <strong>67% Yes</strong>
    </p>

    <p class="participation">
      <span class="sr-only">Participation:</span>
      142 predictions from 98 participants
    </p>
  </div>

  <footer>
    <button
      type="button"
      aria-expanded="false"
      aria-controls="pred-details-123"
    >
      Show details
    </button>
  </footer>
</article>
```

### Modals and Dialogs

```typescript
// Accessible modal pattern
<div
  role="dialog"
  aria-modal="true"
  aria-labelledby="modal-title"
  aria-describedby="modal-desc"
>
  <header>
    <h2 id="modal-title">Confirm Prediction</h2>
    <button
      type="button"
      aria-label="Close dialog"
      class="close-button"
    >
      <span aria-hidden="true">×</span>
    </button>
  </header>

  <div id="modal-desc">
    <p>You are about to submit a prediction with the following details:</p>
    <!-- Details -->
  </div>

  <footer>
    <button type="button">Cancel</button>
    <button type="submit">Confirm</button>
  </footer>
</div>

// Focus management
function openModal(modal) {
  // Store last focused element
  lastFocused = document.activeElement;

  // Show modal
  modal.hidden = false;

  // Focus first focusable element
  modal.querySelector('button, input, [tabindex="0"]').focus();

  // Trap focus in modal
  modal.addEventListener('keydown', trapFocus);
}

function closeModal(modal) {
  modal.hidden = true;
  lastFocused.focus();
}
```

### Live Updates

```typescript
// Accessible live regions
<div aria-live="polite" aria-atomic="true" id="price-updates">
  <!-- Price changes announced here -->
</div>

<div aria-live="assertive" role="alert" id="urgent-alerts">
  <!-- Critical alerts -->
</div>

// Announce updates appropriately
function announceUpdate(message, priority = 'polite') {
  const region = priority === 'assertive'
    ? document.getElementById('urgent-alerts')
    : document.getElementById('price-updates');

  region.textContent = message;

  // Clear after announcement
  setTimeout(() => { region.textContent = ''; }, 1000);
}

// Usage
announceUpdate('Price updated to 72%', 'polite');
announceUpdate('Market closing in 5 minutes!', 'assertive');
```

---

## Part IV: Assistive Technology Support

### Screen Readers

**Tested With**:
- NVDA (Windows)
- JAWS (Windows)
- VoiceOver (macOS, iOS)
- TalkBack (Android)
- Orca (Linux)

**Optimizations**:
```typescript
// Meaningful link text
// Bad
<a href="/market/123">Click here</a>

// Good
<a href="/market/123">View market: Will X happen by Y date?</a>

// Hidden descriptive text
<button>
  <span class="icon" aria-hidden="true">📊</span>
  <span class="sr-only">View statistics for this market</span>
</button>

// Reading order
.visually-hidden-but-readable {
  position: absolute;
  width: 1px;
  height: 1px;
  padding: 0;
  margin: -1px;
  overflow: hidden;
  clip: rect(0, 0, 0, 0);
  border: 0;
}
```

### Voice Control

**Tested With**:
- Dragon NaturallySpeaking
- Voice Control (macOS, iOS)
- Voice Access (Android)
- Windows Speech Recognition

**Optimizations**:
- All interactive elements have visible labels
- Labels match accessible names
- Consistent command vocabulary
- Number overlays for link selection

### Switch Access

**Support**:
- Single switch scanning
- Two-switch step scanning
- Customizable scan timing
- Large activation targets (minimum 44×44px)

### Eye Tracking

**Support**:
- Dwell-to-click compatible
- Large targets
- Minimal precision required
- Gaze-aware layout options

---

## Part V: Cognitive Accessibility

### Plain Language

**Guidelines**:
- Target 8th-grade reading level for core content
- Define technical terms on first use
- Use short sentences and paragraphs
- Active voice preferred
- Avoid jargon where possible

**Example**:
```markdown
// Before (complex)
"The aggregated probability distribution derived from the
MATL-weighted ensemble of predictions suggests a 73%
likelihood of the affirmative outcome materializing."

// After (plain)
"Based on all predictions, there's about a 73% chance
this will happen. Predictions from more experienced
users count more in this calculation."
```

### Consistent Design

**Patterns**:
- Navigation in same place on every page
- Actions always work the same way
- Icons always mean the same thing
- Feedback always appears in the same location

### Error Prevention

```typescript
// Confirmation for significant actions
interface ConfirmationConfig {
  action: string;
  consequences: string;
  reversible: boolean;
  confirmationMethod: 'button' | 'checkbox' | 'type-to-confirm';
}

// Example: High-stake prediction
const confirmHighStake: ConfirmationConfig = {
  action: "Submit prediction with 500 HAM stake",
  consequences: "You will lose 500 HAM if your prediction is wrong",
  reversible: false,
  confirmationMethod: 'checkbox', // "I understand I may lose my stake"
};

// Auto-save to prevent data loss
function autoSaveDraft(formData: FormData) {
  localStorage.setItem('prediction-draft', JSON.stringify(formData));
  showMessage('Draft saved automatically');
}
```

### Timing and Time Limits

**Policy**: No time limits for core tasks.

**Exceptions**:
- Real-time market dynamics (prices change)
- Market closing times (with ample warning)

**Accommodations**:
```typescript
// Warnings before time-sensitive events
const timeWarnings = [
  { before: 24 * 60, message: "Market closes in 24 hours" },
  { before: 60, message: "Market closes in 1 hour" },
  { before: 15, message: "Market closes in 15 minutes" },
  { before: 5, message: "Final warning: 5 minutes until close" },
];

// Extended time option for those who need it
interface AccessibilityPreferences {
  extendedTimeNotifications: boolean;
  additionalWarningTime: number; // Extra minutes before warnings
}
```

### Memory and Attention

**Support**:
- Session persistence (return to where you left off)
- Clear progress indicators
- Step-by-step guidance available
- Undo/redo support
- Clear history of your actions

---

## Part VI: Sensory Accommodations

### Visual

**Low Vision**:
- Minimum 4.5:1 contrast ratio (7:1 in high contrast mode)
- Resizable to 400% without loss of functionality
- No information conveyed by color alone
- Customizable font sizes

**Color Blindness**:
- Patterns and icons supplement colors
- Color palettes tested with simulators
- Labels on color-coded elements

```css
/* High contrast mode */
@media (prefers-contrast: more) {
  :root {
    --text-color: #000000;
    --background-color: #ffffff;
    --link-color: #0000ee;
    --border-color: #000000;
    --focus-outline: 3px solid #000000;
  }
}

/* Reduced motion */
@media (prefers-reduced-motion: reduce) {
  * {
    animation-duration: 0.01ms !important;
    transition-duration: 0.01ms !important;
  }
}
```

**Blindness**:
- Full screen reader support
- Meaningful alt text
- Audio descriptions available
- Tactile/braille display compatible

### Auditory

**Deaf/Hard of Hearing**:
- Captions for all audio content
- Transcripts available
- Visual indicators for audio alerts
- Sign language interpretation for key content (planned)

```typescript
// Visual alternatives to audio
interface Notification {
  audio?: AudioClip;
  visual: {
    message: string;
    icon: string;
    flashColor?: string; // For optional flash notification
    vibration?: VibrationPattern; // For mobile
  };
}

// User preference
function notify(notification: Notification, preferences: UserPrefs) {
  if (preferences.audioEnabled && notification.audio) {
    playAudio(notification.audio);
  }

  showVisualNotification(notification.visual);

  if (preferences.vibrationEnabled && notification.visual.vibration) {
    navigator.vibrate(notification.visual.vibration);
  }
}
```

### Motor

**Limited Mobility**:
- Large click/tap targets (minimum 44×44px)
- Adequate spacing between targets
- No precision movements required
- Keyboard accessible throughout

**Tremors/Limited Dexterity**:
- Debounced inputs
- Confirmation before destructive actions
- Drag-and-drop alternatives
- Voice input support

```typescript
// Large, spaced targets
.button {
  min-width: 44px;
  min-height: 44px;
  padding: 12px 24px;
  margin: 8px;
}

// Debounced input
function debounce(fn, delay) {
  let timeoutId;
  return (...args) => {
    clearTimeout(timeoutId);
    timeoutId = setTimeout(() => fn(...args), delay);
  };
}

const handleInput = debounce((value) => {
  processInput(value);
}, 300);
```

---

## Part VII: Testing and Validation

### Automated Testing

```typescript
// Jest + axe-core for automated accessibility testing
import { axe, toHaveNoViolations } from 'jest-axe';

expect.extend(toHaveNoViolations);

describe('Market Card', () => {
  it('should have no accessibility violations', async () => {
    const { container } = render(<MarketCard market={mockMarket} />);
    const results = await axe(container);
    expect(results).toHaveNoViolations();
  });
});

// Cypress accessibility tests
describe('Prediction Flow', () => {
  it('should be keyboard navigable', () => {
    cy.visit('/markets/123');
    cy.get('body').tab();
    cy.focused().should('have.attr', 'href').and('include', 'skip');
    cy.focused().tab();
    // ... continue testing focus order
  });
});
```

### Manual Testing Checklist

**Keyboard**:
- [ ] All functionality reachable via keyboard
- [ ] Focus visible at all times
- [ ] Logical focus order
- [ ] No keyboard traps
- [ ] Skip links work

**Screen Reader**:
- [ ] All content announced
- [ ] Announcements make sense in context
- [ ] Dynamic updates announced
- [ ] Forms navigable and understandable
- [ ] Errors clearly communicated

**Visual**:
- [ ] Works at 200% zoom
- [ ] Works at 400% zoom
- [ ] High contrast mode usable
- [ ] Color not sole information carrier
- [ ] Text resizable

**Motor**:
- [ ] Targets minimum 44×44px
- [ ] No time-critical actions required
- [ ] Alternatives to drag-and-drop
- [ ] No multi-finger gestures required

### User Testing

We test with users who:
- Use screen readers daily
- Navigate primarily by keyboard
- Have low vision or color blindness
- Have cognitive disabilities
- Use switch access
- Use voice control

Testing frequency: Quarterly user testing sessions

---

## Part VIII: Continuous Improvement

### Accessibility Feedback

**How to Report**:
- In-app accessibility feedback button
- Email: accessibility@epistemic-markets.org
- GitHub issues with "accessibility" label

**Response Commitment**:
- Acknowledge within 48 hours
- Critical issues: fix within 1 week
- Major issues: fix within 1 month
- Minor issues: include in next release

### Accessibility Roadmap

**Current** (Implemented):
- WCAG 2.2 Level AA compliance
- Full keyboard navigation
- Screen reader optimization
- High contrast mode
- Reduced motion support

**Next** (In Progress):
- Sign language interpretation for tutorials
- Simplified language mode
- Customizable interface density
- Voice command interface

**Future** (Planned):
- AI-powered content descriptions
- Personalized accessibility profiles
- Cognitive load detection and adaptation
- Community accessibility champions program

---

## Part IX: Accessibility Statement

### Commitment

Epistemic Markets is committed to ensuring digital accessibility for people with disabilities. We are continually improving the user experience for everyone and applying the relevant accessibility standards.

### Conformance Status

Epistemic Markets is **partially conformant** with WCAG 2.2 Level AA. Partially conformant means that some parts of the content do not fully conform to the accessibility standard.

### Known Limitations

| Area | Issue | Workaround | Fix Target |
|------|-------|------------|------------|
| Complex charts | Limited alt text | Data tables available | Q2 2024 |
| Real-time updates | Rapid announcements | Configurable update frequency | Q1 2024 |
| PDF exports | Not fully accessible | HTML alternative | Q3 2024 |

### Feedback

We welcome your feedback on the accessibility of Epistemic Markets. Please let us know if you encounter accessibility barriers:

- **Email**: accessibility@epistemic-markets.org
- **Response time**: 2 business days

### Technical Specifications

Epistemic Markets relies on the following technologies to work with the particular combination of web browser and any assistive technologies or plugins installed on your computer:

- HTML
- WAI-ARIA
- CSS
- JavaScript

### Assessment Approach

Epistemic Markets assessed the accessibility by the following approaches:
- Self-evaluation
- External audit (annual)
- User testing (quarterly)

---

## Conclusion

Accessibility is not a feature. It's a fundamental requirement for collective intelligence. Every mind excluded is a perspective lost.

We build for:
- The blind seer who perceives patterns others miss
- The deaf analyst who reads markets better than most hear them
- The person with tremors whose careful predictions outperform hasty ones
- The neurodivergent thinker who sees connections invisibly to others

If you can't access Epistemic Markets, we have failed. Tell us, and we will fix it.

---

*"The only disability in life is a bad attitude."*
*— Scott Hamilton*

*We add: "The only disability in software is bad design."*

*We commit to good design, for all.*

