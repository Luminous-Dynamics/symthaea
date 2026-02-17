/**
 * Accessibility Module
 * WCAG 2.1 AA compliant utilities and components
 */

// Core utilities
export {
  A11yProvider,
  useA11y,
  SkipLink,
  VisuallyHidden,
  useFocusTrap,
  useRovingTabindex,
  useFocusRestore,
  IconButton,
  LiveRegion,
  ProgressIndicator,
  LoadingState,
  KeyboardShortcut,
  Tabs,
  getContrastRatio,
  meetsContrastRequirements,
  focusStyles,
} from './accessibility';

// ARIA patterns
export {
  Modal,
  Dropdown,
  Combobox,
  Accordion,
  Toast,
} from './aria-patterns';
