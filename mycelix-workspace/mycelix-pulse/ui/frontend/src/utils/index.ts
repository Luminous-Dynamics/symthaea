// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Utils Index
 *
 * Central export for all utility functions:
 * - Accessibility helpers (ARIA, focus management)
 * - Demo data generators
 */

// Accessibility utilities
export {
  createLiveRegion,
  announce,
  useAnnounce,
  focusFirstElement,
  getFocusableElements,
  useFocusTrap,
  useFocusOnChange,
  useArrowNavigation,
  getListboxProps,
  getOptionProps,
  getDialogProps,
  getTabPanelProps,
  getTabProps,
  srOnlyStyles,
  ScreenReaderOnly,
  formatTrustScoreForSR,
  formatTierForSR,
  formatAssuranceLevelForSR,
} from './accessibility';

// Demo data generators
export {
  generateContact,
  generateContacts,
  generateEmail,
  generateEmails,
  generateClaim,
  generateClaims,
  generateTrustEdge,
  generateTrustNetwork,
  generateAttestation,
  generateAttestations,
  useDemoData,
} from './demoData';

export type {
  DemoContact,
  DemoEmail,
  DemoTrustNetwork,
  DemoAttestation,
} from './demoData';
