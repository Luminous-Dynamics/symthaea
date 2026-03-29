// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * React MFA Components for Mycelix SDK
 *
 * Provides pre-built UI components for multi-factor authentication
 * workflows, including enrollment, verification, and status display.
 *
 * @module frameworks/react/components
 * @packageDocumentation
 *
 * @example Basic MFA Setup
 * ```tsx
 * import {
 *   MfaEnrollmentWizard,
 *   FactorList,
 *   AssuranceLevelBadge,
 *   FlEligibilityBanner,
 * } from '@mycelix/sdk/react/components';
 *
 * function SecuritySettings() {
 *   const [showEnrollment, setShowEnrollment] = useState(false);
 *
 *   return (
 *     <div>
 *       <header>
 *         <h1>Security Settings</h1>
 *         <AssuranceLevelBadge showScore />
 *       </header>
 *
 *       <FlEligibilityBanner
 *         onEnrollClick={() => setShowEnrollment(true)}
 *       />
 *
 *       <FactorList
 *         onEnrollNew={() => setShowEnrollment(true)}
 *         onVerify={(factor) => console.log('Verify', factor)}
 *       />
 *
 *       {showEnrollment && (
 *         <MfaEnrollmentWizard
 *           onComplete={(factor) => {
 *             console.log('Enrolled:', factor);
 *             setShowEnrollment(false);
 *           }}
 *           onCancel={() => setShowEnrollment(false)}
 *         />
 *       )}
 *     </div>
 *   );
 * }
 * ```
 */

// =============================================================================
// Enrollment Components
// =============================================================================

export {
  MfaEnrollmentWizard,
  type MfaEnrollmentWizardProps,
} from './MfaEnrollmentWizard.js';

// =============================================================================
// Verification Components
// =============================================================================

export {
  FactorVerificationModal,
  type FactorVerificationModalProps,
} from './FactorVerificationModal.js';

// =============================================================================
// Status Display Components
// =============================================================================

export {
  AssuranceLevelBadge,
  AssuranceLevelIndicator,
  type AssuranceLevelBadgeProps,
  type AssuranceLevelIndicatorProps,
  type BadgeSize,
} from './AssuranceLevelBadge.js';

export {
  FlEligibilityBanner,
  FlEligibilityIndicator,
  type FlEligibilityBannerProps,
  type FlEligibilityIndicatorProps,
  type BannerVariant,
} from './FlEligibilityBanner.js';

// =============================================================================
// Factor Management Components
// =============================================================================

export {
  FactorList,
  type FactorListProps,
} from './FactorList.js';
