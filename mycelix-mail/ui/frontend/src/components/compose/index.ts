// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Compose Components Index
 *
 * Email composition components with epistemic enhancements
 */

export { default as ComposeWithClaims } from './ComposeWithClaims';
export type { AttachedClaim } from './ComposeWithClaims';
export { CredentialCard, QuickAttachBadges, AttachedClaimsSummary } from './ComposeWithClaims';

// Email templates
export {
  EmailTemplateSelector,
  TemplatePreview,
  CustomTemplateEditor,
  useEmailTemplates,
  builtInTemplates,
} from './EmailTemplates';
export type { EmailTemplate, TemplateVariable } from './EmailTemplates';
