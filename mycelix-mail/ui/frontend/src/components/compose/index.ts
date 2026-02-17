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
