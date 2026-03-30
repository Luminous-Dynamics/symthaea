// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Email Templates System
 *
 * Pre-built templates with claim attachments:
 * - Introduction requests
 * - Attestation requests
 * - Trust verification
 * - Professional outreach
 * - Custom templates
 */

import { useState, useCallback, useMemo } from 'react';

// ============================================
// Types
// ============================================

export interface EmailTemplate {
  id: string;
  name: string;
  description: string;
  category: TemplateCategory;
  icon: string;
  subject: string;
  body: string;
  variables: TemplateVariable[];
  suggestedClaims: string[];
  isCustom?: boolean;
  createdAt?: string;
}

export interface TemplateVariable {
  name: string;
  label: string;
  type: 'text' | 'email' | 'select' | 'textarea';
  required?: boolean;
  defaultValue?: string;
  options?: string[];
  placeholder?: string;
}

export type TemplateCategory =
  | 'introduction'
  | 'attestation'
  | 'verification'
  | 'professional'
  | 'personal'
  | 'custom';

interface FilledTemplate {
  subject: string;
  body: string;
  claims: string[];
}

// ============================================
// Built-in Templates
// ============================================

const builtInTemplates: EmailTemplate[] = [
  {
    id: 'introduction-request',
    name: 'Request Introduction',
    description: 'Ask a contact to introduce you to someone in their network',
    category: 'introduction',
    icon: '🤝',
    subject: 'Introduction Request: {{targetName}}',
    body: `Hi {{recipientName}},

I hope this message finds you well. I noticed that you're connected with {{targetName}}, and I was wondering if you'd be comfortable making an introduction.

I'm interested in connecting because:
{{reason}}

I would really appreciate it if you could facilitate this introduction. I've attached my professional credentials for reference.

Best regards,
{{senderName}}`,
    variables: [
      { name: 'recipientName', label: 'Recipient Name', type: 'text', required: true },
      { name: 'targetName', label: 'Person to Meet', type: 'text', required: true },
      { name: 'reason', label: 'Reason for Introduction', type: 'textarea', required: true },
      { name: 'senderName', label: 'Your Name', type: 'text', required: true },
    ],
    suggestedClaims: ['professional_identity', 'organization_membership'],
  },
  {
    id: 'attestation-request',
    name: 'Request Attestation',
    description: 'Ask someone to attest to your identity or relationship',
    category: 'attestation',
    icon: '✍️',
    subject: 'Trust Attestation Request',
    body: `Hi {{recipientName}},

I'm building my trust network on Mycelix and would greatly appreciate if you could provide an attestation for me.

We've known each other through {{context}}, and I believe your attestation would help establish my credibility in the epistemic network.

The attestation process is simple and only takes a minute. You can attest to:
- Our professional relationship
- My identity verification
- Our history of communication

Thank you for considering this request!

Best regards,
{{senderName}}`,
    variables: [
      { name: 'recipientName', label: 'Recipient Name', type: 'text', required: true },
      { name: 'context', label: 'How You Know Each Other', type: 'text', required: true, placeholder: 'e.g., work colleagues, university' },
      { name: 'senderName', label: 'Your Name', type: 'text', required: true },
    ],
    suggestedClaims: ['email_verification', 'gitcoin_passport'],
  },
  {
    id: 'trust-verification',
    name: 'Trust Verification',
    description: 'Verify your identity to a new contact',
    category: 'verification',
    icon: '🔐',
    subject: 'Identity Verification - {{senderName}}',
    body: `Hi {{recipientName}},

Thank you for connecting with me. To establish trust, I'm sharing my verified credentials:

{{verificationDetails}}

You can verify these claims directly in your Mycelix client. Each claim is cryptographically signed and can be independently verified.

I'm looking forward to building a trusted communication channel with you.

Best regards,
{{senderName}}`,
    variables: [
      { name: 'recipientName', label: 'Recipient Name', type: 'text', required: true },
      { name: 'verificationDetails', label: 'Verification Details', type: 'textarea', placeholder: 'Additional context about your credentials' },
      { name: 'senderName', label: 'Your Name', type: 'text', required: true },
    ],
    suggestedClaims: ['email_verification', 'github_verification', 'domain_ownership'],
  },
  {
    id: 'professional-outreach',
    name: 'Professional Outreach',
    description: 'Reach out to a professional contact with verified credentials',
    category: 'professional',
    icon: '💼',
    subject: '{{subject}}',
    body: `Dear {{recipientName}},

{{opening}}

I'm reaching out because {{reason}}.

{{mainContent}}

I've attached my professional credentials for your reference. I would welcome the opportunity to discuss this further at your convenience.

Best regards,
{{senderName}}
{{senderTitle}}`,
    variables: [
      { name: 'recipientName', label: 'Recipient Name', type: 'text', required: true },
      { name: 'subject', label: 'Email Subject', type: 'text', required: true },
      { name: 'opening', label: 'Opening Line', type: 'text', placeholder: 'e.g., I hope this message finds you well' },
      { name: 'reason', label: 'Reason for Contact', type: 'textarea', required: true },
      { name: 'mainContent', label: 'Main Message', type: 'textarea', required: true },
      { name: 'senderName', label: 'Your Name', type: 'text', required: true },
      { name: 'senderTitle', label: 'Your Title', type: 'text', placeholder: 'e.g., Software Engineer at Acme Corp' },
    ],
    suggestedClaims: ['professional_identity', 'organization_membership', 'domain_ownership'],
  },
  {
    id: 'collaboration-invite',
    name: 'Collaboration Invitation',
    description: 'Invite someone to collaborate on a project',
    category: 'professional',
    icon: '🚀',
    subject: 'Collaboration Opportunity: {{projectName}}',
    body: `Hi {{recipientName}},

I've been following your work on {{theirWork}} and I'm impressed by {{compliment}}.

I'm currently working on {{projectName}} - {{projectDescription}}.

I believe your expertise in {{theirExpertise}} would be invaluable for this project. Would you be interested in exploring a collaboration?

Here's what I'm envisioning:
{{collaborationDetails}}

I'd love to set up a call to discuss this further. Let me know your availability!

Best regards,
{{senderName}}`,
    variables: [
      { name: 'recipientName', label: 'Recipient Name', type: 'text', required: true },
      { name: 'theirWork', label: 'Their Work', type: 'text', placeholder: 'What they\'re known for' },
      { name: 'compliment', label: 'Compliment', type: 'text', placeholder: 'What impressed you' },
      { name: 'projectName', label: 'Project Name', type: 'text', required: true },
      { name: 'projectDescription', label: 'Project Description', type: 'textarea', required: true },
      { name: 'theirExpertise', label: 'Their Expertise', type: 'text', required: true },
      { name: 'collaborationDetails', label: 'Collaboration Details', type: 'textarea', required: true },
      { name: 'senderName', label: 'Your Name', type: 'text', required: true },
    ],
    suggestedClaims: ['github_verification', 'professional_identity'],
  },
  {
    id: 'meeting-request',
    name: 'Meeting Request',
    description: 'Request a meeting with verified identity',
    category: 'professional',
    icon: '📅',
    subject: 'Meeting Request: {{topic}}',
    body: `Hi {{recipientName}},

I would like to schedule a meeting to discuss {{topic}}.

{{context}}

Would you be available for a {{duration}} meeting sometime {{timeframe}}? I'm flexible with timing.

Please let me know what works best for you.

Best regards,
{{senderName}}`,
    variables: [
      { name: 'recipientName', label: 'Recipient Name', type: 'text', required: true },
      { name: 'topic', label: 'Meeting Topic', type: 'text', required: true },
      { name: 'context', label: 'Context', type: 'textarea', placeholder: 'Why you want to meet' },
      { name: 'duration', label: 'Duration', type: 'select', options: ['15 minutes', '30 minutes', '45 minutes', '1 hour'], defaultValue: '30 minutes' },
      { name: 'timeframe', label: 'Timeframe', type: 'text', placeholder: 'e.g., this week, next Monday' },
      { name: 'senderName', label: 'Your Name', type: 'text', required: true },
    ],
    suggestedClaims: ['email_verification'],
  },
];

// ============================================
// Template Card Component
// ============================================

interface TemplateCardProps {
  template: EmailTemplate;
  onSelect: () => void;
  onEdit?: () => void;
  onDelete?: () => void;
}

function TemplateCard({ template, onSelect, onEdit, onDelete }: TemplateCardProps) {
  return (
    <div
      className="p-4 bg-white dark:bg-gray-800 rounded-xl border border-gray-200 dark:border-gray-700 hover:border-blue-300 dark:hover:border-blue-700 transition-colors cursor-pointer group"
      onClick={onSelect}
    >
      <div className="flex items-start gap-3">
        <div className="text-2xl">{template.icon}</div>
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2">
            <h4 className="font-semibold text-gray-900 dark:text-gray-100">
              {template.name}
            </h4>
            {template.isCustom && (
              <span className="px-1.5 py-0.5 text-xs bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded">
                Custom
              </span>
            )}
          </div>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            {template.description}
          </p>
          {template.suggestedClaims.length > 0 && (
            <div className="flex flex-wrap gap-1 mt-2">
              {template.suggestedClaims.slice(0, 3).map((claim) => (
                <span
                  key={claim}
                  className="px-1.5 py-0.5 text-xs bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded"
                >
                  {claim.replace(/_/g, ' ')}
                </span>
              ))}
            </div>
          )}
        </div>
        {template.isCustom && (
          <div className="opacity-0 group-hover:opacity-100 flex gap-1">
            {onEdit && (
              <button
                onClick={(e) => { e.stopPropagation(); onEdit(); }}
                className="p-1 text-gray-400 hover:text-blue-500"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                </svg>
              </button>
            )}
            {onDelete && (
              <button
                onClick={(e) => { e.stopPropagation(); onDelete(); }}
                className="p-1 text-gray-400 hover:text-red-500"
              >
                <svg className="w-4 h-4" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                </svg>
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================
// Variable Form Component
// ============================================

interface VariableFormProps {
  variables: TemplateVariable[];
  values: Record<string, string>;
  onChange: (name: string, value: string) => void;
}

function VariableForm({ variables, values, onChange }: VariableFormProps) {
  return (
    <div className="space-y-4">
      {variables.map((variable) => (
        <div key={variable.name}>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            {variable.label}
            {variable.required && <span className="text-red-500 ml-1">*</span>}
          </label>
          {variable.type === 'textarea' ? (
            <textarea
              value={values[variable.name] || ''}
              onChange={(e) => onChange(variable.name, e.target.value)}
              placeholder={variable.placeholder}
              rows={3}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500"
            />
          ) : variable.type === 'select' ? (
            <select
              value={values[variable.name] || variable.defaultValue || ''}
              onChange={(e) => onChange(variable.name, e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500"
            >
              {variable.options?.map((option) => (
                <option key={option} value={option}>
                  {option}
                </option>
              ))}
            </select>
          ) : (
            <input
              type={variable.type}
              value={values[variable.name] || ''}
              onChange={(e) => onChange(variable.name, e.target.value)}
              placeholder={variable.placeholder}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-800 focus:ring-2 focus:ring-blue-500"
            />
          )}
        </div>
      ))}
    </div>
  );
}

// ============================================
// Main Email Templates Component
// ============================================

interface EmailTemplatesProps {
  isOpen: boolean;
  onClose: () => void;
  onApply: (filled: FilledTemplate) => void;
}

export function EmailTemplates({ isOpen, onClose, onApply }: EmailTemplatesProps) {
  const [selectedTemplate, setSelectedTemplate] = useState<EmailTemplate | null>(null);
  const [variableValues, setVariableValues] = useState<Record<string, string>>({});
  const [customTemplates, setCustomTemplates] = useState<EmailTemplate[]>([]);
  const [activeCategory, setActiveCategory] = useState<TemplateCategory | 'all'>('all');

  // Load custom templates
  useState(() => {
    const saved = localStorage.getItem('mycelix_custom_templates');
    if (saved) {
      setCustomTemplates(JSON.parse(saved));
    }
  });

  const allTemplates = useMemo(
    () => [...builtInTemplates, ...customTemplates],
    [customTemplates]
  );

  const filteredTemplates = useMemo(
    () =>
      activeCategory === 'all'
        ? allTemplates
        : allTemplates.filter((t) => t.category === activeCategory),
    [allTemplates, activeCategory]
  );

  const categories: { id: TemplateCategory | 'all'; label: string; icon: string }[] = [
    { id: 'all', label: 'All', icon: '📋' },
    { id: 'introduction', label: 'Introductions', icon: '🤝' },
    { id: 'attestation', label: 'Attestations', icon: '✍️' },
    { id: 'verification', label: 'Verification', icon: '🔐' },
    { id: 'professional', label: 'Professional', icon: '💼' },
    { id: 'custom', label: 'Custom', icon: '⚡' },
  ];

  const handleVariableChange = useCallback((name: string, value: string) => {
    setVariableValues((prev) => ({ ...prev, [name]: value }));
  }, []);

  const fillTemplate = useCallback((): FilledTemplate | null => {
    if (!selectedTemplate) return null;

    // Check required fields
    const missingRequired = selectedTemplate.variables
      .filter((v) => v.required && !variableValues[v.name])
      .map((v) => v.label);

    if (missingRequired.length > 0) {
      alert(`Please fill in required fields: ${missingRequired.join(', ')}`);
      return null;
    }

    // Replace variables in subject and body
    let subject = selectedTemplate.subject;
    let body = selectedTemplate.body;

    for (const [key, value] of Object.entries(variableValues)) {
      const regex = new RegExp(`{{${key}}}`, 'g');
      subject = subject.replace(regex, value);
      body = body.replace(regex, value);
    }

    return {
      subject,
      body,
      claims: selectedTemplate.suggestedClaims,
    };
  }, [selectedTemplate, variableValues]);

  const handleApply = useCallback(() => {
    const filled = fillTemplate();
    if (filled) {
      onApply(filled);
      onClose();
      setSelectedTemplate(null);
      setVariableValues({});
    }
  }, [fillTemplate, onApply, onClose]);

  const handleBack = useCallback(() => {
    setSelectedTemplate(null);
    setVariableValues({});
  }, []);

  if (!isOpen) return null;

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50 z-50" onClick={onClose} />

      {/* Modal */}
      <div className="fixed inset-4 md:inset-auto md:top-1/2 md:left-1/2 md:-translate-x-1/2 md:-translate-y-1/2 md:w-[800px] md:max-h-[85vh] bg-white dark:bg-gray-900 rounded-xl shadow-2xl z-50 flex flex-col overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between px-6 py-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-3">
            {selectedTemplate && (
              <button
                onClick={handleBack}
                className="p-1 hover:bg-gray-100 dark:hover:bg-gray-800 rounded"
              >
                <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
                </svg>
              </button>
            )}
            <span className="text-2xl">{selectedTemplate?.icon || '📝'}</span>
            <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
              {selectedTemplate?.name || 'Email Templates'}
            </h2>
          </div>
          <button
            onClick={onClose}
            className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors"
          >
            <svg className="w-5 h-5 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Content */}
        <div className="flex-1 overflow-y-auto">
          {selectedTemplate ? (
            // Template filling view
            <div className="p-6 space-y-6">
              <p className="text-gray-600 dark:text-gray-400">
                {selectedTemplate.description}
              </p>

              <VariableForm
                variables={selectedTemplate.variables}
                values={variableValues}
                onChange={handleVariableChange}
              />

              {/* Suggested claims */}
              {selectedTemplate.suggestedClaims.length > 0 && (
                <div>
                  <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Suggested Claims to Attach
                  </h4>
                  <div className="flex flex-wrap gap-2">
                    {selectedTemplate.suggestedClaims.map((claim) => (
                      <span
                        key={claim}
                        className="px-2 py-1 text-sm bg-emerald-100 dark:bg-emerald-900/30 text-emerald-700 dark:text-emerald-300 rounded-lg"
                      >
                        🔐 {claim.replace(/_/g, ' ')}
                      </span>
                    ))}
                  </div>
                </div>
              )}

              {/* Preview */}
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Preview
                </h4>
                <div className="text-sm">
                  <div className="font-medium text-gray-900 dark:text-gray-100 mb-2">
                    Subject: {fillTemplate()?.subject || selectedTemplate.subject}
                  </div>
                  <div className="text-gray-600 dark:text-gray-400 whitespace-pre-wrap">
                    {fillTemplate()?.body || selectedTemplate.body}
                  </div>
                </div>
              </div>
            </div>
          ) : (
            // Template selection view
            <div className="p-6">
              {/* Category tabs */}
              <div className="flex flex-wrap gap-2 mb-6">
                {categories.map((cat) => (
                  <button
                    key={cat.id}
                    onClick={() => setActiveCategory(cat.id)}
                    className={`px-3 py-1.5 text-sm rounded-lg transition-colors ${
                      activeCategory === cat.id
                        ? 'bg-blue-100 dark:bg-blue-900/30 text-blue-700 dark:text-blue-300'
                        : 'bg-gray-100 dark:bg-gray-800 text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
                    }`}
                  >
                    <span className="mr-1">{cat.icon}</span>
                    {cat.label}
                  </button>
                ))}
              </div>

              {/* Template grid */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {filteredTemplates.map((template) => (
                  <TemplateCard
                    key={template.id}
                    template={template}
                    onSelect={() => setSelectedTemplate(template)}
                  />
                ))}
              </div>

              {filteredTemplates.length === 0 && (
                <div className="text-center py-8 text-gray-500">
                  No templates in this category
                </div>
              )}
            </div>
          )}
        </div>

        {/* Footer */}
        {selectedTemplate && (
          <div className="flex justify-end gap-3 px-6 py-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800">
            <button
              onClick={handleBack}
              className="px-4 py-2 text-gray-600 dark:text-gray-400"
            >
              Back
            </button>
            <button
              onClick={handleApply}
              className="px-4 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg"
            >
              Use Template
            </button>
          </div>
        )}
      </div>
    </>
  );
}

// ============================================
// Hook for Templates
// ============================================

export function useEmailTemplates() {
  const [isOpen, setIsOpen] = useState(false);

  return {
    isOpen,
    open: () => setIsOpen(true),
    close: () => setIsOpen(false),
  };
}

export default EmailTemplates;
