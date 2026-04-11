// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Email Template Engine
 *
 * Features:
 * - Variable interpolation with {{variable}} syntax
 * - Conditional blocks with {{#if}} {{/if}}
 * - Loops with {{#each}} {{/each}}
 * - Filters (uppercase, lowercase, date, etc.)
 * - Template inheritance
 * - Preview generation
 */

export interface TemplateVariable {
  name: string;
  type: 'text' | 'number' | 'date' | 'email' | 'select' | 'rich-text';
  label: string;
  description?: string;
  default?: any;
  required?: boolean;
  options?: { label: string; value: string }[];
}

export interface EmailTemplate {
  id: string;
  name: string;
  description?: string;
  category: string;
  subject: string;
  body: string;
  bodyHtml?: string;
  variables: TemplateVariable[];
  createdAt: number;
  updatedAt: number;
  usageCount: number;
  author?: string;
  tags?: string[];
  parentId?: string; // For template inheritance
}

export interface TemplateContext {
  [key: string]: any;
}

export interface RenderOptions {
  escapeHtml?: boolean;
  locale?: string;
  timezone?: string;
}

// Built-in filters
const FILTERS: Record<string, (value: any, ...args: any[]) => any> = {
  uppercase: (v) => String(v).toUpperCase(),
  lowercase: (v) => String(v).toLowerCase(),
  capitalize: (v) => String(v).charAt(0).toUpperCase() + String(v).slice(1),
  trim: (v) => String(v).trim(),
  truncate: (v, length = 50) => {
    const s = String(v);
    return s.length > length ? s.slice(0, length) + '...' : s;
  },
  date: (v, format = 'short') => {
    const d = new Date(v);
    if (format === 'short') return d.toLocaleDateString();
    if (format === 'long') return d.toLocaleDateString(undefined, { dateStyle: 'long' });
    if (format === 'time') return d.toLocaleTimeString();
    if (format === 'datetime') return d.toLocaleString();
    if (format === 'relative') return getRelativeTime(d);
    return d.toISOString();
  },
  number: (v, decimals = 0) => Number(v).toFixed(decimals),
  currency: (v, currency = 'USD') => {
    return new Intl.NumberFormat(undefined, {
      style: 'currency',
      currency,
    }).format(Number(v));
  },
  plural: (v, singular, plural) => (Number(v) === 1 ? singular : plural),
  default: (v, defaultValue) => v ?? defaultValue,
  json: (v) => JSON.stringify(v, null, 2),
  first: (v) => Array.isArray(v) ? v[0] : v,
  last: (v) => Array.isArray(v) ? v[v.length - 1] : v,
  join: (v, separator = ', ') => Array.isArray(v) ? v.join(separator) : v,
  length: (v) => Array.isArray(v) ? v.length : String(v).length,
};

function getRelativeTime(date: Date): string {
  const now = new Date();
  const diff = now.getTime() - date.getTime();
  const seconds = Math.floor(diff / 1000);
  const minutes = Math.floor(seconds / 60);
  const hours = Math.floor(minutes / 60);
  const days = Math.floor(hours / 24);

  if (days > 0) return `${days} day${days > 1 ? 's' : ''} ago`;
  if (hours > 0) return `${hours} hour${hours > 1 ? 's' : ''} ago`;
  if (minutes > 0) return `${minutes} minute${minutes > 1 ? 's' : ''} ago`;
  return 'just now';
}

export class TemplateEngine {
  private templates: Map<string, EmailTemplate> = new Map();
  private customFilters: Map<string, (value: any, ...args: any[]) => any> = new Map();

  constructor() {
    // Load saved templates
    this.loadTemplates();
  }

  /**
   * Register a custom filter
   */
  registerFilter(name: string, fn: (value: any, ...args: any[]) => any): void {
    this.customFilters.set(name, fn);
  }

  /**
   * Add a template
   */
  addTemplate(template: Omit<EmailTemplate, 'id' | 'createdAt' | 'updatedAt' | 'usageCount'>): EmailTemplate {
    const newTemplate: EmailTemplate = {
      ...template,
      id: `tpl-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      createdAt: Date.now(),
      updatedAt: Date.now(),
      usageCount: 0,
    };

    this.templates.set(newTemplate.id, newTemplate);
    this.saveTemplates();

    return newTemplate;
  }

  /**
   * Update a template
   */
  updateTemplate(id: string, updates: Partial<EmailTemplate>): EmailTemplate | null {
    const template = this.templates.get(id);
    if (!template) return null;

    const updated = {
      ...template,
      ...updates,
      id: template.id, // Prevent ID change
      updatedAt: Date.now(),
    };

    this.templates.set(id, updated);
    this.saveTemplates();

    return updated;
  }

  /**
   * Delete a template
   */
  deleteTemplate(id: string): boolean {
    const result = this.templates.delete(id);
    if (result) {
      this.saveTemplates();
    }
    return result;
  }

  /**
   * Get a template by ID
   */
  getTemplate(id: string): EmailTemplate | null {
    return this.templates.get(id) || null;
  }

  /**
   * Get all templates
   */
  getAllTemplates(): EmailTemplate[] {
    return Array.from(this.templates.values());
  }

  /**
   * Get templates by category
   */
  getTemplatesByCategory(category: string): EmailTemplate[] {
    return this.getAllTemplates().filter((t) => t.category === category);
  }

  /**
   * Search templates
   */
  searchTemplates(query: string): EmailTemplate[] {
    const q = query.toLowerCase();
    return this.getAllTemplates().filter(
      (t) =>
        t.name.toLowerCase().includes(q) ||
        t.description?.toLowerCase().includes(q) ||
        t.tags?.some((tag) => tag.toLowerCase().includes(q))
    );
  }

  /**
   * Render a template with context
   */
  render(templateOrId: string | EmailTemplate, context: TemplateContext, options: RenderOptions = {}): { subject: string; body: string; bodyHtml?: string } {
    const template = typeof templateOrId === 'string'
      ? this.templates.get(templateOrId)
      : templateOrId;

    if (!template) {
      throw new Error('Template not found');
    }

    // Handle template inheritance
    let baseTemplate: EmailTemplate | undefined;
    if (template.parentId) {
      baseTemplate = this.templates.get(template.parentId) || undefined;
    }

    // Merge contexts
    const fullContext = this.buildContext(template, context);

    // Render subject
    const subject = this.renderString(template.subject, fullContext, options);

    // Render body
    const body = this.renderString(template.body, fullContext, options);

    // Render HTML body if present
    let bodyHtml: string | undefined;
    if (template.bodyHtml) {
      bodyHtml = this.renderString(template.bodyHtml, fullContext, { ...options, escapeHtml: false });
    } else if (baseTemplate?.bodyHtml) {
      // Inherit HTML template
      bodyHtml = this.renderString(baseTemplate.bodyHtml, fullContext, { ...options, escapeHtml: false });
    }

    // Update usage count
    template.usageCount++;
    this.saveTemplates();

    return { subject, body, bodyHtml };
  }

  /**
   * Preview a template
   */
  preview(template: EmailTemplate): { subject: string; body: string; bodyHtml?: string } {
    // Create sample context from variables
    const context: TemplateContext = {};

    for (const variable of template.variables) {
      context[variable.name] = this.getSampleValue(variable);
    }

    return this.render(template, context);
  }

  /**
   * Validate a template
   */
  validate(template: Pick<EmailTemplate, 'subject' | 'body' | 'bodyHtml' | 'variables'>): { valid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check for undefined variables
    const usedVariables = new Set<string>();
    const variablePattern = /\{\{([^}]+)\}\}/g;

    const checkString = (str: string) => {
      let match;
      while ((match = variablePattern.exec(str)) !== null) {
        const varExpr = match[1].trim();
        // Handle conditionals and loops
        if (varExpr.startsWith('#if ') || varExpr.startsWith('#each ') || varExpr.startsWith('/')) {
          continue;
        }
        // Extract variable name (before any filter pipes)
        const varName = varExpr.split('|')[0].trim().split('.')[0];
        usedVariables.add(varName);
      }
    };

    checkString(template.subject);
    checkString(template.body);
    if (template.bodyHtml) {
      checkString(template.bodyHtml);
    }

    const definedVariables = new Set(template.variables.map((v) => v.name));

    for (const used of usedVariables) {
      if (!definedVariables.has(used) && !['else', 'this'].includes(used)) {
        errors.push(`Undefined variable: ${used}`);
      }
    }

    // Check for unclosed blocks
    const openBlocks: string[] = [];
    const blockPattern = /\{\{(#(if|each)|\/)\s*(\w*)/g;

    const checkBlocks = (str: string) => {
      let match;
      while ((match = blockPattern.exec(str)) !== null) {
        if (match[1].startsWith('#')) {
          openBlocks.push(match[2]);
        } else if (match[1] === '/') {
          const expected = openBlocks.pop();
          if (expected !== match[3] && match[3] !== '') {
            errors.push(`Mismatched block: expected /${expected}, got /${match[3]}`);
          }
        }
      }
    };

    checkBlocks(template.subject);
    checkBlocks(template.body);
    if (template.bodyHtml) {
      checkBlocks(template.bodyHtml);
    }

    if (openBlocks.length > 0) {
      errors.push(`Unclosed blocks: ${openBlocks.join(', ')}`);
    }

    return { valid: errors.length === 0, errors };
  }

  /**
   * Get template categories
   */
  getCategories(): string[] {
    const categories = new Set<string>();
    for (const template of this.templates.values()) {
      categories.add(template.category);
    }
    return Array.from(categories);
  }

  // Private methods

  private renderString(template: string, context: TemplateContext, options: RenderOptions): string {
    let result = template;

    // Handle conditionals: {{#if condition}}...{{else}}...{{/if}}
    result = this.processConditionals(result, context);

    // Handle loops: {{#each items}}...{{/each}}
    result = this.processLoops(result, context);

    // Handle variable interpolation: {{variable|filter:arg}}
    result = this.processVariables(result, context, options);

    return result;
  }

  private processConditionals(template: string, context: TemplateContext): string {
    const pattern = /\{\{#if\s+(.+?)\}\}([\s\S]*?)(?:\{\{else\}\}([\s\S]*?))?\{\{\/if\}\}/g;

    return template.replace(pattern, (_, condition, ifBlock, elseBlock = '') => {
      const value = this.resolveValue(condition.trim(), context);
      const isTruthy = Array.isArray(value) ? value.length > 0 : Boolean(value);
      return isTruthy ? ifBlock : elseBlock;
    });
  }

  private processLoops(template: string, context: TemplateContext): string {
    const pattern = /\{\{#each\s+(\w+)\}\}([\s\S]*?)\{\{\/each\}\}/g;

    return template.replace(pattern, (_, arrayName, loopContent) => {
      const array = context[arrayName];
      if (!Array.isArray(array)) return '';

      return array
        .map((item, index) => {
          const loopContext = {
            ...context,
            this: item,
            '@index': index,
            '@first': index === 0,
            '@last': index === array.length - 1,
          };
          return this.processVariables(loopContent, loopContext, {});
        })
        .join('');
    });
  }

  private processVariables(template: string, context: TemplateContext, options: RenderOptions): string {
    const pattern = /\{\{([^}]+)\}\}/g;

    return template.replace(pattern, (_, expr) => {
      const trimmed = expr.trim();

      // Skip block markers
      if (trimmed.startsWith('#') || trimmed.startsWith('/') || trimmed === 'else') {
        return `{{${expr}}}`;
      }

      // Parse expression: variable|filter:arg|filter2
      const parts = trimmed.split('|').map((p) => p.trim());
      const varPath = parts[0];
      const filters = parts.slice(1);

      // Resolve value
      let value = this.resolveValue(varPath, context);

      // Apply filters
      for (const filterExpr of filters) {
        const [filterName, ...args] = filterExpr.split(':').map((p) => p.trim());
        const filter = this.customFilters.get(filterName) || FILTERS[filterName];

        if (filter) {
          value = filter(value, ...args);
        }
      }

      // Escape HTML if needed
      if (options.escapeHtml !== false && typeof value === 'string') {
        value = this.escapeHtml(value);
      }

      return value ?? '';
    });
  }

  private resolveValue(path: string, context: TemplateContext): any {
    const parts = path.split('.');
    let value: any = context;

    for (const part of parts) {
      if (value == null) return undefined;
      value = value[part];
    }

    return value;
  }

  private escapeHtml(str: string): string {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#039;');
  }

  private buildContext(template: EmailTemplate, userContext: TemplateContext): TemplateContext {
    const context: TemplateContext = {};

    // Add defaults from variable definitions
    for (const variable of template.variables) {
      if (variable.default !== undefined) {
        context[variable.name] = variable.default;
      }
    }

    // Override with user context
    Object.assign(context, userContext);

    // Add built-in variables
    context._now = new Date();
    context._year = new Date().getFullYear();

    return context;
  }

  private getSampleValue(variable: TemplateVariable): any {
    if (variable.default !== undefined) return variable.default;

    switch (variable.type) {
      case 'text':
        return `[${variable.label}]`;
      case 'number':
        return 42;
      case 'date':
        return new Date().toISOString();
      case 'email':
        return 'example@email.com';
      case 'select':
        return variable.options?.[0]?.value || '';
      case 'rich-text':
        return `<p>[${variable.label}]</p>`;
      default:
        return '';
    }
  }

  private loadTemplates(): void {
    const saved = localStorage.getItem('mycelix-email-templates');
    if (saved) {
      const templates = JSON.parse(saved) as EmailTemplate[];
      for (const template of templates) {
        this.templates.set(template.id, template);
      }
    }

    // Add default templates if empty
    if (this.templates.size === 0) {
      this.addDefaultTemplates();
    }
  }

  private saveTemplates(): void {
    const templates = Array.from(this.templates.values());
    localStorage.setItem('mycelix-email-templates', JSON.stringify(templates));
  }

  private addDefaultTemplates(): void {
    this.addTemplate({
      name: 'Welcome Email',
      description: 'A friendly welcome email for new contacts',
      category: 'Onboarding',
      subject: 'Welcome, {{name}}!',
      body: `Hi {{name}},

Welcome to our community! We're excited to have you with us.

{{#if company}}
We noticed you're from {{company}}. We have some great resources specifically for teams like yours.
{{/if}}

If you have any questions, feel free to reach out.

Best regards,
{{sender_name}}`,
      variables: [
        { name: 'name', type: 'text', label: 'Recipient Name', required: true },
        { name: 'company', type: 'text', label: 'Company Name' },
        { name: 'sender_name', type: 'text', label: 'Your Name', required: true },
      ],
      tags: ['welcome', 'onboarding'],
    });

    this.addTemplate({
      name: 'Meeting Request',
      description: 'Request a meeting with someone',
      category: 'Business',
      subject: 'Meeting Request: {{topic}}',
      body: `Hi {{name}},

I'd like to schedule a meeting to discuss {{topic}}.

Proposed time: {{date|date:long}} at {{time}}
Duration: {{duration}} minutes

{{#if notes}}
Additional notes:
{{notes}}
{{/if}}

Please let me know if this works for you or suggest an alternative time.

Best,
{{sender_name}}`,
      variables: [
        { name: 'name', type: 'text', label: 'Recipient Name', required: true },
        { name: 'topic', type: 'text', label: 'Meeting Topic', required: true },
        { name: 'date', type: 'date', label: 'Proposed Date', required: true },
        { name: 'time', type: 'text', label: 'Time', default: '10:00 AM' },
        { name: 'duration', type: 'number', label: 'Duration (minutes)', default: 30 },
        { name: 'notes', type: 'rich-text', label: 'Additional Notes' },
        { name: 'sender_name', type: 'text', label: 'Your Name', required: true },
      ],
      tags: ['meeting', 'scheduling'],
    });

    this.addTemplate({
      name: 'Follow-up',
      description: 'Follow up after a meeting or conversation',
      category: 'Business',
      subject: 'Following up: {{topic}}',
      body: `Hi {{name}},

Thank you for taking the time to {{context}} on {{date|date:short}}.

{{#if action_items}}
Here are the action items we discussed:
{{#each action_items}}
- {{this}}
{{/each}}
{{/if}}

{{#if next_steps}}
Next steps: {{next_steps}}
{{/if}}

Looking forward to continuing our conversation.

Best,
{{sender_name}}`,
      variables: [
        { name: 'name', type: 'text', label: 'Recipient Name', required: true },
        { name: 'topic', type: 'text', label: 'Topic', required: true },
        { name: 'context', type: 'text', label: 'Context', default: 'meet with me' },
        { name: 'date', type: 'date', label: 'Meeting Date', required: true },
        { name: 'action_items', type: 'text', label: 'Action Items (comma-separated)' },
        { name: 'next_steps', type: 'text', label: 'Next Steps' },
        { name: 'sender_name', type: 'text', label: 'Your Name', required: true },
      ],
      tags: ['follow-up', 'meeting'],
    });
  }
}

export const templateEngine = new TemplateEngine();
export default TemplateEngine;
