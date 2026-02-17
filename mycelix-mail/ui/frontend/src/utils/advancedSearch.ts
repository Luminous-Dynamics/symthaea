import type { Email, Label } from '@/types';

export interface SearchOperator {
  type: 'from' | 'to' | 'subject' | 'has' | 'is' | 'before' | 'after' | 'label' | 'text';
  value: string;
  negate?: boolean;
}

export interface ParsedQuery {
  operators: SearchOperator[];
  freeText: string[];
}

/**
 * Parse search query string into structured operators
 * Supports Gmail-style operators:
 * - from:sender@example.com
 * - to:recipient@example.com
 * - subject:keyword
 * - has:attachment, has:link
 * - is:unread, is:read, is:starred, is:important
 * - before:2024-01-01, after:2024-01-01
 * - label:labelname
 * - -operator:value (negation with -)
 */
export const parseSearchQuery = (query: string): ParsedQuery => {
  const operators: SearchOperator[] = [];
  const freeText: string[] = [];

  // Regex to match operators: operator:value or -operator:value
  const operatorRegex = /(-)?(\w+):("([^"]+)"|(\S+))/g;

  let lastIndex = 0;
  let match;

  while ((match = operatorRegex.exec(query)) !== null) {
    // Extract free text before this operator
    const textBefore = query.slice(lastIndex, match.index).trim();
    if (textBefore) {
      freeText.push(...textBefore.split(/\s+/).filter(Boolean));
    }

    const negate = !!match[1];
    const operatorType = match[2].toLowerCase();
    const value = match[4] || match[5]; // Quoted or unquoted value

    // Validate operator type
    const validOperators = ['from', 'to', 'subject', 'has', 'is', 'before', 'after', 'label'];
    if (validOperators.includes(operatorType)) {
      operators.push({
        type: operatorType as SearchOperator['type'],
        value,
        negate,
      });
    } else {
      // Invalid operator, treat as free text
      freeText.push(match[0]);
    }

    lastIndex = operatorRegex.lastIndex;
  }

  // Extract remaining free text
  const textAfter = query.slice(lastIndex).trim();
  if (textAfter) {
    freeText.push(...textAfter.split(/\s+/).filter(Boolean));
  }

  return { operators, freeText };
};

/**
 * Apply parsed search query to filter emails
 */
export const filterEmailsBySearch = (
  emails: Email[],
  query: string,
  getLabelsForEmail?: (emailId: string) => Label[]
): Email[] => {
  if (!query.trim()) return emails;

  const parsed = parseSearchQuery(query);

  return emails.filter((email) => {
    // Check all operators
    for (const operator of parsed.operators) {
      const matches = matchOperator(email, operator, getLabelsForEmail);
      if (!matches) return false;
    }

    // Check free text (search in subject, from, body)
    if (parsed.freeText.length > 0) {
      const searchableText = [
        email.subject,
        email.from,
        email.body || '',
        ...(email.to || []),
        ...(email.cc || []),
      ]
        .join(' ')
        .toLowerCase();

      for (const text of parsed.freeText) {
        if (!searchableText.includes(text.toLowerCase())) {
          return false;
        }
      }
    }

    return true;
  });
};

/**
 * Check if email matches a single operator
 */
const matchOperator = (
  email: Email,
  operator: SearchOperator,
  getLabelsForEmail?: (emailId: string) => Label[]
): boolean => {
  let matches = false;

  switch (operator.type) {
    case 'from':
      matches = email.from.toLowerCase().includes(operator.value.toLowerCase());
      break;

    case 'to':
      matches = (email.to || []).some((recipient) =>
        recipient.toLowerCase().includes(operator.value.toLowerCase())
      );
      break;

    case 'subject':
      matches = email.subject.toLowerCase().includes(operator.value.toLowerCase());
      break;

    case 'has':
      if (operator.value.toLowerCase() === 'attachment') {
        matches = email.hasAttachments || false;
      } else if (operator.value.toLowerCase() === 'link') {
        matches = (email.body || '').includes('http');
      }
      break;

    case 'is':
      const value = operator.value.toLowerCase();
      if (value === 'unread') {
        matches = !email.isRead;
      } else if (value === 'read') {
        matches = email.isRead;
      } else if (value === 'starred') {
        matches = email.isStarred;
      } else if (value === 'important') {
        // Check if email has Important label
        const labels = getLabelsForEmail?.(email.id) || [];
        matches = labels.some((l) => l.name === 'Important');
      }
      break;

    case 'before':
      try {
        const beforeDate = new Date(operator.value);
        const emailDate = new Date(email.date);
        matches = emailDate < beforeDate;
      } catch {
        matches = false;
      }
      break;

    case 'after':
      try {
        const afterDate = new Date(operator.value);
        const emailDate = new Date(email.date);
        matches = emailDate > afterDate;
      } catch {
        matches = false;
      }
      break;

    case 'label':
      const labels = getLabelsForEmail?.(email.id) || [];
      matches = labels.some((l) => l.name.toLowerCase().includes(operator.value.toLowerCase()));
      break;
  }

  // Apply negation if present
  return operator.negate ? !matches : matches;
};

/**
 * Format search query from structured operators
 * (for UI -> query string conversion)
 */
export const formatSearchQuery = (operators: SearchOperator[], freeText?: string[]): string => {
  const parts: string[] = [];

  // Add operators
  for (const op of operators) {
    const prefix = op.negate ? '-' : '';
    const value = op.value.includes(' ') ? `"${op.value}"` : op.value;
    parts.push(`${prefix}${op.type}:${value}`);
  }

  // Add free text
  if (freeText && freeText.length > 0) {
    parts.push(...freeText);
  }

  return parts.join(' ');
};

/**
 * Get search suggestions based on current query
 */
export const getSearchSuggestions = (
  query: string,
  emails: Email[],
  labels: Label[]
): string[] => {
  const suggestions: string[] = [];
  const lowerQuery = query.toLowerCase();

  // Operator suggestions
  const operators = [
    { op: 'from:', desc: 'Emails from specific sender' },
    { op: 'to:', desc: 'Emails sent to specific recipient' },
    { op: 'subject:', desc: 'Emails with subject containing...' },
    { op: 'has:attachment', desc: 'Emails with attachments' },
    { op: 'is:unread', desc: 'Unread emails' },
    { op: 'is:starred', desc: 'Starred emails' },
    { op: 'is:important', desc: 'Important emails' },
    { op: 'label:', desc: 'Emails with specific label' },
  ];

  for (const { op, desc } of operators) {
    if (op.startsWith(lowerQuery) && !query.includes(':')) {
      suggestions.push(op);
    }
  }

  // Label suggestions
  if (query.startsWith('label:')) {
    const labelQuery = query.slice(6).toLowerCase();
    for (const label of labels) {
      if (label.name.toLowerCase().includes(labelQuery)) {
        suggestions.push(`label:${label.name}`);
      }
    }
  }

  return suggestions;
};
