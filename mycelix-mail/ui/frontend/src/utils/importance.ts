import type { Email } from '@/types';
import type { Contact } from '@/store/contactStore';

/**
 * Calculate email importance score
 * Higher score = more important
 */
export const calculateEmailImportance = (
  email: Email,
  userEmail?: string,
  vipContacts?: Contact[]
): number => {
  let score = 0;

  // 1. Direct recipient (not CC) - very important
  if (userEmail) {
    const isDirectRecipient = email.to.some(
      (to) => to.address.toLowerCase() === userEmail.toLowerCase()
    );
    const isCcRecipient = email.cc?.some(
      (cc) => cc.address.toLowerCase() === userEmail.toLowerCase()
    );

    if (isDirectRecipient) {
      score += 20; // Directly addressed to user
    } else if (isCcRecipient) {
      score += 5; // Only CC'd
    }
  }

  // 2. VIP contact - important
  if (vipContacts && vipContacts.length > 0) {
    const isFromVip = vipContacts.some(
      (contact) => contact.email.toLowerCase() === email.from.address.toLowerCase()
    );
    if (isFromVip) {
      score += 30;
    }
  }

  // 3. Urgent keywords in subject or body
  const urgentKeywords = [
    'urgent',
    'asap',
    'important',
    'critical',
    'emergency',
    'deadline',
    'time-sensitive',
    'action required',
    'immediate',
    'priority',
  ];

  const subjectLower = email.subject.toLowerCase();
  const bodyLower = (email.bodyText || '').toLowerCase();

  urgentKeywords.forEach((keyword) => {
    if (subjectLower.includes(keyword)) {
      score += 15;
    } else if (bodyLower.includes(keyword)) {
      score += 10;
    }
  });

  // 4. Work domain (assuming common work domains)
  const workDomains = [
    '.com',
    '.org',
    '.net',
    '.io',
    '.co',
    // Add custom work domains as needed
  ];

  const fromDomain = email.from.address.split('@')[1]?.toLowerCase();
  if (fromDomain && userEmail) {
    const userDomain = userEmail.split('@')[1]?.toLowerCase();

    // Same domain as user (likely coworker)
    if (fromDomain === userDomain) {
      score += 15;
    }

    // Known work domain
    if (workDomains.some((domain) => fromDomain.endsWith(domain))) {
      score += 5;
    }
  }

  // 5. Few recipients (more personal)
  const totalRecipients = email.to.length + (email.cc?.length || 0);
  if (totalRecipients === 1) {
    score += 10; // Only to you
  } else if (totalRecipients <= 3) {
    score += 5; // Small group
  }

  // 6. Has attachments (might be important)
  if (email.attachments && email.attachments.length > 0) {
    score += 5;
  }

  // 7. Reply in thread (active conversation)
  if (email.subject.match(/^(Re:|RE:|Fwd:|FWD:)/i)) {
    score += 5;
  }

  // 8. Recent email (last 24 hours)
  const emailDate = new Date(email.date);
  const now = new Date();
  const hoursSinceReceived = (now.getTime() - emailDate.getTime()) / (1000 * 60 * 60);

  if (hoursSinceReceived < 1) {
    score += 10; // Very recent
  } else if (hoursSinceReceived < 24) {
    score += 5; // Today
  }

  // 9. Not unread but starred (user marked important)
  if (email.isStarred) {
    score += 25;
  }

  // 10. From a frequent contact
  // This would require contact store integration
  // Handled externally via vipContacts parameter

  return Math.min(score, 100); // Cap at 100
};

/**
 * Classify email importance level
 */
export type ImportanceLevel = 'critical' | 'high' | 'medium' | 'low';

export const getImportanceLevel = (score: number): ImportanceLevel => {
  if (score >= 70) return 'critical';
  if (score >= 50) return 'high';
  if (score >= 30) return 'medium';
  return 'low';
};

/**
 * Get importance badge properties
 */
export const getImportanceBadge = (level: ImportanceLevel) => {
  switch (level) {
    case 'critical':
      return {
        label: 'Critical',
        color: 'bg-red-500 dark:bg-red-600',
        textColor: 'text-white',
        icon: '🔥',
      };
    case 'high':
      return {
        label: 'Important',
        color: 'bg-orange-500 dark:bg-orange-600',
        textColor: 'text-white',
        icon: '❗',
      };
    case 'medium':
      return {
        label: 'Medium',
        color: 'bg-yellow-500 dark:bg-yellow-600',
        textColor: 'text-white',
        icon: '⚠️',
      };
    case 'low':
      return {
        label: 'Low',
        color: 'bg-gray-400 dark:bg-gray-600',
        textColor: 'text-white',
        icon: '📬',
      };
  }
};

/**
 * Should auto-label as important?
 */
export const shouldAutoLabelImportant = (score: number): boolean => {
  return score >= 50; // High or critical
};

/**
 * Get importance tooltip
 */
export const getImportanceTooltip = (
  email: Email,
  score: number,
  level: ImportanceLevel
): string => {
  const reasons: string[] = [];

  if (email.isStarred) {
    reasons.push('Starred by you');
  }

  if (email.subject.match(/urgent|asap|important/i)) {
    reasons.push('Contains urgent keywords');
  }

  if (email.to.length === 1) {
    reasons.push('Sent only to you');
  }

  if (email.attachments && email.attachments.length > 0) {
    reasons.push(`Has ${email.attachments.length} attachment(s)`);
  }

  const hoursSinceReceived =
    (Date.now() - new Date(email.date).getTime()) / (1000 * 60 * 60);
  if (hoursSinceReceived < 1) {
    reasons.push('Received very recently');
  }

  if (reasons.length === 0) {
    return `Importance: ${level} (Score: ${score})`;
  }

  return `${level.charAt(0).toUpperCase() + level.slice(1)} importance (${score})\n\nReasons:\n- ${reasons.join('\n- ')}`;
};

/**
 * Sort emails by importance
 */
export const sortByImportance = (
  emails: Email[],
  userEmail?: string,
  vipContacts?: Contact[]
): Email[] => {
  return [...emails].sort((a, b) => {
    const scoreA = calculateEmailImportance(a, userEmail, vipContacts);
    const scoreB = calculateEmailImportance(b, userEmail, vipContacts);
    return scoreB - scoreA; // Higher score first
  });
};

/**
 * Filter only important emails
 */
export const filterImportantEmails = (
  emails: Email[],
  minScore: number = 50,
  userEmail?: string,
  vipContacts?: Contact[]
): Email[] => {
  return emails.filter((email) => {
    const score = calculateEmailImportance(email, userEmail, vipContacts);
    return score >= minScore;
  });
};
