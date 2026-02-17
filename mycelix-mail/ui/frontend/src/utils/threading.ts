import type { Email } from '@/types';
import type { Label } from '@/store/labelStore';

export interface EmailThread {
  id: string;
  subject: string; // Original subject from newest email
  normalizedSubject: string; // Normalized for grouping
  emails: Email[];
  participants: string[]; // All email addresses involved
  lastMessageDate: string;
  unreadCount: number;
  isStarred: boolean; // True if any email in thread is starred
  hasAttachments: boolean; // True if any email has attachments
  messageCount: number;
  preview: string; // Preview from latest email
}

/**
 * Normalize email subject for threading
 * Removes Re:, Fwd:, Fw:, etc. and trims whitespace
 */
export const normalizeSubject = (subject: string): string => {
  if (!subject) return '';

  return subject
    .replace(/^(Re:|RE:|Fwd:|FWD:|Fw:|FW:)\s*/gi, '')
    .replace(/\s+/g, ' ') // Normalize whitespace
    .trim()
    .toLowerCase();
};

/**
 * Group emails into conversation threads based on subject
 * Returns threads sorted by latest message date
 */
export const groupEmailsIntoThreads = (
  emails: Email[],
  getLabelsForEmail?: (emailId: string) => Label[]
): EmailThread[] => {
  const threadMap = new Map<string, Email[]>();

  // Group emails by normalized subject
  emails.forEach((email) => {
    const normalized = normalizeSubject(email.subject);
    if (!normalized) return; // Skip emails with no subject

    if (!threadMap.has(normalized)) {
      threadMap.set(normalized, []);
    }
    threadMap.get(normalized)!.push(email);
  });

  // Convert map to thread objects
  const threads: EmailThread[] = Array.from(threadMap.entries()).map(([normalizedSubject, threadEmails]) => {
    // Sort emails in thread by date (oldest first)
    const sortedEmails = threadEmails.sort((a, b) =>
      new Date(a.date).getTime() - new Date(b.date).getTime()
    );

    // Get latest email for subject and preview
    const latestEmail = sortedEmails[sortedEmails.length - 1];

    // Collect all unique participants
    const participantSet = new Set<string>();
    sortedEmails.forEach((email) => {
      participantSet.add(email.from.address);
      email.to.forEach((to) => participantSet.add(to.address));
      email.cc?.forEach((cc) => participantSet.add(cc.address));
    });

    return {
      id: `thread_${normalizedSubject}_${latestEmail.id}`,
      subject: latestEmail.subject, // Use original subject from latest email
      normalizedSubject,
      emails: sortedEmails,
      participants: Array.from(participantSet),
      lastMessageDate: latestEmail.date,
      unreadCount: sortedEmails.filter((e) => !e.isRead).length,
      isStarred: sortedEmails.some((e) => e.isStarred),
      hasAttachments: sortedEmails.some((e) => e.attachments && e.attachments.length > 0),
      messageCount: sortedEmails.length,
      preview: latestEmail.bodyText?.substring(0, 150) || '',
    };
  });

  // Sort threads by latest message date (newest first)
  return threads.sort((a, b) =>
    new Date(b.lastMessageDate).getTime() - new Date(a.lastMessageDate).getTime()
  );
};

/**
 * Check if two emails belong to the same thread
 */
export const isSameThread = (email1: Email, email2: Email): boolean => {
  return normalizeSubject(email1.subject) === normalizeSubject(email2.subject);
};

/**
 * Get participant names for display
 * Returns formatted string like "John, Jane, +3 others"
 */
export const formatParticipants = (participants: string[], maxShow: number = 3): string => {
  if (participants.length === 0) return '';
  if (participants.length <= maxShow) {
    return participants.join(', ');
  }

  const shown = participants.slice(0, maxShow);
  const remaining = participants.length - maxShow;
  return `${shown.join(', ')}, +${remaining} other${remaining > 1 ? 's' : ''}`;
};

/**
 * Calculate thread importance score
 * Higher score = more important
 */
export const calculateThreadImportance = (thread: EmailThread): number => {
  let score = 0;

  // Unread messages are important
  score += thread.unreadCount * 10;

  // Starred threads are important
  if (thread.isStarred) score += 20;

  // Active conversations (many messages) are important
  if (thread.messageCount > 5) score += 15;
  else if (thread.messageCount > 2) score += 10;

  // Recent threads are more important
  const hoursSinceLastMessage =
    (Date.now() - new Date(thread.lastMessageDate).getTime()) / (1000 * 60 * 60);

  if (hoursSinceLastMessage < 1) score += 20;
  else if (hoursSinceLastMessage < 24) score += 10;
  else if (hoursSinceLastMessage < 72) score += 5;

  // Attachments might be important
  if (thread.hasAttachments) score += 5;

  return score;
};
