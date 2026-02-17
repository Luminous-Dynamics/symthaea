import { format, formatDistanceToNow, isToday, isYesterday, isThisWeek, isThisYear } from 'date-fns';

/**
 * Format email date for display in email list
 * Shows time for today, "Yesterday" for yesterday, day name for this week,
 * and full date for older emails
 *
 * @param date - The date to format
 * @returns Formatted date string
 *
 * @example
 * formatEmailDate(new Date()) // "2:30 PM"
 * formatEmailDate(yesterdayDate) // "Yesterday"
 * formatEmailDate(lastWeekDate) // "Monday"
 * formatEmailDate(lastMonthDate) // "Jan 15"
 */
export function formatEmailDate(date: Date | string): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;

  if (isToday(dateObj)) {
    return format(dateObj, 'h:mm a');
  } else if (isYesterday(dateObj)) {
    return 'Yesterday';
  } else if (isThisWeek(dateObj, { weekStartsOn: 0 })) {
    return format(dateObj, 'EEEE'); // Day name
  } else if (isThisYear(dateObj)) {
    return format(dateObj, 'MMM d');
  } else {
    return format(dateObj, 'MMM d, yyyy');
  }
}

/**
 * Format email date for detailed view
 * Shows full date and time with weekday
 *
 * @param date - The date to format
 * @returns Formatted date string
 *
 * @example
 * formatEmailDateDetailed(date) // "Monday, January 15, 2024 at 2:30 PM"
 */
export function formatEmailDateDetailed(date: Date | string): string {
  const dateObj = typeof date === 'string' ? new Date(date) : date;
  return format(dateObj, 'EEEE, MMMM d, yyyy \'at\' h:mm a');
}

/**
 * Format bytes to human-readable file size
 *
 * @param bytes - File size in bytes
 * @param decimals - Number of decimal places (default: 2)
 * @returns Formatted size string
 *
 * @example
 * formatBytes(1024) // "1 KB"
 * formatBytes(1536, 1) // "1.5 KB"
 * formatBytes(1048576) // "1 MB"
 */
export function formatBytes(bytes: number, decimals: number = 2): string {
  if (bytes === 0) return '0 Bytes';

  const k = 1024;
  const dm = decimals < 0 ? 0 : decimals;
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB'];

  const i = Math.floor(Math.log(bytes) / Math.log(k));

  return parseFloat((bytes / Math.pow(k, i)).toFixed(dm)) + ' ' + sizes[i];
}

/**
 * Format email recipient list
 * Shows first N recipients and "+X more" for remaining
 *
 * @param recipients - Array of email recipients
 * @param maxDisplay - Maximum number of recipients to display (default: 2)
 * @returns Formatted recipient string
 *
 * @example
 * formatRecipients([{name: 'John'}, {name: 'Jane'}, {name: 'Bob'}], 2)
 * // "John, Jane +1 more"
 */
export function formatRecipients(
  recipients: Array<{ name?: string; address: string }>,
  maxDisplay: number = 2
): string {
  if (recipients.length === 0) return '';
  if (recipients.length <= maxDisplay) {
    return recipients.map((r) => r.name || r.address).join(', ');
  }

  const displayed = recipients
    .slice(0, maxDisplay)
    .map((r) => r.name || r.address)
    .join(', ');
  const remaining = recipients.length - maxDisplay;

  return `${displayed} +${remaining} more`;
}

/**
 * Truncate text to specified length with ellipsis
 *
 * @param text - Text to truncate
 * @param maxLength - Maximum length (default: 100)
 * @returns Truncated text with ellipsis if needed
 *
 * @example
 * truncateText("This is a very long email subject that needs truncating", 30)
 * // "This is a very long email s..."
 */
export function truncateText(text: string, maxLength: number = 100): string {
  if (text.length <= maxLength) return text;
  return text.substring(0, maxLength) + '...';
}
