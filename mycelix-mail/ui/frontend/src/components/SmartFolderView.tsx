import { useMemo } from 'react';
import { useLabelStore } from '@/store/labelStore';
import type { Email } from '@/types';
import EmptyState from './EmptyState';

interface SmartFolderViewProps {
  folderId: string;
  emails: Email[];
  onSelectEmail: (emailId: string) => void;
  selectedEmailId: string | null;
}

type SmartFolderType = 'all-mail' | 'starred' | 'important' | 'unread' | 'attachments';

export default function SmartFolderView({
  folderId,
  emails,
  onSelectEmail,
  selectedEmailId,
}: SmartFolderViewProps) {
  const { labels, getLabelsForEmail } = useLabelStore();

  // Get Important label ID
  const importantLabel = labels.find(l => l.name === 'Important');

  // Filter emails based on smart folder type
  const filteredEmails = useMemo(() => {
    switch (folderId) {
      case '__all_mail__':
        // All emails except Trash and Spam
        return emails.filter(email =>
          email.folder?.type !== 'TRASH' && email.folder?.type !== 'SPAM'
        );

      case '__starred__':
        // Only starred emails
        return emails.filter(email => email.isStarred);

      case '__important__':
        // Emails with Important label
        if (!importantLabel) return [];
        return emails.filter(email => {
          const emailLabels = getLabelsForEmail(email.id);
          return emailLabels.some(l => l.id === importantLabel.id);
        });

      case '__unread__':
        // Only unread emails
        return emails.filter(email => !email.isRead);

      case '__attachments__':
        // Emails with attachments
        return emails.filter(email => email.attachments && email.attachments.length > 0);

      default:
        return emails;
    }
  }, [folderId, emails, importantLabel, getLabelsForEmail]);

  const getFolderInfo = () => {
    switch (folderId) {
      case '__all_mail__':
        return { icon: '📧', title: 'All Mail', emptyMessage: 'No emails found' };
      case '__starred__':
        return { icon: '⭐', title: 'Starred', emptyMessage: 'No starred emails' };
      case '__important__':
        return { icon: '❗', title: 'Important', emptyMessage: 'No important emails' };
      case '__unread__':
        return { icon: '🔵', title: 'Unread', emptyMessage: 'No unread emails' };
      case '__attachments__':
        return { icon: '📎', title: 'Has Attachments', emptyMessage: 'No emails with attachments' };
      default:
        return { icon: '📧', title: 'Emails', emptyMessage: 'No emails' };
    }
  };

  const folderInfo = getFolderInfo();

  if (filteredEmails.length === 0) {
    return (
      <div className="h-full flex items-center justify-center p-4">
        <EmptyState
          icon={folderInfo.icon}
          title={folderInfo.emptyMessage}
          description="Try checking other folders or adjusting your filters"
        />
      </div>
    );
  }

  return (
    <div className="h-full overflow-y-auto">
      {/* Header */}
      <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3 z-10">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 flex items-center space-x-2">
            <span>{folderInfo.icon}</span>
            <span>{folderInfo.title}</span>
          </h2>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {filteredEmails.length} email{filteredEmails.length !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {/* Email List */}
      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        {filteredEmails.map((email) => {
          const isSelected = selectedEmailId === email.id;

          return (
            <button
              key={email.id}
              onClick={() => onSelectEmail(email.id)}
              className={`w-full text-left px-4 py-3 hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors ${
                isSelected ? 'bg-primary-50 dark:bg-primary-900/30' : ''
              } ${!email.isRead ? 'font-semibold' : ''}`}
            >
              <div className="flex items-start justify-between mb-1">
                <span className="text-sm text-gray-900 dark:text-gray-100 truncate flex-1">
                  {email.from.name || email.from.address}
                </span>
                <div className="flex items-center space-x-2 ml-2 flex-shrink-0">
                  {email.isStarred && <span className="text-yellow-500">⭐</span>}
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {new Date(email.date).toLocaleDateString()}
                  </span>
                </div>
              </div>
              <div className="text-sm text-gray-900 dark:text-gray-100 truncate mb-1">
                {email.subject}
              </div>
              <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
                {email.bodyText?.substring(0, 100)}
              </div>
              {email.attachments && email.attachments.length > 0 && (
                <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  📎 {email.attachments.length} attachment{email.attachments.length > 1 ? 's' : ''}
                </div>
              )}
              {email.folder && (
                <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  📁 {email.folder.name}
                </div>
              )}
            </button>
          );
        })}
      </div>
    </div>
  );
}
