import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';
import { useSnoozeStore } from '@/store/snoozeStore';
import { formatDistanceToNow } from 'date-fns';
import { toast } from '@/store/toastStore';

interface SnoozedFolderViewProps {
  onSelectEmail?: (emailId: string) => void;
  selectedEmailId?: string | null;
}

export default function SnoozedFolderView({ onSelectEmail, selectedEmailId }: SnoozedFolderViewProps) {
  const { snoozedEmails, unsnoozeEmail } = useSnoozeStore();

  // Fetch all snoozed emails
  const { data: emails, isLoading } = useQuery({
    queryKey: ['snoozed-emails', snoozedEmails.map(s => s.emailId)],
    queryFn: async () => {
      if (snoozedEmails.length === 0) return [];

      // Fetch all snoozed emails
      const emailPromises = snoozedEmails.map(async (snoozed) => {
        try {
          const email = await api.getEmail(snoozed.emailId);
          return { ...email, snoozeInfo: snoozed };
        } catch (error) {
          console.error(`Failed to fetch email ${snoozed.emailId}:`, error);
          return null;
        }
      });

      const results = await Promise.all(emailPromises);
      return results.filter((email) => email !== null);
    },
    enabled: snoozedEmails.length > 0,
  });

  // Sort by snooze time (soonest first)
  const sortedEmails = emails?.sort((a, b) => {
    const timeA = new Date(a.snoozeInfo.snoozedUntil).getTime();
    const timeB = new Date(b.snoozeInfo.snoozedUntil).getTime();
    return timeA - timeB;
  }) || [];

  const handleUnsnooze = (emailId: string, e: React.MouseEvent) => {
    e.stopPropagation();
    unsnoozeEmail(emailId);
    toast.success('Email returned to inbox');
  };

  const getTimeUntilDue = (snoozedUntil: string) => {
    const dueTime = new Date(snoozedUntil);
    const now = new Date();

    if (dueTime <= now) {
      return { text: 'Ready now', color: 'text-green-600 dark:text-green-400', isOverdue: true };
    }

    const hoursUntil = Math.round((dueTime.getTime() - now.getTime()) / (1000 * 60 * 60));

    if (hoursUntil < 1) {
      return {
        text: `${Math.round((dueTime.getTime() - now.getTime()) / (1000 * 60))} min`,
        color: 'text-orange-600 dark:text-orange-400',
        isOverdue: false
      };
    }

    if (hoursUntil < 24) {
      return {
        text: `${hoursUntil}h`,
        color: 'text-blue-600 dark:text-blue-400',
        isOverdue: false
      };
    }

    return {
      text: formatDistanceToNow(dueTime, { addSuffix: true }),
      color: 'text-gray-600 dark:text-gray-400',
      isOverdue: false
    };
  };

  if (isLoading) {
    return (
      <div className="p-4">
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600"></div>
        </div>
      </div>
    );
  }

  if (sortedEmails.length === 0) {
    return (
      <div className="p-4">
        <div className="text-center py-12">
          <svg
            className="w-16 h-16 mx-auto text-gray-400 dark:text-gray-600 mb-4"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
            />
          </svg>
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No snoozed emails
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Emails you snooze will appear here
          </p>
        </div>
      </div>
    );
  }

  return (
    <div className="overflow-y-auto h-full">
      {/* Header */}
      <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-4 py-3 z-10">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Snoozed Emails
          </h2>
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {sortedEmails.length} email{sortedEmails.length !== 1 ? 's' : ''}
          </span>
        </div>
      </div>

      {/* Email List */}
      <div className="divide-y divide-gray-200 dark:divide-gray-700">
        {sortedEmails.map((email) => {
          const timeInfo = getTimeUntilDue(email.snoozeInfo.snoozedUntil);
          const isSelected = selectedEmailId === email.id;
          const dueDate = new Date(email.snoozeInfo.snoozedUntil);

          return (
            <div
              key={email.id}
              onClick={() => onSelectEmail?.(email.id)}
              className={`p-4 cursor-pointer transition-colors ${
                isSelected
                  ? 'bg-primary-50 dark:bg-primary-900/20 border-l-4 border-primary-600'
                  : 'hover:bg-gray-50 dark:hover:bg-gray-700/50'
              }`}
            >
              <div className="flex items-start justify-between mb-2">
                <div className="flex-1 min-w-0">
                  <div className="flex items-center space-x-2 mb-1">
                    <h3 className={`text-sm font-semibold truncate ${
                      email.isRead
                        ? 'text-gray-700 dark:text-gray-300'
                        : 'text-gray-900 dark:text-gray-100'
                    }`}>
                      {email.from.name || email.from.email}
                    </h3>
                    {email.isStarred && <span className="text-yellow-500">⭐</span>}
                  </div>
                  <p className={`text-sm truncate ${
                    email.isRead
                      ? 'text-gray-600 dark:text-gray-400'
                      : 'text-gray-800 dark:text-gray-200 font-medium'
                  }`}>
                    {email.subject}
                  </p>
                </div>

                <button
                  onClick={(e) => handleUnsnooze(email.id, e)}
                  className="ml-3 px-3 py-1 text-xs font-medium text-primary-600 dark:text-primary-400 hover:bg-primary-100 dark:hover:bg-primary-900/30 rounded-md transition-colors flex-shrink-0"
                  title="Unsnooze email"
                >
                  Unsnooze
                </button>
              </div>

              {/* Snooze Info */}
              <div className="flex items-center justify-between text-xs">
                <div className="flex items-center space-x-3">
                  <div className="flex items-center space-x-1">
                    <svg className="w-3.5 h-3.5 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
                      />
                    </svg>
                    <span className={timeInfo.color}>
                      {timeInfo.text}
                    </span>
                  </div>

                  {timeInfo.isOverdue && (
                    <span className="px-2 py-0.5 bg-green-100 dark:bg-green-900/30 text-green-700 dark:text-green-300 rounded-full text-xs font-medium">
                      Ready
                    </span>
                  )}
                </div>

                <span className="text-gray-500 dark:text-gray-400">
                  Until {dueDate.toLocaleDateString([], {
                    month: 'short',
                    day: 'numeric',
                    hour: '2-digit',
                    minute: '2-digit'
                  })}
                </span>
              </div>

              {/* Preview */}
              {email.preview && (
                <p className="mt-2 text-xs text-gray-600 dark:text-gray-400 line-clamp-2">
                  {email.preview}
                </p>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
