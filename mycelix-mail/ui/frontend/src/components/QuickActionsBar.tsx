import { useState } from 'react';

interface QuickActionsBarProps {
  selectedEmailId: string | null;
  isVisible: boolean;
  onReply?: () => void;
  onReplyAll?: () => void;
  onForward?: () => void;
  onArchive?: () => void;
  onDelete?: () => void;
  onMarkRead?: () => void;
  onMarkUnread?: () => void;
  onStar?: () => void;
  onLabel?: () => void;
  onSnooze?: () => void;
  isRead?: boolean;
  isStarred?: boolean;
}

export default function QuickActionsBar({
  selectedEmailId,
  isVisible,
  onReply,
  onReplyAll,
  onForward,
  onArchive,
  onDelete,
  onMarkRead,
  onMarkUnread,
  onStar,
  onLabel,
  onSnooze,
  isRead = false,
  isStarred = false,
}: QuickActionsBarProps) {
  const [showTooltip, setShowTooltip] = useState<string | null>(null);

  if (!isVisible || !selectedEmailId) {
    return null;
  }

  const actions = [
    {
      id: 'reply',
      label: 'Reply',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6"
          />
        </svg>
      ),
      onClick: onReply,
      shortcut: 'R',
    },
    {
      id: 'reply-all',
      label: 'Reply All',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M3 10h10a8 8 0 018 8v2M3 10l6 6m-6-6l6-6M13 10l6-6"
          />
        </svg>
      ),
      onClick: onReplyAll,
      shortcut: 'Shift+R',
    },
    {
      id: 'forward',
      label: 'Forward',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M13 7l5 5m0 0l-5 5m5-5H6"
          />
        </svg>
      ),
      onClick: onForward,
      shortcut: 'F',
    },
    {
      id: 'divider-1',
      isDivider: true,
    },
    {
      id: 'star',
      label: isStarred ? 'Unstar' : 'Star',
      icon: isStarred ? (
        <svg className="w-5 h-5 fill-yellow-500" viewBox="0 0 24 24">
          <path d="M12 2l3.09 6.26L22 9.27l-5 4.87 1.18 6.88L12 17.77l-6.18 3.25L7 14.14 2 9.27l6.91-1.01L12 2z" />
        </svg>
      ) : (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M11.049 2.927c.3-.921 1.603-.921 1.902 0l1.519 4.674a1 1 0 00.95.69h4.915c.969 0 1.371 1.24.588 1.81l-3.976 2.888a1 1 0 00-.363 1.118l1.518 4.674c.3.922-.755 1.688-1.538 1.118l-3.976-2.888a1 1 0 00-1.176 0l-3.976 2.888c-.783.57-1.838-.197-1.538-1.118l1.518-4.674a1 1 0 00-.363-1.118l-3.976-2.888c-.784-.57-.38-1.81.588-1.81h4.914a1 1 0 00.951-.69l1.519-4.674z"
          />
        </svg>
      ),
      onClick: onStar,
      shortcut: 'S',
    },
    {
      id: 'mark-read',
      label: isRead ? 'Mark Unread' : 'Mark Read',
      icon: isRead ? (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M3 19v-8.93a2 2 0 01.89-1.664l7-4.666a2 2 0 012.22 0l7 4.666A2 2 0 0121 10.07V19M3 19a2 2 0 002 2h14a2 2 0 002-2M3 19l6.75-4.5M21 19l-6.75-4.5M3 10l6.75 4.5M21 10l-6.75 4.5m0 0l-1.14.76a2 2 0 01-2.22 0l-1.14-.76"
          />
        </svg>
      ) : (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M3 8l7.89 5.26a2 2 0 002.22 0L21 8M5 19h14a2 2 0 002-2V7a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z"
          />
        </svg>
      ),
      onClick: isRead ? onMarkUnread : onMarkRead,
      shortcut: 'U',
    },
    {
      id: 'label',
      label: 'Label',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
          />
        </svg>
      ),
      onClick: onLabel,
      shortcut: 'L',
    },
    {
      id: 'snooze',
      label: 'Snooze',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      ),
      onClick: onSnooze,
      shortcut: 'Z',
    },
    {
      id: 'divider-2',
      isDivider: true,
    },
    {
      id: 'archive',
      label: 'Archive',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M5 8h14M5 8a2 2 0 110-4h14a2 2 0 110 4M5 8v10a2 2 0 002 2h10a2 2 0 002-2V8m-9 4h4"
          />
        </svg>
      ),
      onClick: onArchive,
      shortcut: 'E',
    },
    {
      id: 'delete',
      label: 'Delete',
      icon: (
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
          />
        </svg>
      ),
      onClick: onDelete,
      shortcut: '#',
      className: 'text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20',
    },
  ];

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 pointer-events-none z-40" />

      {/* Floating Action Bar */}
      <div
        className="fixed bottom-6 left-1/2 -translate-x-1/2 z-50 animate-slide-up"
        role="toolbar"
        aria-label="Quick actions"
      >
        <div className="bg-white dark:bg-gray-800 rounded-full shadow-2xl border border-gray-200 dark:border-gray-700 px-4 py-3 flex items-center space-x-2">
          {actions.map((action) => {
            if ('isDivider' in action && action.isDivider) {
              return (
                <div
                  key={action.id}
                  className="w-px h-6 bg-gray-300 dark:bg-gray-600 mx-1"
                  aria-hidden="true"
                />
              );
            }

            return (
              <div key={action.id} className="relative">
                <button
                  onClick={action.onClick}
                  onMouseEnter={() => setShowTooltip(action.id)}
                  onMouseLeave={() => setShowTooltip(null)}
                  className={`p-2.5 rounded-full transition-all duration-200 hover:scale-110 ${
                    action.className ||
                    'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                  }`}
                  aria-label={action.label}
                  title={action.label}
                >
                  {action.icon}
                </button>

                {/* Tooltip */}
                {showTooltip === action.id && (
                  <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-2 px-3 py-2 bg-gray-900 dark:bg-gray-700 text-white text-xs rounded-lg shadow-lg whitespace-nowrap animate-fade-in">
                    <div className="font-medium">{action.label}</div>
                    {action.shortcut && (
                      <div className="text-gray-300 dark:text-gray-400 mt-0.5">
                        Press <kbd className="px-1 py-0.5 bg-gray-800 dark:bg-gray-600 rounded text-xs">{action.shortcut}</kbd>
                      </div>
                    )}
                    {/* Arrow */}
                    <div className="absolute top-full left-1/2 -translate-x-1/2 -mt-px">
                      <div className="border-4 border-transparent border-t-gray-900 dark:border-t-gray-700" />
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Styles for animations */}
      <style>{`
        @keyframes slide-up {
          from {
            opacity: 0;
            transform: translate(-50%, 20px);
          }
          to {
            opacity: 1;
            transform: translate(-50%, 0);
          }
        }

        @keyframes fade-in {
          from {
            opacity: 0;
          }
          to {
            opacity: 1;
          }
        }

        .animate-slide-up {
          animation: slide-up 0.3s cubic-bezier(0.16, 1, 0.3, 1);
        }

        .animate-fade-in {
          animation: fade-in 0.2s ease-out;
        }
      `}</style>
    </>
  );
}
