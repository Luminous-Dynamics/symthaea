import SnoozeMenu from './SnoozeMenu';
import type { Email } from '@/types';

interface EmailViewToolbarProps {
  email: Email;
  onBack: () => void;
  onDelete: () => void;
  onStar: () => void;
  onMarkRead: () => void;
  onReply: () => void;
  onReplyAll: () => void;
  onForward: () => void;
  onPrint: () => void;
  currentFolderId?: string;
  onDeleteAndNext?: () => void;
  onNextEmail?: () => void;
  onPreviousEmail?: () => void;
}

export default function EmailViewToolbar({
  email,
  onBack,
  onDelete,
  onStar,
  onMarkRead,
  onReply,
  onReplyAll,
  onForward,
  onPrint,
  currentFolderId,
  onDeleteAndNext,
  onNextEmail,
  onPreviousEmail,
}: EmailViewToolbarProps) {
  return (
    <div className="flex items-center justify-between px-6 py-3 border-b border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800">
      <div className="flex items-center space-x-2">
        <button
          onClick={onBack}
          className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
          title="Back to list (Esc)"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>

        <div className="w-px h-6 bg-gray-300 dark:bg-gray-600 mx-1" />

        {/* Navigation buttons */}
        {onPreviousEmail && (
          <button
            onClick={onPreviousEmail}
            className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            title="Previous email ([)"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
            </svg>
          </button>
        )}

        {onNextEmail && (
          <button
            onClick={onNextEmail}
            className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            title="Next email (])"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
            </svg>
          </button>
        )}
      </div>

      <div className="flex items-center space-x-1">
        <button
          onClick={onReply}
          className="px-3 py-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
          title="Reply (R)"
        >
          <span className="mr-1.5">↩️</span>
          Reply
        </button>

        <button
          onClick={onReplyAll}
          className="px-3 py-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
          title="Reply All (A)"
        >
          <span className="mr-1.5">↩️↩️</span>
          Reply All
        </button>

        <button
          onClick={onForward}
          className="px-3 py-1.5 text-sm font-medium text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
          title="Forward (F)"
        >
          <span className="mr-1.5">➡️</span>
          Forward
        </button>

        <div className="w-px h-6 bg-gray-300 dark:bg-gray-600 mx-1" />

        <button
          onClick={onStar}
          className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
          title={email.isStarred ? 'Unstar (S)' : 'Star (S)'}
        >
          {email.isStarred ? '⭐' : '☆'}
        </button>

        <button
          onClick={onMarkRead}
          className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
          title={email.isRead ? 'Mark unread (U)' : 'Mark read (U)'}
        >
          {email.isRead ? '⚫' : '✉️'}
        </button>

        {currentFolderId && (
          <SnoozeMenu
            emailId={email.id}
            currentFolderId={currentFolderId}
            onSnooze={onBack}
          />
        )}

        <button
          onClick={onPrint}
          className="p-2 text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
          title="Print"
        >
          🖨️
        </button>

        {onDeleteAndNext ? (
          <button
            onClick={onDeleteAndNext}
            className="px-3 py-1.5 text-sm font-medium text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded transition-colors"
            title="Delete and next (#)"
          >
            <span className="mr-1.5">🗑️</span>
            Delete & Next
          </button>
        ) : (
          <button
            onClick={onDelete}
            className="p-2 text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 rounded transition-colors"
            title="Delete (Delete)"
          >
            🗑️
          </button>
        )}
      </div>
    </div>
  );
}
