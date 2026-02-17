interface BulkActionsToolbarProps {
  selectedCount: number;
  isPending?: boolean;
  onMarkRead: () => void;
  onMarkUnread: () => void;
  onStar: () => void;
  onUnstar: () => void;
  onDelete: () => void;
  onDeselectAll: () => void;
}

export default function BulkActionsToolbar({
  selectedCount,
  isPending = false,
  onMarkRead,
  onMarkUnread,
  onStar,
  onUnstar,
  onDelete,
  onDeselectAll,
}: BulkActionsToolbarProps) {
  if (selectedCount === 0) return null;

  return (
    <div className="sticky top-0 z-20 bg-primary-100 dark:bg-primary-900/30 border-b border-primary-300 dark:border-primary-700 px-4 py-3">
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <button
            onClick={onDeselectAll}
            disabled={isPending}
            className="text-sm text-gray-600 dark:text-gray-300 hover:text-gray-900 dark:hover:text-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
            aria-label="Deselect all"
          >
            ✕
          </button>
          <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
            {isPending ? 'Processing...' : `${selectedCount} email${selectedCount !== 1 ? 's' : ''} selected`}
          </span>
        </div>

        <div className="flex items-center space-x-2">
          <button
            onClick={onMarkRead}
            disabled={isPending}
            className="px-3 py-1.5 text-xs font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 rounded border border-gray-300 dark:border-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Mark as read"
          >
            ✉️ Mark Read
          </button>

          <button
            onClick={onMarkUnread}
            disabled={isPending}
            className="px-3 py-1.5 text-xs font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 rounded border border-gray-300 dark:border-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Mark as unread"
          >
            ⚫ Mark Unread
          </button>

          <button
            onClick={onStar}
            disabled={isPending}
            className="px-3 py-1.5 text-xs font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 rounded border border-gray-300 dark:border-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Star all"
          >
            ⭐ Star
          </button>

          <button
            onClick={onUnstar}
            disabled={isPending}
            className="px-3 py-1.5 text-xs font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-700 rounded border border-gray-300 dark:border-gray-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Unstar all"
          >
            ☆ Unstar
          </button>

          <div className="w-px h-6 bg-gray-300 dark:bg-gray-600" />

          <button
            onClick={() => {
              if (confirm(`Delete ${selectedCount} email${selectedCount !== 1 ? 's' : ''}?`)) {
                onDelete();
              }
            }}
            disabled={isPending}
            className="px-3 py-1.5 text-xs font-medium text-red-700 dark:text-red-400 bg-white dark:bg-gray-800 hover:bg-red-50 dark:hover:bg-red-900/30 rounded border border-red-300 dark:border-red-700 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            title="Delete selected"
          >
            🗑️ Delete
          </button>
        </div>
      </div>
    </div>
  );
}
