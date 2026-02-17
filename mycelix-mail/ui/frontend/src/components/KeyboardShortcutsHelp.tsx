interface KeyboardShortcutsHelpProps {
  onClose: () => void;
}

interface ShortcutGroup {
  title: string;
  shortcuts: Array<{ key: string; description: string }>;
}

const shortcutGroups: ShortcutGroup[] = [
  {
    title: 'Composition',
    shortcuts: [
      { key: 'c', description: 'Compose new email' },
      { key: 'r', description: 'Reply' },
      { key: 'a', description: 'Reply all' },
      { key: 'f', description: 'Forward' },
    ],
  },
  {
    title: 'Navigation',
    shortcuts: [
      { key: 'j', description: 'Next email' },
      { key: 'k', description: 'Previous email' },
      { key: '/', description: 'Focus search' },
      { key: 'Esc', description: 'Close modal / Clear selection' },
    ],
  },
  {
    title: 'Actions',
    shortcuts: [
      { key: 's', description: 'Star / Unstar' },
      { key: 'u', description: 'Mark read / unread' },
      { key: 'Delete', description: 'Delete email' },
      { key: 'z', description: 'Snooze selected email' },
      { key: 'l', description: 'Add/remove labels' },
    ],
  },
  {
    title: 'Productivity',
    shortcuts: [
      { key: 'Ctrl/Cmd + K', description: 'Open template picker' },
      { key: 'Ctrl/Cmd + ,', description: 'Open settings' },
    ],
  },
  {
    title: 'Bulk Operations',
    shortcuts: [
      { key: 'Ctrl/Cmd + A', description: 'Select all emails' },
      { key: 'Shift + L', description: 'Label selected emails' },
      { key: 'Shift + R', description: 'Mark selected as read' },
      { key: 'Shift + U', description: 'Mark selected as unread' },
      { key: 'Shift + S', description: 'Star selected' },
      { key: 'Shift + D', description: 'Deselect all' },
      { key: 'Shift + Delete', description: 'Delete selected' },
    ],
  },
  {
    title: 'Help',
    shortcuts: [{ key: '?', description: 'Show this help' }],
  },
];

export default function KeyboardShortcutsHelp({ onClose }: KeyboardShortcutsHelpProps) {
  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-hidden"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">Keyboard Shortcuts</h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">Navigate faster with these shortcuts</p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 text-2xl"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-6 max-h-[70vh] overflow-y-auto">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {shortcutGroups.map((group) => (
              <div key={group.title}>
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-3">{group.title}</h3>
                <div className="space-y-2">
                  {group.shortcuts.map((shortcut) => (
                    <div key={shortcut.key} className="flex items-center justify-between">
                      <span className="text-sm text-gray-600 dark:text-gray-400">{shortcut.description}</span>
                      <kbd className="px-3 py-1 text-xs font-semibold text-gray-800 dark:text-gray-200 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded">
                        {shortcut.key}
                      </kbd>
                    </div>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700">
          <p className="text-xs text-gray-500 dark:text-gray-400 text-center">
            Press <kbd className="px-2 py-0.5 text-xs font-semibold text-gray-800 dark:text-gray-200 bg-gray-100 dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded">Esc</kbd> or click outside to close
          </p>
        </div>
      </div>
    </div>
  );
}
