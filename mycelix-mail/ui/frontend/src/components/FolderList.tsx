// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
import type { Folder } from '@/types';

interface FolderListProps {
  folders: Folder[];
  selectedFolderId: string | null;
  onSelectFolder: (folderId: string) => void;
}

export default function FolderList({ folders, selectedFolderId, onSelectFolder }: FolderListProps) {
  const folderIcons: Record<string, string> = {
    INBOX: '📥',
    SENT: '📤',
    DRAFTS: '✏️',
    TRASH: '🗑️',
    SPAM: '🚫',
    ARCHIVE: '📦',
    CUSTOM: '📁',
  };

  const getFolderIcon = (folder: Folder) => {
    // Special cases for virtual and smart folders
    const specialIcons: Record<string, string> = {
      '__snoozed__': '⏰',
      '__all_mail__': '📧',
      '__starred__': '⭐',
      '__important__': '❗',
      '__unread__': '🔵',
      '__attachments__': '📎',
    };

    if (specialIcons[folder.id]) {
      return specialIcons[folder.id];
    }

    return folderIcons[folder.type] || folderIcons.CUSTOM;
  };

  return (
    <nav role="navigation" aria-label="Folder navigation">
      <div className="space-y-1">
        {folders.map((folder) => (
          <button
            key={folder.id}
            onClick={() => onSelectFolder(folder.id)}
            className={`w-full flex items-center justify-between px-3 py-2 text-sm rounded-md transition-colors ${
              selectedFolderId === folder.id
                ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-900 dark:text-primary-100'
                : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
            }`}
            aria-label={`${folder.name} folder${folder.unreadCount > 0 ? `, ${folder.unreadCount} unread` : ''}`}
            aria-current={selectedFolderId === folder.id ? 'page' : undefined}
          >
            <span className="flex items-center">
              <span className="mr-2">{getFolderIcon(folder)}</span>
              {folder.name}
            </span>
            {folder.unreadCount > 0 && (
              <span className="bg-primary-600 dark:bg-primary-500 text-white text-xs font-medium px-2 py-0.5 rounded-full">
                {folder.unreadCount > 99 ? '99+' : folder.unreadCount}
              </span>
            )}
          </button>
        ))}
      </div>
    </nav>
  );
}
