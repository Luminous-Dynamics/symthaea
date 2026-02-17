import { render, screen, fireEvent } from '@testing-library/react';
import { describe, it, expect, vi } from 'vitest';
import FolderList from '../FolderList';
import type { Folder } from '@/types';

const mockFolders: Folder[] = [
  {
    id: '1',
    name: 'Inbox',
    path: 'INBOX',
    type: 'INBOX',
    unreadCount: 5,
    totalCount: 20,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '2',
    name: 'Sent',
    path: 'INBOX/Sent',
    type: 'SENT',
    unreadCount: 0,
    totalCount: 10,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
  {
    id: '3',
    name: 'Drafts',
    path: 'INBOX/Drafts',
    type: 'DRAFTS',
    unreadCount: 2,
    totalCount: 3,
    createdAt: '2024-01-01T00:00:00Z',
    updatedAt: '2024-01-01T00:00:00Z',
  },
];

describe('FolderList', () => {
  it('should render all folders', () => {
    render(
      <FolderList
        folders={mockFolders}
        selectedFolderId={null}
        onSelectFolder={vi.fn()}
      />
    );

    expect(screen.getByText('Inbox')).toBeInTheDocument();
    expect(screen.getByText('Sent')).toBeInTheDocument();
    expect(screen.getByText('Drafts')).toBeInTheDocument();
  });

  it('should display unread count badges', () => {
    render(
      <FolderList
        folders={mockFolders}
        selectedFolderId={null}
        onSelectFolder={vi.fn()}
      />
    );

    expect(screen.getByText('5')).toBeInTheDocument(); // Inbox unread
    expect(screen.getByText('2')).toBeInTheDocument(); // Drafts unread
  });

  it('should not display badge for folders with no unread emails', () => {
    render(
      <FolderList
        folders={mockFolders}
        selectedFolderId={null}
        onSelectFolder={vi.fn()}
      />
    );

    const sentButton = screen.getByText('Sent').closest('button');
    expect(sentButton).toBeInTheDocument();
    expect(sentButton).not.toHaveTextContent('0');
  });

  it('should call onSelectFolder when a folder is clicked', () => {
    const handleSelectFolder = vi.fn();
    render(
      <FolderList
        folders={mockFolders}
        selectedFolderId={null}
        onSelectFolder={handleSelectFolder}
      />
    );

    fireEvent.click(screen.getByText('Inbox'));
    expect(handleSelectFolder).toHaveBeenCalledWith('1');
  });

  it('should highlight selected folder', () => {
    render(
      <FolderList
        folders={mockFolders}
        selectedFolderId="1"
        onSelectFolder={vi.fn()}
      />
    );

    const inboxButton = screen.getByText('Inbox').closest('button');
    expect(inboxButton).toHaveClass('bg-primary-100', 'text-primary-900');
  });

  it('should render empty state when no folders', () => {
    const { container } = render(
      <FolderList
        folders={[]}
        selectedFolderId={null}
        onSelectFolder={vi.fn()}
      />
    );

    expect(container.querySelector('.space-y-1')).toBeEmptyDOMElement();
  });
});
