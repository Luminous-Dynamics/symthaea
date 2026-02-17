import { useState } from 'react';

export type SortBy = 'date' | 'sender' | 'subject';
export type SortOrder = 'asc' | 'desc';
export type FilterType = 'all' | 'unread' | 'starred' | 'attachments';

interface EmailSortFilterProps {
  sortBy: SortBy;
  sortOrder: SortOrder;
  filterType: FilterType;
  onSortChange: (sortBy: SortBy, sortOrder: SortOrder) => void;
  onFilterChange: (filterType: FilterType) => void;
}

export default function EmailSortFilter({
  sortBy,
  sortOrder,
  filterType,
  onSortChange,
  onFilterChange,
}: EmailSortFilterProps) {
  const [showSortMenu, setShowSortMenu] = useState(false);

  const sortOptions: { value: SortBy; label: string }[] = [
    { value: 'date', label: 'Date' },
    { value: 'sender', label: 'Sender' },
    { value: 'subject', label: 'Subject' },
  ];

  const filterOptions: { value: FilterType; label: string; icon: string }[] = [
    { value: 'all', label: 'All', icon: '📧' },
    { value: 'unread', label: 'Unread', icon: '⚫' },
    { value: 'starred', label: 'Starred', icon: '⭐' },
    { value: 'attachments', label: 'Has Attachments', icon: '📎' },
  ];

  const toggleSortOrder = () => {
    onSortChange(sortBy, sortOrder === 'asc' ? 'desc' : 'asc');
  };

  return (
    <div className="flex items-center justify-between px-3 py-2 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-800/50">
      {/* Filter Buttons */}
      <div className="flex items-center space-x-1">
        {filterOptions.map((option) => (
          <button
            key={option.value}
            onClick={() => onFilterChange(option.value)}
            className={`px-2 py-1 text-xs rounded transition-colors ${
              filterType === option.value
                ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 font-medium'
                : 'text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700'
            }`}
            title={option.label}
          >
            <span className="mr-1">{option.icon}</span>
            <span className="hidden sm:inline">{option.label}</span>
          </button>
        ))}
      </div>

      {/* Sort Dropdown */}
      <div className="relative">
        <button
          onClick={() => setShowSortMenu(!showSortMenu)}
          className="flex items-center space-x-1 px-2 py-1 text-xs text-gray-600 dark:text-gray-400 hover:bg-gray-200 dark:hover:bg-gray-700 rounded transition-colors"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M3 4h13M3 8h9m-9 4h6m4 0l4-4m0 0l4 4m-4-4v12"
            />
          </svg>
          <span>Sort: {sortOptions.find((o) => o.value === sortBy)?.label}</span>
          <svg
            className={`w-3 h-3 transition-transform ${sortOrder === 'asc' ? 'rotate-180' : ''}`}
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>

        {showSortMenu && (
          <>
            <div
              className="fixed inset-0 z-10"
              onClick={() => setShowSortMenu(false)}
            />
            <div className="absolute right-0 mt-1 w-48 bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 z-20">
              <div className="py-1">
                {sortOptions.map((option) => (
                  <button
                    key={option.value}
                    onClick={() => {
                      onSortChange(option.value, sortOrder);
                      setShowSortMenu(false);
                    }}
                    className={`w-full px-4 py-2 text-left text-sm hover:bg-gray-100 dark:hover:bg-gray-700 ${
                      sortBy === option.value
                        ? 'text-primary-600 dark:text-primary-400 font-medium'
                        : 'text-gray-700 dark:text-gray-300'
                    }`}
                  >
                    {option.label}
                  </button>
                ))}
                <div className="border-t border-gray-200 dark:border-gray-700 mt-1 pt-1">
                  <button
                    onClick={() => {
                      toggleSortOrder();
                      setShowSortMenu(false);
                    }}
                    className="w-full px-4 py-2 text-left text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700"
                  >
                    {sortOrder === 'asc' ? '↑ Ascending' : '↓ Descending'}
                  </button>
                </div>
              </div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
