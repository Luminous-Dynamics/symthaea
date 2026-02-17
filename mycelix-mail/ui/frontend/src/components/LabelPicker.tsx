import { useState, useEffect, useRef } from 'react';
import { useLabelStore, LABEL_COLORS, LABEL_ICONS } from '@/store/labelStore';
import LabelChip from './LabelChip';

interface LabelPickerProps {
  emailId: string;
  onClose: () => void;
  mode?: 'single' | 'bulk';
  emailIds?: string[]; // For bulk operations
}

export default function LabelPicker({
  emailId,
  onClose,
  mode = 'single',
  emailIds = [],
}: LabelPickerProps) {
  const {
    labels,
    getLabelsForEmail,
    toggleLabelOnEmail,
    bulkLabelEmails,
    bulkUnlabelEmails,
    addLabel,
  } = useLabelStore();

  const [searchQuery, setSearchQuery] = useState('');
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [newLabelName, setNewLabelName] = useState('');
  const [newLabelColor, setNewLabelColor] = useState(LABEL_COLORS[0].value);
  const [newLabelIcon, setNewLabelIcon] = useState(LABEL_ICONS[0]);
  const searchInputRef = useRef<HTMLInputElement>(null);

  const currentLabels = mode === 'single' ? getLabelsForEmail(emailId) : [];
  const targetIds = mode === 'bulk' ? emailIds : [emailId];

  // Filter labels by search query
  const filteredLabels = labels.filter((label) =>
    label.name.toLowerCase().includes(searchQuery.toLowerCase())
  );

  // Focus search input on mount
  useEffect(() => {
    searchInputRef.current?.focus();
  }, []);

  // Handle keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [onClose]);

  const handleToggleLabel = (labelId: string) => {
    if (mode === 'single') {
      toggleLabelOnEmail(emailId, labelId);
    } else {
      // For bulk mode, check if any email has this label
      const hasLabel = currentLabels.some((l) => l.id === labelId);
      if (hasLabel) {
        bulkUnlabelEmails(targetIds, labelId);
      } else {
        bulkLabelEmails(targetIds, labelId);
      }
    }
  };

  const handleCreateLabel = (e: React.FormEvent) => {
    e.preventDefault();

    if (!newLabelName.trim()) return;

    addLabel({
      name: newLabelName.trim(),
      color: newLabelColor,
      icon: newLabelIcon,
    });

    // Reset form
    setNewLabelName('');
    setNewLabelColor(LABEL_COLORS[0].value);
    setNewLabelIcon(LABEL_ICONS[0]);
    setShowCreateForm(false);
    setSearchQuery('');
  };

  const isLabelApplied = (labelId: string) => {
    return currentLabels.some((l) => l.id === labelId);
  };

  return (
    <div
      className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4"
      onClick={onClose}
    >
      <div
        className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full max-h-[80vh] overflow-hidden flex flex-col"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              {mode === 'bulk' ? `Label ${targetIds.length} Emails` : 'Label Email'}
            </h2>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-0.5">
              Select labels to organize this email
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 text-2xl"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {/* Search */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <input
            ref={searchInputRef}
            type="text"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            placeholder="Search labels..."
            className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 placeholder-gray-500 dark:placeholder-gray-400 focus:outline-none focus:ring-2 focus:ring-primary-500"
          />
        </div>

        {/* Labels List */}
        <div className="flex-1 overflow-y-auto p-4">
          {filteredLabels.length > 0 ? (
            <div className="space-y-2">
              {filteredLabels.map((label) => {
                const applied = isLabelApplied(label.id);

                return (
                  <button
                    key={label.id}
                    onClick={() => handleToggleLabel(label.id)}
                    className={`w-full flex items-center justify-between p-3 rounded-lg border transition-colors ${
                      applied
                        ? 'bg-primary-50 dark:bg-primary-900/20 border-primary-300 dark:border-primary-700'
                        : 'bg-gray-50 dark:bg-gray-700/50 border-gray-200 dark:border-gray-600 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }`}
                  >
                    <div className="flex items-center space-x-3">
                      <div
                        className="w-3 h-3 rounded-full flex-shrink-0"
                        style={{ backgroundColor: label.color }}
                      />
                      <span className="flex items-center space-x-2">
                        {label.icon && <span>{label.icon}</span>}
                        <span className="font-medium text-gray-900 dark:text-gray-100">
                          {label.name}
                        </span>
                      </span>
                      {label.emailCount > 0 && (
                        <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                          ({label.emailCount})
                        </span>
                      )}
                    </div>
                    {applied && (
                      <svg
                        className="w-5 h-5 text-primary-600 dark:text-primary-400"
                        fill="none"
                        stroke="currentColor"
                        viewBox="0 0 24 24"
                      >
                        <path
                          strokeLinecap="round"
                          strokeLinejoin="round"
                          strokeWidth={2}
                          d="M5 13l4 4L19 7"
                        />
                      </svg>
                    )}
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="text-center py-8 text-gray-500 dark:text-gray-400">
              No labels found
            </div>
          )}
        </div>

        {/* Create New Label */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700">
          {!showCreateForm ? (
            <button
              onClick={() => setShowCreateForm(true)}
              className="w-full flex items-center justify-center space-x-2 px-4 py-2 text-primary-600 dark:text-primary-400 hover:bg-primary-50 dark:hover:bg-primary-900/20 rounded-md transition-colors"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  strokeWidth={2}
                  d="M12 4v16m8-8H4"
                />
              </svg>
              <span className="font-medium">Create new label</span>
            </button>
          ) : (
            <form onSubmit={handleCreateLabel} className="space-y-3">
              <input
                type="text"
                value={newLabelName}
                onChange={(e) => setNewLabelName(e.target.value)}
                placeholder="Label name"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                autoFocus
              />

              {/* Color Picker */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Color
                </label>
                <div className="grid grid-cols-6 gap-2">
                  {LABEL_COLORS.map((color) => (
                    <button
                      key={color.value}
                      type="button"
                      onClick={() => setNewLabelColor(color.value)}
                      className={`w-8 h-8 rounded-full transition-transform ${
                        newLabelColor === color.value ? 'ring-2 ring-offset-2 ring-primary-500 scale-110' : ''
                      }`}
                      style={{ backgroundColor: color.value }}
                      title={color.name}
                    />
                  ))}
                </div>
              </div>

              {/* Icon Picker */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Icon
                </label>
                <div className="grid grid-cols-10 gap-1">
                  {LABEL_ICONS.map((icon) => (
                    <button
                      key={icon}
                      type="button"
                      onClick={() => setNewLabelIcon(icon)}
                      className={`w-8 h-8 rounded transition-colors ${
                        newLabelIcon === icon
                          ? 'bg-primary-100 dark:bg-primary-900/30'
                          : 'hover:bg-gray-100 dark:hover:bg-gray-700'
                      }`}
                    >
                      {icon}
                    </button>
                  ))}
                </div>
              </div>

              {/* Preview */}
              <div className="pt-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Preview
                </label>
                <LabelChip
                  label={{
                    id: 'preview',
                    name: newLabelName || 'New Label',
                    color: newLabelColor,
                    icon: newLabelIcon,
                    emailCount: 0,
                    createdAt: '',
                    updatedAt: '',
                  }}
                  size="md"
                />
              </div>

              <div className="flex space-x-2">
                <button
                  type="submit"
                  className="flex-1 btn btn-primary"
                  disabled={!newLabelName.trim()}
                >
                  Create
                </button>
                <button
                  type="button"
                  onClick={() => setShowCreateForm(false)}
                  className="flex-1 btn btn-secondary"
                >
                  Cancel
                </button>
              </div>
            </form>
          )}
        </div>
      </div>
    </div>
  );
}
