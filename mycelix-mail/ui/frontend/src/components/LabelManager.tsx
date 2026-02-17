import { useState } from 'react';
import { useLabelStore, LABEL_COLORS, LABEL_ICONS } from '@/store/labelStore';
import type { Label } from '@/store/labelStore';
import LabelChip from './LabelChip';

export default function LabelManager() {
  const { labels, addLabel, updateLabel, deleteLabel } = useLabelStore();

  const [isCreating, setIsCreating] = useState(false);
  const [editingId, setEditingId] = useState<string | null>(null);
  const [formData, setFormData] = useState({
    name: '',
    color: LABEL_COLORS[0].value,
    icon: LABEL_ICONS[0],
  });

  const handleCreate = (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.name.trim()) return;

    addLabel({
      name: formData.name.trim(),
      color: formData.color,
      icon: formData.icon,
    });

    // Reset form
    setFormData({
      name: '',
      color: LABEL_COLORS[0].value,
      icon: LABEL_ICONS[0],
    });
    setIsCreating(false);
  };

  const handleEdit = (label: Label) => {
    setEditingId(label.id);
    setFormData({
      name: label.name,
      color: label.color,
      icon: label.icon || LABEL_ICONS[0],
    });
  };

  const handleUpdate = (e: React.FormEvent) => {
    e.preventDefault();

    if (!editingId || !formData.name.trim()) return;

    updateLabel(editingId, {
      name: formData.name.trim(),
      color: formData.color,
      icon: formData.icon,
    });

    // Reset
    setEditingId(null);
    setFormData({
      name: '',
      color: LABEL_COLORS[0].value,
      icon: LABEL_ICONS[0],
    });
  };

  const handleDelete = (id: string) => {
    if (confirm('Are you sure you want to delete this label? It will be removed from all emails.')) {
      deleteLabel(id);
    }
  };

  const handleCancelEdit = () => {
    setEditingId(null);
    setIsCreating(false);
    setFormData({
      name: '',
      color: LABEL_COLORS[0].value,
      icon: LABEL_ICONS[0],
    });
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
            Email Labels
          </h3>
          <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
            Organize your emails with custom labels
          </p>
        </div>
        {!isCreating && !editingId && (
          <button
            onClick={() => setIsCreating(true)}
            className="btn btn-primary flex items-center space-x-2"
          >
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 4v16m8-8H4"
              />
            </svg>
            <span>New Label</span>
          </button>
        )}
      </div>

      {/* Create/Edit Form */}
      {(isCreating || editingId) && (
        <div className="bg-gray-50 dark:bg-gray-700/50 rounded-lg p-6 border border-gray-200 dark:border-gray-600">
          <h4 className="text-md font-semibold text-gray-900 dark:text-gray-100 mb-4">
            {editingId ? 'Edit Label' : 'Create New Label'}
          </h4>
          <form onSubmit={editingId ? handleUpdate : handleCreate} className="space-y-4">
            {/* Name Input */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Label Name
              </label>
              <input
                type="text"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                placeholder="Enter label name"
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                autoFocus
              />
            </div>

            {/* Color Picker */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Color
              </label>
              <div className="grid grid-cols-12 gap-2">
                {LABEL_COLORS.map((color) => (
                  <button
                    key={color.value}
                    type="button"
                    onClick={() => setFormData({ ...formData, color: color.value })}
                    className={`w-10 h-10 rounded-lg transition-all ${
                      formData.color === color.value
                        ? 'ring-2 ring-offset-2 ring-primary-500 dark:ring-offset-gray-800 scale-110'
                        : 'hover:scale-105'
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
                    onClick={() => setFormData({ ...formData, icon })}
                    className={`w-10 h-10 rounded-lg text-xl transition-colors ${
                      formData.icon === icon
                        ? 'bg-primary-100 dark:bg-primary-900/30 ring-2 ring-primary-500'
                        : 'bg-white dark:bg-gray-800 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }`}
                  >
                    {icon}
                  </button>
                ))}
              </div>
            </div>

            {/* Preview */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Preview
              </label>
              <LabelChip
                label={{
                  id: 'preview',
                  name: formData.name || 'Label Name',
                  color: formData.color,
                  icon: formData.icon,
                  emailCount: 0,
                  createdAt: '',
                  updatedAt: '',
                }}
                size="md"
              />
            </div>

            {/* Actions */}
            <div className="flex space-x-3 pt-2">
              <button type="submit" className="btn btn-primary" disabled={!formData.name.trim()}>
                {editingId ? 'Update Label' : 'Create Label'}
              </button>
              <button type="button" onClick={handleCancelEdit} className="btn btn-secondary">
                Cancel
              </button>
            </div>
          </form>
        </div>
      )}

      {/* Labels List */}
      {labels.length > 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {labels.map((label) => (
            <div
              key={label.id}
              className="bg-white dark:bg-gray-800 rounded-lg p-4 border border-gray-200 dark:border-gray-700 hover:border-gray-300 dark:hover:border-gray-600 transition-colors"
            >
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center space-x-3 mb-2">
                    <div
                      className="w-4 h-4 rounded-full flex-shrink-0"
                      style={{ backgroundColor: label.color }}
                    />
                    <div className="flex items-center space-x-2">
                      {label.icon && <span className="text-lg">{label.icon}</span>}
                      <h4 className="font-semibold text-gray-900 dark:text-gray-100">
                        {label.name}
                      </h4>
                    </div>
                  </div>
                  <div className="flex items-center space-x-4 text-sm text-gray-600 dark:text-gray-400">
                    <span>{label.emailCount} emails</span>
                    <span>•</span>
                    <span className="flex items-center space-x-1">
                      <LabelChip label={label} size="sm" />
                    </span>
                  </div>
                </div>
                <div className="flex items-center space-x-2 ml-4">
                  <button
                    onClick={() => handleEdit(label)}
                    className="p-2 text-gray-600 dark:text-gray-400 hover:text-primary-600 dark:hover:text-primary-400 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-md transition-colors"
                    title="Edit label"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                      />
                    </svg>
                  </button>
                  <button
                    onClick={() => handleDelete(label.id)}
                    className="p-2 text-gray-600 dark:text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/20 rounded-md transition-colors"
                    title="Delete label"
                  >
                    <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
                      />
                    </svg>
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      ) : (
        <div className="text-center py-12 bg-gray-50 dark:bg-gray-800/50 rounded-lg border border-gray-200 dark:border-gray-700">
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
              d="M7 7h.01M7 3h5c.512 0 1.024.195 1.414.586l7 7a2 2 0 010 2.828l-7 7a2 2 0 01-2.828 0l-7-7A1.994 1.994 0 013 12V7a4 4 0 014-4z"
            />
          </svg>
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
            No labels yet
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-4">
            Create your first label to start organizing emails
          </p>
          <button onClick={() => setIsCreating(true)} className="btn btn-primary">
            Create Label
          </button>
        </div>
      )}
    </div>
  );
}
