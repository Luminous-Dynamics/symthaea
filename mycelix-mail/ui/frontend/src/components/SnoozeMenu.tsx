import { useState } from 'react';
import { useSnoozeStore, getSnoozeDate, type SnoozePreset } from '@/store/snoozeStore';

interface SnoozeMenuProps {
  emailId: string;
  currentFolderId: string;
  onSnooze?: () => void;
}

const snoozeOptions: Array<{ preset: SnoozePreset; label: string; icon: string }> = [
  { preset: 'later-today', label: 'Later today', icon: '🕐' },
  { preset: 'tomorrow', label: 'Tomorrow', icon: '📅' },
  { preset: 'this-weekend', label: 'This weekend', icon: '🎯' },
  { preset: 'next-week', label: 'Next week', icon: '📆' },
  { preset: 'custom', label: 'Custom...', icon: '🎨' },
];

export default function SnoozeMenu({ emailId, currentFolderId, onSnooze }: SnoozeMenuProps) {
  const { snoozeEmail } = useSnoozeStore();
  const [isOpen, setIsOpen] = useState(false);
  const [showCustomPicker, setShowCustomPicker] = useState(false);
  const [customDate, setCustomDate] = useState('');
  const [customTime, setCustomTime] = useState('09:00');

  const handleSnoozePreset = (preset: SnoozePreset) => {
    if (preset === 'custom') {
      setShowCustomPicker(true);
      return;
    }

    const snoozeDate = getSnoozeDate(preset);
    if (snoozeDate) {
      snoozeEmail(emailId, snoozeDate, currentFolderId);
      setIsOpen(false);
      onSnooze?.();
    }
  };

  const handleCustomSnooze = () => {
    if (!customDate || !customTime) {
      return;
    }

    const dateTime = new Date(`${customDate}T${customTime}`);
    if (dateTime > new Date()) {
      snoozeEmail(emailId, dateTime, currentFolderId);
      setIsOpen(false);
      setShowCustomPicker(false);
      onSnooze?.();
    }
  };

  const getPresetDescription = (preset: SnoozePreset): string => {
    const date = getSnoozeDate(preset);
    if (!date) return '';

    const today = new Date();
    const isToday = date.toDateString() === today.toDateString();

    if (isToday) {
      return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    return date.toLocaleDateString([], {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  if (showCustomPicker) {
    return (
      <div className="relative">
        <button
          onClick={() => {
            setShowCustomPicker(false);
            setIsOpen(false);
          }}
          className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
          title="Cancel snooze"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>

        <div className="absolute right-0 mt-2 w-80 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 z-50">
          <div className="p-4">
            <h3 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-3">
              Custom Snooze
            </h3>
            <div className="space-y-3">
              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Date
                </label>
                <input
                  type="date"
                  value={customDate}
                  onChange={(e) => setCustomDate(e.target.value)}
                  min={new Date().toISOString().split('T')[0]}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
              </div>
              <div>
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Time
                </label>
                <input
                  type="time"
                  value={customTime}
                  onChange={(e) => setCustomTime(e.target.value)}
                  className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                />
              </div>
              <div className="flex space-x-2 pt-2">
                <button
                  onClick={() => setShowCustomPicker(false)}
                  className="flex-1 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg transition-colors"
                >
                  Cancel
                </button>
                <button
                  onClick={handleCustomSnooze}
                  disabled={!customDate || !customTime}
                  className="flex-1 px-3 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-lg transition-colors"
                >
                  Snooze
                </button>
              </div>
            </div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="relative">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="p-2 text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
        title="Snooze email"
      >
        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            strokeWidth={2}
            d="M12 8v4l3 3m6-3a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
      </button>

      {isOpen && (
        <>
          {/* Overlay */}
          <div
            className="fixed inset-0 z-40"
            onClick={() => setIsOpen(false)}
          />

          {/* Menu */}
          <div className="absolute right-0 mt-2 w-64 bg-white dark:bg-gray-800 rounded-lg shadow-xl border border-gray-200 dark:border-gray-700 z-50">
            <div className="p-2">
              <div className="px-3 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Snooze until
              </div>
              {snoozeOptions.map((option) => (
                <button
                  key={option.preset}
                  onClick={() => handleSnoozePreset(option.preset)}
                  className="w-full flex items-center justify-between px-3 py-2.5 text-sm text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700 rounded-lg transition-colors"
                >
                  <span className="flex items-center space-x-3">
                    <span className="text-lg">{option.icon}</span>
                    <span className="font-medium">{option.label}</span>
                  </span>
                  {option.preset !== 'custom' && (
                    <span className="text-xs text-gray-500 dark:text-gray-400">
                      {getPresetDescription(option.preset)}
                    </span>
                  )}
                </button>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  );
}
