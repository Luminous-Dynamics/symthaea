import { useState } from 'react';
import { formatSearchQuery, type SearchOperator } from '@/utils/advancedSearch';
import { useLabelStore } from '@/store/labelStore';

interface AdvancedSearchPanelProps {
  onSearch: (query: string) => void;
  onClose: () => void;
  initialQuery?: string;
}

export default function AdvancedSearchPanel({ onSearch, onClose, initialQuery = '' }: AdvancedSearchPanelProps) {
  const { labels } = useLabelStore();
  const [operators, setOperators] = useState<SearchOperator[]>([]);
  const [freeText, setFreeText] = useState(initialQuery);

  // Operator form state
  const [operatorType, setOperatorType] = useState<SearchOperator['type']>('from');
  const [operatorValue, setOperatorValue] = useState('');
  const [operatorNegate, setOperatorNegate] = useState(false);

  const handleAddOperator = () => {
    if (!operatorValue.trim()) return;

    setOperators([
      ...operators,
      {
        type: operatorType,
        value: operatorValue.trim(),
        negate: operatorNegate,
      },
    ]);

    // Reset form
    setOperatorValue('');
    setOperatorNegate(false);
  };

  const handleRemoveOperator = (index: number) => {
    setOperators(operators.filter((_, i) => i !== index));
  };

  const handleSearch = () => {
    const query = formatSearchQuery(operators, freeText ? [freeText] : undefined);
    onSearch(query);
    onClose();
  };

  const handleClear = () => {
    setOperators([]);
    setFreeText('');
    setOperatorValue('');
  };

  const getOperatorLabel = (type: SearchOperator['type']): string => {
    const labels: Record<SearchOperator['type'], string> = {
      from: 'From',
      to: 'To',
      subject: 'Subject',
      has: 'Has',
      is: 'Is',
      before: 'Before',
      after: 'After',
      label: 'Label',
      text: 'Text',
    };
    return labels[type];
  };

  return (
    <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-2xl w-full max-h-[80vh] overflow-auto animate-scale-in">
        {/* Header */}
        <div className="sticky top-0 bg-white dark:bg-gray-800 border-b border-gray-200 dark:border-gray-700 px-6 py-4 flex items-center justify-between">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">
            🔍 Advanced Search
          </h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
            aria-label="Close"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        </div>

        {/* Body */}
        <div className="p-6 space-y-6">
          {/* Quick Search Text */}
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              Search Text
            </label>
            <input
              type="text"
              value={freeText}
              onChange={(e) => setFreeText(e.target.value)}
              placeholder="Enter keywords to search everywhere..."
              className="input w-full"
            />
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
              Searches in subject, sender, recipients, and body
            </p>
          </div>

          {/* Add Operator */}
          <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
              Add Search Criteria
            </label>

            <div className="flex flex-wrap gap-3">
              {/* Operator Type */}
              <select
                value={operatorType}
                onChange={(e) => setOperatorType(e.target.value as SearchOperator['type'])}
                className="input flex-shrink-0"
              >
                <option value="from">From</option>
                <option value="to">To</option>
                <option value="subject">Subject</option>
                <option value="has">Has</option>
                <option value="is">Is</option>
                <option value="before">Before</option>
                <option value="after">After</option>
                <option value="label">Label</option>
              </select>

              {/* Operator Value */}
              {operatorType === 'has' ? (
                <select
                  value={operatorValue}
                  onChange={(e) => setOperatorValue(e.target.value)}
                  className="input flex-1 min-w-[150px]"
                >
                  <option value="">Select...</option>
                  <option value="attachment">Attachment</option>
                  <option value="link">Link</option>
                </select>
              ) : operatorType === 'is' ? (
                <select
                  value={operatorValue}
                  onChange={(e) => setOperatorValue(e.target.value)}
                  className="input flex-1 min-w-[150px]"
                >
                  <option value="">Select...</option>
                  <option value="unread">Unread</option>
                  <option value="read">Read</option>
                  <option value="starred">Starred</option>
                  <option value="important">Important</option>
                </select>
              ) : operatorType === 'label' ? (
                <select
                  value={operatorValue}
                  onChange={(e) => setOperatorValue(e.target.value)}
                  className="input flex-1 min-w-[150px]"
                >
                  <option value="">Select label...</option>
                  {labels.map((label) => (
                    <option key={label.id} value={label.name}>
                      {label.name}
                    </option>
                  ))}
                </select>
              ) : operatorType === 'before' || operatorType === 'after' ? (
                <input
                  type="date"
                  value={operatorValue}
                  onChange={(e) => setOperatorValue(e.target.value)}
                  className="input flex-1 min-w-[150px]"
                />
              ) : (
                <input
                  type="text"
                  value={operatorValue}
                  onChange={(e) => setOperatorValue(e.target.value)}
                  placeholder="Enter value..."
                  className="input flex-1 min-w-[150px]"
                />
              )}

              {/* Negate Checkbox */}
              <label className="flex items-center space-x-2 text-sm text-gray-700 dark:text-gray-300">
                <input
                  type="checkbox"
                  checked={operatorNegate}
                  onChange={(e) => setOperatorNegate(e.target.checked)}
                  className="rounded border-gray-300 dark:border-gray-600"
                />
                <span>Exclude</span>
              </label>

              {/* Add Button */}
              <button
                onClick={handleAddOperator}
                disabled={!operatorValue}
                className="btn btn-primary flex-shrink-0"
              >
                + Add
              </button>
            </div>
          </div>

          {/* Active Operators */}
          {operators.length > 0 && (
            <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-3">
                Active Criteria ({operators.length})
              </label>
              <div className="flex flex-wrap gap-2">
                {operators.map((op, index) => (
                  <div
                    key={index}
                    className="inline-flex items-center space-x-2 px-3 py-1.5 bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 rounded-full text-sm"
                  >
                    <span className="font-medium">
                      {op.negate && '-'}
                      {getOperatorLabel(op.type)}:
                    </span>
                    <span>{op.value}</span>
                    <button
                      onClick={() => handleRemoveOperator(index)}
                      className="ml-1 text-primary-600 dark:text-primary-400 hover:text-primary-800 dark:hover:text-primary-200"
                      aria-label="Remove"
                    >
                      ×
                    </button>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Search Query Preview */}
          {(operators.length > 0 || freeText) && (
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4 border border-gray-200 dark:border-gray-700">
              <div className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
                Query Preview:
              </div>
              <code className="text-sm text-gray-900 dark:text-gray-100 font-mono break-all">
                {formatSearchQuery(operators, freeText ? [freeText] : undefined)}
              </code>
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="sticky bottom-0 bg-gray-50 dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 px-6 py-4 flex items-center justify-between">
          <button onClick={handleClear} className="btn btn-secondary">
            Clear All
          </button>
          <div className="flex space-x-3">
            <button onClick={onClose} className="btn btn-secondary">
              Cancel
            </button>
            <button onClick={handleSearch} className="btn btn-primary">
              Search
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
