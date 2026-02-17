import { useState } from 'react';
import { useTemplateStore, type EmailTemplate, type TemplateCategory } from '@/store/templateStore';

interface TemplatePickerProps {
  isOpen: boolean;
  onClose: () => void;
  onSelectTemplate: (template: EmailTemplate) => void;
}

const categoryIcons: Record<TemplateCategory, string> = {
  greeting: '👋',
  'follow-up': '📧',
  meeting: '📅',
  support: '🛠️',
  custom: '⭐',
};

const categoryLabels: Record<TemplateCategory, string> = {
  greeting: 'Greetings',
  'follow-up': 'Follow-ups',
  meeting: 'Meetings',
  support: 'Support',
  custom: 'Custom',
};

export default function TemplatePicker({ isOpen, onClose, onSelectTemplate }: TemplatePickerProps) {
  const { templates, getTemplatesByCategory, getPopularTemplates, incrementUseCount } = useTemplateStore();
  const [activeCategory, setActiveCategory] = useState<'all' | TemplateCategory>('all');
  const [searchQuery, setSearchQuery] = useState('');

  const handleSelectTemplate = (template: EmailTemplate) => {
    incrementUseCount(template.id);
    onSelectTemplate(template);
    onClose();
  };

  // Filter templates
  const filteredTemplates = templates.filter((template) => {
    const matchesCategory = activeCategory === 'all' || template.category === activeCategory;
    const matchesSearch =
      !searchQuery ||
      template.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.subject.toLowerCase().includes(searchQuery.toLowerCase()) ||
      template.body.toLowerCase().includes(searchQuery.toLowerCase());

    return matchesCategory && matchesSearch;
  });

  const popularTemplates = getPopularTemplates(3);
  const categories: Array<'all' | TemplateCategory> = ['all', 'greeting', 'follow-up', 'meeting', 'support', 'custom'];

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-4xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">
              Email Templates
            </h2>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              Select a template to quickly compose your email
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            aria-label="Close"
          >
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M6 18L18 6M6 6l12 12"
              />
            </svg>
          </button>
        </div>

        {/* Search */}
        <div className="p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="relative">
            <input
              type="text"
              placeholder="Search templates..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
            />
            <svg
              className="absolute left-3 top-1/2 transform -translate-y-1/2 w-4 h-4 text-gray-400"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M21 21l-6-6m2-5a7 7 0 11-14 0 7 7 0 0114 0z"
              />
            </svg>
          </div>
        </div>

        {/* Category Tabs */}
        <div className="flex items-center space-x-2 px-4 py-3 border-b border-gray-200 dark:border-gray-700 overflow-x-auto">
          {categories.map((category) => (
            <button
              key={category}
              onClick={() => setActiveCategory(category)}
              className={`px-4 py-2 text-sm font-medium rounded-md whitespace-nowrap transition-colors ${
                activeCategory === category
                  ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300'
                  : 'text-gray-600 dark:text-gray-400 hover:bg-gray-100 dark:hover:bg-gray-700'
              }`}
            >
              {category === 'all' ? '📋 All' : `${categoryIcons[category]} ${categoryLabels[category]}`}
            </button>
          ))}
        </div>

        {/* Popular Templates Section */}
        {activeCategory === 'all' && !searchQuery && popularTemplates.length > 0 && (
          <div className="p-4 bg-gray-50 dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700">
            <h3 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
              ⭐ Most Used
            </h3>
            <div className="flex flex-wrap gap-2">
              {popularTemplates.map((template) => (
                <button
                  key={template.id}
                  onClick={() => handleSelectTemplate(template)}
                  className="px-3 py-2 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors text-left"
                >
                  <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                    {template.name}
                  </div>
                  <div className="text-xs text-gray-500 dark:text-gray-400">
                    Used {template.useCount} times
                  </div>
                </button>
              ))}
            </div>
          </div>
        )}

        {/* Templates List */}
        <div className="flex-1 overflow-y-auto p-4">
          {filteredTemplates.length === 0 ? (
            <div className="text-center py-12">
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
                  d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                />
              </svg>
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-2">
                No templates found
              </h3>
              <p className="text-gray-600 dark:text-gray-400">
                {searchQuery ? 'Try a different search term' : 'No templates in this category'}
              </p>
            </div>
          ) : (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {filteredTemplates.map((template) => (
                <div
                  key={template.id}
                  onClick={() => handleSelectTemplate(template)}
                  className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md hover:border-primary-300 dark:hover:border-primary-700 transition-all cursor-pointer group"
                >
                  {/* Header */}
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex-1">
                      <div className="flex items-center space-x-2 mb-1">
                        <span className="text-lg">{categoryIcons[template.category]}</span>
                        <h3 className="font-semibold text-gray-900 dark:text-gray-100">
                          {template.name}
                        </h3>
                      </div>
                      <p className="text-xs text-gray-600 dark:text-gray-400 font-medium">
                        Subject: {template.subject}
                      </p>
                    </div>
                    <svg
                      className="w-5 h-5 text-gray-400 group-hover:text-primary-600 dark:group-hover:text-primary-400 transition-colors"
                      fill="none"
                      stroke="currentColor"
                      viewBox="0 0 24 24"
                    >
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M9 5l7 7-7 7"
                      />
                    </svg>
                  </div>

                  {/* Preview */}
                  <div className="mb-3">
                    <pre className="text-xs text-gray-700 dark:text-gray-300 whitespace-pre-wrap font-sans line-clamp-3 bg-gray-50 dark:bg-gray-900 p-2 rounded">
                      {template.body.substring(0, 150)}...
                    </pre>
                  </div>

                  {/* Footer */}
                  <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                    <div className="flex flex-wrap gap-1">
                      {template.variables.length > 0 && (
                        <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">
                          {template.variables.length} variables
                        </span>
                      )}
                      {template.useCount > 0 && (
                        <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">
                          Used {template.useCount}×
                        </span>
                      )}
                    </div>
                    <span className="text-primary-600 dark:text-primary-400 opacity-0 group-hover:opacity-100 transition-opacity">
                      Click to use
                    </span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
          <p className="text-xs text-gray-600 dark:text-gray-400 text-center">
            <span className="font-medium">Tip:</span> Variables like {`{{name}}`} will need to be replaced with actual values
          </p>
        </div>
      </div>
    </div>
  );
}
