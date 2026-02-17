import { useState } from 'react';
import { useTemplateStore, type EmailTemplate, type TemplateCategory } from '@/store/templateStore';
import { toast } from '@/store/toastStore';
import { formatDistanceToNow } from 'date-fns';

const categoryIcons: Record<TemplateCategory, string> = {
  greeting: '👋',
  'follow-up': '📧',
  meeting: '📅',
  support: '🛠️',
  custom: '⭐',
};

export default function TemplateManager() {
  const { templates, addTemplate, updateTemplate, deleteTemplate } = useTemplateStore();
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<EmailTemplate | null>(null);
  const [formData, setFormData] = useState({
    name: '',
    subject: '',
    body: '',
    category: 'custom' as TemplateCategory,
    variables: [] as string[],
  });

  const handleEdit = (template: EmailTemplate) => {
    setEditingTemplate(template);
    setFormData({
      name: template.name,
      subject: template.subject,
      body: template.body,
      category: template.category,
      variables: template.variables,
    });
    setIsEditorOpen(true);
  };

  const handleDelete = (template: EmailTemplate) => {
    if (confirm(`Delete template "${template.name}"?`)) {
      deleteTemplate(template.id);
      toast.success('Template deleted');
    }
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.name.trim()) {
      toast.error('Please enter a template name');
      return;
    }

    // Extract variables from subject and body
    const variableRegex = /{{(.*?)}}/g;
    const variablesSet = new Set<string>();

    [...formData.subject.matchAll(variableRegex), ...formData.body.matchAll(variableRegex)].forEach((match) => {
      variablesSet.add(match[1].trim());
    });

    const extractedVariables = Array.from(variablesSet);

    if (editingTemplate) {
      updateTemplate(editingTemplate.id, { ...formData, variables: extractedVariables });
      toast.success('Template updated');
    } else {
      addTemplate({ ...formData, variables: extractedVariables });
      toast.success('Template created');
    }

    handleCloseEditor();
  };

  const handleNewTemplate = () => {
    setEditingTemplate(null);
    setFormData({
      name: '',
      subject: '',
      body: '',
      category: 'custom',
      variables: [],
    });
    setIsEditorOpen(true);
  };

  const handleCloseEditor = () => {
    setIsEditorOpen(false);
    setEditingTemplate(null);
  };

  const insertVariable = (variable: string) => {
    const textarea = document.getElementById('template-body') as HTMLTextAreaElement;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const text = formData.body;
    const before = text.substring(0, start);
    const after = text.substring(end);

    const newContent = before + `{{${variable}}}` + after;
    setFormData({ ...formData, body: newContent });

    // Set cursor position after inserted variable
    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(start + variable.length + 4, start + variable.length + 4);
    }, 0);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">
            Email Templates
          </h2>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Create quick reply templates for common emails
          </p>
        </div>
        <button onClick={handleNewTemplate} className="btn btn-primary">
          <svg
            className="w-5 h-5 mr-2"
            fill="none"
            stroke="currentColor"
            viewBox="0 0 24 24"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={2}
              d="M12 4v16m8-8H4"
            />
          </svg>
          New Template
        </button>
      </div>

      {/* Templates List */}
      {templates.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 dark:bg-gray-800 rounded-lg">
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
            No templates yet
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Create your first template to speed up email composition
          </p>
          <button onClick={handleNewTemplate} className="btn btn-primary">
            Create Template
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {templates.map((template) => (
            <div
              key={template.id}
              className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow"
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
                  <p className="text-xs text-gray-600 dark:text-gray-400">
                    {template.subject}
                  </p>
                </div>
                <div className="flex space-x-1">
                  <button
                    onClick={() => handleEdit(template)}
                    className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                    title="Edit template"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path
                        strokeLinecap="round"
                        strokeLinejoin="round"
                        strokeWidth={2}
                        d="M11 5H6a2 2 0 00-2 2v11a2 2 0 002 2h11a2 2 0 002-2v-5m-1.414-9.414a2 2 0 112.828 2.828L11.828 15H9v-2.828l8.586-8.586z"
                      />
                    </svg>
                  </button>
                  <button
                    onClick={() => handleDelete(template)}
                    className="p-1 text-gray-400 hover:text-red-600 dark:hover:text-red-400"
                    title="Delete template"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
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

              {/* Preview */}
              <div className="mb-3 p-3 bg-gray-50 dark:bg-gray-900 rounded border border-gray-200 dark:border-gray-700">
                <pre className="text-xs text-gray-700 dark:text-gray-300 whitespace-pre-wrap font-sans line-clamp-4">
                  {template.body || 'No content'}
                </pre>
              </div>

              {/* Stats */}
              <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                <div className="flex items-center space-x-2">
                  {template.useCount > 0 && (
                    <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">
                      Used {template.useCount}×
                    </span>
                  )}
                  {template.variables.length > 0 && (
                    <span className="px-2 py-1 bg-gray-100 dark:bg-gray-700 rounded">
                      {template.variables.length} vars
                    </span>
                  )}
                </div>
                {template.lastUsed && (
                  <span>
                    {formatDistanceToNow(new Date(template.lastUsed), { addSuffix: true })}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Editor Modal */}
      {isEditorOpen && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
              <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">
                {editingTemplate ? 'Edit Template' : 'New Template'}
              </h2>
              <button
                onClick={handleCloseEditor}
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

            {/* Form */}
            <form onSubmit={handleSubmit} className="flex-1 overflow-y-auto p-6">
              <div className="space-y-6">
                {/* Name & Category */}
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Template Name *
                    </label>
                    <input
                      type="text"
                      value={formData.name}
                      onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                      placeholder="e.g., Quick Thanks"
                    />
                  </div>
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                      Category
                    </label>
                    <select
                      value={formData.category}
                      onChange={(e) => setFormData({ ...formData, category: e.target.value as TemplateCategory })}
                      className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    >
                      <option value="greeting">👋 Greeting</option>
                      <option value="follow-up">📧 Follow-up</option>
                      <option value="meeting">📅 Meeting</option>
                      <option value="support">🛠️ Support</option>
                      <option value="custom">⭐ Custom</option>
                    </select>
                  </div>
                </div>

                {/* Subject */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Subject Line
                  </label>
                  <input
                    type="text"
                    value={formData.subject}
                    onChange={(e) => setFormData({ ...formData, subject: e.target.value })}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                    placeholder="e.g., Thank you for {{topic}}"
                  />
                </div>

                {/* Variables Helper */}
                <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Insert Variables:
                  </p>
                  <div className="flex flex-wrap gap-2">
                    {['name', 'email', 'date', 'time', 'topic', 'company'].map((variable) => (
                      <button
                        key={variable}
                        type="button"
                        onClick={() => insertVariable(variable)}
                        className="px-3 py-1 text-xs font-medium bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300"
                      >
                        {`{{${variable}}}`}
                      </button>
                    ))}
                  </div>
                </div>

                {/* Body */}
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    Email Body
                  </label>
                  <textarea
                    id="template-body"
                    value={formData.body}
                    onChange={(e) => setFormData({ ...formData, body: e.target.value })}
                    rows={12}
                    className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 font-mono text-sm"
                    placeholder="Hi {{name}},&#10;&#10;Thank you for...&#10;&#10;Best regards"
                  />
                  <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                    Use variables like {`{{name}}`} which will need to be replaced when using the template
                  </p>
                </div>
              </div>
            </form>

            {/* Footer */}
            <div className="flex justify-end space-x-3 p-6 border-t border-gray-200 dark:border-gray-700">
              <button
                type="button"
                onClick={handleCloseEditor}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleSubmit}
                className="btn btn-primary"
              >
                {editingTemplate ? 'Update Template' : 'Create Template'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
