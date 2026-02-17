import { useState, useEffect } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';
import { useSignatureStore, type Signature } from '@/store/signatureStore';
import { toast } from '@/store/toastStore';

interface SignatureEditorProps {
  isOpen: boolean;
  onClose: () => void;
  editingSignature?: Signature | null;
}

export default function SignatureEditor({
  isOpen,
  onClose,
  editingSignature,
}: SignatureEditorProps) {
  const { addSignature, updateSignature } = useSignatureStore();
  const { data: accountsData } = useQuery({
    queryKey: ['accounts'],
    queryFn: () => api.getAccounts(),
  });

  const accounts = accountsData || [];

  const [formData, setFormData] = useState({
    name: '',
    accountId: '',
    content: '',
    isDefault: false,
  });

  useEffect(() => {
    if (editingSignature) {
      setFormData({
        name: editingSignature.name,
        accountId: editingSignature.accountId,
        content: editingSignature.content,
        isDefault: editingSignature.isDefault,
      });
    } else {
      setFormData({
        name: '',
        accountId: accounts[0]?.id || '',
        content: '',
        isDefault: false,
      });
    }
  }, [editingSignature, isOpen, accounts]);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    if (!formData.name.trim()) {
      toast.error('Please enter a signature name');
      return;
    }

    if (!formData.accountId) {
      toast.error('Please select an email account');
      return;
    }

    if (editingSignature) {
      updateSignature(editingSignature.id, formData);
      toast.success('Signature updated');
    } else {
      addSignature(formData);
      toast.success('Signature created');
    }

    onClose();
  };

  const insertVariable = (variable: string) => {
    const textarea = document.getElementById('signature-content') as HTMLTextAreaElement;
    if (!textarea) return;

    const start = textarea.selectionStart;
    const end = textarea.selectionEnd;
    const text = formData.content;
    const before = text.substring(0, start);
    const after = text.substring(end);

    const newContent = before + `{{${variable}}}` + after;
    setFormData({ ...formData, content: newContent });

    // Set cursor position after inserted variable
    setTimeout(() => {
      textarea.focus();
      textarea.setSelectionRange(start + variable.length + 4, start + variable.length + 4);
    }, 0);
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">
            {editingSignature ? 'Edit Signature' : 'New Signature'}
          </h2>
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

        {/* Form */}
        <form onSubmit={handleSubmit} className="flex-1 overflow-y-auto p-6">
          <div className="space-y-6">
            {/* Name */}
            <div>
              <label
                htmlFor="name"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
              >
                Signature Name *
              </label>
              <input
                type="text"
                id="name"
                value={formData.name}
                onChange={(e) => setFormData({ ...formData, name: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
                placeholder="e.g., Professional, Casual, Support"
              />
            </div>

            {/* Account */}
            <div>
              <label
                htmlFor="account"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
              >
                Email Account *
              </label>
              <select
                id="account"
                value={formData.accountId}
                onChange={(e) => setFormData({ ...formData, accountId: e.target.value })}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100"
              >
                <option value="">Select an account</option>
                {accounts.map((account) => (
                  <option key={account.id} value={account.id}>
                    {account.email}
                  </option>
                ))}
              </select>
            </div>

            {/* Variables Helper */}
            <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Insert Variables:
              </p>
              <div className="flex flex-wrap gap-2">
                {['name', 'email', 'date', 'time', 'company'].map((variable) => (
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

            {/* Content */}
            <div>
              <label
                htmlFor="signature-content"
                className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2"
              >
                Signature Content
              </label>
              <textarea
                id="signature-content"
                value={formData.content}
                onChange={(e) => setFormData({ ...formData, content: e.target.value })}
                rows={10}
                className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg focus:outline-none focus:ring-2 focus:ring-primary-500 bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 font-mono text-sm"
                placeholder="Best regards,&#10;John Doe&#10;{{email}}&#10;{{company}}"
              />
              <p className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                Use variables like {`{{name}}`} which will be replaced with actual values
              </p>
            </div>

            {/* Preview */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Preview
              </label>
              <div className="border border-gray-300 dark:border-gray-600 rounded-lg p-4 bg-gray-50 dark:bg-gray-900 min-h-[100px]">
                <pre className="whitespace-pre-wrap font-sans text-sm text-gray-900 dark:text-gray-100">
                  {formData.content || 'Your signature preview will appear here...'}
                </pre>
              </div>
            </div>

            {/* Default Checkbox */}
            <div className="flex items-center">
              <input
                type="checkbox"
                id="isDefault"
                checked={formData.isDefault}
                onChange={(e) => setFormData({ ...formData, isDefault: e.target.checked })}
                className="rounded border-gray-300 dark:border-gray-600 text-primary-600 focus:ring-primary-500"
              />
              <label
                htmlFor="isDefault"
                className="ml-2 text-sm text-gray-700 dark:text-gray-300"
              >
                Set as default signature for this account
              </label>
            </div>
          </div>
        </form>

        {/* Footer */}
        <div className="flex justify-end space-x-3 p-6 border-t border-gray-200 dark:border-gray-700">
          <button
            type="button"
            onClick={onClose}
            className="btn btn-secondary"
          >
            Cancel
          </button>
          <button
            onClick={handleSubmit}
            className="btn btn-primary"
          >
            {editingSignature ? 'Update Signature' : 'Create Signature'}
          </button>
        </div>
      </div>
    </div>
  );
}
