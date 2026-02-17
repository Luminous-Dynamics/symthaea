import { useState } from 'react';
import { useQuery } from '@tanstack/react-query';
import { api } from '@/services/api';
import { useSignatureStore, type Signature } from '@/store/signatureStore';
import SignatureEditor from './SignatureEditor';
import { toast } from '@/store/toastStore';
import { formatDistanceToNow } from 'date-fns';

export default function SignatureManager() {
  const { signatures, deleteSignature, setDefaultSignature } = useSignatureStore();
  const [isEditorOpen, setIsEditorOpen] = useState(false);
  const [editingSignature, setEditingSignature] = useState<Signature | null>(null);

  const { data: accountsData } = useQuery({
    queryKey: ['accounts'],
    queryFn: () => api.getAccounts(),
  });

  const accounts = accountsData || [];

  const getAccountEmail = (accountId: string) => {
    return accounts.find((acc) => acc.id === accountId)?.email || 'Unknown';
  };

  const handleEdit = (signature: Signature) => {
    setEditingSignature(signature);
    setIsEditorOpen(true);
  };

  const handleDelete = (signature: Signature) => {
    if (confirm(`Delete signature "${signature.name}"?`)) {
      deleteSignature(signature.id);
      toast.success('Signature deleted');
    }
  };

  const handleSetDefault = (signature: Signature) => {
    setDefaultSignature(signature.accountId, signature.id);
    toast.success(`"${signature.name}" set as default`);
  };

  const handleNewSignature = () => {
    setEditingSignature(null);
    setIsEditorOpen(true);
  };

  const handleCloseEditor = () => {
    setIsEditorOpen(false);
    setEditingSignature(null);
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-semibold text-gray-900 dark:text-gray-100">
            Email Signatures
          </h2>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            Create and manage custom signatures for your email accounts
          </p>
        </div>
        <button onClick={handleNewSignature} className="btn btn-primary">
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
          New Signature
        </button>
      </div>

      {/* Signatures List */}
      {signatures.length === 0 ? (
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
            No signatures yet
          </h3>
          <p className="text-gray-600 dark:text-gray-400 mb-6">
            Create your first signature to personalize your emails
          </p>
          <button onClick={handleNewSignature} className="btn btn-primary">
            Create Signature
          </button>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {signatures.map((signature) => (
            <div
              key={signature.id}
              className="bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg p-4 hover:shadow-md transition-shadow"
            >
              {/* Header */}
              <div className="flex items-start justify-between mb-3">
                <div className="flex-1">
                  <h3 className="font-semibold text-gray-900 dark:text-gray-100 mb-1">
                    {signature.name}
                    {signature.isDefault && (
                      <span className="ml-2 inline-flex items-center px-2 py-0.5 rounded text-xs font-medium bg-primary-100 dark:bg-primary-900/30 text-primary-800 dark:text-primary-200">
                        Default
                      </span>
                    )}
                  </h3>
                  <p className="text-sm text-gray-600 dark:text-gray-400">
                    {getAccountEmail(signature.accountId)}
                  </p>
                </div>
                <div className="flex space-x-1">
                  <button
                    onClick={() => handleEdit(signature)}
                    className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                    title="Edit signature"
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
                    onClick={() => handleDelete(signature)}
                    className="p-1 text-gray-400 hover:text-red-600 dark:hover:text-red-400"
                    title="Delete signature"
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
                  {signature.content || 'No content'}
                </pre>
              </div>

              {/* Footer */}
              <div className="flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                <span>
                  Updated {formatDistanceToNow(new Date(signature.updatedAt), { addSuffix: true })}
                </span>
                {!signature.isDefault && (
                  <button
                    onClick={() => handleSetDefault(signature)}
                    className="text-primary-600 dark:text-primary-400 hover:underline"
                  >
                    Set as default
                  </button>
                )}
              </div>
            </div>
          ))}
        </div>
      )}

      {/* Editor Modal */}
      <SignatureEditor
        isOpen={isEditorOpen}
        onClose={handleCloseEditor}
        editingSignature={editingSignature}
      />
    </div>
  );
}
