import { useState, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { api } from '@/services/api';
import { toast } from '@/store/toastStore';
import { useAuthStore } from '@/store/authStore';
import { useSignatureStore } from '@/store/signatureStore';
import { useDraftStore } from '@/store/draftStore';
import TemplatePicker from './TemplatePicker';
import RichTextEditor from './RichTextEditor';
import type { Email } from '@/types';
import type { EmailTemplate } from '@/store/templateStore';

type ComposeMode = 'new' | 'reply' | 'replyAll' | 'forward';

interface ComposeEmailProps {
  onClose: () => void;
  mode?: ComposeMode;
  originalEmail?: Email;
}

export default function ComposeEmail({ onClose, mode = 'new', originalEmail }: ComposeEmailProps) {
  const queryClient = useQueryClient();
  const user = useAuthStore((state) => state.user);
  const { getSignatureForAccount, signatures } = useSignatureStore();
  const { saveDraft, updateDraft, deleteDraft, currentDraftId, setCurrentDraft } = useDraftStore();

  const { data: accounts } = useQuery({
    queryKey: ['accounts'],
    queryFn: () => api.getAccounts(),
  });

  const defaultAccount = accounts?.find((acc) => acc.isDefault) || accounts?.[0];
  const accountSignatures = signatures.filter(sig => sig.accountId === defaultAccount?.id);
  const defaultSignature = defaultAccount ? getSignatureForAccount(defaultAccount.id) : null;
  const userSignature = defaultSignature?.content ? `\n\n${defaultSignature.content}` : (user?.signature ? `\n\n${user.signature}` : '');

  // Prepare initial form data based on mode
  const getInitialFormData = () => {
    // Try to restore draft for new emails
    if (mode === 'new' && !originalEmail) {
      const savedDraft = localStorage.getItem('email-draft');
      if (savedDraft) {
        try {
          const draft = JSON.parse(savedDraft);
          // Show toast to indicate draft was restored
          toast.info('Draft restored');
          return draft;
        } catch (e) {
          // Invalid draft, ignore
          console.error('Failed to restore draft:', e);
        }
      }
      return { to: '', cc: '', bcc: '', subject: '', body: userSignature };
    }

    if (!originalEmail) {
      return { to: '', cc: '', bcc: '', subject: '', body: mode === 'new' ? userSignature : '' };
    }

    const quotedBody = originalEmail.bodyText
      ? `\n\n---\nOn ${new Date(originalEmail.date).toLocaleString()}, ${originalEmail.from.address} wrote:\n\n${originalEmail.bodyText.split('\n').map(line => `> ${line}`).join('\n')}`
      : '';

    switch (mode) {
      case 'reply':
        return {
          to: originalEmail.from.address,
          cc: '',
          bcc: '',
          subject: originalEmail.subject.startsWith('Re:')
            ? originalEmail.subject
            : `Re: ${originalEmail.subject}`,
          body: quotedBody,
        };

      case 'replyAll':
        const ccAddresses = [
          ...originalEmail.to.filter(addr => addr.address !== originalEmail.from.address).map(addr => addr.address),
          ...(originalEmail.cc || []).map(addr => addr.address),
        ].join(', ');

        return {
          to: originalEmail.from.address,
          cc: ccAddresses,
          bcc: '',
          subject: originalEmail.subject.startsWith('Re:')
            ? originalEmail.subject
            : `Re: ${originalEmail.subject}`,
          body: quotedBody,
        };

      case 'forward':
        return {
          to: '',
          cc: '',
          bcc: '',
          subject: originalEmail.subject.startsWith('Fwd:')
            ? originalEmail.subject
            : `Fwd: ${originalEmail.subject}`,
          body: `\n\n---\nForwarded message:\nFrom: ${originalEmail.from.address}\nDate: ${new Date(originalEmail.date).toLocaleString()}\nSubject: ${originalEmail.subject}\nTo: ${originalEmail.to.map(t => t.address).join(', ')}\n\n${originalEmail.bodyText || ''}`,
        };

      default:
        return { to: '', cc: '', bcc: '', subject: '', body: '' };
    }
  };

  const [formData, setFormData] = useState(getInitialFormData());
  const [error, setError] = useState('');
  const [showCcBcc, setShowCcBcc] = useState(false);
  const [includeSignature, setIncludeSignature] = useState(mode === 'new');
  const [showTemplatePicker, setShowTemplatePicker] = useState(false);

  // Auto-save draft every 30 seconds
  useEffect(() => {
    const interval = setInterval(() => {
      // Only save if there's actual content
      if (formData.to || formData.subject || formData.body.trim() !== userSignature.trim()) {
        if (currentDraftId) {
          // Update existing draft
          updateDraft(currentDraftId, {
            to: formData.to,
            cc: formData.cc,
            bcc: formData.bcc,
            subject: formData.subject,
            body: formData.body,
            mode,
            inReplyTo: originalEmail?.id,
          });
        } else if (mode === 'new') {
          // Create new draft
          const draftId = saveDraft({
            to: formData.to,
            cc: formData.cc,
            bcc: formData.bcc,
            subject: formData.subject,
            body: formData.body,
            isHtml: true,
            mode,
            inReplyTo: originalEmail?.id,
          });
        }
      }
    }, 30000); // 30 seconds

    return () => clearInterval(interval);
  }, [formData, mode, originalEmail, currentDraftId, saveDraft, updateDraft, userSignature]);

  // Keyboard shortcuts
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      // Cmd/Ctrl+Enter to send
      if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (formData.to.trim() && formData.subject.trim()) {
          sendMutation.mutate(formData);
        } else {
          setError('Please fill in recipient and subject');
        }
      }

      // Cmd/Ctrl+K to open template picker
      if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        setShowTemplatePicker(true);
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [formData, sendMutation]);

  // Remove old query for accounts as it's now at the top
  const sendMutation = useMutation({
    mutationFn: (data: typeof formData) => {
      if (!defaultAccount) throw new Error('No email account configured');
      return api.sendEmail({
        accountId: defaultAccount.id,
        to: data.to.split(',').map((e) => e.trim()),
        subject: data.subject,
        body: data.body,
        cc: data.cc ? data.cc.split(',').map((e) => e.trim()) : undefined,
        bcc: data.bcc ? data.bcc.split(',').map((e) => e.trim()) : undefined,
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['emails'] });
      toast.success('Email sent successfully!');
      // Clear draft after successful send
      if (currentDraftId) {
        deleteDraft(currentDraftId);
        setCurrentDraft(null);
      }
      onClose();
    },
    onError: (err: any) => {
      const errorMessage = err.response?.data?.message || 'Failed to send email';
      setError(errorMessage);
      toast.error(errorMessage);
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!formData.to.trim()) {
      setError('Please enter at least one recipient');
      return;
    }

    if (!formData.subject.trim()) {
      setError('Please enter a subject');
      return;
    }

    sendMutation.mutate(formData);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement | HTMLTextAreaElement>) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const handleBodyChange = (html: string) => {
    setFormData({ ...formData, body: html });
  };

  const toggleSignature = () => {
    if (!defaultSignature) {
      toast.info('No signature configured for this account');
      return;
    }

    const signatureText = `\n\n${defaultSignature.content}`;

    if (includeSignature) {
      // Remove signature
      setFormData({
        ...formData,
        body: formData.body.replace(signatureText, '').trim(),
      });
      setIncludeSignature(false);
    } else {
      // Add signature
      setFormData({
        ...formData,
        body: formData.body.trim() + signatureText,
      });
      setIncludeSignature(true);
    }
  };

  const handleSelectTemplate = (template: EmailTemplate) => {
    // Apply template to form
    setFormData({
      ...formData,
      subject: template.subject,
      body: template.body + (includeSignature && defaultSignature ? `\n\n${defaultSignature.content}` : ''),
    });
    toast.success(`Template "${template.name}" applied`);
  };

  const handleDiscard = () => {
    // Only confirm if there's content
    if (formData.to || formData.subject || formData.body.trim() !== userSignature.trim()) {
      if (confirm('Discard this draft?')) {
        if (currentDraftId) {
          deleteDraft(currentDraftId);
          setCurrentDraft(null);
        }
        onClose();
      }
    } else {
      if (currentDraftId) {
        deleteDraft(currentDraftId);
        setCurrentDraft(null);
      }
      onClose();
    }
  };

  const isValid = formData.to.trim() !== '' && formData.subject.trim() !== '';

  const getTitle = () => {
    switch (mode) {
      case 'reply':
        return 'Reply';
      case 'replyAll':
        return 'Reply All';
      case 'forward':
        return 'Forward';
      default:
        return 'New Message';
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-3xl w-full max-h-[90vh] overflow-hidden flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <h2 className="text-xl font-semibold text-gray-900 dark:text-gray-100">{getTitle()}</h2>
          <button
            onClick={handleDiscard}
            className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 text-2xl"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="flex-1 flex flex-col overflow-hidden">
          <div className="flex-shrink-0 p-4 space-y-3 border-b border-gray-200 dark:border-gray-700">
            {/* From */}
            <div className="flex items-center">
              <label className="text-sm font-medium text-gray-700 dark:text-gray-300 w-16">From:</label>
              <span className="text-sm text-gray-900 dark:text-gray-100">
                {defaultAccount?.email || 'No account configured'}
              </span>
            </div>

            {/* To */}
            <div className="flex items-center">
              <label htmlFor="to" className="text-sm font-medium text-gray-700 dark:text-gray-300 w-16">
                To:
              </label>
              <div className="flex-1 flex items-center">
                <input
                  id="to"
                  name="to"
                  type="text"
                  required
                  className="flex-1 px-2 py-1 text-sm border-0 focus:outline-none focus:ring-0 bg-transparent text-gray-900 dark:text-gray-100"
                  value={formData.to}
                  onChange={handleChange}
                  placeholder="recipient@example.com"
                />
                {!showCcBcc && (
                  <button
                    type="button"
                    onClick={() => setShowCcBcc(true)}
                    className="text-xs text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 px-2"
                  >
                    Cc/Bcc
                  </button>
                )}
              </div>
            </div>

            {/* Cc */}
            {showCcBcc && (
              <div className="flex items-center">
                <label htmlFor="cc" className="text-sm font-medium text-gray-700 dark:text-gray-300 w-16">
                  Cc:
                </label>
                <input
                  id="cc"
                  name="cc"
                  type="text"
                  className="flex-1 px-2 py-1 text-sm border-0 focus:outline-none focus:ring-0 bg-transparent text-gray-900 dark:text-gray-100"
                  value={formData.cc}
                  onChange={handleChange}
                  placeholder="cc@example.com"
                />
              </div>
            )}

            {/* Bcc */}
            {showCcBcc && (
              <div className="flex items-center">
                <label htmlFor="bcc" className="text-sm font-medium text-gray-700 dark:text-gray-300 w-16">
                  Bcc:
                </label>
                <input
                  id="bcc"
                  name="bcc"
                  type="text"
                  className="flex-1 px-2 py-1 text-sm border-0 focus:outline-none focus:ring-0 bg-transparent text-gray-900 dark:text-gray-100"
                  value={formData.bcc}
                  onChange={handleChange}
                  placeholder="bcc@example.com"
                />
              </div>
            )}

            {/* Subject */}
            <div className="flex items-center">
              <label htmlFor="subject" className="text-sm font-medium text-gray-700 dark:text-gray-300 w-16">
                Subject:
              </label>
              <div className="flex-1">
                <input
                  id="subject"
                  name="subject"
                  type="text"
                  required
                  maxLength={100}
                  className="w-full px-2 py-1 text-sm border-0 focus:outline-none focus:ring-0 bg-transparent text-gray-900 dark:text-gray-100"
                  value={formData.subject}
                  onChange={handleChange}
                  placeholder="Email subject"
                />
                <div className={`text-xs mt-1 ${formData.subject.length > 60 ? 'text-yellow-600 dark:text-yellow-500' : 'text-gray-400 dark:text-gray-500'}`}>
                  {formData.subject.length}/100 {formData.subject.length > 60 && '(recommend < 60 chars)'}
                </div>
              </div>
            </div>
          </div>

          {/* Error Message */}
          {error && (
            <div className="mx-4 mt-4 p-3 bg-red-50 dark:bg-red-900/30 border border-red-200 dark:border-red-700 text-red-700 dark:text-red-400 rounded text-sm">
              {error}
            </div>
          )}

          {/* Body */}
          <div className="flex-1 overflow-y-auto">
            <RichTextEditor
              value={formData.body}
              onChange={handleBodyChange}
              placeholder="Compose your email..."
              minHeight="300px"
              className="h-full"
            />
          </div>

          {/* Footer */}
          <div className="flex-shrink-0 flex items-center justify-between p-4 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900">
            <div className="flex items-center space-x-2">
              <button
                type="submit"
                disabled={sendMutation.isPending || !isValid}
                className="btn btn-primary flex items-center space-x-2 disabled:opacity-50 disabled:cursor-not-allowed"
                title={isValid ? 'Send email (Ctrl/Cmd+Enter)' : 'Please fill in recipient and subject'}
              >
                {sendMutation.isPending ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
                    <span>Sending...</span>
                  </>
                ) : (
                  <>
                    <span>Send</span>
                    <span>✉️</span>
                  </>
                )}
              </button>
              <button
                type="button"
                onClick={handleDiscard}
                className="btn btn-secondary"
                disabled={sendMutation.isPending}
              >
                {mode === 'new' ? 'Discard' : 'Cancel'}
              </button>
              <button
                type="button"
                onClick={() => setShowTemplatePicker(true)}
                className="px-3 py-2 text-sm font-medium bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 rounded-md hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
                title="Use template (Ctrl+T)"
              >
                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                </svg>
              </button>
              {defaultSignature && (
                <button
                  type="button"
                  onClick={toggleSignature}
                  className={`px-3 py-2 text-sm font-medium rounded-md transition-colors ${
                    includeSignature
                      ? 'bg-primary-100 dark:bg-primary-900/30 text-primary-700 dark:text-primary-300 border border-primary-300 dark:border-primary-700'
                      : 'bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-300 border border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-700'
                  }`}
                  title={includeSignature ? 'Remove signature' : 'Add signature'}
                >
                  <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15.232 5.232l3.536 3.536m-2.036-5.036a2.5 2.5 0 113.536 3.536L6.5 21.036H3v-3.572L16.732 3.732z" />
                  </svg>
                </button>
              )}
            </div>

            {/* Template Picker Modal */}
            <TemplatePicker
              isOpen={showTemplatePicker}
              onClose={() => setShowTemplatePicker(false)}
              onSelectTemplate={handleSelectTemplate}
            />

            <div className="flex items-center space-x-4">
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {mode === 'new' && 'Auto-saving...'}
              </span>
              <span className="text-xs text-gray-500 dark:text-gray-400">
                {formData.body.length} chars
              </span>
              <span className="text-xs text-gray-400 dark:text-gray-500 hidden sm:inline">
                Ctrl/Cmd+Enter to send
              </span>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
