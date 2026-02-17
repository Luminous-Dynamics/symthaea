import { useState } from 'react';
import { useMutation, useQueryClient } from '@tanstack/react-query';
import { api } from '@/services/api';
import { EMAIL_PROVIDERS, getProviderById, type EmailProvider } from '@/data/emailProviders';

interface AccountWizardProps {
  onClose: () => void;
}

type Step = 'provider' | 'credentials' | 'settings' | 'testing';

export default function AccountWizard({ onClose }: AccountWizardProps) {
  const queryClient = useQueryClient();
  const [step, setStep] = useState<Step>('provider');
  const [selectedProvider, setSelectedProvider] = useState<EmailProvider | null>(null);
  const [formData, setFormData] = useState({
    email: '',
    password: '',
    imapHost: '',
    imapPort: 993,
    imapSecure: true,
    imapUser: '',
    imapPassword: '',
    smtpHost: '',
    smtpPort: 587,
    smtpSecure: true,
    smtpUser: '',
    smtpPassword: '',
  });
  const [error, setError] = useState('');
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle');

  const createAccountMutation = useMutation({
    mutationFn: (data: any) => api.createAccount(data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ['accounts'] });
      onClose();
    },
    onError: (err: any) => {
      setError(err.response?.data?.message || 'Failed to create account');
      setTestStatus('error');
    },
  });

  const handleProviderSelect = (provider: EmailProvider) => {
    setSelectedProvider(provider);

    if (provider.id !== 'custom') {
      setFormData((prev) => ({
        ...prev,
        imapHost: provider.imapHost,
        imapPort: provider.imapPort,
        imapSecure: provider.imapSecure,
        smtpHost: provider.smtpHost,
        smtpPort: provider.smtpPort,
        smtpSecure: provider.smtpSecure,
      }));
    }

    setStep('credentials');
  };

  const handleCredentialsSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');

    if (!formData.email || !formData.password) {
      setError('Please enter both email and password');
      return;
    }

    // Auto-fill IMAP/SMTP users with email if not set
    setFormData((prev) => ({
      ...prev,
      imapUser: prev.imapUser || prev.email,
      imapPassword: prev.imapPassword || prev.password,
      smtpUser: prev.smtpUser || prev.email,
      smtpPassword: prev.smtpPassword || prev.password,
    }));

    if (selectedProvider?.id === 'custom') {
      setStep('settings');
    } else {
      setStep('testing');
      testAndCreate();
    }
  };

  const handleSettingsSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setStep('testing');
    testAndCreate();
  };

  const testAndCreate = () => {
    setTestStatus('testing');

    // Simulate connection test (in real app, make a test API call)
    setTimeout(() => {
      createAccountMutation.mutate({
        email: formData.email,
        provider: selectedProvider?.name || 'Custom',
        imapHost: formData.imapHost,
        imapPort: formData.imapPort,
        imapSecure: formData.imapSecure,
        imapUser: formData.imapUser || formData.email,
        imapPassword: formData.imapPassword || formData.password,
        smtpHost: formData.smtpHost,
        smtpPort: formData.smtpPort,
        smtpSecure: formData.smtpSecure,
        smtpUser: formData.smtpUser || formData.email,
        smtpPassword: formData.smtpPassword || formData.password,
      });
    }, 1500);
  };

  const handleChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value, type, checked } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: type === 'checkbox' ? checked : type === 'number' ? parseInt(value) : value,
    }));
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <div className="bg-white rounded-lg shadow-xl max-w-2xl w-full max-h-[90vh] overflow-hidden">
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div>
            <h2 className="text-2xl font-semibold text-gray-900">Add Email Account</h2>
            <p className="text-sm text-gray-500 mt-1">
              {step === 'provider' && 'Choose your email provider'}
              {step === 'credentials' && 'Enter your email credentials'}
              {step === 'settings' && 'Configure server settings'}
              {step === 'testing' && 'Testing connection...'}
            </p>
          </div>
          <button
            onClick={onClose}
            className="text-gray-400 hover:text-gray-600 text-2xl"
            aria-label="Close"
          >
            ×
          </button>
        </div>

        {/* Progress Indicator */}
        <div className="flex items-center justify-center p-4 bg-gray-50 border-b border-gray-200">
          <div className="flex items-center space-x-2">
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              step === 'provider' ? 'bg-primary-600 text-white' : 'bg-green-500 text-white'
            }`}>
              1
            </div>
            <div className="w-12 h-1 bg-gray-300"></div>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              step === 'credentials' ? 'bg-primary-600 text-white' : step === 'provider' ? 'bg-gray-300 text-gray-600' : 'bg-green-500 text-white'
            }`}>
              2
            </div>
            {selectedProvider?.id === 'custom' && (
              <>
                <div className="w-12 h-1 bg-gray-300"></div>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
                  step === 'settings' ? 'bg-primary-600 text-white' : step === 'testing' ? 'bg-green-500 text-white' : 'bg-gray-300 text-gray-600'
                }`}>
                  3
                </div>
              </>
            )}
            <div className="w-12 h-1 bg-gray-300"></div>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm font-medium ${
              step === 'testing' ? 'bg-primary-600 text-white' : 'bg-gray-300 text-gray-600'
            }`}>
              {selectedProvider?.id === 'custom' ? '4' : '3'}
            </div>
          </div>
        </div>

        {/* Content */}
        <div className="p-6 max-h-[60vh] overflow-y-auto">
          {error && (
            <div className="mb-4 p-3 bg-red-50 border border-red-200 text-red-700 rounded text-sm">
              {error}
            </div>
          )}

          {/* Step 1: Provider Selection */}
          {step === 'provider' && (
            <div className="grid grid-cols-1 gap-3">
              {EMAIL_PROVIDERS.map((provider) => (
                <button
                  key={provider.id}
                  onClick={() => handleProviderSelect(provider)}
                  className="flex items-center p-4 border-2 border-gray-200 rounded-lg hover:border-primary-500 hover:bg-primary-50 transition-colors text-left"
                >
                  <div className="flex-1">
                    <h3 className="font-semibold text-gray-900">{provider.name}</h3>
                    <p className="text-sm text-gray-500">{provider.description}</p>
                  </div>
                  <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
                  </svg>
                </button>
              ))}
            </div>
          )}

          {/* Step 2: Credentials */}
          {step === 'credentials' && (
            <form onSubmit={handleCredentialsSubmit} className="space-y-4">
              {selectedProvider?.setupInstructions && (
                <div className="p-4 bg-blue-50 border border-blue-200 rounded-lg">
                  <h4 className="font-medium text-blue-900 mb-2">Setup Instructions</h4>
                  <div className="text-sm text-blue-700 whitespace-pre-line">
                    {selectedProvider.setupInstructions}
                  </div>
                </div>
              )}

              <div>
                <label htmlFor="email" className="block text-sm font-medium text-gray-700 mb-1">
                  Email Address
                </label>
                <input
                  id="email"
                  name="email"
                  type="email"
                  required
                  className="input"
                  value={formData.email}
                  onChange={handleChange}
                  placeholder="you@example.com"
                />
              </div>

              <div>
                <label htmlFor="password" className="block text-sm font-medium text-gray-700 mb-1">
                  Password {selectedProvider?.id !== 'custom' && '(App Password recommended)'}
                </label>
                <input
                  id="password"
                  name="password"
                  type="password"
                  required
                  className="input"
                  value={formData.password}
                  onChange={handleChange}
                  placeholder="••••••••"
                />
              </div>

              <div className="flex space-x-3 pt-4">
                <button
                  type="button"
                  onClick={() => setStep('provider')}
                  className="btn btn-secondary flex-1"
                >
                  Back
                </button>
                <button type="submit" className="btn btn-primary flex-1">
                  {selectedProvider?.id === 'custom' ? 'Next' : 'Test Connection'}
                </button>
              </div>
            </form>
          )}

          {/* Step 3: Custom Settings */}
          {step === 'settings' && (
            <form onSubmit={handleSettingsSubmit} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div className="col-span-2">
                  <h3 className="font-medium text-gray-900 mb-2">IMAP Settings (Incoming)</h3>
                </div>
                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    IMAP Server
                  </label>
                  <input
                    name="imapHost"
                    type="text"
                    required
                    className="input"
                    value={formData.imapHost}
                    onChange={handleChange}
                    placeholder="imap.example.com"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Port
                  </label>
                  <input
                    name="imapPort"
                    type="number"
                    required
                    className="input"
                    value={formData.imapPort}
                    onChange={handleChange}
                  />
                </div>
                <div className="flex items-center pt-6">
                  <input
                    id="imapSecure"
                    name="imapSecure"
                    type="checkbox"
                    className="mr-2"
                    checked={formData.imapSecure}
                    onChange={handleChange}
                  />
                  <label htmlFor="imapSecure" className="text-sm text-gray-700">
                    Use SSL/TLS
                  </label>
                </div>

                <div className="col-span-2 mt-4">
                  <h3 className="font-medium text-gray-900 mb-2">SMTP Settings (Outgoing)</h3>
                </div>
                <div className="col-span-2">
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    SMTP Server
                  </label>
                  <input
                    name="smtpHost"
                    type="text"
                    required
                    className="input"
                    value={formData.smtpHost}
                    onChange={handleChange}
                    placeholder="smtp.example.com"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Port
                  </label>
                  <input
                    name="smtpPort"
                    type="number"
                    required
                    className="input"
                    value={formData.smtpPort}
                    onChange={handleChange}
                  />
                </div>
                <div className="flex items-center pt-6">
                  <input
                    id="smtpSecure"
                    name="smtpSecure"
                    type="checkbox"
                    className="mr-2"
                    checked={formData.smtpSecure}
                    onChange={handleChange}
                  />
                  <label htmlFor="smtpSecure" className="text-sm text-gray-700">
                    Use SSL/TLS
                  </label>
                </div>
              </div>

              <div className="flex space-x-3 pt-4">
                <button
                  type="button"
                  onClick={() => setStep('credentials')}
                  className="btn btn-secondary flex-1"
                >
                  Back
                </button>
                <button type="submit" className="btn btn-primary flex-1">
                  Test Connection
                </button>
              </div>
            </form>
          )}

          {/* Step 4: Testing */}
          {step === 'testing' && (
            <div className="text-center py-8">
              {testStatus === 'testing' && (
                <>
                  <div className="animate-spin rounded-full h-16 w-16 border-b-4 border-primary-600 mx-auto mb-4"></div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Testing Connection</h3>
                  <p className="text-sm text-gray-500">
                    Verifying IMAP and SMTP settings...
                  </p>
                </>
              )}

              {testStatus === 'success' && (
                <>
                  <div className="w-16 h-16 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-10 h-10 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Success!</h3>
                  <p className="text-sm text-gray-500">
                    Your email account has been added successfully.
                  </p>
                </>
              )}

              {testStatus === 'error' && (
                <>
                  <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center mx-auto mb-4">
                    <svg className="w-10 h-10 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </div>
                  <h3 className="text-lg font-medium text-gray-900 mb-2">Connection Failed</h3>
                  <p className="text-sm text-gray-500 mb-4">
                    Please check your credentials and try again.
                  </p>
                  <button
                    onClick={() => {
                      setTestStatus('idle');
                      setStep('credentials');
                    }}
                    className="btn btn-primary"
                  >
                    Try Again
                  </button>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
