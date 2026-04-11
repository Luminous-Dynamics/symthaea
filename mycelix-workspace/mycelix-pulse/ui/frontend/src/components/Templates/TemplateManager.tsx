// Copyright (C) 2024-2026 Tristan Stoltz / Luminous Dynamics
// SPDX-License-Identifier: AGPL-3.0-or-later
// Commercial licensing: see COMMERCIAL_LICENSE.md at repository root
/**
 * Template Manager Component
 *
 * Email templates, text snippets, and signatures management
 */

import React, { useState, useEffect } from 'react';

interface EmailTemplate {
  id: string;
  name: string;
  description?: string;
  subject: string;
  bodyHtml: string;
  category?: string;
  fields: TemplateField[];
  isShared: boolean;
  useCount: number;
}

interface TemplateField {
  name: string;
  fieldType: string;
  label: string;
  placeholder?: string;
  defaultValue?: string;
  required: boolean;
}

interface TextSnippet {
  id: string;
  trigger: string;
  expansion: string;
  description?: string;
  category?: string;
  useCount: number;
}

interface Signature {
  id: string;
  name: string;
  contentHtml: string;
  isDefault: boolean;
}

type Tab = 'templates' | 'snippets' | 'signatures';

export default function TemplateManager() {
  const [activeTab, setActiveTab] = useState<Tab>('templates');

  return (
    <div className="h-full flex flex-col">
      <div className="border-b border-border">
        <div className="flex gap-4 px-6 pt-6">
          {(['templates', 'snippets', 'signatures'] as Tab[]).map((tab) => (
            <button
              key={tab}
              onClick={() => setActiveTab(tab)}
              className={`pb-3 px-2 border-b-2 capitalize ${
                activeTab === tab ? 'border-primary font-medium' : 'border-transparent text-muted'
              }`}
            >
              {tab}
            </button>
          ))}
        </div>
      </div>

      <div className="flex-1 overflow-y-auto p-6">
        {activeTab === 'templates' && <TemplatesTab />}
        {activeTab === 'snippets' && <SnippetsTab />}
        {activeTab === 'signatures' && <SignaturesTab />}
      </div>
    </div>
  );
}

function TemplatesTab() {
  const [templates, setTemplates] = useState<EmailTemplate[]>([]);
  const [loading, setLoading] = useState(true);
  const [showEditor, setShowEditor] = useState(false);
  const [editingTemplate, setEditingTemplate] = useState<EmailTemplate | null>(null);
  const [searchQuery, setSearchQuery] = useState('');
  const [categoryFilter, setCategoryFilter] = useState<string>('');

  useEffect(() => {
    fetchTemplates();
  }, []);

  const fetchTemplates = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/templates');
      if (response.ok) setTemplates(await response.json());
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this template?')) return;
    await fetch(`/api/templates/${id}`, { method: 'DELETE' });
    fetchTemplates();
  };

  const handleDuplicate = async (id: string) => {
    await fetch(`/api/templates/${id}/duplicate`, { method: 'POST' });
    fetchTemplates();
  };

  const categories = [...new Set(templates.map((t) => t.category).filter(Boolean))];

  const filteredTemplates = templates.filter((t) => {
    const matchesSearch = !searchQuery ||
      t.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      t.subject.toLowerCase().includes(searchQuery.toLowerCase());
    const matchesCategory = !categoryFilter || t.category === categoryFilter;
    return matchesSearch && matchesCategory;
  });

  if (showEditor) {
    return (
      <TemplateEditor
        template={editingTemplate}
        onSave={() => { setShowEditor(false); fetchTemplates(); }}
        onCancel={() => setShowEditor(false)}
      />
    );
  }

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <input
            type="text"
            placeholder="Search templates..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="px-3 py-2 border border-border rounded w-64"
          />
          <select
            value={categoryFilter}
            onChange={(e) => setCategoryFilter(e.target.value)}
            className="px-3 py-2 border border-border rounded"
          >
            <option value="">All Categories</option>
            {categories.map((cat) => (
              <option key={cat} value={cat}>{cat}</option>
            ))}
          </select>
        </div>
        <button
          onClick={() => { setEditingTemplate(null); setShowEditor(true); }}
          className="px-4 py-2 bg-primary text-white rounded"
        >
          New Template
        </button>
      </div>

      {loading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full" />
        </div>
      ) : filteredTemplates.length === 0 ? (
        <div className="text-center py-12 text-muted">
          {searchQuery || categoryFilter ? 'No templates match your search' : 'No templates yet. Create your first one!'}
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredTemplates.map((template) => (
            <div key={template.id} className="border border-border rounded-lg p-4 hover:shadow-md transition-shadow">
              <div className="flex items-start justify-between mb-2">
                <div>
                  <h3 className="font-semibold">{template.name}</h3>
                  {template.category && (
                    <span className="text-xs px-2 py-0.5 bg-muted/30 rounded">{template.category}</span>
                  )}
                </div>
                {template.isShared && <span className="text-xs text-primary">Shared</span>}
              </div>
              <p className="text-sm text-muted mb-2">{template.subject}</p>
              {template.description && (
                <p className="text-sm text-muted mb-3 line-clamp-2">{template.description}</p>
              )}
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted">{template.useCount} uses</span>
                <div className="flex gap-2">
                  <button onClick={() => handleDuplicate(template.id)} className="text-primary hover:underline">
                    Duplicate
                  </button>
                  <button onClick={() => { setEditingTemplate(template); setShowEditor(true); }} className="text-primary hover:underline">
                    Edit
                  </button>
                  <button onClick={() => handleDelete(template.id)} className="text-red-500 hover:underline">
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function TemplateEditor({
  template,
  onSave,
  onCancel,
}: {
  template: EmailTemplate | null;
  onSave: () => void;
  onCancel: () => void;
}) {
  const [name, setName] = useState(template?.name || '');
  const [description, setDescription] = useState(template?.description || '');
  const [subject, setSubject] = useState(template?.subject || '');
  const [bodyHtml, setBodyHtml] = useState(template?.bodyHtml || '');
  const [category, setCategory] = useState(template?.category || '');
  const [isShared, setIsShared] = useState(template?.isShared || false);
  const [saving, setSaving] = useState(false);

  const handleSave = async () => {
    setSaving(true);
    try {
      const method = template ? 'PUT' : 'POST';
      const url = template ? `/api/templates/${template.id}` : '/api/templates';

      await fetch(url, {
        method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, description, subject, body_html: bodyHtml, category, is_shared: isShared }),
      });

      onSave();
    } finally {
      setSaving(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-xl font-semibold">{template ? 'Edit Template' : 'New Template'}</h2>
        <button onClick={onCancel} className="text-muted hover:text-foreground">Cancel</button>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div>
          <label className="block text-sm font-medium mb-1">Name *</label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full px-3 py-2 border border-border rounded"
            placeholder="Follow-up email"
          />
        </div>
        <div>
          <label className="block text-sm font-medium mb-1">Category</label>
          <input
            type="text"
            value={category}
            onChange={(e) => setCategory(e.target.value)}
            className="w-full px-3 py-2 border border-border rounded"
            placeholder="Sales"
          />
        </div>
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">Description</label>
        <input
          type="text"
          value={description}
          onChange={(e) => setDescription(e.target.value)}
          className="w-full px-3 py-2 border border-border rounded"
          placeholder="Template for following up after meetings"
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">Subject *</label>
        <input
          type="text"
          value={subject}
          onChange={(e) => setSubject(e.target.value)}
          className="w-full px-3 py-2 border border-border rounded"
          placeholder="Following up on our conversation, {{recipient.firstName}}"
        />
      </div>

      <div>
        <label className="block text-sm font-medium mb-1">Body *</label>
        <textarea
          value={bodyHtml}
          onChange={(e) => setBodyHtml(e.target.value)}
          className="w-full px-3 py-2 border border-border rounded h-64 font-mono text-sm"
          placeholder="Hi {{recipient.firstName}},

Thank you for taking the time to meet with me today...

Best regards,
{{sender.firstName}}"
        />
        <p className="text-xs text-muted mt-1">
          Use {'{{field.name}}'} for merge fields. Available: recipient.firstName, recipient.lastName, recipient.email, recipient.company, sender.firstName, sender.lastName, sender.email, sender.signature, date.today
        </p>
      </div>

      <label className="flex items-center gap-2">
        <input type="checkbox" checked={isShared} onChange={(e) => setIsShared(e.target.checked)} />
        <span>Share with team</span>
      </label>

      <div className="flex justify-end gap-3">
        <button onClick={onCancel} className="px-4 py-2 border border-border rounded">Cancel</button>
        <button
          onClick={handleSave}
          disabled={!name || !subject || !bodyHtml || saving}
          className="px-4 py-2 bg-primary text-white rounded disabled:opacity-50"
        >
          {saving ? 'Saving...' : 'Save Template'}
        </button>
      </div>
    </div>
  );
}

function SnippetsTab() {
  const [snippets, setSnippets] = useState<TextSnippet[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [editingSnippet, setEditingSnippet] = useState<TextSnippet | null>(null);
  const [trigger, setTrigger] = useState('');
  const [expansion, setExpansion] = useState('');
  const [description, setDescription] = useState('');

  useEffect(() => {
    fetchSnippets();
  }, []);

  const fetchSnippets = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/snippets');
      if (response.ok) setSnippets(await response.json());
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    const method = editingSnippet ? 'PUT' : 'POST';
    const url = editingSnippet ? `/api/snippets/${editingSnippet.id}` : '/api/snippets';

    await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ trigger, expansion, description }),
    });

    setShowForm(false);
    setEditingSnippet(null);
    setTrigger('');
    setExpansion('');
    setDescription('');
    fetchSnippets();
  };

  const handleEdit = (snippet: TextSnippet) => {
    setEditingSnippet(snippet);
    setTrigger(snippet.trigger);
    setExpansion(snippet.expansion);
    setDescription(snippet.description || '');
    setShowForm(true);
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this snippet?')) return;
    await fetch(`/api/snippets/${id}`, { method: 'DELETE' });
    fetchSnippets();
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-muted">Type a trigger (e.g., ;sig) to expand text automatically</p>
        </div>
        <button
          onClick={() => setShowForm(!showForm)}
          className="px-4 py-2 bg-primary text-white rounded"
        >
          {showForm ? 'Cancel' : 'New Snippet'}
        </button>
      </div>

      {showForm && (
        <div className="border border-border rounded-lg p-4 space-y-4">
          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1">Trigger *</label>
              <input
                type="text"
                value={trigger}
                onChange={(e) => setTrigger(e.target.value)}
                className="w-full px-3 py-2 border border-border rounded"
                placeholder=";sig"
                maxLength={20}
              />
              <p className="text-xs text-muted mt-1">Short code to trigger expansion</p>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1">Description</label>
              <input
                type="text"
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                className="w-full px-3 py-2 border border-border rounded"
                placeholder="My email signature"
              />
            </div>
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Expansion *</label>
            <textarea
              value={expansion}
              onChange={(e) => setExpansion(e.target.value)}
              className="w-full px-3 py-2 border border-border rounded h-32"
              placeholder="Best regards,
John Doe
Product Manager"
            />
          </div>
          <div className="flex justify-end gap-3">
            <button onClick={() => { setShowForm(false); setEditingSnippet(null); }} className="px-4 py-2 border border-border rounded">
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={!trigger || !expansion}
              className="px-4 py-2 bg-primary text-white rounded disabled:opacity-50"
            >
              {editingSnippet ? 'Update' : 'Create'} Snippet
            </button>
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full" />
        </div>
      ) : snippets.length === 0 ? (
        <div className="text-center py-12 text-muted">
          No snippets yet. Create your first one!
        </div>
      ) : (
        <div className="divide-y divide-border rounded-lg border border-border">
          {snippets.map((snippet) => (
            <div key={snippet.id} className="p-4 hover:bg-muted/30">
              <div className="flex items-start justify-between">
                <div className="flex-1">
                  <div className="flex items-center gap-2 mb-1">
                    <code className="px-2 py-1 bg-muted/50 rounded text-sm font-mono">{snippet.trigger}</code>
                    {snippet.description && <span className="text-muted">- {snippet.description}</span>}
                  </div>
                  <p className="text-sm text-muted whitespace-pre-wrap line-clamp-2">{snippet.expansion}</p>
                </div>
                <div className="flex items-center gap-2 ml-4">
                  <span className="text-xs text-muted">{snippet.useCount} uses</span>
                  <button onClick={() => handleEdit(snippet)} className="text-primary hover:underline text-sm">
                    Edit
                  </button>
                  <button onClick={() => handleDelete(snippet.id)} className="text-red-500 hover:underline text-sm">
                    Delete
                  </button>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function SignaturesTab() {
  const [signatures, setSignatures] = useState<Signature[]>([]);
  const [loading, setLoading] = useState(true);
  const [showEditor, setShowEditor] = useState(false);
  const [editingSignature, setEditingSignature] = useState<Signature | null>(null);
  const [name, setName] = useState('');
  const [content, setContent] = useState('');

  useEffect(() => {
    fetchSignatures();
  }, []);

  const fetchSignatures = async () => {
    setLoading(true);
    try {
      const response = await fetch('/api/signatures');
      if (response.ok) setSignatures(await response.json());
    } finally {
      setLoading(false);
    }
  };

  const handleSave = async () => {
    const method = editingSignature ? 'PUT' : 'POST';
    const url = editingSignature ? `/api/signatures/${editingSignature.id}` : '/api/signatures';

    await fetch(url, {
      method,
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, content_html: content, is_default: signatures.length === 0 }),
    });

    setShowEditor(false);
    setEditingSignature(null);
    setName('');
    setContent('');
    fetchSignatures();
  };

  const handleSetDefault = async (id: string) => {
    await fetch(`/api/signatures/${id}/default`, { method: 'POST' });
    fetchSignatures();
  };

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this signature?')) return;
    await fetch(`/api/signatures/${id}`, { method: 'DELETE' });
    fetchSignatures();
  };

  const handleEdit = (sig: Signature) => {
    setEditingSignature(sig);
    setName(sig.name);
    setContent(sig.contentHtml);
    setShowEditor(true);
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div />
        <button
          onClick={() => setShowEditor(!showEditor)}
          className="px-4 py-2 bg-primary text-white rounded"
        >
          {showEditor ? 'Cancel' : 'New Signature'}
        </button>
      </div>

      {showEditor && (
        <div className="border border-border rounded-lg p-4 space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1">Name *</label>
            <input
              type="text"
              value={name}
              onChange={(e) => setName(e.target.value)}
              className="w-full px-3 py-2 border border-border rounded"
              placeholder="Work Signature"
            />
          </div>
          <div>
            <label className="block text-sm font-medium mb-1">Signature *</label>
            <textarea
              value={content}
              onChange={(e) => setContent(e.target.value)}
              className="w-full px-3 py-2 border border-border rounded h-32"
              placeholder="<p>Best regards,</p><p><strong>John Doe</strong></p><p>Product Manager | Acme Inc.</p>"
            />
            <p className="text-xs text-muted mt-1">HTML formatting supported</p>
          </div>
          <div className="flex justify-end gap-3">
            <button onClick={() => { setShowEditor(false); setEditingSignature(null); }} className="px-4 py-2 border border-border rounded">
              Cancel
            </button>
            <button
              onClick={handleSave}
              disabled={!name || !content}
              className="px-4 py-2 bg-primary text-white rounded disabled:opacity-50"
            >
              {editingSignature ? 'Update' : 'Create'} Signature
            </button>
          </div>
        </div>
      )}

      {loading ? (
        <div className="flex justify-center py-12">
          <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full" />
        </div>
      ) : signatures.length === 0 ? (
        <div className="text-center py-12 text-muted">
          No signatures yet. Create your first one!
        </div>
      ) : (
        <div className="space-y-4">
          {signatures.map((sig) => (
            <div key={sig.id} className="border border-border rounded-lg p-4">
              <div className="flex items-center justify-between mb-3">
                <div className="flex items-center gap-2">
                  <h3 className="font-semibold">{sig.name}</h3>
                  {sig.isDefault && (
                    <span className="px-2 py-0.5 bg-primary/10 text-primary rounded text-xs">Default</span>
                  )}
                </div>
                <div className="flex gap-2">
                  {!sig.isDefault && (
                    <button onClick={() => handleSetDefault(sig.id)} className="text-primary hover:underline text-sm">
                      Set Default
                    </button>
                  )}
                  <button onClick={() => handleEdit(sig)} className="text-primary hover:underline text-sm">
                    Edit
                  </button>
                  <button onClick={() => handleDelete(sig.id)} className="text-red-500 hover:underline text-sm">
                    Delete
                  </button>
                </div>
              </div>
              <div
                className="prose prose-sm max-w-none bg-muted/20 p-3 rounded"
                dangerouslySetInnerHTML={{ __html: sig.contentHtml }}
              />
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

// Template Picker Modal
export function TemplatePicker({
  onSelect,
  onClose,
}: {
  onSelect: (template: EmailTemplate) => void;
  onClose: () => void;
}) {
  const [templates, setTemplates] = useState<EmailTemplate[]>([]);
  const [loading, setLoading] = useState(true);
  const [search, setSearch] = useState('');

  useEffect(() => {
    fetch('/api/templates').then((r) => r.json()).then(setTemplates).finally(() => setLoading(false));
  }, []);

  const filtered = templates.filter((t) =>
    t.name.toLowerCase().includes(search.toLowerCase()) ||
    t.subject.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
      <div className="bg-background rounded-lg shadow-xl w-full max-w-2xl max-h-[80vh] flex flex-col">
        <div className="p-4 border-b border-border">
          <div className="flex items-center justify-between mb-3">
            <h2 className="text-lg font-semibold">Choose Template</h2>
            <button onClick={onClose} className="text-muted hover:text-foreground">X</button>
          </div>
          <input
            type="text"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Search templates..."
            className="w-full px-3 py-2 border border-border rounded"
            autoFocus
          />
        </div>
        <div className="flex-1 overflow-y-auto p-4">
          {loading ? (
            <div className="flex justify-center py-12">
              <div className="animate-spin h-8 w-8 border-2 border-primary border-t-transparent rounded-full" />
            </div>
          ) : filtered.length === 0 ? (
            <div className="text-center py-12 text-muted">No templates found</div>
          ) : (
            <div className="space-y-2">
              {filtered.map((template) => (
                <div
                  key={template.id}
                  className="p-3 border border-border rounded hover:bg-muted/30 cursor-pointer"
                  onClick={() => { onSelect(template); onClose(); }}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium">{template.name}</span>
                    {template.category && (
                      <span className="text-xs px-2 py-0.5 bg-muted/30 rounded">{template.category}</span>
                    )}
                  </div>
                  <p className="text-sm text-muted">{template.subject}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
