import { useRef, useEffect, useState, useCallback } from 'react';

interface RichTextEditorProps {
  value: string;
  onChange: (html: string) => void;
  placeholder?: string;
  className?: string;
  minHeight?: string;
}

type FormatCommand =
  | 'bold'
  | 'italic'
  | 'underline'
  | 'strikeThrough'
  | 'insertUnorderedList'
  | 'insertOrderedList'
  | 'formatBlock'
  | 'justifyLeft'
  | 'justifyCenter'
  | 'justifyRight'
  | 'removeFormat';

export default function RichTextEditor({
  value,
  onChange,
  placeholder = 'Type your message...',
  className = '',
  minHeight = '200px',
}: RichTextEditorProps) {
  const editorRef = useRef<HTMLDivElement>(null);
  const [isPlainText, setIsPlainText] = useState(false);
  const [showLinkDialog, setShowLinkDialog] = useState(false);
  const [linkUrl, setLinkUrl] = useState('');

  // Set initial content
  useEffect(() => {
    if (editorRef.current && editorRef.current.innerHTML !== value) {
      editorRef.current.innerHTML = value || '';
    }
  }, []);

  const handleInput = () => {
    if (editorRef.current) {
      onChange(editorRef.current.innerHTML);
    }
  };

  const executeCommand = (command: FormatCommand, value?: string) => {
    document.execCommand(command, false, value);
    editorRef.current?.focus();
    handleInput();
  };

  const handleBold = () => executeCommand('bold');
  const handleItalic = () => executeCommand('italic');
  const handleUnderline = () => executeCommand('underline');
  const handleStrikethrough = () => executeCommand('strikeThrough');
  const handleUnorderedList = () => executeCommand('insertUnorderedList');
  const handleOrderedList = () => executeCommand('insertOrderedList');
  const handleClearFormat = () => executeCommand('removeFormat');

  const handleHeading = (level: string) => {
    executeCommand('formatBlock', level);
  };

  const handleAlign = (alignment: 'left' | 'center' | 'right') => {
    const command = `justify${alignment.charAt(0).toUpperCase() + alignment.slice(1)}` as FormatCommand;
    executeCommand(command);
  };

  const handleLink = () => {
    const selection = window.getSelection();
    if (selection && selection.toString()) {
      setShowLinkDialog(true);
    } else {
      alert('Please select some text to create a link');
    }
  };

  const insertLink = () => {
    if (linkUrl) {
      const url = linkUrl.startsWith('http') ? linkUrl : `https://${linkUrl}`;
      document.execCommand('createLink', false, url);
      setShowLinkDialog(false);
      setLinkUrl('');
      editorRef.current?.focus();
      handleInput();
    }
  };

  const handleCodeBlock = () => {
    const selection = window.getSelection();
    if (selection && selection.toString()) {
      const code = `<code style="background: #f3f4f6; padding: 2px 6px; border-radius: 3px; font-family: monospace;">${selection.toString()}</code>`;
      document.execCommand('insertHTML', false, code);
      handleInput();
    }
  };

  const handleQuote = () => {
    executeCommand('formatBlock', 'blockquote');
  };

  const togglePlainText = () => {
    if (!editorRef.current) return;

    if (isPlainText) {
      // Convert plain text back to HTML
      const plainText = editorRef.current.innerText;
      editorRef.current.innerHTML = plainText.replace(/\n/g, '<br>');
    } else {
      // Convert HTML to plain text
      const plainText = editorRef.current.innerText;
      editorRef.current.innerText = plainText;
    }
    setIsPlainText(!isPlainText);
    handleInput();
  };

  // Keyboard shortcuts
  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.ctrlKey || e.metaKey) {
      switch (e.key.toLowerCase()) {
        case 'b':
          e.preventDefault();
          handleBold();
          break;
        case 'i':
          e.preventDefault();
          handleItalic();
          break;
        case 'u':
          e.preventDefault();
          handleUnderline();
          break;
        case 'k':
          e.preventDefault();
          handleLink();
          break;
      }
    }
  }, []);

  const ToolbarButton = ({
    onClick,
    icon,
    title,
    active = false
  }: {
    onClick: () => void;
    icon: React.ReactNode;
    title: string;
    active?: boolean;
  }) => (
    <button
      type="button"
      onClick={onClick}
      title={title}
      className={`p-2 rounded hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors ${
        active ? 'bg-gray-200 dark:bg-gray-600' : ''
      }`}
      onMouseDown={(e) => e.preventDefault()} // Prevent focus loss
    >
      {icon}
    </button>
  );

  return (
    <div className={`border border-gray-300 dark:border-gray-600 rounded-lg overflow-hidden ${className}`}>
      {/* Toolbar */}
      {!isPlainText && (
        <div className="bg-gray-100 dark:bg-gray-700 border-b border-gray-300 dark:border-gray-600 p-2 flex flex-wrap gap-1">
          {/* Text Formatting */}
          <div className="flex gap-1 pr-2 border-r border-gray-300 dark:border-gray-600">
            <ToolbarButton onClick={handleBold} title="Bold (Ctrl+B)" icon={<b>B</b>} />
            <ToolbarButton onClick={handleItalic} title="Italic (Ctrl+I)" icon={<i>I</i>} />
            <ToolbarButton onClick={handleUnderline} title="Underline (Ctrl+U)" icon={<u>U</u>} />
            <ToolbarButton onClick={handleStrikethrough} title="Strikethrough" icon={<s>S</s>} />
          </div>

          {/* Headings */}
          <div className="flex gap-1 pr-2 border-r border-gray-300 dark:border-gray-600">
            <ToolbarButton onClick={() => handleHeading('p')} title="Normal text" icon={<span className="text-sm">¶</span>} />
            <ToolbarButton onClick={() => handleHeading('h1')} title="Heading 1" icon={<span className="font-bold">H1</span>} />
            <ToolbarButton onClick={() => handleHeading('h2')} title="Heading 2" icon={<span className="font-bold text-sm">H2</span>} />
          </div>

          {/* Lists */}
          <div className="flex gap-1 pr-2 border-r border-gray-300 dark:border-gray-600">
            <ToolbarButton onClick={handleUnorderedList} title="Bullet list" icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            } />
            <ToolbarButton onClick={handleOrderedList} title="Numbered list" icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 20l4-16m2 16l4-16M6 9h14M4 15h14" />
              </svg>
            } />
          </div>

          {/* Alignment */}
          <div className="flex gap-1 pr-2 border-r border-gray-300 dark:border-gray-600">
            <ToolbarButton onClick={() => handleAlign('left')} title="Align left" icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h10M4 18h16" />
              </svg>
            } />
            <ToolbarButton onClick={() => handleAlign('center')} title="Align center" icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M7 12h10M4 18h16" />
              </svg>
            } />
            <ToolbarButton onClick={() => handleAlign('right')} title="Align right" icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M10 12h10M4 18h16" />
              </svg>
            } />
          </div>

          {/* Insert */}
          <div className="flex gap-1 pr-2 border-r border-gray-300 dark:border-gray-600">
            <ToolbarButton onClick={handleLink} title="Insert link (Ctrl+K)" icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13.828 10.172a4 4 0 00-5.656 0l-4 4a4 4 0 105.656 5.656l1.102-1.101m-.758-4.899a4 4 0 005.656 0l4-4a4 4 0 00-5.656-5.656l-1.1 1.1" />
              </svg>
            } />
            <ToolbarButton onClick={handleCodeBlock} title="Code" icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10 20l4-16m4 4l4 4-4 4M6 16l-4-4 4-4" />
              </svg>
            } />
            <ToolbarButton onClick={handleQuote} title="Quote" icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 8h10M7 12h4m1 8l-4-4H5a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v8a2 2 0 01-2 2h-3l-4 4z" />
              </svg>
            } />
          </div>

          {/* Utilities */}
          <div className="flex gap-1">
            <ToolbarButton onClick={handleClearFormat} title="Clear formatting" icon={
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
              </svg>
            } />
            <ToolbarButton
              onClick={togglePlainText}
              title={isPlainText ? "Switch to rich text" : "Switch to plain text"}
              icon={<span className="text-xs font-mono">{isPlainText ? 'RT' : 'TXT'}</span>}
              active={isPlainText}
            />
          </div>
        </div>
      )}

      {/* Editor */}
      <div
        ref={editorRef}
        contentEditable
        onInput={handleInput}
        onKeyDown={handleKeyDown}
        className="p-4 focus:outline-none bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 prose dark:prose-invert max-w-none"
        style={{ minHeight }}
        data-placeholder={placeholder}
        suppressContentEditableWarning
      />

      {/* Link Dialog */}
      {showLinkDialog && (
        <div className="fixed inset-0 bg-black/50 z-50 flex items-center justify-center p-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl p-6 max-w-md w-full">
            <h3 className="text-lg font-semibold mb-4 text-gray-900 dark:text-gray-100">Insert Link</h3>
            <input
              type="url"
              value={linkUrl}
              onChange={(e) => setLinkUrl(e.target.value)}
              placeholder="https://example.com"
              className="input w-full mb-4"
              autoFocus
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  insertLink();
                } else if (e.key === 'Escape') {
                  setShowLinkDialog(false);
                  setLinkUrl('');
                }
              }}
            />
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => {
                  setShowLinkDialog(false);
                  setLinkUrl('');
                }}
                className="btn btn-secondary"
              >
                Cancel
              </button>
              <button onClick={insertLink} className="btn btn-primary">
                Insert
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`
        [contenteditable]:empty:before {
          content: attr(data-placeholder);
          color: #9ca3af;
          pointer-events: none;
        }

        [contenteditable] h1 {
          font-size: 2em;
          font-weight: bold;
          margin: 0.5em 0;
        }

        [contenteditable] h2 {
          font-size: 1.5em;
          font-weight: bold;
          margin: 0.5em 0;
        }

        [contenteditable] blockquote {
          border-left: 4px solid #e5e7eb;
          padding-left: 1em;
          margin: 1em 0;
          color: #6b7280;
        }

        [contenteditable] code {
          background: #f3f4f6;
          padding: 2px 6px;
          border-radius: 3px;
          font-family: monospace;
        }

        [contenteditable] a {
          color: #2563eb;
          text-decoration: underline;
        }

        [contenteditable] ul, [contenteditable] ol {
          padding-left: 2em;
          margin: 0.5em 0;
        }
      `}</style>
    </div>
  );
}
