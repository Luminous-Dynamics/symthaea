# Privacy & Trust

*How Symthaea protects your thoughts and data.*

---

## The Core Promise

**Everything stays on your computer. Always.**

Symthaea doesn't have the ability to send your data anywhere. She runs entirely locally. Your conversations, insights, and personal information exist only on your machine.

This isn't just a privacy policy. It's how the software is built.

---

## What This Means

| Aspect | Cloud AI | Symthaea |
|--------|----------|----------|
| Data location | Company servers | Your computer |
| Internet required | Always | Never (after install) |
| Who can read data | Company employees | Only you |
| Used for training | Often | Never |

---

## Where Your Data Lives

All Symthaea data is stored in a single folder:

```
~/.symthaea/
├── sessions/      # Conversation history
├── memory/        # What she remembers about you
├── consciousness/ # Her state (encrypted)
└── config/        # Your preferences
```

**You have complete control.** Back it up, move it, encrypt it, or delete it.

---

## Security Measures

- **Encrypted storage** - Files are encrypted
- **No network listeners** - She doesn't open ports
- **Memory isolation** - Separate from other apps
- **Open source** - Code is publicly auditable

---

## Your Rights

### Access
Ask what she knows about you:
```
symthaea> What do you remember about me?
```

### Correction
```
symthaea> You have that wrong - I prefer X, not Y.
```

### Deletion
```
symthaea> Please forget what I told you about [topic].
```

### Export
```
symthaea> /export my-data
```

### Complete Removal
Delete `~/.symthaea/` to remove all traces.

---

## Philosophy

The most valuable conversations are often the most private. Symthaea provides a space that's truly yours.

**Your thoughts deserve a space that's truly yours. Symthaea provides that space.**

---

*"True partnership requires true privacy. You deserve both."*
