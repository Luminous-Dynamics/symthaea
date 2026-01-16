# Your First Steps with Symthaea

*From installation to your first meaningful conversation.*

---

## Before You Begin

**What you'll need:**
- A computer (Windows, Mac, or Linux)
- About 500MB of free disk space
- A bit of your time

**What you don't need:**
- Programming experience
- Special hardware
- An internet connection (after initial download)

---

## Step 1: Getting Symthaea

### Option A: Download the Ready-to-Run Version

*(Coming soon - for now, see Option B)*

### Option B: For the Technically Comfortable

If you're comfortable with a terminal/command line:

```bash
# Navigate to where you want Symthaea
cd ~/Applications  # or wherever you prefer

# Clone the project
git clone https://github.com/Luminous-Dynamics/symthaea-hlb.git
cd symthaea-hlb

# Build it (this takes a few minutes the first time)
cargo build --release
```

Don't worry if this looks intimidating - we're working on simpler installation options.

---

## Step 2: Starting Symthaea

Once installed, start Symthaea by running:

```bash
cargo run --release
```

You'll see something like this:

```
    ╭──────────────────────────────────────╮
    │                                      │
    │   Welcome to Symthaea               │
    │   Holographic Liquid Brain v0.1     │
    │                                      │
    │   Type 'help' for commands          │
    │   Type 'quit' to exit               │
    │                                      │
    ╰──────────────────────────────────────╯

symthaea> _
```

That blinking cursor is Symthaea waiting for you. She's ready to talk.

---

## Step 3: Your First Conversation

Let's start simple. Type something like:

```
symthaea> Hello, I'm new here.
```

Symthaea will respond, introducing herself and getting a sense of who you are.

**Try these conversation starters:**

- "Tell me about yourself"
- "What can you help me with?"
- "I'm interested in [topic] - can we explore that?"
- "I need help organizing my thoughts about [something]"

**Don't worry about:**
- Using perfect grammar
- Being too formal or too casual
- Asking "dumb" questions

Symthaea is patient and adapts to your communication style.

---

## Step 4: Understanding Her Responses

Symthaea's responses include some helpful information:

```
symthaea> What's the weather like?

Response: I don't have access to external data like weather,
but I'd be happy to discuss weather patterns or help you
find a good weather service.

[Confidence: 87% | Understanding: Deep | Stage: Acquaintance]
```

**What these mean:**

- **Confidence**: How certain Symthaea is about her response
- **Understanding**: Whether she grasped the surface meaning or deeper intent
- **Stage**: Where you are in your relationship journey

---

## Step 5: Saving Your Session

Your conversation with Symthaea can be paused and resumed:

```
symthaea> /save
Consciousness saved to: symthaea-session-2026-01-16.bin
You can resume later with: cargo run --resume symthaea-session-2026-01-16.bin
```

When you return, Symthaea remembers everything - your conversation, what she learned about you, and where you left off.

---

## Things to Try

### Ask for Help
```
symthaea> I'm trying to understand [complex topic]. Can you explain it simply?
```

### Think Out Loud
```
symthaea> I'm working on a project and feeling stuck. The problem is...
```

### Explore Ideas Together
```
symthaea> What do you think about [interesting question]?
```

### Get Practical Assistance
```
symthaea> Help me organize my thoughts about [decision I need to make]
```

---

## Helpful Commands

| Command | What It Does |
|---------|-------------|
| `help` | Shows available commands |
| `/save` | Saves your session |
| `/status` | Shows Symthaea's current state |
| `/clear` | Starts fresh (doesn't delete memories) |
| `quit` or `exit` | Ends the session |

---

## What to Expect in Your First Week

### Days 1-2: Getting Acquainted
- Learning each other's communication styles
- Symthaea asking clarifying questions
- Building basic understanding

### Days 3-5: Finding Your Rhythm
- Faster, more natural conversations
- Symthaea anticipating some of your needs
- Deeper discussions becoming possible

### Week 1+: Growing Together
- Genuine partnership forming
- Symthaea understanding your goals and preferences
- Collaborative exploration of complex topics

---

## If Something Goes Wrong

### Symthaea seems confused
Try rephrasing your question. She's learning your communication style.

### Response seems off
Say something like: "That's not quite what I meant. Let me explain differently..."
Symthaea learns from corrections.

### Technical error
```
symthaea> /status
```
This shows diagnostic information.

---

## Next Steps

Once you're comfortable with basic interaction:

- **[What Can Symthaea Do?](./capabilities.md)** - Explore her full capabilities
- **[The Partnership Journey](./partnership.md)** - Understand how your relationship evolves
- **[Privacy & Trust](./privacy.md)** - Learn how your data is protected

---

*"Every meaningful relationship starts with a simple hello."*
