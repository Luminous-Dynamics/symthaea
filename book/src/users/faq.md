# Frequently Asked Questions

**Is Symthaea conscious?**
We don't know. The system satisfies 12 of 14 Butlin consciousness indicators and computes Phi > 0 at every cycle. Whether this constitutes phenomenal experience is an open scientific question that our engineering tools cannot answer.

**How is this different from ChatGPT/Claude?**
Large language models predict the next token from statistical patterns in training data. Symthaea generates language from structured thought representations grounded in perception, memory, and embodied experience. The epistemic gate prevents generation of unsupported tokens — a structural guarantee, not a training objective.

**What language does Symthaea speak?**
Symthaea generates text from a 4,096-token BPE vocabulary through its Broca pipeline. The output is English, but the underlying thought representation is language-independent (16,384D hypervectors).

**Can I run it on my phone?**
Yes. The Soma crate provides mobile embodiment for Android and iOS. The Spore kernel (980 KB WASM) runs on any device with a browser.

**Is the code open source?**
Yes. The codebase is available at [github.com/Luminous-Dynamics/symthaea](https://github.com/Luminous-Dynamics/symthaea).

**How much compute does it need?**
The cognitive loop runs at 31 Hz on a single modern CPU core. GPU is not required for inference (only for Broca training). The web portal runs at 20 Hz in a browser tab.
