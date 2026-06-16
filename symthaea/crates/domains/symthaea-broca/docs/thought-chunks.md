# Thought Chunks

Thought Chunks are the experimental alternative to next-token training.

Instead of asking Broca to predict the next token directly, the branch predicts
the next semantic HDC vector, grouped into variable-length chunks. A lightweight
decoder then turns each chunk into text, code, an action, or structured data.

## Contract

- `ThoughtChunk`: one semantic unit with an HDC vector, output kind, psi,
  confidence, optional token span, and optional target string.
- `ThoughtChunkSequence`: a variable-length list of chunks for one source
  example.
- `ThoughtChunkDecoder`: trait for decoders that map chunks to external forms.

## Training Hypothesis

High-psi thought chunks should be easier for the motor language system to learn
than long fixed BPTT windows because the target is semantic continuity first,
surface realization second.

## First Experiment

1. Convert canonical/training pairs into one or more `ThoughtChunk`s.
2. Train a predictor from current thought HV plus previous chunk HV to next chunk
   HV.
3. Train a small decoder per `ThoughtChunkKind`.
4. Evaluate chunk cosine similarity, decoder validity, and canonical text
   quality separately.

The current token trainer remains the production path until this branch beats it
on canonical eval and structured validity.
