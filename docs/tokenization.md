# Tokenizer Approach

The chunking service operates behind the `Tokenizer` protocol defined in
`app/services/tokenization.py`. This keeps the chunking logic independent
of which tokenizer is actually used.

## Currently in use: `ApproximateTokenizer`

- **Method:** Unicode word/punctuation splitting (the `\w+|[^\w\s]` regex).
  Each word and each punctuation mark counts as one "token".
- **Why this approach was chosen:** At Phase 3, no embedding model had
  been selected yet (hardware/compatibility checks weren't done). A real
  model tokenizer (e.g. the HuggingFace tokenizer for BGE-M3,
  multilingual-E5, or tiktoken/BPE) requires both an extra dependency and,
  in some cases, downloading from the internet — which conflicts with the
  "as local as possible" goal. A dependency-free, fully deterministic,
  offline-capable approach was preferred instead.
- **Limitations:** This does not reproduce any real embedding/LLM model's
  subword vocabulary exactly. Token counts are directionally correct
  (more text → more tokens) but will not exactly match any real model's
  tokenizer output. This is a design choice, not a measured performance
  claim (see project rule 16).

## Swappability

`ChunkingService` accepts any object conforming to the `Tokenizer`
protocol (`count_tokens`, `token_boundaries`). Once an embedding model is
selected in Phase 4, a `Tokenizer` implementation wrapping that model's
real tokenizer (e.g. one based on HuggingFace `AutoTokenizer`) can be
injected without any change to the chunking logic itself.

## Chunking configurations

`app/services/chunking_presets.py` defines three named experimental
configurations: `300_50`, `500_75`, `800_100` (max_tokens/overlap_tokens).
These names are designed to be selectable from benchmark scripts.
